"""
parallel.py — Parallel prototype update kernel for GTSOM.

Provides ``update_prototypes_kernel``, which performs the batch SOM
scatter-accumulate update rule. If numba is available, a JIT-compiled
parallel kernel is used (``PARALLEL = True``). Otherwise a pure-numpy
fallback with identical semantics is used (``PARALLEL = False``).

The numba kernel uses pre-allocated thread-local accumulators
(shape ``(n_threads, M, d)``) to avoid race conditions during the
parallel scatter, mirroring the ``parallelReduce`` + ``join()`` pattern
from the original C++ implementation. Thread count is determined at
runtime by ``numba.get_num_threads()``.

Imports
-------
This module is imported by ``gtsom.py`` as::

    from .parallel import update_prototypes_kernel, PARALLEL

The caller never needs to know which implementation is active.

Also exports ``resolve_n_jobs`` for converting the user-facing ``n_jobs``
parameter to a concrete thread count, and ``set_num_threads`` (when numba
is available) for applying it.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Attempt numba import
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange, get_num_threads, set_num_threads, get_thread_id
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Thread count resolver — used by GTSOM.__init__() and fit()
# ---------------------------------------------------------------------------

def resolve_n_jobs(n_jobs):
    """
    Resolve ``n_jobs`` to a concrete positive thread count.

    Parameters
    ----------
    n_jobs : int or None
        ``None`` or ``-1`` → all logical CPU cores.
        Positive int → that many threads, clamped to ``os.cpu_count()``.
        ``1`` → single-threaded.

    Returns
    -------
    int
        Resolved thread count, always >= 1.
    """
    max_cores = os.cpu_count() or 1
    if n_jobs is None or n_jobs == -1:
        return max_cores
    return max(1, min(int(n_jobs), max_cores))


# ---------------------------------------------------------------------------
# Numba path
# ---------------------------------------------------------------------------

if _NUMBA_AVAILABLE:

    @njit(parallel=True, nogil=True)
    def _update_W_numba(
        X,              # (N, d)  float64 — data matrix
        order,          # (N,)    int64   — argsort of BMU
        boundaries,     # (n_unique+1,) int64 — slice boundaries per unique BMU
        unique_bmus,    # (n_unique,)   int64 — sorted unique BMU indices
        nbr_indptr,     # (M+1,)  int32  — CSR row pointers of nbr_W
        nbr_indices,    # (nnz,)  int32  — CSR column indices of nbr_W
        nbr_data,       # (nnz,)  float64 — CSR values of nbr_W
        W,              # (M, d)  float64 — prototype matrix (modified in place)
    ):
        """
        Parallel batch SOM prototype update via true thread-local accumulators.

        Thread count is read from numba at runtime via get_num_threads().
        Each parallel iteration uses get_thread_id() to index into its own
        slice of the pre-allocated (n_threads, M, d) accumulator, guaranteeing
        no write conflicts. This mirrors the C++ parallelReduce + join() pattern.

        For each unique BMU b (processed in parallel across threads):
          1. Accumulate X_b_sum = sum of X[i] for all i whose BMU is b.
          2. For each neighbor j of b (from CSR row b of nbr_W):
               thread_sum_HX[tid, j] += nbr_W[b, j] * X_b_sum
               thread_sum_H[tid, j]  += nbr_W[b, j] * n_b

        After all BMUs are processed, reduce thread-local arrays into
        global sum_HX / sum_H and update W where sum_H > 0.
        """
        N, d     = X.shape
        M        = W.shape[0]
        n_unique = unique_bmus.shape[0]
        n_threads = get_num_threads()

        # Thread-local accumulators — indexed by true thread ID
        # Shape (n_threads, M, d) and (n_threads, M) — each thread owns one slice
        thread_sum_HX = np.zeros((n_threads, M, d), dtype=np.float64)
        thread_sum_H  = np.zeros((n_threads, M),    dtype=np.float64)

        # Parallel scatter over unique BMUs.
        # get_thread_id() returns the true ID of the executing thread (0..n_threads-1),
        # guaranteed unique per concurrently executing iteration — no race conditions.
        for ui in prange(n_unique):
            tid   = get_thread_id()
            b     = unique_bmus[ui]
            start = boundaries[ui]
            end   = boundaries[ui + 1]
            n_b   = end - start

            # Accumulate X_b_sum over this BMU's data slice
            X_b_sum = np.zeros(d, dtype=np.float64)
            for k in range(start, end):
                row = order[k]
                for dim in range(d):
                    X_b_sum[dim] += X[row, dim]

            # Scatter to neighbors via CSR row b
            row_start = nbr_indptr[b]
            row_end   = nbr_indptr[b + 1]
            for nz in range(row_start, row_end):
                j = nbr_indices[nz]
                h = nbr_data[nz]
                for dim in range(d):
                    thread_sum_HX[tid, j, dim] += h * X_b_sum[dim]
                thread_sum_H[tid, j] += h * n_b

        # Serial reduction across threads -> global accumulators
        sum_HX = np.zeros((M, d), dtype=np.float64)
        sum_H  = np.zeros(M,      dtype=np.float64)
        for t in range(n_threads):
            for j in range(M):
                sum_H[j] += thread_sum_H[t, j]
                for dim in range(d):
                    sum_HX[j, dim] += thread_sum_HX[t, j, dim]

        # Update W where support exists
        for j in range(M):
            if sum_H[j] > 0.0:
                for dim in range(d):
                    W[j, dim] = sum_HX[j, dim] / sum_H[j]

    def update_prototypes_kernel(X, BMU, nbr_W, W):
        """
        Parallel prototype update kernel (numba backend).

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
        BMU : np.ndarray, shape (N,), dtype int
            First BMU index for each datum.
        nbr_W : scipy.sparse.csr_matrix, shape (M, M)
            Neighbourhood weight matrix.
        W : np.ndarray, shape (M, d)
            Prototype matrix, updated in place. May be float32 or float64;
            computation is always done in float64 and cast back on exit.
        """
        W_dtype = W.dtype   # remember original dtype for cast-back

        # Group data points by BMU via argsort
        order       = np.argsort(BMU).astype(np.int64)
        sorted_bmu  = BMU[order]
        unique_bmus, first_occ = np.unique(sorted_bmu, return_index=True)
        boundaries  = np.empty(len(unique_bmus) + 1, dtype=np.int64)
        boundaries[:-1] = first_occ
        boundaries[-1]  = len(BMU)

        # Extract CSR arrays as contiguous int32/float64 for numba
        indptr  = nbr_W.indptr.astype(np.int32)
        indices = nbr_W.indices.astype(np.int32)
        data    = nbr_W.data.astype(np.float64)

        # Both X and W must be float64 inside the kernel.
        # W_f64 is a fresh array; the kernel writes into it, then we
        # cast back to the original dtype and copy into W in-place.
        X_f64 = np.ascontiguousarray(X, dtype=np.float64)
        W_f64 = W.astype(np.float64)   # copy — kernel writes here

        _update_W_numba(
            X_f64,
            order,
            boundaries,
            unique_bmus,
            indptr,
            indices,
            data,
            W_f64,
        )

        # Cast result back to original dtype and write into W in-place
        np.copyto(W, W_f64.astype(W_dtype))

    PARALLEL = True
    _BACKEND = 'numba'


# ---------------------------------------------------------------------------
# Numpy fallback path
# ---------------------------------------------------------------------------

else:

    def update_prototypes_kernel(X, BMU, nbr_W, W):
        """
        Serial prototype update kernel (numpy fallback).

        Identical semantics to the numba backend but runs on a single thread.
        Used when numba is not installed.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
        BMU : np.ndarray, shape (N,), dtype int
        nbr_W : scipy.sparse.csr_matrix, shape (M, M)
        W : np.ndarray, shape (M, d)
            Updated in place.
        """
        M, d = W.shape

        sum_HX = np.zeros((M, d), dtype=np.float64)
        sum_H  = np.zeros(M,      dtype=np.float64)

        order       = np.argsort(BMU)
        sorted_bmu  = BMU[order]
        unique_bmus, first_occ = np.unique(sorted_bmu, return_index=True)
        boundaries  = np.empty(len(unique_bmus) + 1, dtype=np.intp)
        boundaries[:-1] = first_occ
        boundaries[-1]  = len(BMU)

        for i, b in enumerate(unique_bmus):
            X_b     = X[order[boundaries[i]:boundaries[i + 1]]]
            nbr_row = nbr_W.getrow(b)
            js      = nbr_row.indices
            hs      = nbr_row.data

            if js.size == 0:
                continue

            X_b_sum = X_b.sum(axis=0)
            n_b     = X_b.shape[0]

            sum_HX[js] += hs[:, None] * X_b_sum[None, :]
            sum_H[js]  += hs * n_b

        active = sum_H > 0.0
        W[active] = (sum_HX[active] / sum_H[active, None]).astype(W.dtype)

    PARALLEL = False
    _BACKEND = 'numpy'