import numpy as np
from scipy.sparse import csr_matrix


__all__ = ["NeighborKernel"]


class NeighborKernel:
    """
    Computes sparse neighborhood activation weights for SOM prototype updates.

    At each learning step, the kernel takes the pairwise distance matrix
    between prototypes (from Embedding.dist) and a per-prototype bandwidth
    vector rho, and returns a sparse matrix of neighborhood influence weights.

    The kernel function is set at construction and applied uniformly via
    compute(). Per-prototype rho values allow spatially adaptive bandwidths,
    though in practice a single global rho (broadcast to length M) is typical.

    Parameters
    ----------
    kind : str, default 'exp'
        Kernel function to use. Currently supported: 'exp' (exponential decay).
        Future options: 'gaussian'.
    h_min : float, default 0.01
        Minimum activation threshold. Weights below this value are set to
        zero and excluded from the sparse output. Controls the sparsity of
        nbr_w — lower h_min means broader (denser) neighborhoods.

    Attributes
    ----------
    kind : str
        Kernel function identifier.
    h_min : float
        Activation threshold.

    Examples
    --------
    >>> kernel = NeighborKernel(kind='exp', h_min=0.01)
    >>> nbr_w = kernel.compute(embedding.dist, rho=3.0)
    >>> nbr_w.shape
    (M, M)
    """

    SUPPORTED_KINDS = {"exp"}

    def __init__(self, kind="exp", h_min=0.01):
        if kind not in self.SUPPORTED_KINDS:
            raise ValueError(
                f"kind '{kind}' not supported. "
                f"Choose from {self.SUPPORTED_KINDS}."
            )
        if not (0.0 < h_min < 1.0):
            raise ValueError(
                f"h_min must be in (0, 1), got {h_min}."
            )
        self.kind = kind
        self.h_min = h_min

    def compute(self, dist, rho):
        """
        Compute sparse neighborhood activation weights.

        For each prototype i, computes the activation weight for every
        other prototype j using the kernel function applied to dist[i, j]
        and rho[i]. Weights below h_min are zeroed out, and the result
        is returned as a sparse CSR matrix.

        Row i of the returned matrix contains the nonzero activation
        weights of prototype i's neighborhood. The nonzero column indices
        give the active neighbor prototypes, making a separate nbr_idx
        array unnecessary.

        Parameters
        ----------
        dist : np.ndarray, shape (M, M)
            Pairwise prototype distance matrix (from Embedding.dist).
        rho : float or array-like of shape (M,)
            Neighborhood bandwidth. A scalar is broadcast to all M
            prototypes. An array allows per-prototype bandwidths.

        Returns
        -------
        nbr_w : scipy.sparse.csr_matrix, shape (M, M)
            Sparse matrix of neighborhood activation weights.
            nbr_w[i] gives the activation weights for prototype i's
            active neighbors. Self-activation (diagonal) is included.
        """
        M = dist.shape[0]
        rho = self._broadcast_rho(rho, M)

        if self.kind == "exp":
            W = self._exp_kernel(dist, rho)

        W[W < self.h_min] = 0.0
        return csr_matrix(W)

    # ------------------------------------------------------------------
    # Kernel functions
    # ------------------------------------------------------------------

    def _exp_kernel(self, dist, rho):
        """
        Exponential decay kernel.

        W[i, j] = exp(-dist[i, j] / rho[i])

        Parameters
        ----------
        dist : np.ndarray, shape (M, M)
        rho : np.ndarray, shape (M,)

        Returns
        -------
        W : np.ndarray, shape (M, M)
        """
        return np.exp(-dist / rho[:, None])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _broadcast_rho(self, rho, M):
        """
        Ensure rho is a float64 array of length M.

        Scalars are broadcast to all M prototypes. Arrays are validated
        to have length M.

        Parameters
        ----------
        rho : float or array-like
        M : int

        Returns
        -------
        rho : np.ndarray, shape (M,)
        """
        rho = np.asarray(rho, dtype=np.float64)
        if rho.ndim == 0:
            return np.full(M, rho.item())
        if rho.shape != (M,):
            raise ValueError(
                f"rho must be a scalar or array of length M={M}, "
                f"got shape {rho.shape}."
            )
        return rho

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self):
        return f"NeighborKernel(kind='{self.kind}', h_min={self.h_min})"