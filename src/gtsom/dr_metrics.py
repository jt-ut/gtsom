"""
dr_metrics.py — Dimensionality Reduction quality metrics for the gtsom package.

Provides a standalone function :func:`compute_dr_metrics` that evaluates how
faithfully a low-dimensional embedding preserves the structure of a
high-dimensional point set. Designed for use with SOM prototype vectors and
their 2-D output-space coordinates, but applicable to any (high-d, low-d) pair.

The primary entry point is :func:`compute_dr_metrics`, which returns a
:class:`DRMetricsResult` dataclass. This function can be used independently of
GTSOM — it requires only the high-d matrix, the low-d matrix, and optionally
a CONN adjacency matrix for the CONN_WAFL metric.

Metrics overview
----------------
Two families of metrics are computed:

**Co-ranking family** (requires ``pyDRMetrics``):
  Q_local, Q_global   — neighbourhood preservation at local and global scales
  Q_AUC               — preservation across all scales simultaneously
  LCMC_AUC            — chance-corrected Q_AUC (most interpretable scalar)
  Trust_AUC           — absence of false neighbours, averaged over all scales

**Folding metric** (requires embed_geodesic_dist and CONN; numpy/scipy only):
  CONN_WAFL           — weighted average embedding hops per high-d CONN edge
  CONN_WAFL_SE        — standard error of CONN_WAFL

Soft dependency
---------------
Co-ranking metrics require the ``pyDRMetrics`` package::

    pip install pyDRMetrics

CONN_WAFL requires only numpy and scipy, which are already core dependencies
of the gtsom package.

References
----------
Zhang, Y. et al. pyDRMetrics — A Python toolkit for dimensionality reduction
quality assessment. Heliyon, 2021.
https://github.com/zhangys11/pyDRMetrics

Tasdemir, K. & Merényi, E. (2009). Exploiting data topology in visualization
and clustering of self-organizing maps. IEEE Transactions on Neural Networks.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np


__all__ = ["DRMetricsResult", "compute_dr_metrics"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DRMetricsResult:
    """
    Container for dimensionality reduction quality metrics.

    All values are floats. Fields that could not be computed (e.g. CONN_WAFL
    when no CONN was provided, or co-ranking metrics when pyDRMetrics is not
    installed) are stored as ``None``.

    The co-ranking metrics (Q_local through Trust_AUC) are derived from the
    co-ranking matrix of the (high-d, low-d) point pair. They measure how
    well the embedding preserves neighbourhood structure at various scales.
    All co-ranking metrics are in [0, 1]; higher is better.

    Attributes
    ----------
    Q_local : float or None
        Local neighbourhood preservation, averaged over neighbourhood sizes
        k ≤ kmax, where kmax is the scale at which LCMC is maximised (a
        data-driven split between local and global scales). Measures whether
        the nearest neighbours in high-d are also near neighbours in low-d.
        Higher is better.

    Q_global : float or None
        Global neighbourhood preservation, averaged over neighbourhood sizes
        k > kmax. Measures whether points that are far apart in high-d are
        also far apart in low-d. Complements Q_local: an embedding can score
        well locally but fold or compress at larger scales (or vice versa).
        Higher is better.

    Q_AUC : float or None
        Mean QNN over all neighbourhood sizes k — the area under the QNN(k)
        curve (computed as a simple mean, mirroring pyDRMetrics). Summarises
        neighbourhood preservation across all scales simultaneously. Higher
        is better.

    LCMC_AUC : float or None
        Chance-corrected AUC of the Local Continuity Meta-Criterion (LCMC).

        LCMC(k) = QNN(k) - k / (M - 1)

        The subtracted term is the expected QNN under a random embedding, so
        LCMC(k) measures how much better than chance the embedding is at scale
        k. LCMC_AUC integrates LCMC over all k and normalises by the
        theoretical maximum AUC (integral of 1 - k/K), giving a score in
        [0, 1] where 1 = perfect preservation and 0 = chance level. This is
        the most interpretable single scalar summary of embedding quality.

    Trust_AUC : float or None
        Trustworthiness, averaged over all neighbourhood sizes k. T(k)
        measures the absence of false neighbours introduced by the embedding:
        points that appear as k-nearest neighbours in low-d but are *not*
        k-nearest neighbours in high-d. T(k) = 1 is perfect (no false
        neighbours at scale k); T(k) < 1 indicates intrusions. T(k) is
        typically monotonically decreasing with k (intrusions become
        inevitable at large k), so no local/global split is applied — the
        mean over all k is the natural summary. Higher is better.

    CONN_WAFL : float or None
        Weighted Average Folding Length, computed over the CONN prototype
        co-adjacency graph. Measures how many hops in the embedding (low-d
        geodesic distance) are needed, on average, to span each pair of
        prototypes that are neighbours in high-d according to CONN:

            CONN_WAFL = Σ_{(i,j): CONN[i,j]>0} CONN[i,j] * embed_geodesic_dist[i,j]
                      / Σ_{(i,j): CONN[i,j]>0} CONN[i,j]

        where embed_geodesic_dist[i,j] is the geodesic hop-count distance between prototypes
        i and j in the embedding graph. CONN_WAFL = 1 is ideal (every high-d
        CONN-neighbour pair is also a direct embedding neighbour). Values > 1
        indicate folding: high-d neighbours are separated in the embedding,
        requiring multiple hops to connect them. Only computed when embed_geodesic_dist and
        CONN are provided.

    CONN_WAFL_SE : float or None
        Weighted standard error of the CONN_WAFL estimate. None when
        CONN_WAFL is None.
    """

    Q_local      : Optional[float] = field(default=None)
    Q_global     : Optional[float] = field(default=None)
    Q_AUC        : Optional[float] = field(default=None)
    LCMC_AUC     : Optional[float] = field(default=None)
    Trust_AUC    : Optional[float] = field(default=None)
    CONN_WAFL    : Optional[float] = field(default=None)
    CONN_WAFL_SE : Optional[float] = field(default=None)

    def as_dict(self):
        """Return metric values as a plain dict."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __repr__(self):
        lines = ["DRMetricsResult("]
        for f in fields(self):
            val = getattr(self, f.name)
            if val is None:
                lines.append(f"    {f.name}=None,")
            else:
                lines.append(f"    {f.name}={val:.6f},")
        lines.append(")")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_coranking_metrics(X, Y):
    """
    Compute co-ranking family metrics via pyDRMetrics.

    Extracts Q_local, Q_global, Q_AUC, LCMC_AUC, and Trust_AUC from the
    co-ranking matrix of (X, Y). The local/global split for Q is determined
    by kmax — the neighbourhood size at which LCMC is maximised. Trust_AUC
    is the mean of T(k) over all k independently of kmax, since T(k) is
    typically monotonically decreasing and has no natural internal maximum.

    Parameters
    ----------
    X : np.ndarray, shape (M, d)
        High-dimensional coordinates.
    Y : np.ndarray, shape (M, dim)
        Low-dimensional coordinates.

    Returns
    -------
    Q_local, Q_global, Q_AUC, LCMC_AUC, Trust_AUC : float

    Raises
    ------
    ImportError
        If pyDRMetrics is not installed.
    """
    try:
        from pyDRMetrics.pyDRMetrics import DRMetrics
    except ImportError:
        raise ImportError(
            "pyDRMetrics is required for co-ranking metrics. "
            "Install it with: pip install pyDRMetrics"
        )

    drm = DRMetrics(X=X, Z=Y)

    # Scalars returned directly by pyDRMetrics
    Q_local  = float(drm.Qlocal)
    Q_global = float(drm.Qglobal)
    Q_AUC    = float(drm.AUC)

    # LCMC_AUC: integrate LCMC(k) and normalise by theoretical maximum AUC.
    # The maximum AUC is the integral of (1 - k/K), the ceiling of LCMC(k).
    # krng is 1-based to mirror the R reference implementation (seq_along).
    lcmc = np.asarray(drm.LCMC, dtype=np.float64)
    K    = len(lcmc)
    krng = np.arange(1, K + 1, dtype=np.float64)
    auc_lcmc = np.trapz(lcmc,         x=krng)
    auc_max  = np.trapz(1 - krng / K, x=krng)
    LCMC_AUC = float(auc_lcmc / auc_max)

    # Trust_AUC: mean of T(k) over all k.
    # T(k) is typically monotonically decreasing — no natural internal
    # maximum — so no local/global split is applied.
    T         = np.asarray(drm.T, dtype=np.float64)
    Trust_AUC = float(np.mean(T))

    return Q_local, Q_global, Q_AUC, LCMC_AUC, Trust_AUC


def _compute_conn_wafl(embed_geodesic_dist, CONN):
    """
    Compute the CONN Weighted Average Folding Length (CONN_WAFL).

    CONN_WAFL is the CONN-weighted mean of the embedding geodesic distances
    embed_geodesic_dist[i,j] over all prototype pairs (i,j) with CONN[i,j] > 0. A value of
    1 is ideal; values > 1 indicate that high-d manifold neighbours are
    separated in the embedding (folding).

    CONN is symmetric, so each pair (i,j) and (j,i) both appear as nonzero
    entries in the sparse matrix. Both are included in the weighted mean,
    consistent with the R reference implementation (``which(CONN > 0)`` on
    the full symmetric matrix). The double-counting cancels exactly in the
    numerator and denominator of the weighted mean.

    Parameters
    ----------
    embed_geodesic_dist : np.ndarray, shape (M, M)
        Pairwise geodesic hop-count distances in the embedding graph
        (e.g. ``Embedding.dist``).
    CONN : scipy.sparse matrix, shape (M, M)
        Prototype co-adjacency matrix. Nonzero entries indicate high-d
        manifold neighbours; values are co-occurrence counts used as weights.

    Returns
    -------
    conn_wafl : float
    conn_wafl_se : float
        Weighted standard error of the CONN_WAFL estimate.
    """
    coo     = CONN.tocoo()
    weights = coo.data.astype(np.float64)
    dists   = embed_geodesic_dist[coo.row, coo.col].astype(np.float64)

    total_w      = weights.sum()
    conn_wafl    = float((weights * dists).sum() / total_w)
    weighted_var = (weights * (dists - conn_wafl) ** 2).sum() / total_w
    conn_wafl_se = float(np.sqrt(weighted_var) / np.sqrt(len(weights)))

    return conn_wafl, conn_wafl_se


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_dr_metrics(
    X,
    Y,
    embed_geodesic_dist=None,
    CONN=None,
    compute_coranking=True,
):
    """
    Compute dimensionality reduction quality metrics for a (high-d, low-d) pair.

    Evaluates how faithfully the low-dimensional embedding ``Y`` preserves the
    structure of the high-dimensional point set ``X``. Designed for use with
    SOM prototype vectors and their 2-D output-space coordinates, but
    applicable to any matched (high-d, low-d) pair.

    Two families of metrics are available:

    **Co-ranking family** (requires ``pyDRMetrics``): measures of neighbourhood
    preservation at local scales, global scales, and across all scales, plus a
    chance-corrected summary (LCMC_AUC) and a trustworthiness summary
    (Trust_AUC). Set ``compute_coranking=False`` to skip if ``pyDRMetrics``
    is not installed.

    **CONN_WAFL** (requires ``embed_geodesic_dist`` and ``CONN``): a CONN-weighted measure of
    embedding folding — how many embedding hops are needed, on average, to
    span each high-d CONN neighbourhood. Requires only numpy/scipy.

    Parameters
    ----------
    X : array-like, shape (M, d)
        High-dimensional coordinates (e.g. SOM prototype vectors ``W``).
    Y : array-like, shape (M, dim)
        Low-dimensional coordinates (e.g. SOM embedding coords
        ``Embedding.coords``).
    embed_geodesic_dist : np.ndarray, shape (M, M), optional
        Pairwise geodesic hop-count distances in the embedding graph.
        Required for CONN_WAFL. In GTSOM, this is ``Embedding.dist``.
    CONN : scipy.sparse matrix, shape (M, M), optional
        Prototype co-adjacency matrix. Required for CONN_WAFL. In GTSOM,
        this is ``VQRecaller.CONN``.
    compute_coranking : bool, default True
        Whether to compute co-ranking family metrics via ``pyDRMetrics``.
        Set to False if ``pyDRMetrics`` is not installed or if only
        CONN_WAFL is needed.

    Returns
    -------
    DRMetricsResult
        Dataclass with the following fields. Fields that were not computed
        are None.

        ================  ==========  =============  =======================
        Field             Range       Better         Requires
        ================  ==========  =============  =======================
        Q_local           [0, 1]      higher         pyDRMetrics
        Q_global          [0, 1]      higher         pyDRMetrics
        Q_AUC             [0, 1]      higher         pyDRMetrics
        LCMC_AUC          [0, 1]      higher         pyDRMetrics
        Trust_AUC         [0, 1]      higher         pyDRMetrics
        CONN_WAFL         [1, ∞)      lower (→ 1)   embed_geodesic_dist + CONN
        CONN_WAFL_SE      [0, ∞)      lower          embed_geodesic_dist + CONN
        ================  ==========  =============  =======================

        See :class:`DRMetricsResult` for full descriptions of each field.

    Raises
    ------
    ImportError
        If ``compute_coranking=True`` and ``pyDRMetrics`` is not installed.
    ValueError
        If ``CONN`` is provided without ``embed_geodesic_dist``, or vice versa.
        If ``X`` and ``Y`` have different numbers of rows.

    Examples
    --------
    Full metrics (requires pyDRMetrics and a CONN matrix):

    >>> result = compute_dr_metrics(X=W, Y=coords, embed_geodesic_dist=embed.dist,
    ...                             CONN=recaller.CONN)
    >>> result.LCMC_AUC
    0.812...
    >>> result.CONN_WAFL
    1.043...

    Co-ranking only (no CONN needed):

    >>> result = compute_dr_metrics(X=W, Y=coords)
    >>> result.Q_local
    0.734...
    >>> result.Trust_AUC
    0.891...

    CONN_WAFL only (no pyDRMetrics needed):

    >>> result = compute_dr_metrics(X=W, Y=coords, embed_geodesic_dist=embed.dist,
    ...                             CONN=recaller.CONN,
    ...                             compute_coranking=False)
    >>> result.CONN_WAFL
    1.043...

    Notes
    -----
    ``X`` and ``Y`` must have the same number of rows M (one row per point).
    No internal scaling is applied — pass pre-processed arrays if needed.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of rows. "
            f"Got X.shape={X.shape}, Y.shape={Y.shape}."
        )

    if (embed_geodesic_dist is None) != (CONN is None):
        raise ValueError(
            "embed_geodesic_dist and CONN must be supplied together — both or neither."
        )

    result = DRMetricsResult()

    # ------------------------------------------------------------------
    # Co-ranking family metrics
    # ------------------------------------------------------------------
    if compute_coranking:
        (
            result.Q_local,
            result.Q_global,
            result.Q_AUC,
            result.LCMC_AUC,
            result.Trust_AUC,
        ) = _compute_coranking_metrics(X, Y)

    # ------------------------------------------------------------------
    # CONN_WAFL
    # ------------------------------------------------------------------
    if embed_geodesic_dist is not None and CONN is not None:
        result.CONN_WAFL, result.CONN_WAFL_SE = _compute_conn_wafl(
            np.asarray(embed_geodesic_dist, dtype=np.float64), CONN
        )

    return result