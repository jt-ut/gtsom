"""
gtsom.py — General Topology Self-Organizing Map

The GTSOM class implements batch SOM learning over an arbitrary low-dimensional
output topology defined by an Embedding instance. Prototype vectors live in
high-dimensional feature space; their neighbourhood relationships are governed
by geodesic distances on the output manifold.

Typical usage
-------------
# Grid-based (classical SOM style)
som = GTSOM(rho_0=5.0, rho_f=0.5, halflife_epochs=50)
som.from_grid(X, shape=(10, 10), coord_init='hex', W_init='random')
som.fit(X, n_epochs=20)

# Data-driven (general topology)
som = GTSOM(rho_0=5.0, rho_f=0.5, halflife_epochs=50)
som.from_data(X, M=100, coord_dim=2, coord_init='pca', W_init='kmeans')
som.fit(X, n_epochs=20)

# Hybrid lattice + CONN-based neighbourhood update
som = GTSOM(
    rho_0=5.0, rho_f=0.5, halflife_epochs=50,
    nbr_topo_alpha_0=1.0, nbr_topo_alpha_f=0.3,
)
som.from_data(X, M=100, coord_dim=2, coord_init='pca', W_init='kmeans')
som.fit(X, n_epochs=20)

Notes
-----
GTSOM does not perform any internal scaling of X. It is the caller's
responsibility to whiten, standardise, or otherwise preprocess X before
passing it to any method. Distances computed during prototype updates and
BMU search will reflect the scale of the input features.
"""

import time
import numpy as np
from scipy.sparse.csgraph import shortest_path

from .embedding import Embedding
from .kernel import NeighborKernel
from .utils import ExponentialAnneal, DataValidator, reduce_coords_pca, reduce_coords_le, reduce_coords_random, reduce_coords_random_proj
from .parallel import update_prototypes_kernel, PARALLEL, resolve_n_jobs
from vqlp import VQRecaller


# ---------------------------------------------------------------------------
# Module-level prototype initialisation helpers
# ---------------------------------------------------------------------------

def _init_W_random(X, M, rng):
    """
    Initialise prototype matrix W by random sampling from X.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
    M : int
    rng : np.random.Generator

    Returns
    -------
    W : np.ndarray, shape (M, d)
    """
    idx = rng.choice(X.shape[0], size=M, replace=False)
    return X[idx].copy()


def _init_W_pca(X, embed, random_state):
    """
    Initialise prototype matrix W via PCA back-projection.

    Projects embed.coords (low-d neuron positions) back into high-d feature
    space using the top-k principal components of X, where k = embed.dim.
    This places each prototype at the point in high-d space that corresponds
    to its low-d coordinate under the linear PCA map.

    embed.coords are first rescaled to span the same range as the PCA score
    space, so that back-projected prototypes land inside the data cloud.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
    embed : Embedding
        Must already have coords set, shape (M, embed.dim).
    random_state : int or None

    Returns
    -------
    W : np.ndarray, shape (M, d), same dtype as X
    """
    from sklearn.decomposition import PCA
    k = embed.dim  # 2 or 3
    pca = PCA(n_components=k, svd_solver='randomized', random_state=random_state)
    pca.fit(X)

    # Rescale embed.coords to match the range of PCA score space per component
    scores = pca.transform(X)      # (N, k)
    coords = embed.coords.copy()   # (M, k)
    for dim in range(k):
        s_min, s_max = scores[:, dim].min(), scores[:, dim].max()
        c_min, c_max = coords[:, dim].min(), coords[:, dim].max()
        if c_max - c_min > 0:
            coords[:, dim] = (coords[:, dim] - c_min) / (c_max - c_min)
            coords[:, dim] = coords[:, dim] * (s_max - s_min) + s_min

    # Back-project: W = coords @ components + mean
    W = coords @ pca.components_ + pca.mean_
    return W.astype(X.dtype)


def _init_W_kmeans(X, M, random_state):
    """
    Initialise prototype matrix W via FAISS k-means (requires vqlp).

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
    M : int
    random_state : int or None

    Returns
    -------
    W : np.ndarray, shape (M, d), same dtype as X
    """
    try:
        from vqlp import VQFitter
    except ImportError:
        raise ImportError(
            "vqlp package is required for W_init='kmeans'. "
            "Install it or use W_init='random'."
        )
    vqf = VQFitter(M=M, p=2, max_bmu=1, random_state=random_state)
    vqf.fit(X, method='kmeans')
    return vqf.W.astype(X.dtype)


# ---------------------------------------------------------------------------
# Neighbourhood kernel helpers
# ---------------------------------------------------------------------------

def _cadj_zelnik_kernel(W, CADJ, CADJ_nhbs, rho, nbr_influence_min):
    """
    Self-tuning neighbourhood kernel based on CADJ-weighted local radii.

    Computes a symmetric M×M similarity matrix where the similarity between
    prototypes i and j is::

        H[i,j] = exp(-dist²(W[i], W[j]) / (sigma_i * sigma_j * rho))

    where ``sigma_i`` is the CADJ-weighted mean Euclidean distance from
    prototype i to its CADJ neighbours — a locally adaptive scale reflecting
    the typical size of i's data-manifold neighbourhood in feature space.

    CADJ (rather than CONN) is used for sigma computation deliberately:
    ``CADJ[i,j]`` counts data points for which i was the 1st BMU and j the
    2nd, giving a directed, i-centric view of i's local neighbourhood. This
    asymmetry is desirable — ``sigma_i`` should reflect i's own outward reach,
    not a symmetrised average. The symmetry of the final kernel is recovered
    via the geometric mean ``sigma_i * sigma_j`` in the denominator.

    The quantity ``dist²(i,j) / (sigma_i * sigma_j)`` is dimensionless and
    can be interpreted as a locally-normalised distance: j is approximately
    "one hop" from i when ``dist(i,j) ≈ sqrt(sigma_i * sigma_j)``, the
    geometric mean of the two local scales. Dividing further by ``rho``
    controls how many such normalised hops exert meaningful neighbourhood
    influence, analogously to rho's role in the lattice kernel.

    Unlike ``'CONN'`` mode (which uses CONN as a graph and computes shortest-
    path hop counts), ``'CONN_STK'`` uses CADJ only to compute local radii
    sigma_i, then applies the self-tuning kernel to *full* Euclidean prototype
    distances. Every prototype pair receives a nonzero similarity; sparsity
    is imposed only by thresholding at ``nbr_influence_min``.

    Parameters
    ----------
    W : np.ndarray, shape (M, d)
        Current prototype matrix.
    CADJ : scipy.sparse matrix, shape (M, M)
        Asymmetric co-adjacency matrix. CADJ[i,j] counts data points for
        which i was the 1st BMU and j the 2nd. Used to weight the local
        radius computation for each prototype.
    CADJ_nhbs : list of list of int
        CADJ_nhbs[i] = column indices of nonzero entries in row i.
        Precomputed from the recaller. Prototypes with no CADJ neighbours
        (empty RFs throughout training) receive a fallback sigma equal to
        the median sigma of well-connected prototypes.
    rho : float
        Current neighbourhood bandwidth (annealed). Acts as a multiplier
        on the local radius unit: larger rho broadens the neighbourhood
        across more CADJ-radius units.
    nbr_influence_min : float
        Entries below this value are set to zero, producing a sparse output
        consistent with ``NeighborKernel`` behaviour.

    Returns
    -------
    H : scipy.sparse.csr_matrix, shape (M, M)
        Symmetric neighbourhood weight matrix with values in (0, 1].
    """
    from scipy.spatial.distance import cdist
    from scipy.sparse import csr_matrix as _csr

    M = W.shape[0]

    # ------------------------------------------------------------------
    # Step 1: compute sigma_i — CADJ-weighted mean Euclidean distance
    #         from prototype i to its CADJ neighbours.
    #         Uses CADJ (directed) so sigma_i reflects i's own outward
    #         neighbourhood reach, not a symmetrised average.
    # ------------------------------------------------------------------
    sigma = np.empty(M, dtype=np.float32)
    for i in range(M):
        nhbs = CADJ_nhbs[i]
        if len(nhbs) == 0:
            sigma[i] = -1.0    # flagged for fallback
            continue
        dists   = np.linalg.norm(W[nhbs] - W[i], axis=1)
        weights = np.asarray(CADJ[i, nhbs].todense()).ravel().astype(np.float32)
        wsum    = weights.sum()
        sigma[i] = float(np.dot(weights, dists) / wsum) if wsum > 0 else -1.0

    # Prototypes with no CADJ neighbours get the median sigma of well-
    # connected prototypes, falling back to 1.0 if all are degenerate.
    valid    = sigma > 0
    fallback = float(np.median(sigma[valid])) if valid.any() else 1.0
    sigma[~valid] = fallback

    # ------------------------------------------------------------------
    # Step 2: full M×M squared Euclidean distance matrix
    # ------------------------------------------------------------------
    D2 = cdist(W, W, metric='sqeuclidean')    # (M, M)

    # ------------------------------------------------------------------
    # Step 3: self-tuning kernel.
    #         Row-then-column broadcast division avoids constructing the
    #         full sigma_outer matrix:
    #           D2 / sigma[:, None]  →  D2[i,j] / sigma_i   (row-wise)
    #           / sigma[None, :]     →  / sigma_j            (col-wise)
    #           / rho                →  scale by bandwidth
    #         Result is exp(-dist²_ij / (sigma_i * sigma_j * rho)).
    # ------------------------------------------------------------------
    H = np.exp(-D2 / sigma[:, None] / sigma[None, :] / rho).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 4: threshold, fix diagonal, return sparse CSR
    # ------------------------------------------------------------------
    H[H < nbr_influence_min] = 0.0
    np.fill_diagonal(H, 1.0)
    return _csr(H)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GTSOM:
    """
    General Topology Self-Organizing Map.

    The constructor sets all learning parameters (schedules, parallelism,
    diagnostics). After construction, call :meth:`from_grid` or
    :meth:`from_data` to set the output topology and initialise prototypes,
    then call :meth:`fit` to train.

    Parameters
    ----------
    rho_0 : float
        Initial neighbourhood bandwidth.
    rho_f : float
        Final (minimum) neighbourhood bandwidth.
    tau : float, optional
        Exponential decay time constant in epochs. Mutually exclusive with
        ``halflife_epochs``.
    halflife_epochs : float, optional
        Epoch at which each annealed parameter reaches the geometric midpoint
        between its initial and final values. Mutually exclusive with ``tau``.
    anneal : {'exponential'}, default 'exponential'
        Annealing schedule type. Currently only exponential decay is supported.
    n_jobs : int or None, default None
        Number of parallel threads for prototype updates. ``None`` or ``-1``
        uses all available CPU cores. Ignored with a warning if numba is not
        installed.
    nbr_topo_alpha_0 : float, default 1.0
        Initial value of the topology-blending parameter, in [0, 1]. When
        1.0, pure lattice-based neighbourhood (classical SOM). When < 1.0,
        a blend of lattice and CONN-graph neighbourhood is used. See
        :meth:`_compute_neighborhood` for the full blending rule.
    nbr_topo_alpha_f : float, optional
        Final value of the topology-blending parameter, in [0, 1]. Defaults
        to ``nbr_topo_alpha_0`` for a flat (non-annealing) schedule.
    compute_dr_metrics : bool, default False
        Whether to compute dimensionality reduction quality metrics at each
        snapshot. When True, each entry in ``learn_history_`` will contain a
        :class:`~gtsom.dr_metrics.DRMetricsResult` with co-ranking and
        CONN_WAFL metrics.
    nbr_influence_min : float, default 0.01
        Neighbourhood kernel activation floor. Kernel weights below this
        value are zeroed. Shared across all architecture configurations built
        on this instance.
    proto_topo : {'CONN', 'CONN_STK'}, default 'CONN'
        Input-space (prototype) topology used to compute the high-dimensional
        neighbourhood signal when ``nbr_topo_alpha < 1.0``. Has no effect
        when ``nbr_topo_alpha_0 == nbr_topo_alpha_f == 1.0``.

        ``'CONN'``
            CONN graph geodesics. The CONN matrix is treated as an
            unweighted graph; shortest-path hop counts are computed and
            fed through the standard exponential kernel. Purely
            topological — ignores Euclidean distances between prototypes.
        ``'CONN_STK'``
            CONN self-tuning kernel. CONN weights are used to compute a
            locally adaptive radius ``sigma_i`` (CONN-weighted mean
            Euclidean distance from i to its CONN neighbours) for each
            prototype. The full M×M Euclidean distance matrix is then
            transformed via ``exp(-dist²(i,j) / (sigma_i * sigma_j * rho))``,
            where ``dist²(i,j) / (sigma_i * sigma_j)`` is a dimensionless
            locally-normalised distance analogous to a hop count.
    random_state : int or None, default None
        Seed for all random operations.

    Notes
    -----
    GTSOM does not perform any internal scaling of X. It is the caller's
    responsibility to whiten, standardise, or otherwise preprocess X before
    passing it to any method.

    Raises
    ------
    ValueError
        If neither or both of ``tau`` and ``halflife_epochs`` are supplied,
        if ``anneal`` is not recognised, if either alpha parameter is
        outside [0, 1], if ``nbr_influence_min`` is not in (0, 1), or if
        ``proto_topo`` is not recognised.
    """

    def __init__(
        self,
        rho_0,
        rho_f,
        tau=None,
        halflife_epochs=None,
        anneal='exponential',
        n_jobs=None,
        nbr_topo_alpha_0=1.0,
        nbr_topo_alpha_f=None,
        proto_topo='CONN',
        compute_dr_metrics=False,
        nbr_influence_min=0.01,
        random_state=None,
    ):
        # ------------------------------------------------------------------
        # Validate schedule parameters
        # ------------------------------------------------------------------
        if anneal not in ('exponential',):
            raise ValueError(
                f"anneal must be 'exponential', got {anneal!r}. "
                "Additional schedules may be added in future."
            )
        if tau is None and halflife_epochs is None:
            raise ValueError(
                "Exactly one of tau or halflife_epochs must be supplied."
            )
        if tau is not None and halflife_epochs is not None:
            raise ValueError(
                "Supply either tau or halflife_epochs, not both."
            )

        alpha_f = nbr_topo_alpha_f if nbr_topo_alpha_f is not None else nbr_topo_alpha_0
        if not (0.0 <= nbr_topo_alpha_0 <= 1.0):
            raise ValueError(
                f"nbr_topo_alpha_0 must be in [0, 1], got {nbr_topo_alpha_0}."
            )
        if not (0.0 <= alpha_f <= 1.0):
            raise ValueError(
                f"nbr_topo_alpha_f must be in [0, 1], got {alpha_f}."
            )
        if not (0.0 < nbr_influence_min < 1.0):
            raise ValueError(
                f"nbr_influence_min must be in (0, 1), got {nbr_influence_min}."
            )
        if proto_topo not in ('CONN', 'CONN_STK'):
            raise ValueError(
                f"proto_topo must be 'CONN' or 'CONN_STK', got {proto_topo!r}."
            )

        # ------------------------------------------------------------------
        # Build annealing schedules
        # ------------------------------------------------------------------
        if tau is not None:
            self.rho_schedule   = ExponentialAnneal(
                initial=rho_0, final=rho_f, tau=tau
            )
            self.alpha_schedule = ExponentialAnneal(
                initial=nbr_topo_alpha_0, final=alpha_f, tau=tau
            )
        else:
            self.rho_schedule   = ExponentialAnneal.from_halflife(
                initial=rho_0, final=rho_f, halflife_epochs=halflife_epochs
            )
            self.alpha_schedule = ExponentialAnneal.from_halflife(
                initial=nbr_topo_alpha_0, final=alpha_f,
                halflife_epochs=halflife_epochs
            )

        # ------------------------------------------------------------------
        # Store learning configuration
        # ------------------------------------------------------------------
        self.nbr_influence_min = nbr_influence_min
        self.proto_topo         = proto_topo
        self.random_state       = random_state
        self.compute_dr_metrics = bool(compute_dr_metrics)

        # Resolve and store thread count
        if not PARALLEL and n_jobs is not None and n_jobs != 1:
            import warnings
            warnings.warn(
                f"n_jobs={n_jobs} ignored: numba is not installed. "
                "Install numba for parallel prototype updates.",
                UserWarning, stacklevel=2,
            )
        self.n_jobs = resolve_n_jobs(n_jobs)

        # ------------------------------------------------------------------
        # Architecture attributes — None until from_grid / from_data is called
        # ------------------------------------------------------------------
        self.W          = None   # prototype matrix (M, d)
        self.embed      = None   # Embedding instance
        self.recaller   = None   # VQRecaller instance
        self.kernel     = None   # NeighborKernel instance
        self._validator = None   # DataValidator instance
        self.prevBMU    = None   # (N,) previous-epoch BMU assignments
        self.W0         = None   # initial prototype matrix (never modified)
        self.coords0    = None   # initial embedding coords (never modified)
        self.nbr_W      = None   # neighbourhood weight matrix (M, M)

        # ------------------------------------------------------------------
        # Training state
        # ------------------------------------------------------------------
        self.age         = 0     # total epochs completed
        self.train_time  = 0.0   # cumulative wall-clock training time (s)

        # Learning history: list of snapshot dicts, one per recorded epoch.
        # learn_history_[0] is the post-architecture-init state (age=0),
        # populated by from_grid / from_data. Subsequent entries are
        # appended by fit() after each epoch.
        # Each dict contains:
        #   'age'        : int              — self.age at snapshot time
        #   'mqe'        : float            — global MQE over all data
        #   'W_mqe'      : ndarray (M,)     — per-prototype MQE, nan for empty RFs
        #   'delBMU'     : float            — fraction of data whose BMU changed
        #   'dr_metrics' : DRMetricsResult  — DR quality metrics, or None
        #   'fig'        : ggplot or None   — plot captured this epoch
        self.learn_history_ = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def M(self):
        """Number of prototypes (neurons)."""
        if self.W is None:
            raise RuntimeError(
                "Architecture not initialised. "
                "Call from_grid() or from_data() first."
            )
        return self.W.shape[0]

    @property
    def d(self):
        """Dimensionality of the high-dimensional feature space."""
        if self.W is None:
            raise RuntimeError(
                "Architecture not initialised. "
                "Call from_grid() or from_data() first."
            )
        return self.W.shape[1]

    # ------------------------------------------------------------------
    # Architecture initialisation — instance methods
    # ------------------------------------------------------------------

    def from_grid(
        self,
        X,
        shape,
        coord_init='hex',
        W_init='random',
        labels=None,
    ):
        """
        Set a regular grid output topology and initialise prototypes.

        Wipes any existing architecture (W, embed, recaller, learn_history_,
        etc.) and builds a fresh grid-based configuration. All learning
        parameters (rho schedules, nbr_influence_min, etc.) are inherited from the
        values set in ``__init__`` and are not changed.

        Parameters
        ----------
        X : array-like, shape (N, d)
            Training data. Used for prototype initialisation; not stored.
        shape : tuple of int
            Grid dimensions, e.g. ``(10, 10)`` or ``(5, 10, 4)``. The
            length of the tuple sets the embedding dimension (2 or 3).
        coord_init : {'hex', 'rect'}, default 'hex'
            Grid layout. ``'hex'`` uses a hexagonal arrangement (Delaunay
            adjacency); ``'rect'`` uses an 8-connected rectangular grid.
        W_init : {'random', 'pca'}, default 'random'
            How to initialise prototype vectors W.

            ``'random'``
                Random sample of M rows from X.
            ``'pca'``
                Back-project grid neuron coordinates through the PCA of X,
                placing each prototype at the corresponding point in high-d
                feature space.
        labels : array-like, shape (N,), default None
            Optional observation labels. If provided, prototype-level label
            summaries (WL, WL_Dist, WL_Purity) are computed during the
            initial recall and stored on ``self.recaller``.

        Returns
        -------
        self : GTSOM
            Returns self for optional method chaining.

        Raises
        ------
        ValueError
            If ``shape`` is not a 2- or 3-tuple, ``coord_init`` or
            ``W_init`` are not recognised, or the grid has more neurons
            than data points.
        """
        if coord_init not in ('hex', 'rect'):
            raise ValueError(
                f"coord_init must be 'hex' or 'rect' for from_grid, "
                f"got {coord_init!r}"
            )
        if W_init not in ('random', 'pca'):
            raise ValueError(
                f"W_init must be 'random' or 'pca' for from_grid, "
                f"got {W_init!r}"
            )
        if len(shape) not in (2, 3):
            raise ValueError(
                f"shape must be a 2- or 3-tuple, got length {len(shape)}"
            )

        X = np.asarray(X)
        rng = np.random.default_rng(self.random_state)

        # Build embedding — grid kind matches coord_init
        embed = Embedding.from_grid(shape, kind=coord_init)
        M = embed.coords.shape[0]

        if M > X.shape[0]:
            raise ValueError(
                f"Grid has {M} neurons but X has only {X.shape[0]} rows. "
                f"Reduce shape or provide more data."
            )

        # Initialise W
        if W_init == 'random':
            W = _init_W_random(X, M, rng)
        else:  # 'pca'
            W = _init_W_pca(X, embed, self.random_state)

        self._init_architecture(X, W, embed, labels)
        return self

    def from_data(
        self,
        X,
        M,
        coord_dim=2,
        coord_init='pca',
        W_init='kmeans',
        coord_topo='delaunay',
        labels=None,
    ):
        """
        Set a data-driven output topology and initialise prototypes.

        Wipes any existing architecture (W, embed, recaller, learn_history_,
        etc.) and builds a fresh data-driven configuration. All learning
        parameters (rho schedules, nbr_influence_min, etc.) are inherited from the
        values set in ``__init__`` and are not changed.

        Prototype vectors are found via vector quantisation of X, projected
        to ``coord_dim`` dimensions to define neuron positions, and either
        a Delaunay triangulation or Gabriel graph of those positions defines
        the output topology.

        Parameters
        ----------
        X : array-like, shape (N, d)
            Training data.
        M : int
            Number of prototypes (neurons). Must be >= 3 (required for
            Delaunay triangulation) and <= N.
        coord_dim : {2, 3}, default 2
            Dimensionality of the output embedding.
        coord_init : {'pca', 'le', 'random', 'random_proj'}, default 'pca'
            Dimensionality reduction method for computing neuron coordinates
            from the initial prototype vectors.

            ``'pca'``
                Randomised PCA (fast, linear). The natural default when
                ``W_init='kmeans'``, since k-means centroids carry data
                structure that PCA can meaningfully summarise.
            ``'le'``
                Laplacian Eigenmaps via sklearn SpectralEmbedding
                (nonlinear, slower). Useful when the data manifold is
                curved.
            ``'random'``
                Sample ``M`` points uniformly at random from the unit
                hypercube in ``coord_dim`` dimensions. Produces a
                completely unstructured initial layout — the strongest
                possible "worst-case" starting state for illustrating
                how self-organising learning imposes order. Particularly
                effective for exposition when combined with
                ``W_init='random'``.
            ``'random_proj'``
                Johnson-Lindenstrauss random projection: multiply ``W``
                by a random Gaussian matrix scaled by
                ``1 / sqrt(coord_dim)``. Preserves approximate pairwise
                distances in expectation at zero computational cost,
                giving a random but structurally grounded starting layout.
        W_init : {'kmeans', 'random'}, default 'kmeans'
            How to find initial prototype vectors.

            ``'kmeans'``
                FAISS k-means via VQFitter (requires vqlp).
            ``'random'``
                Random sample of M rows from X.
        coord_topo : {'delaunay', 'gabriel'}, default 'delaunay'
            Graph used to define the output-space topology from the
            projected prototype coordinates.

            ``'delaunay'``
                Delaunay triangulation. Denser connectivity.
            ``'gabriel'``
                Gabriel graph (subgraph of Delaunay). Sparser connectivity
                that more closely reflects local proximity structure.
        labels : array-like, shape (N,), default None
            Optional observation labels. If provided, prototype-level label
            summaries (WL, WL_Dist, WL_Purity) are computed during the
            initial recall and stored on ``self.recaller``.

        Returns
        -------
        self : GTSOM
            Returns self for optional method chaining.

        Raises
        ------
        ValueError
            If ``coord_dim`` is not 2 or 3, or if any of ``coord_init``,
            ``W_init``, or ``coord_topo`` are not recognised, or M is out
            of range.
        ImportError
            If ``W_init='kmeans'`` and vqlp is not installed.
        """
        if coord_dim not in (2, 3):
            raise ValueError(f"coord_dim must be 2 or 3, got {coord_dim!r}")
        if coord_init not in ('pca', 'le', 'random', 'random_proj'):
            raise ValueError(
                f"coord_init must be 'pca', 'le', 'random', or 'random_proj' "
                f"for from_data, got {coord_init!r}"
            )
        if coord_topo not in ('delaunay', 'gabriel'):
            raise ValueError(
                f"coord_topo must be 'delaunay' or 'gabriel', "
                f"got {coord_topo!r}"
            )
        if W_init not in ('kmeans', 'random'):
            raise ValueError(
                f"W_init must be 'kmeans' or 'random' for from_data, "
                f"got {W_init!r}"
            )
        if M > X.shape[0]:
            raise ValueError(
                f"M={M} exceeds number of data points N={X.shape[0]}."
            )
        if M < 3:
            raise ValueError(
                f"M must be at least 3 for Delaunay triangulation, got {M}."
            )

        X = np.asarray(X)
        rng = np.random.default_rng(self.random_state)

        # Step 1: initialise prototype vectors in high-d
        if W_init == 'kmeans':
            W = _init_W_kmeans(X, M, self.random_state)
        else:  # 'random'
            W = _init_W_random(X, M, rng)

        # Step 2: project W to coord_dim to define neuron positions
        if coord_init == 'pca':
            coords = reduce_coords_pca(W, coord_dim, self.random_state)
        elif coord_init == 'le':
            coords = reduce_coords_le(W, coord_dim, self.random_state)
        elif coord_init == 'random':
            coords = reduce_coords_random(M, coord_dim, self.random_state)
        else:  # 'random_proj'
            coords = reduce_coords_random_proj(W, coord_dim, self.random_state)

        # Step 3: build embedding from projected coords using chosen topology
        if coord_topo == 'delaunay':
            embed = Embedding.from_delaunay(coords)
        else:  # 'gabriel'
            embed = Embedding.from_gabriel(coords)

        self._init_architecture(X, W, embed, labels)
        return self

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, n_epochs, labels=None, verbose=True, plot_every=5):
        """
        Fit the GTSOM to data X using batch SOM learning.

        :meth:`from_grid` or :meth:`from_data` must be called before
        :meth:`fit`. The rho annealing schedule advances based on
        ``self.age``, so successive calls to :meth:`fit` continue from
        where the previous call left off.

        Parameters
        ----------
        X : array-like, shape (N, d)
            Training data. Should be the same dataset passed to
            :meth:`from_grid` or :meth:`from_data`.
        n_epochs : int
            Number of training epochs to run in this call.
        labels : array-like, shape (N,), default None
            Optional observation labels. If provided, prototype-level
            label summaries (WL, WL_Dist, WL_Purity) are recomputed
            each epoch and stored on ``self.recaller``.
        verbose : bool, default True
            If True, print a formatted table of training metrics after
            each epoch.
        plot_every : int, default 5
            Capture a SOM plot into ``self.learn_history_`` every this
            many epochs. Set to 0 to disable in-training plotting.

        Raises
        ------
        RuntimeError
            If :meth:`from_grid` or :meth:`from_data` has not been called.

        Notes
        -----
        GTSOM does not perform any internal scaling of X.
        """
        if self.W is None:
            raise RuntimeError(
                "Architecture not initialised. "
                "Call from_grid() or from_data() before fit()."
            )

        X = np.asarray(X)
        if self._validator is not None:
            self._validator.check(X, context="fit")

        _HDR = f"{'Epoch':>7}   {'rho':>6}   {'alpha':>6}   {'MQE':>8}   {'delBMU':>7}"
        _SEP = f"{'-----':>7}   {'------':>6}   {'------':>6}   {'--------':>8}   {'-------':>7}"

        if verbose:
            print(_HDR)
            print(_SEP)

        # Apply thread count for this fit() call; restore afterward
        if PARALLEL:
            from numba import set_num_threads, get_num_threads
            _prev_threads = get_num_threads()
            set_num_threads(self.n_jobs)

        _t0 = time.perf_counter()
        for local_epoch in range(n_epochs):
            rho   = self.rho_schedule(self.age)
            alpha = self.alpha_schedule(self.age)
            self._compute_neighborhood(rho, alpha)
            self._update_prototypes(X)
            self._recall(X, labels=labels)
            self.age += 1
            include_fig = plot_every > 0 and (self.age % plot_every == 0)
            self._snapshot(include_fig=include_fig)
            if verbose:
                snap = self.learn_history_[-1]
                print(
                    f"{self.age:>7d}   {rho:>6.4f}   {alpha:>6.4f}   "
                    f"{snap['mqe']:>8.4f}   {snap['delBMU']:>7.4f}"
                )
                if (local_epoch + 1) % 10 == 0:
                    print(_SEP)

        if verbose:
            print(_SEP)

        self.train_time += time.perf_counter() - _t0

        if PARALLEL:
            set_num_threads(_prev_threads)

    def transform(self, X):
        """
        Map data points to output coordinates via BMU lookup.

        Parameters
        ----------
        X : array-like, shape (N, d)

        Returns
        -------
        coords : np.ndarray, shape (N, embed.dim)

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.W is None:
            raise RuntimeError(
                "Architecture not initialised. "
                "Call from_grid() or from_data() before transform()."
            )
        if self.age == 0:
            raise RuntimeError(
                "transform() called before fit(). "
                "Call fit() first to train the model."
            )
        X = np.asarray(X)
        self.recaller.recall(X=X, W=self.W)
        return self.embed.coords[self.recaller.BMU[:, 0]]

    def fit_transform(self, X, **fit_kwargs):
        """
        Fit the GTSOM and return output coordinates.

        Equivalent to calling :meth:`fit` followed by :meth:`transform`.

        Parameters
        ----------
        X : array-like, shape (N, d)
        **fit_kwargs
            Passed to :meth:`fit`.

        Returns
        -------
        coords : np.ndarray, shape (N, embed.dim)
        """
        self.fit(X, **fit_kwargs)
        return self.transform(X)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(
        self,
        color_by='auto',
        cmap='viridis',
        rf_size_log_threshold=10,
        legend_pos='right',
        title='SOUMAP',
        subtitle=None,
        xlab=r'SOM$_1$',
        ylab=r'SOM$_2$',
    ):
        """
        Plot the SOM in output (lattice) space using plotnine.

        Delegates to :func:`~gtsom.vis_tools.vis_embedding_discrete` when
        labels are available, and to
        :func:`~gtsom.vis_tools.vis_embedding_continuous` otherwise (coloring
        by MQE). Neurons are drawn at their ``embed.coords`` positions,
        connected by edges from ``embed.adjacency``. Point size is mapped to
        receptive field size (RFSize), with the size transform (linear or
        log10) chosen automatically based on the distribution of RFSize values.
        If DR metrics have been computed for the current snapshot, they are
        formatted and shown as a plot caption.

        Returns a ``plotnine.ggplot`` object — nothing is displayed
        automatically. Call ``.save(path)`` to write to disk, or evaluate
        the object in a notebook to display it inline.

        Parameters
        ----------
        color_by : {'auto', 'mqe', 'rfsize', 'labels'}, default 'auto'
            Quantity used to colour neurons.

            ``'auto'``
                Use ``'labels'`` if ``recaller.WL`` is available,
                otherwise fall back to ``'mqe'``.
            ``'mqe'``
                Per-prototype mean quantization error. Continuous scale.
            ``'rfsize'``
                Receptive field size. Continuous scale.
            ``'labels'``
                Winning label per prototype. Discrete scale. Raises
                ``ValueError`` if ``recaller.WL`` is None.
        cmap : str, default 'viridis'
            Colormap name for continuous quantities ('mqe', 'rfsize').
            Any matplotlib-compatible string is accepted (e.g. 'plasma',
            'magma'). Ignored for label plots.
        rf_size_log_threshold : float, default 10
            Controls whether RFSize is mapped to point size on a linear
            or log10 scale. If ``max(RFSize) / median(RFSize)`` exceeds
            this threshold, ``point_size_wts_trans='log10'`` is passed to
            the vis_tools function; otherwise ``'identity'`` is used. Set
            to ``inf`` to always use linear scaling.
        legend_pos : str, default 'right'
            Legend / colorbar position. One of ``'right'``, ``'bottom'``,
            or ``'none'``.
        title : str or None, default 'SOUMAP'
            Main title line.
        subtitle : str or None, default None
            Subtitle line. If None, auto-generates ``'Epoch {self.age}'``.
        xlab : str, default 'SOM$_1$'
            Label for the horizontal axis.
        ylab : str, default 'SOM$_2$'
            Label for the vertical axis.

        Returns
        -------
        plotnine.ggplot

        Raises
        ------
        RuntimeError
            If :meth:`from_grid` or :meth:`from_data` has not been called.
        NotImplementedError
            If ``embed.dim != 2``.
        ValueError
            If ``color_by='labels'`` but no labels have been provided.
        """
        if self.W is None:
            raise RuntimeError(
                "Architecture not initialised. "
                "Call from_grid() or from_data() before plot()."
            )

        from .vis_tools import vis_embedding_discrete, vis_embedding_continuous

        if self.embed.dim != 2:
            raise NotImplementedError(
                f"plot() only supports 2-D embeddings; "
                f"embed.dim={self.embed.dim}."
            )

        # Resolve 'auto' color_by
        if color_by == 'auto':
            color_by = 'labels' if self.recaller.WL is not None else 'mqe'

        if color_by not in ('mqe', 'rfsize', 'labels'):
            raise ValueError(
                f"color_by must be 'auto', 'mqe', 'rfsize', or 'labels', "
                f"got {color_by!r}."
            )
        if color_by == 'labels' and self.recaller.WL is None:
            raise ValueError(
                "color_by='labels' requires labels to have been provided "
                "at construction or during fit(). recaller.WL is currently None."
            )

        # ------------------------------------------------------------------
        # RFSize → point_size_wts and transform
        # ------------------------------------------------------------------
        rf       = self.recaller.RFSize.astype(float)
        use_log  = (rf.max() / (np.median(rf) + 1e-8)) > rf_size_log_threshold
        rf_trans = 'log10' if use_log else 'identity'
        rf_norm  = (rf - rf.min()) / (rf.max() - rf.min() + 1e-8)

        # ------------------------------------------------------------------
        # Point size: scale down with M so points don't overlap at large M
        # ------------------------------------------------------------------
        point_size = max(2.0, 20.0 / np.sqrt(self.M))

        # ------------------------------------------------------------------
        # Subtitle and caption
        # ------------------------------------------------------------------
        sub     = subtitle if subtitle is not None else f'Epoch {self.age}'
        caption = None
        snap    = self.learn_history_[-1] if self.learn_history_ else None
        if snap is not None and snap.get('dr_metrics') is not None:
            dm = snap['dr_metrics']
            lines = []
            qnn_parts = []
            for name in ('Q_local', 'Q_global', 'Q_AUC', 'LCMC_AUC', 'Trust_AUC'):
                val = getattr(dm, name)
                if val is not None:
                    qnn_parts.append(f'{name}={val:.3f}')
            if qnn_parts:
                lines.append('  '.join(qnn_parts))
            wafl = getattr(dm, 'CONN_WAFL', None)
            se   = getattr(dm, 'CONN_WAFL_SE', None)
            if wafl is not None:
                wafl_str = f'CONN_WAFL={wafl:.3f}'
                if se is not None:
                    wafl_str += f' \u00b1 {se:.3f}'
                lines.append(wafl_str)
            if lines:
                caption = '\n'.join(lines)

        # ------------------------------------------------------------------
        # Shared kwargs for both vis functions
        # ------------------------------------------------------------------
        shared = dict(
            x                    = self.embed.coords[:, 0],
            y                    = self.embed.coords[:, 1],
            point_size           = point_size,
            point_size_wts       = rf_norm,
            point_size_wts_trans = rf_trans,
            graph                = self.embed.adjacency,
            edge_size            = 0.3,
            edge_color           = '#BBBBBB',
            xlab                 = xlab,
            ylab                 = ylab,
            title                = title,
            subtitle             = sub,
            caption              = caption,
            legend_pos           = legend_pos,
        )

        # ------------------------------------------------------------------
        # Dispatch to discrete or continuous
        # ------------------------------------------------------------------
        if color_by == 'labels':
            return vis_embedding_discrete(
                z            = self.recaller.WL,
                legend_title = 'Label',
                **shared,
            )
        else:
            z = (
                self.learn_history_[-1]['W_mqe']
                if color_by == 'mqe'
                else rf
            )
            legend_title = 'MQE' if color_by == 'mqe' else 'RF Size'
            return vis_embedding_continuous(
                z            = z,
                cmap         = cmap,
                legend_title = legend_title,
                **shared,
            )

    # ------------------------------------------------------------------
    # Coordinate update
    # ------------------------------------------------------------------

    def update_embedding(self, coords):
        """
        Replace the output-space embedding coordinates and refresh the
        neighbourhood weight matrix atomically.

        Calls ``self.embed.update_coords(coords)`` to validate the new
        positions, rebuild adjacency, and recompute geodesic distances.
        Then immediately recomputes ``self.nbr_W`` from the updated
        topology using the current annealing bandwidth
        ``self.rho_schedule(self.age)``, so that the next :meth:`fit`
        call starts with a consistent neighbourhood structure.

        Parameters
        ----------
        coords : array-like, shape (M, dim)
            New low-dimensional prototype positions.

        Returns
        -------
        self : GTSOM

        Raises
        ------
        RuntimeError
            If :meth:`from_grid` or :meth:`from_data` has not been called.
        ValueError
            Propagated from :meth:`Embedding.update_coords` if the shape
            of ``coords`` does not match the current embedding shape.
        """
        if self.W is None:
            raise RuntimeError(
                "Architecture not initialised. "
                "Call from_grid() or from_data() before update_embedding()."
            )
        self.embed.update_coords(coords)
        self._compute_neighborhood(
            self.rho_schedule(self.age),
            self.alpha_schedule(self.age),
        )
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_architecture(self, X, W, embed, labels):
        """
        Wipe all architecture state and reinitialise from (X, W, embed).

        Called at the end of both :meth:`from_grid` and :meth:`from_data`
        after the topology and prototype matrix have been assembled. Handles
        the full "clean slate" reset — clearing training history, resetting
        age, constructing fresh recaller and kernel instances — then runs
        the initial recall and takes the age-0 snapshot (including the
        initial figure and, if ``self.compute_dr_metrics`` is True, the
        baseline DR metrics).

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
        W : np.ndarray, shape (M, d)
        embed : Embedding
        labels : array-like or None
        """
        # Wipe all mutable architecture state
        self.W          = W
        self.embed      = embed
        self.recaller   = VQRecaller(p=2, max_bmu=2, verbose=False)
        self.kernel     = NeighborKernel(h_min=self.nbr_influence_min)
        self._validator = DataValidator(X)
        self.prevBMU    = np.full(X.shape[0], -1, dtype=int)
        self.W0         = W.copy()
        self.coords0    = embed.coords.copy()
        self.nbr_W      = None

        # Wipe training state
        self.age         = 0
        self.train_time  = 0.0
        self.learn_history_ = []

        # Initial recall then complete age-0 snapshot (including DR metrics
        # and figure, since all learning parameters are already set)
        self._recall(X, labels=labels)
        self._snapshot(include_fig=True)

    def _recall(self, X, labels=None):
        """
        Perform a full recall against current prototypes.

        Calls ``recaller.recall(X=X, W=self.W, labels=labels)``, which
        computes BMU, QE, receptive fields (RF), and the co-adjacency
        matrix (CADJ/CONN). If labels are provided, also computes
        prototype-level label summaries (WL, WL_Dist, WL_Purity).

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
        labels : array-like, shape (N,), default None
        """
        self.recaller.recall(X=X, W=self.W, labels=labels)

    def _compute_W_mqe(self):
        """
        Compute per-prototype mean quantization error from the current recall.

        Returns
        -------
        W_mqe : np.ndarray, shape (M,)
            Per-prototype MQE; np.nan for empty receptive fields.
        """
        bmu    = self.recaller.BMU[:, 0]
        qe     = self.recaller.QE[:, 0]
        counts = np.bincount(bmu, minlength=self.M)
        sums   = np.bincount(bmu, weights=qe, minlength=self.M)
        W_mqe  = np.full(self.M, np.nan)
        active = counts > 0
        W_mqe[active] = np.sqrt(sums[active] / counts[active])
        return W_mqe

    def _snapshot(self, include_fig=False):
        """
        Append a monitoring snapshot for the current state to
        ``self.learn_history_``.

        Operations are performed in this strict order:

        1. Compute ``delBMU`` from ``self.prevBMU`` (before overwriting).
        2. Append the snapshot dict — including ``delBMU`` and ``W_mqe`` —
           so that :meth:`plot` can read from ``learn_history_[-1]``.
        3. Overwrite ``self.prevBMU`` with the current BMU assignments.
        4. Optionally compute DR metrics and store in the snapshot.
        5. Optionally call :meth:`plot` and store the figure.

        Because all learning parameters (including ``compute_dr_metrics``)
        are set in ``__init__`` before any architecture method is called,
        this method produces a complete snapshot on every call — including
        the age-0 call from :meth:`_init_architecture`. No backfill is
        needed.

        Parameters
        ----------
        include_fig : bool, default False
            If True, call :meth:`plot` and store the figure under 'fig'.
        """
        # Step 1
        current_bmu = self.recaller.BMU[:, 0]
        delBMU = float((current_bmu != self.prevBMU).mean())

        # Step 2
        self.learn_history_.append({
            'age'        : self.age,
            'mqe'        : float(np.sqrt(self.recaller.QE[:, 0].mean())),
            'W_mqe'      : self._compute_W_mqe(),
            'delBMU'     : delBMU,
            'dr_metrics' : None,
            'fig'        : None,
        })

        # Step 3
        self.prevBMU = current_bmu.copy()

        # Step 4
        if self.compute_dr_metrics:
            self.learn_history_[-1]['dr_metrics'] = self._snapshot_dr_metrics()

        # Step 5
        if include_fig:
            self.learn_history_[-1]['fig'] = self.plot(color_by='auto')

    def _snapshot_dr_metrics(self):
        """
        Compute DR quality metrics for the current prototype / embedding state.

        Returns
        -------
        DRMetricsResult
        """
        from .dr_metrics import compute_dr_metrics as _compute_dr_metrics
        try:
            return _compute_dr_metrics(
                X                   = self.W,
                Y                   = self.embed.coords,
                embed_geodesic_dist = self.embed.dist,
                CONN                = self.recaller.CONN,
                compute_coranking   = True,
            )
        except ImportError:
            return _compute_dr_metrics(
                X                   = self.W,
                Y                   = self.embed.coords,
                embed_geodesic_dist = self.embed.dist,
                CONN                = self.recaller.CONN,
                compute_coranking   = False,
            )

    def _compute_neighborhood(self, rho, alpha):
        """
        Compute neighbourhood activation weights for the current epoch.

        When ``alpha == 1.0``, pure lattice-based neighbourhood (standard SOM)::

            nbr_W = H_lat   where   H_lat[i, j] = exp(-D_lat[i, j] / rho)

        When ``alpha < 1.0``, a hybrid blend of the lattice neighbourhood and
        an input-space (prototype) neighbourhood is used::

            nbr_W = alpha * H_lat + (1 - alpha) * H_proto

        The form of ``H_proto`` is controlled by ``self.proto_topo``:

        ``'CONN'``
            ``H_proto = H_CONN``: shortest-path hop counts on the unweighted
            CONN graph, fed through the standard exponential kernel.
            ``H_CONN[i,j] = exp(-D_CONN[i,j] / rho)``
            where ``D_CONN[i,j]`` is the hop count on the CONN graph.

        ``'CONN_STK'``
            ``H_proto = H_STK``: CADJ self-tuning kernel. Uses CADJ-weighted
            local radii ``sigma_i`` (CADJ-weighted mean Euclidean distance
            from i to its CADJ neighbours) to compute a locally-normalised
            Euclidean similarity.
            ``H_STK[i,j] = exp(-dist²(W[i],W[j]) / (sigma_i * sigma_j * rho))``

        The result is stored in ``self.nbr_W``.

        Parameters
        ----------
        rho : float
            Neighbourhood bandwidth (current annealed value).
        alpha : float
            Current topology-blending weight. When 1.0, only ``H_lat`` is
            used and ``H_proto`` is not computed.
        """
        H_lat = self.kernel.compute(self.embed.dist, rho)

        if alpha < 1.0 and self.recaller.CONN is not None:
            if self.proto_topo == 'CONN':
                D_CONN  = shortest_path(
                    self.recaller.CONN, method="D",
                    directed=False, unweighted=True,
                ).astype(np.float32)
                H_proto = self.kernel.compute(D_CONN, rho)
            else:  # 'CONN_STK'
                H_proto = _cadj_zelnik_kernel(
                    W                 = self.W,
                    CADJ              = self.recaller.CADJ,
                    CADJ_nhbs         = self.recaller.CADJ_nhbs,
                    rho               = rho,
                    nbr_influence_min = self.nbr_influence_min,
                )
            self.nbr_W = alpha * H_lat + (1 - alpha) * H_proto
        else:
            self.nbr_W = H_lat

    def _update_prototypes(self, X):
        """
        Perform one batch SOM prototype update step.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
        """
        update_prototypes_kernel(
            X,
            self.recaller.BMU[:, 0],
            self.nbr_W,
            self.W,
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self):
        built   = self.W is not None
        fitted  = self.age > 0
        n_snaps = len(self.learn_history_)
        backend = 'numba' if PARALLEL else 'numpy'
        if built:
            arch = f"M={self.M}, d={self.d}, embed_dim={self.embed.dim}, "
        else:
            arch = "not built, "
        return (
            f"GTSOM({arch}"
            f"fitted={fitted}, "
            f"age={self.age}, snapshots={n_snaps}, "
            f"backend={backend!r}, n_jobs={self.n_jobs}, "
            f"nbr_influence_min={self.nbr_influence_min}, "
            f"proto_topo={self.proto_topo!r}, "
            f"alpha_schedule={self.alpha_schedule}, "
            f"compute_dr_metrics={self.compute_dr_metrics}, "
            f"train_time={self.train_time:.3f}s)"
        )