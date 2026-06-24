"""
gtsom.py — General Topology Self-Organizing Map

The GTSOM class implements batch SOM learning over an arbitrary low-dimensional
output topology defined by an Embedding instance. Prototype vectors live in
high-dimensional feature space; their neighbourhood relationships are governed
by geodesic distances on the output manifold.

Typical usage
-------------
# Grid-based (classical SOM style)
som = GTSOM.from_grid(X, shape=(10, 10), coord_init='hex', W_init='random')
som.compile(rho_0=5.0, rho_f=0.5, tau=20)
som.fit(X, n_epochs=20)

# Data-driven (general topology)
som = GTSOM.from_data(X, M=100, coord_dim=2, coord_init='pca', W_init='kmeans')
som.compile(rho_0=5.0, rho_f=0.5, tau=20)
som.fit(X, n_epochs=20)

# Hybrid lattice + CONN-based neighbourhood update
som = GTSOM.from_data(X, M=100, coord_dim=2, coord_init='pca', W_init='kmeans')
som.compile(rho_0=5.0, rho_f=0.5, tau=20,
            nbr_topo_alpha_0=1.0, nbr_topo_alpha_f=0.3)
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
from .utils import ExponentialAnneal, DataValidator, reduce_coords_pca, reduce_coords_le
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
# Main class
# ---------------------------------------------------------------------------

class GTSOM:
    """
    General Topology Self-Organizing Map.

    Do not construct directly for typical use — prefer the classmethods
    :meth:`from_grid` and :meth:`from_data`, which handle all initialisation
    internally and return a ready-to-fit instance.

    Parameters
    ----------
    W : array-like, shape (M, d)
        Initial prototype matrix.
    embed : Embedding
        Output-space embedding. Must already have coords, adjacency, and dist
        populated (i.e. ``compute_topo()`` already called).
    recaller : VQRecaller
        Nearest-prototype search object. Should be freshly constructed
        (unfitted); :meth:`fit` will call ``update_BMU`` each epoch.
    kernel : NeighborKernel
        Neighbourhood activation kernel.
    random_state : int or None, default None
        Seed for all random operations. Stored as an instance variable and
        passed to any internal method that requires a random number generator.

    Notes
    -----
    GTSOM does not perform any internal scaling of X. It is the caller's
    responsibility to whiten, standardise, or otherwise preprocess X before
    passing it to any method. Distances computed during prototype updates and
    BMU search will reflect the scale of the input features.
    """

    def __init__(self, W, embed, recaller, kernel, random_state=None):
        W = np.asarray(W)
        if W.ndim != 2:
            raise ValueError(f"W must be 2-D, got shape {W.shape}")

        M_w = W.shape[0]
        M_e = embed.coords.shape[0]
        if M_w != M_e:
            raise ValueError(
                f"W has {M_w} prototypes but embed has {M_e} neurons."
            )

        self.W = W
        self.embed = embed
        self.recaller = recaller
        self.kernel = kernel
        self.random_state = random_state

        # DataValidator — set by classmethods after X is first seen;
        # used to warn if a different X is passed to fit() or transform()
        self._validator = None

        # Annealing schedule and parallelism — set by compile()
        self.rho_schedule   = None
        self.alpha_schedule = None   # topology-blending annealing; set by compile()
        self.n_jobs         = None   # resolved thread count; None = all cores

        # Fit state — populated during fit()
        self.nbr_W = None

        # Current training age (total epochs completed across all fit() calls)
        self.age = 0

        # Cumulative wall-clock time spent in fit() calls, in seconds.
        # Accumulated with += across successive fit() calls.
        self.train_time = 0.0

        # Initial prototype matrix and embedding coords, stored at construction
        # for reference (e.g. visualising how prototypes evolved from their
        # starting positions). Never modified after initialisation.
        self.W0 = None        # set by classmethods after W is finalised
        self.coords0 = None   # set by classmethods after embed is finalised

        # Previous-epoch BMU assignments, shape (N,), used to compute delBMU.
        # Initialised to a vector of -1s by the classmethods (using N from
        # DataValidator) so that delBMU = 1.0 at age=0 by construction,
        # since -1 never matches a valid BMU index.
        self.prevBMU = None   # set by classmethods to np.full(N, -1, dtype=int)

        # Learning history: list of snapshot dicts, one per recorded epoch.
        # learn_history_[0] is always the post-initialisation state (age=0).
        # Subsequent entries are appended by fit() after each epoch.
        # Each dict contains:
        #   'age'    : int      — self.age at the time of the snapshot
        #   'mqe'    : float    — global mean quantization error over all data
        #   'W_mqe'  : ndarray (M,) — per-prototype MQE, nan for empty RFs
        #   'delBMU' : float    — proportion of data whose BMU changed since
        #                         the previous epoch; 1.0 at age=0 by definition
        #   'fig'    : Figure or None — plot captured this epoch (or None)
        self.learn_history_ = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def M(self):
        """Number of prototypes (neurons)."""
        return self.W.shape[0]

    @property
    def d(self):
        """Dimensionality of the high-dimensional feature space."""
        return self.W.shape[1]

    # ------------------------------------------------------------------
    # Classmethods — initialisation factories
    # ------------------------------------------------------------------

    @classmethod
    def from_grid(
        cls,
        X,
        shape,
        coord_init='hex',
        W_init='random',
        h_min=0.01,
        random_state=None,
        labels=None,
    ):
        """
        Initialise a GTSOM on a regular grid output topology.

        Parameters
        ----------
        X : array-like, shape (N, d)
            Training data. Used for prototype initialisation; not stored.
        shape : tuple of int
            Grid dimensions, e.g. ``(10, 10)`` or ``(5, 10, 4)``. The length
            of the tuple sets the embedding dimension (2 or 3).
        coord_init : {'hex', 'rect'}, default 'hex'
            Grid layout. ``'hex'`` uses a hexagonal arrangement (Delaunay);
            ``'rect'`` uses an 8-connected rectangular grid.
        W_init : {'random', 'pca'}, default 'random'
            How to initialise prototype vectors W.

            ``'random'``
                Random sample of M rows from X.
            ``'pca'``
                Back-project grid neuron coordinates through the PCA of X,
                placing each prototype at the corresponding point in high-d
                feature space.
        h_min : float, default 0.01
            Neighbourhood kernel activation threshold; weights below this
            value are zeroed. Passed to :class:`NeighborKernel`.
        random_state : int or None, default None
            Seed for reproducibility.
        labels : array-like, shape (N,), default None
            Optional observation labels. Any hashable type is accepted
            (int, str, float, etc.). If provided, prototype-level label
            summaries (WL, WL_Dist, WL_Purity) are computed during the
            initial recall and stored on ``self.recaller``.

        Returns
        -------
        GTSOM

        Raises
        ------
        ValueError
            If ``shape`` is not a 2- or 3-tuple, ``coord_init`` or
            ``W_init`` are not recognised, or the grid has more neurons
            than data points.
        """
        validator = DataValidator(X)
        X = np.asarray(X)
        rng = np.random.default_rng(random_state)

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
            W = _init_W_pca(X, embed, random_state)

        recaller = VQRecaller(p=2, max_bmu=2, verbose=False)
        kernel = NeighborKernel(h_min=h_min)

        instance = cls(W, embed, recaller, kernel, random_state=random_state)
        instance._validator = validator
        instance.prevBMU = np.full(validator.shape[0], -1, dtype=int)
        instance.W0 = W.copy()
        instance.coords0 = embed.coords.copy()
        instance._recall(X, labels=labels)
        instance._snapshot(include_fig=True)
        return instance

    @classmethod
    def from_data(
        cls,
        X,
        M,
        coord_dim=2,
        coord_init='pca',
        W_init='kmeans',
        coord_topo='delaunay',
        h_min=0.01,
        random_state=None,
        labels=None,
    ):
        """
        Initialise a GTSOM with a data-driven output topology.

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
        coord_init : {'pca', 'le'}, default 'pca'
            Dimensionality reduction method for computing neuron coordinates
            from the initial prototype vectors.

            ``'pca'``
                Randomised PCA (fast, linear).
            ``'le'``
                Laplacian Eigenmaps via sklearn SpectralEmbedding
                (nonlinear, slower).
        coord_topo : {'delaunay', 'gabriel'}, default 'delaunay'
            Graph used to define the output-space topology from the
            projected prototype coordinates.

            ``'delaunay'``
                Delaunay triangulation. Denser connectivity; every
                prototype is guaranteed at least a few neighbours.
            ``'gabriel'``
                Gabriel graph (a subgraph of Delaunay). Sparser
                connectivity that more closely reflects local proximity
                structure. Guaranteed to be connected for any finite
                point set in general position.
        W_init : {'kmeans', 'random'}, default 'kmeans'
            How to find initial prototype vectors.

            ``'kmeans'``
                FAISS k-means via VQFitter (requires vqlp).
            ``'random'``
                Random sample of M rows from X.
        h_min : float, default 0.01
            Neighbourhood kernel activation threshold; weights below this
            value are zeroed. Passed to :class:`NeighborKernel`.
        random_state : int or None, default None
            Seed for reproducibility. Passed to VQ, dim-reduction, and
            any random sampling steps.
        labels : array-like, shape (N,), default None
            Optional observation labels. Any hashable type is accepted
            (int, str, float, etc.). If provided, prototype-level label
            summaries (WL, WL_Dist, WL_Purity) are computed during the
            initial recall and stored on ``self.recaller``.

        Returns
        -------
        GTSOM

        Raises
        ------
        ValueError
            If ``coord_dim`` is not 2 or 3, ``coord_init``, ``W_init``,
            or ``coord_topo`` are not recognised, or M is out of range.
        ImportError
            If ``W_init='kmeans'`` and vqlp is not installed.
        """
        validator = DataValidator(X)
        X = np.asarray(X)
        rng = np.random.default_rng(random_state)

        if coord_dim not in (2, 3):
            raise ValueError(f"coord_dim must be 2 or 3, got {coord_dim!r}")
        if coord_init not in ('pca', 'le'):
            raise ValueError(
                f"coord_init must be 'pca' or 'le' for from_data, "
                f"got {coord_init!r}"
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

        # Step 1: initialise prototype vectors in high-d
        if W_init == 'kmeans':
            W = _init_W_kmeans(X, M, random_state)
        else:  # 'random'
            W = _init_W_random(X, M, rng)

        # Step 2: project W to coord_dim to define neuron positions
        if coord_init == 'pca':
            coords = reduce_coords_pca(W, coord_dim, random_state)
        else:  # 'le'
            coords = reduce_coords_le(W, coord_dim, random_state)

        # Step 3: build embedding from projected coords using chosen topology
        if coord_topo == 'delaunay':
            embed = Embedding.from_delaunay(coords)
        else:  # 'gabriel'
            embed = Embedding.from_gabriel(coords)

        # Step 4: build recaller and kernel
        recaller = VQRecaller(p=2, max_bmu=2, verbose=False)
        kernel = NeighborKernel(h_min=h_min)

        instance = cls(W, embed, recaller, kernel, random_state=random_state)
        instance._validator = validator
        instance.prevBMU = np.full(validator.shape[0], -1, dtype=int)
        instance.W0 = W.copy()
        instance.coords0 = embed.coords.copy()
        instance._recall(X, labels=labels)
        instance._snapshot(include_fig=True)
        return instance

    # ------------------------------------------------------------------
    # Training configuration
    # ------------------------------------------------------------------

    def compile(self, rho_0, rho_f, tau=None, halflife_epochs=None,
                anneal='exponential', n_jobs=None,
                nbr_topo_alpha_0=1.0, nbr_topo_alpha_f=None):
        """
        Set the neighbourhood bandwidth and topology-blending annealing schedules.

        Must be called at least once before :meth:`fit`. Can be called
        again at any time to change the schedule or blending parameters —
        for example, to slow the decay rate for a refinement phase — without
        affecting ``self.age``, ``self.W``, or ``self.learn_history_``.

        To start learning from scratch, construct a new instance via
        :meth:`from_grid` or :meth:`from_data` rather than calling
        ``compile()`` again.

        Exactly one of ``tau`` or ``halflife_epochs`` must be supplied.
        The resolved time constant is shared across all annealed parameters
        (``rho`` and ``nbr_topo_alpha``).

        Parameters
        ----------
        rho_0 : float
            Initial neighbourhood bandwidth, evaluated at the current
            ``self.age`` on the first subsequent :meth:`fit` epoch.
        rho_f : float
            Final (minimum) neighbourhood bandwidth.
        tau : float, optional
            Exponential decay time constant in epochs, shared across all
            annealed parameters. Mutually exclusive with ``halflife_epochs``.
        halflife_epochs : float, optional
            Convenience alternative to ``tau``. The epoch at which each
            annealed parameter reaches the geometric midpoint between its
            initial and final values — i.e. the half-life of the decay.
            The schedule clips at the final value at ``2 * halflife_epochs``.
            Mutually exclusive with ``tau``.
        anneal : {'exponential'}, default 'exponential'
            Annealing schedule type. Currently only exponential decay
            is supported.
        n_jobs : int or None, default None
            Number of parallel threads for prototype updates.
            ``None`` or ``-1`` uses all available CPU cores.
            ``1`` forces single-threaded execution regardless of whether
            numba is installed (useful for debugging or benchmarking).
            Values > 1 are silently clamped to ``os.cpu_count()``.
            Ignored with a warning if numba is not installed.
        nbr_topo_alpha_0 : float, default 1.0
            Initial value of the topology-blending parameter. Must be
            in [0, 1]. See :meth:`_compute_neighborhood` for a full
            description of the blending rule.
        nbr_topo_alpha_f : float, optional
            Final (minimum) value of the topology-blending parameter.
            Must be in [0, 1] and <= ``nbr_topo_alpha_0``. If not
            supplied, defaults to ``nbr_topo_alpha_0``, producing a
            flat (non-annealing) schedule — equivalent to the fixed
            ``nbr_topo_alpha`` behaviour of earlier versions.

            Annealing ``nbr_topo_alpha`` downward over training reflects
            the intuition that CONN becomes a more reliable guide to
            manifold structure as prototypes settle, so its influence on
            the neighbourhood update should increase over time.

        Raises
        ------
        ValueError
            If neither or both of ``tau`` and ``halflife_epochs`` are
            supplied, if ``anneal`` is not recognised, or if either
            ``nbr_topo_alpha_0`` or ``nbr_topo_alpha_f`` is not in [0, 1].

        Examples
        --------
        >>> som.compile(rho_0=5.0, rho_f=0.3, halflife_epochs=50)
        >>> som.compile(rho_0=5.0, rho_f=0.3, halflife_epochs=50, n_jobs=4)
        >>> # Fixed blending weight (no annealing):
        >>> som.compile(rho_0=5.0, rho_f=0.3, halflife_epochs=50,
        ...             nbr_topo_alpha_0=0.6)
        >>> # Annealing from pure lattice toward CONN-dominated:
        >>> som.compile(rho_0=5.0, rho_f=0.3, halflife_epochs=50,
        ...             nbr_topo_alpha_0=1.0, nbr_topo_alpha_f=0.3)
        """
        if anneal not in ('exponential',):
            raise ValueError(
                f"anneal must be 'exponential', got {anneal!r}. "
                f"Additional schedules may be added in future."
            )
        if tau is None and halflife_epochs is None:
            raise ValueError(
                "Exactly one of tau or halflife_epochs must be supplied."
            )
        if tau is not None and halflife_epochs is not None:
            raise ValueError(
                "Supply either tau or halflife_epochs, not both."
            )

        # Resolve alpha_f: default to alpha_0 for a flat (non-annealing) schedule
        alpha_f = nbr_topo_alpha_f if nbr_topo_alpha_f is not None else nbr_topo_alpha_0

        if not (0.0 <= nbr_topo_alpha_0 <= 1.0):
            raise ValueError(
                f"nbr_topo_alpha_0 must be in [0, 1], got {nbr_topo_alpha_0}."
            )
        if not (0.0 <= alpha_f <= 1.0):
            raise ValueError(
                f"nbr_topo_alpha_f must be in [0, 1], got {alpha_f}."
            )

        if anneal == 'exponential':
            if tau is not None:
                self.rho_schedule = ExponentialAnneal(
                    initial=rho_0, final=rho_f, tau=tau
                )
                self.alpha_schedule = ExponentialAnneal(
                    initial=nbr_topo_alpha_0, final=alpha_f, tau=tau
                )
            else:
                self.rho_schedule = ExponentialAnneal.from_halflife(
                    initial=rho_0, final=rho_f, halflife_epochs=halflife_epochs
                )
                self.alpha_schedule = ExponentialAnneal.from_halflife(
                    initial=nbr_topo_alpha_0, final=alpha_f,
                    halflife_epochs=halflife_epochs
                )

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
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, n_epochs, labels=None, verbose=True, plot_every=5):
        """
        Fit the GTSOM to data X using batch SOM learning.

        :meth:`compile` must be called before :meth:`fit`. The rho
        annealing schedule is owned by ``self.rho_schedule`` and
        advances based on ``self.age``, so successive calls to
        ``fit()`` continue from where the previous call left off.

        Parameters
        ----------
        X : array-like, shape (N, d)
            Training data. Should be the same dataset passed to the
            classmethod used to construct this instance.
        n_epochs : int
            Number of training epochs to run in this call.
        labels : array-like, shape (N,), default None
            Optional observation labels. If provided, prototype-level
            label summaries (WL, WL_Dist, WL_Purity) are recomputed
            each epoch and stored on ``self.recaller``.
        verbose : bool, default True
            If True, print a formatted table of training metrics after
            each epoch. The table header is printed at the start of each
            ``fit()`` call and the separator is reprinted every 10 epochs
            for readability in long terminal runs. VQRecaller output is
            always suppressed regardless of this flag.
        plot_every : int, default 5
            Capture a SOM plot into ``self.learn_history_`` every this many
            epochs. Set to 0 to disable in-training plotting entirely.
            Plots are Figure objects stored in memory; nothing is displayed
            unless the caller explicitly calls ``plt.show()`` or
            ``fig.savefig()``.

        Raises
        ------
        RuntimeError
            If :meth:`compile` has not been called.

        Notes
        -----
        GTSOM does not perform any internal scaling of X. It is the
        caller's responsibility to whiten, standardise, or otherwise
        preprocess X before calling fit.
        """
        if self.rho_schedule is None:
            raise RuntimeError(
                "compile() must be called before fit(). "
                "Call som.compile(rho_0=..., rho_f=..., halflife_epochs=...) first."
            )

        X = np.asarray(X)
        if self._validator is not None:
            self._validator.check(X, context="fit")

        # Column widths are fixed; epoch width scales with total epochs run
        _HDR  = f"{'Epoch':>7}   {'rho':>6}   {'MQE':>8}   {'delBMU':>7}"
        _SEP  = f"{'-----':>7}   {'------':>6}   {'--------':>8}   {'-------':>7}"

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
                    f"{self.age:>7d}   {rho:>6.4f}   "
                    f"{snap['mqe']:>8.4f}   {snap['delBMU']:>7.4f}"
                )
                # Reprint separator every 10 rows for long-run readability
                if (local_epoch + 1) % 10 == 0:
                    print(_SEP)

        if verbose:
            print(_SEP)

        self.train_time += time.perf_counter() - _t0

        # Restore previous thread count
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
            Output coordinates of the BMU for each data point.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.age == 0:
            raise RuntimeError(
                "transform() called before fit(). "
                "Call fit() first to train the model."
            )
        X = np.asarray(X)
        # Recall against current W to get fresh BMU assignments for X
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
        color_by='mqe',
        cmap_continuous='viridis',
        cmap_categorical='tab10',
        ax=None,
        title='SOM Learning',
        subtitle=None,
        xlabel=r'SOM$_1$',
        ylabel=r'SOM$_2$',
    ):
        """
        Plot the SOM in output (lattice) space.

        Neurons are drawn at their ``embed.coords`` positions, connected
        by edges from ``embed.adjacency``, and coloured according to
        ``color_by``. Nothing is displayed automatically — the returned
        Figure must be explicitly shown (``plt.show()``) or saved
        (``fig.savefig()``).

        Parameters
        ----------
        color_by : {'mqe', 'rfsize', 'labels'}, default 'mqe'
            Quantity used to colour neurons.

            ``'mqe'``
                Per-prototype mean quantization error from ``learn_history_``.
                Always available after any recall. Continuous colormap.
            ``'rfsize'``
                Receptive field size (``recaller.RFSize``). Always
                available after any recall. Continuous colormap.
            ``'labels'``
                Winning label per prototype (``recaller.WL``). Requires
                that labels were passed to the constructor or to
                ``fit()``. Categorical colormap. Raises ``ValueError``
                if ``recaller.WL`` is None.
        cmap_continuous : str or Colormap, default 'viridis'
            Matplotlib colormap for continuous quantities ('mqe',
            'rfsize').
        cmap_categorical : str or Colormap, default 'tab10'
            Matplotlib colormap for categorical quantities ('labels').
        ax : matplotlib.axes.Axes or None, default None
            Axes to draw into. If None, a new Figure and Axes are
            created internally.
        title : str or None, default 'SOM Learning'
            Main title line. Pass None to suppress.
        subtitle : str or None, default None
            Second title line. If None, auto-generates
            ``f'Epoch = {self.age}'``.
        xlabel : str, default 'SOM_1'
            Label for the horizontal axis.
        ylabel : str, default 'SOM_2'
            Label for the vertical axis.

        Returns
        -------
        fig : matplotlib.figure.Figure

        Raises
        ------
        NotImplementedError
            If ``embed.dim != 2``.
        ValueError
            If ``color_by='labels'`` but no labels have been provided.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.patches import Patch

        if self.embed.dim != 2:
            raise NotImplementedError(
                f"plot() only supports 2-D embeddings; "
                f"embed.dim={self.embed.dim}."
            )

        if color_by not in ('mqe', 'rfsize', 'labels'):
            raise ValueError(
                f"color_by must be 'mqe', 'rfsize', or 'labels', "
                f"got {color_by!r}."
            )

        if color_by == 'labels' and self.recaller.WL is None:
            raise ValueError(
                "color_by='labels' requires labels to have been provided "
                "at construction or during fit(). "
                "recaller.WL is currently None."
            )

        # ------------------------------------------------------------------
        # Resolve axes / figure
        # ------------------------------------------------------------------
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
        else:
            fig = ax.figure

        # ------------------------------------------------------------------
        # Draw adjacency edges (upper triangle only — matrix is symmetric)
        # ------------------------------------------------------------------
        cx = self.embed.adjacency.tocoo()
        coords = self.embed.coords
        for i, j in zip(cx.row, cx.col):
            if i < j:
                ax.plot(
                    [coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    color='dimgrey', lw=0.8, alpha=0.5, zorder=1,
                )

        # ------------------------------------------------------------------
        # Colour neurons
        # ------------------------------------------------------------------
        if color_by in ('mqe', 'rfsize'):
            if color_by == 'mqe':
                values = self.learn_history_[-1]['W_mqe']   # (M,), nan for empty RFs
                cb_label = 'MQE'
            else:
                values = self.recaller.RFSize.astype(float)
                cb_label = 'RF Size'

            cmap = plt.get_cmap(cmap_continuous)
            finite = np.isfinite(values)
            vmin = float(np.nanmin(values)) if finite.any() else 0.0
            vmax = float(np.nanmax(values)) if finite.any() else 1.0
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            # Finite-valued neurons: coloured by cmap
            sc = ax.scatter(
                coords[finite, 0], coords[finite, 1],
                c=values[finite], cmap=cmap, norm=norm,
                s=60, zorder=2, edgecolors='black', linewidths=0.4,
            )
            fig.colorbar(sc, ax=ax, label=cb_label, shrink=0.85)

            # Empty-RF neurons (nan): grey
            empty = ~finite
            if empty.any():
                ax.scatter(
                    coords[empty, 0], coords[empty, 1],
                    c='lightgrey', s=60, zorder=2,
                    edgecolors='black', linewidths=0.4,
                )

        else:  # color_by == 'labels'
            WL = self.recaller.WL          # object array (M,), None for empty
            unq_labels = self.recaller.WL_unq
            n_labels = len(unq_labels)
            cmap = plt.get_cmap(cmap_categorical, n_labels)
            label_to_color = {lbl: cmap(i) for i, lbl in enumerate(unq_labels)}

            node_colors = [
                label_to_color[lbl] if lbl is not None else 'lightgrey'
                for lbl in WL
            ]
            ax.scatter(
                coords[:, 0], coords[:, 1],
                c=node_colors, s=60, zorder=2,
                edgecolors='black', linewidths=0.4,
            )

            # Legend
            legend_elements = [
                Patch(facecolor=label_to_color[lbl], edgecolor='black',
                      label=str(lbl))
                for lbl in unq_labels
            ]
            has_empty = any(lbl is None for lbl in WL)
            if has_empty:
                legend_elements.append(
                    Patch(facecolor='lightgrey', edgecolor='black',
                          label='Empty RF')
                )
            ax.legend(handles=legend_elements, loc='best', fontsize=8,
                      title='Label')

        # ------------------------------------------------------------------
        # Titles and axis labels
        # ------------------------------------------------------------------
        sub = subtitle if subtitle is not None else f'Epoch = {self.age}'
        if title is not None:
            ax.set_title(f'{title}\n{sub}', fontsize=10)
        else:
            ax.set_title(sub, fontsize=10)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal', adjustable='datalim')

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recall(self, X, labels=None):
        """
        Perform a full recall against current prototypes.

        Calls ``recaller.recall(X=X, W=self.W, labels=labels)``, which
        computes BMU, QE, receptive fields (RF), and the co-adjacency
        matrix (CADJ/CONN). If labels are provided, also computes
        prototype-level label summaries (WL, WL_Dist, WL_Purity).
        Should be called after any change to ``self.W``.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
        labels : array-like, shape (N,), default None
            Optional observation labels passed through to
            ``recaller.recall_labels()``.
        """
        self.recaller.recall(X=X, W=self.W, labels=labels)

    def _compute_W_mqe(self):
        """
        Compute per-prototype mean quantization error from the current recall.

        Averages ``recaller.QE[:, 0]`` over each prototype's receptive
        field using ``np.bincount`` for efficiency. Prototypes with empty
        RFs receive ``np.nan``.

        Returns
        -------
        W_mqe : np.ndarray, shape (M,)
        """
        bmu    = self.recaller.BMU[:, 0]   # (N,) first BMU per datum
        qe     = self.recaller.QE[:, 0]    # (N,) per-datum QE
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
        4. Optionally call :meth:`plot` and store the Figure.

        Returns nothing; callers must not wrap this call in ``append()``.

        Called by classmethods (after initialisation) and by :meth:`fit`
        after each epoch.

        Parameters
        ----------
        include_fig : bool, default False
            If True, call :meth:`plot` and store the Figure under the
            'fig' key. If False, 'fig' is None.

        Each snapshot dict contains:
            'age'    : int      — self.age at snapshot time
            'mqe'    : float    — global MQE over all data
            'W_mqe'  : ndarray (M,) — per-prototype MQE, nan for empty RFs
            'delBMU' : float    — proportion of data whose BMU changed since
                                   the previous epoch; 1.0 at age=0
            'fig'    : Figure or None
        """
        # Step 1: compute delBMU before prevBMU is overwritten
        current_bmu = self.recaller.BMU[:, 0]
        delBMU = float((current_bmu != self.prevBMU).mean())

        # Step 2: append snapshot (plot() can now read W_mqe and delBMU)
        self.learn_history_.append({
            'age'    : self.age,
            'mqe'    : float(np.sqrt(self.recaller.QE[:, 0].mean())),
            'W_mqe'  : self._compute_W_mqe(),
            'delBMU' : delBMU,
            'fig'    : None,
        })

        # Step 3: overwrite prevBMU for the next epoch's delBMU computation
        self.prevBMU = current_bmu.copy()

        # Step 4: optionally capture plot (reads learn_history_[-1])
        if include_fig:
            self.learn_history_[-1]['fig'] = self.plot()

    def _compute_neighborhood(self, rho, alpha):
        """
        Compute neighbourhood activation weights for the current epoch.

        When ``alpha == 1.0``, this reduces to the standard SOM rule:
        neighbourhood weights are derived solely from geodesic hop-count
        distances on the lattice embedding::

            nbr_W = H_lat   where   H_lat[i, j] = exp(-D_lat[i, j] / rho)

        When ``alpha < 1.0``, a hybrid update rule is used that blends
        lattice-based and CONN-based neighbourhood influences::

            nbr_W = alpha * H_lat + (1 - alpha) * H_CONN

        ``H_CONN[i, j] = exp(-D_CONN[i, j] / rho)``, where ``D_CONN`` is
        the matrix of shortest hop-count paths on the binary CONN graph
        (computed from ``self.recaller.CONN`` with edge weights ignored).
        Prototype pairs that are disconnected in CONN have
        ``D_CONN[i, j] = inf``, so ``H_CONN[i, j] = 0`` for those pairs
        — they receive only the dampened lattice signal
        ``alpha * H_lat[i, j]``.

        Both ``H_lat`` and ``H_CONN`` use the same bandwidth ``rho`` and
        the same exponential decay kernel, so their values lie in [0, 1]
        and the linear blend is well-posed.

        The result is stored in ``self.nbr_W`` as a sparse CSR matrix of
        shape (M, M) and consumed by :meth:`_update_prototypes`.

        Parameters
        ----------
        rho : float or np.ndarray, shape (M,)
            Neighbourhood bandwidth; scalar or per-prototype array.
        alpha : float
            Current topology-blending weight, evaluated from
            ``self.alpha_schedule`` at the current epoch. When 1.0,
            only ``H_lat`` is computed (no CONN overhead).
        """
        H_lat = self.kernel.compute(self.embed.dist, rho)
        if alpha < 1.0 and self.recaller.CONN is not None:
            D_CONN = shortest_path(
                self.recaller.CONN, method="D", directed=False, unweighted=True
            ).astype(np.float32)
            H_CONN = self.kernel.compute(D_CONN, rho)
            self.nbr_W = alpha * H_lat + (1 - alpha) * H_CONN
        else:
            self.nbr_W = H_lat

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

        This method is the correct way to inject an externally computed
        coordinate update (e.g. from a tSNE or UMAP step) into a fitted
        GTSOM without leaving ``nbr_W`` stale.

        Parameters
        ----------
        coords : array-like, shape (M, dim)
            New low-dimensional prototype positions. Must match the
            current embedding shape ``(M, dim)`` exactly.

        Returns
        -------
        self : GTSOM
            Returns self for method chaining.

        Raises
        ------
        RuntimeError
            If :meth:`compile` has not been called yet (``self.rho_schedule``
            is None). The neighbourhood weight matrix cannot be recomputed
            without a valid annealing schedule.
        ValueError
            Propagated from :meth:`Embedding.update_coords` if the shape
            of ``coords`` does not match the current embedding shape.
        """
        if self.rho_schedule is None:
            raise RuntimeError(
                "compile() must be called before update_embedding(). "
                "Call som.compile(rho_0=..., rho_f=..., halflife_epochs=...) "
                "first so that a valid rho_schedule is available for "
                "recomputing the neighbourhood weight matrix."
            )
        self.embed.update_coords(coords)
        self._compute_neighborhood(
            self.rho_schedule(self.age),
            self.alpha_schedule(self.age),
        )
        return self

    def _update_prototypes(self, X):
        """
        Perform one batch SOM prototype update step.

        Delegates to ``update_prototypes_kernel`` from ``parallel.py``,
        which selects either the numba parallel backend or the numpy
        serial fallback depending on whether numba is installed.

        In both cases the update rule is identical: for each datum i
        with BMU b, scatter its contribution through b's neighbourhood::

            sum_HX[j] += nbr_W[b, j] * X[i]
            sum_H[j]  += nbr_W[b, j]

        Then update prototypes as::

            W[j] = sum_HX[j] / sum_H[j]   (only where sum_H[j] > 0)

        Prototypes with zero support are left unchanged.

        See ``parallel.py`` for full implementation notes on the
        parallel scatter-accumulate and thread-local accumulator design.

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
        compiled = self.rho_schedule is not None
        fitted   = self.age > 0
        n_snaps  = len(self.learn_history_)
        backend  = 'numba' if PARALLEL else 'numpy'
        return (
            f"GTSOM(M={self.M}, d={self.d}, "
            f"embed_dim={self.embed.dim}, "
            f"compiled={compiled}, fitted={fitted}, "
            f"age={self.age}, snapshots={n_snaps}, "
            f"backend={backend!r}, n_jobs={self.n_jobs}, "
            f"alpha_schedule={self.alpha_schedule}, "
            f"train_time={self.train_time:.3f}s)"
        )