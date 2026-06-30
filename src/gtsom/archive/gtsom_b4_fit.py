"""
gtsom.py — General Topology Self-Organizing Map

The GTSOM class implements batch SOM learning over an arbitrary low-dimensional
output topology defined by an Embedding instance. Prototype vectors live in
high-dimensional feature space; their neighbourhood relationships are governed
by geodesic distances on the output manifold.

Typical usage
-------------
# Grid-based (classical SOM style)
som = GTSOM.from_grid(X, shape=(10, 10), coord_init='hex', w_init='random')
som.compile(rho_0=5.0, rho_f=0.5, rho_tau=20)
som.fit(X, n_epochs=20)

# Data-driven (general topology)
som = GTSOM.from_data(X, M=100, coord_dim=2, coord_init='pca', w_init='kmeans')
som.compile(rho_0=5.0, rho_f=0.5, rho_tau=20)
som.fit(X, n_epochs=20)

Notes
-----
GTSOM does not perform any internal scaling of X. It is the caller's
responsibility to whiten, standardise, or otherwise preprocess X before
passing it to any method. Distances computed during prototype updates and
BMU search will reflect the scale of the input features.
"""

import warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding

from .embedding import Embedding
from .kernel import NeighborKernel
from .utils import ExponentialAnneal
from vqlp import VQRecaller


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _fingerprint_X(X):
    """
    Compute a lightweight summary of X for consistency checking across calls.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, assumed already validated as 2-D.

    Returns
    -------
    tuple
        (shape, dtype, mean, std, min, max). Sufficient to catch most cases
        of mismatched datasets without being expensive to compute. Note that
        if X is pre-whitened, mean≈0 and std≈1 by construction, but shape,
        dtype, min, and max still discriminate between datasets.
    """
    return (
        X.shape,
        X.dtype,
        float(X.mean()),
        float(X.std()),
        float(X.min()),
        float(X.max()),
    )


def _validate_X(X):
    """
    Validate and return X as a 2-D numpy array.

    Does not cast dtype — the caller is responsible for any required
    type conversion (e.g. FAISS requires float32, handled internally
    by VQRecaller).

    Parameters
    ----------
    X : array-like

    Returns
    -------
    X : np.ndarray, shape (N, d)
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    return X


def _check_X(X, fingerprint, context=""):
    """
    Warn if X does not match a previously recorded fingerprint.

    Parameters
    ----------
    X : np.ndarray
        Incoming data matrix, already validated as 2-D.
    fingerprint : tuple
        Fingerprint produced by _fingerprint_X at initialisation time.
    context : str
        Optional caller name included in the warning message.
    """
    current = _fingerprint_X(X)
    if current != fingerprint:
        prefix = f"{context}: " if context else ""
        warnings.warn(
            f"{prefix}X does not match the dataset used at initialisation. "
            f"Expected shape={fingerprint[0]}, dtype={fingerprint[1]}, "
            f"mean={fingerprint[2]:.4g}, std={fingerprint[3]:.4g}, "
            f"min={fingerprint[4]:.4g}, max={fingerprint[5]:.4g}. "
            f"Got shape={current[0]}, dtype={current[1]}, "
            f"mean={current[2]:.4g}, std={current[3]:.4g}, "
            f"min={current[4]:.4g}, max={current[5]:.4g}.",
            UserWarning,
            stacklevel=3,
        )


def _random_subset(X, M, rng):
    """
    Draw M rows from X without replacement.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
    M : int
    rng : np.random.Generator

    Returns
    -------
    np.ndarray, shape (M, d)
    """
    idx = rng.choice(X.shape[0], size=M, replace=False)
    return X[idx].copy()


def _init_w_random(X, M, rng):
    """
    Initialise prototype matrix by random sampling from X.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
    M : int
    rng : np.random.Generator

    Returns
    -------
    W : np.ndarray, shape (M, d)
    """
    return _random_subset(X, M, rng)


def _init_w_pca(X, embed, random_state):
    """
    Initialise prototype matrix via PCA back-projection.

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
    k = embed.dim  # 2 or 3
    pca = PCA(n_components=k, svd_solver='randomized', random_state=random_state)
    pca.fit(X)

    # Rescale embed.coords to match the range of PCA score space per component
    scores = pca.transform(X)   # (N, k)
    coords = embed.coords.copy()  # (M, k)

    for dim in range(k):
        s_min, s_max = scores[:, dim].min(), scores[:, dim].max()
        c_min, c_max = coords[:, dim].min(), coords[:, dim].max()
        if c_max - c_min > 0:
            coords[:, dim] = (coords[:, dim] - c_min) / (c_max - c_min)
            coords[:, dim] = coords[:, dim] * (s_max - s_min) + s_min

    # Back-project: W = coords @ components + mean
    W = coords @ pca.components_ + pca.mean_
    return W.astype(X.dtype)


def _init_w_kmeans(X, M, random_state):
    """
    Initialise prototype matrix via FAISS k-means (requires vqlp).

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
            "vqlp package is required for w_init='kmeans'. "
            "Install it or use w_init='random'."
        )
    vqf = VQFitter(M=M, p=2, max_bmu=1, random_state=random_state)
    vqf.fit(X, method='kmeans')
    return vqf.W.astype(X.dtype)


def _reduce_coords_pca(W, coord_dim, random_state):
    """
    Reduce M prototype vectors to coord_dim dimensions via randomised PCA.

    Parameters
    ----------
    W : np.ndarray, shape (M, d)
    coord_dim : int
        2 or 3.
    random_state : int or None

    Returns
    -------
    coords : np.ndarray, shape (M, coord_dim), float32
    """
    pca = PCA(n_components=coord_dim, svd_solver='randomized',
              random_state=random_state)
    coords = pca.fit_transform(W)
    return coords.astype(np.float32)


def _reduce_coords_le(W, coord_dim, random_state):
    """
    Reduce M prototype vectors to coord_dim dimensions via Laplacian Eigenmaps.

    Uses sklearn's SpectralEmbedding with a nearest-neighbours affinity graph
    built from the high-d prototype vectors.

    Parameters
    ----------
    W : np.ndarray, shape (M, d)
    coord_dim : int
        2 or 3.
    random_state : int or None

    Returns
    -------
    coords : np.ndarray, shape (M, coord_dim), float32
    """
    se = SpectralEmbedding(
        n_components=coord_dim,
        affinity='nearest_neighbors',
        random_state=random_state,
    )
    coords = se.fit_transform(W)
    return coords.astype(np.float32)


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

        # X fingerprint — set by classmethods, checked in fit/transform
        self._X_fingerprint = None

        # Annealing schedule — set by compile()
        self.rho_schedule = None

        # Fit state — populated during fit()
        self.nbr_w = None

        # Fit history
        self.age = 0
        self.mqe_history_ = []

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
        w_init='random',
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
        w_init : {'random', 'pca'}, default 'random'
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
            ``w_init`` are not recognised, or the grid has more neurons
            than data points.
        """
        X = _validate_X(X)
        rng = np.random.default_rng(random_state)

        if coord_init not in ('hex', 'rect'):
            raise ValueError(
                f"coord_init must be 'hex' or 'rect' for from_grid, "
                f"got {coord_init!r}"
            )
        if w_init not in ('random', 'pca'):
            raise ValueError(
                f"w_init must be 'random' or 'pca' for from_grid, "
                f"got {w_init!r}"
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
        if w_init == 'random':
            W = _init_w_random(X, M, rng)
        else:  # 'pca'
            W = _init_w_pca(X, embed, random_state)

        recaller = VQRecaller(p=2, max_bmu=2)
        kernel = NeighborKernel(h_min=h_min)

        instance = cls(W, embed, recaller, kernel, random_state=random_state)
        instance._X_fingerprint = _fingerprint_X(X)
        instance._recall(X, labels=labels)
        return instance

    @classmethod
    def from_data(
        cls,
        X,
        M,
        coord_dim=2,
        coord_init='pca',
        w_init='kmeans',
        h_min=0.01,
        random_state=None,
        labels=None,
    ):
        """
        Initialise a GTSOM with a data-driven output topology.

        Prototype vectors are found via vector quantisation of X, projected
        to ``coord_dim`` dimensions to define neuron positions, and a Delaunay
        triangulation of those positions defines the output topology.

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
        w_init : {'kmeans', 'random'}, default 'kmeans'
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
            If ``coord_dim`` is not 2 or 3, ``coord_init`` or ``w_init``
            are not recognised, or M is out of range.
        ImportError
            If ``w_init='kmeans'`` and vqlp is not installed.
        """
        X = _validate_X(X)
        rng = np.random.default_rng(random_state)

        if coord_dim not in (2, 3):
            raise ValueError(f"coord_dim must be 2 or 3, got {coord_dim!r}")
        if coord_init not in ('pca', 'le'):
            raise ValueError(
                f"coord_init must be 'pca' or 'le' for from_data, "
                f"got {coord_init!r}"
            )
        if w_init not in ('kmeans', 'random'):
            raise ValueError(
                f"w_init must be 'kmeans' or 'random' for from_data, "
                f"got {w_init!r}"
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
        if w_init == 'kmeans':
            W = _init_w_kmeans(X, M, random_state)
        else:  # 'random'
            W = _init_w_random(X, M, rng)

        # Step 2: project prototypes to coord_dim to define neuron positions
        if coord_init == 'pca':
            coords = _reduce_coords_pca(W, coord_dim, random_state)
        else:  # 'le'
            coords = _reduce_coords_le(W, coord_dim, random_state)

        # Step 3: build embedding via Delaunay triangulation of those coords
        embed = Embedding.from_delaunay(coords)

        # Step 4: build recaller and kernel
        recaller = VQRecaller(p=2, max_bmu=2)
        kernel = NeighborKernel(h_min=h_min)

        instance = cls(W, embed, recaller, kernel, random_state=random_state)
        instance._X_fingerprint = _fingerprint_X(X)
        instance._recall(X, labels=labels)
        return instance

    # ------------------------------------------------------------------
    # Training configuration
    # ------------------------------------------------------------------

    def compile(self, rho_0, rho_f, rho_tau=None, target_epochs=None,
                anneal='exponential'):
        """
        Set the learning schedule before calling :meth:`fit`.

        Must be called at least once before :meth:`fit`. Calling
        ``compile()`` again resets ``age`` to 0 and clears
        ``mqe_history_``, starting a fresh training run.

        Exactly one of ``rho_tau`` or ``target_epochs`` must be supplied.

        Parameters
        ----------
        rho_0 : float
            Initial neighbourhood bandwidth at age=0.
        rho_f : float
            Final (minimum) neighbourhood bandwidth. The schedule is
            clipped at this value once ``target_epochs`` has been
            reached.
        rho_tau : float, optional
            Exponential decay time constant in epochs. After ``rho_tau``
            epochs, rho has decayed to approximately 37% of its initial
            range. Mutually exclusive with ``target_epochs``.
        target_epochs : float, optional
            Convenience alternative to ``rho_tau``. Specifies the number
            of epochs over which rho should decay from ``rho_0`` to
            ``rho_f``. ``rho_tau`` is computed internally. Mutually
            exclusive with ``rho_tau``.
        anneal : {'exponential'}, default 'exponential'
            Annealing schedule type. Currently only exponential decay
            is supported.

        Raises
        ------
        ValueError
            If neither or both of ``rho_tau`` and ``target_epochs`` are
            supplied, or if ``anneal`` is not recognised.

        Examples
        --------
        Direct tau:

        >>> som.compile(rho_0=5.0, rho_f=0.3, rho_tau=20)

        From target epoch count:

        >>> som.compile(rho_0=5.0, rho_f=0.3, target_epochs=50)
        """
        # Validate anneal type
        if anneal not in ('exponential',):
            raise ValueError(
                f"anneal must be 'exponential', got {anneal!r}. "
                f"Additional schedules may be added in future."
            )

        # Exactly one of rho_tau / target_epochs must be given
        if rho_tau is None and target_epochs is None:
            raise ValueError(
                "Exactly one of rho_tau or target_epochs must be supplied."
            )
        if rho_tau is not None and target_epochs is not None:
            raise ValueError(
                "Supply either rho_tau or target_epochs, not both."
            )

        # Build the schedule
        if anneal == 'exponential':
            if rho_tau is not None:
                self.rho_schedule = ExponentialAnneal(
                    initial=rho_0, final=rho_f, tau=rho_tau
                )
            else:
                self.rho_schedule = ExponentialAnneal.from_target(
                    initial=rho_0, final=rho_f, target_epochs=target_epochs
                )

        # Reset training state — fresh run
        self.age = 0
        self.mqe_history_ = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, n_epochs, labels=None, verbose=True):
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
            If True, print rho and MQE after each epoch.

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
        raise NotImplementedError("fit() not yet implemented.")

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
        """
        raise NotImplementedError("transform() not yet implemented.")

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
    # Private helpers — stubs
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

    def _anneal(self, epoch, n_epochs, rho_0, rho_f, schedule):
        """
        Return rho for the current epoch under the chosen annealing schedule.

        Parameters
        ----------
        epoch : int
            Current epoch (0-based).
        n_epochs : int
            Total number of epochs.
        rho_0 : float
            Initial bandwidth.
        rho_f : float
            Final bandwidth.
        schedule : {'exponential', 'linear'}

        Returns
        -------
        rho : float
        """
        raise NotImplementedError

    def _compute_neighborhood(self, rho):
        """
        Compute neighbourhood activation weights for the current epoch.

        Calls ``kernel.compute(embed.dist, rho)`` and stores the result
        in ``self.nbr_w``.

        Parameters
        ----------
        rho : float or np.ndarray, shape (M,)
            Neighbourhood bandwidth; scalar or per-prototype array.
        """
        raise NotImplementedError

    def _update_prototypes(self, X):
        """
        Perform one batch SOM prototype update step.

        For each datum, scatters its contribution through the neighbourhood
        of its BMU, then normalises to compute new prototype positions.
        Prototypes with no data support in the current epoch are left
        unchanged.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self):
        compiled = self.rho_schedule is not None
        fitted = self.age > 0
        return (
            f"GTSOM(M={self.M}, d={self.d}, "
            f"embed_dim={self.embed.dim}, "
            f"compiled={compiled}, "
            f"fitted={fitted}, age={self.age})"
        )

        