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
som.compile(rho_0=5.0, rho_f=0.5, rho_tau=20)
som.fit(X, n_epochs=20)

# Data-driven (general topology)
som = GTSOM.from_data(X, M=100, coord_dim=2, coord_init='pca', W_init='kmeans')
som.compile(rho_0=5.0, rho_f=0.5, rho_tau=20)
som.fit(X, n_epochs=20)

Notes
-----
GTSOM does not perform any internal scaling of X. It is the caller's
responsibility to whiten, standardise, or otherwise preprocess X before
passing it to any method. Distances computed during prototype updates and
BMU search will reflect the scale of the input features.
"""

import numpy as np

from .embedding import Embedding
from .kernel import NeighborKernel
from .utils import ExponentialAnneal, DataValidator, reduce_coords_pca, reduce_coords_le
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

        # Annealing schedule — set by compile()
        self.rho_schedule = None

        # Fit state — populated during fit()
        self.nbr_W = None

        # Current training age (total epochs completed across all fit() calls)
        self.age = 0

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
            If ``coord_dim`` is not 2 or 3, ``coord_init`` or ``W_init``
            are not recognised, or M is out of range.
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

        # Step 3: build embedding via Delaunay triangulation of those coords
        embed = Embedding.from_delaunay(coords)

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

    def compile(self, rho_0, rho_f, rho_tau=None, target_epochs=None,
                anneal='exponential'):
        """
        Set the neighbourhood bandwidth annealing schedule.

        Must be called at least once before :meth:`fit`. Can be called
        again at any time to change the schedule — for example, to slow
        the decay rate for a refinement phase — without affecting
        ``self.age``, ``self.W``, or ``self.learn_history_``.

        To start learning from scratch, construct a new instance via
        :meth:`from_grid` or :meth:`from_data` rather than calling
        ``compile()`` again.

        Exactly one of ``rho_tau`` or ``target_epochs`` must be supplied.

        Parameters
        ----------
        rho_0 : float
            Initial neighbourhood bandwidth, evaluated at the current
            ``self.age`` on the first subsequent :meth:`fit` epoch.
        rho_f : float
            Final (minimum) neighbourhood bandwidth.
        rho_tau : float, optional
            Exponential decay time constant in epochs. Mutually exclusive
            with ``target_epochs``.
        target_epochs : float, optional
            Convenience alternative to ``rho_tau``. The number of epochs
            over which rho decays from ``rho_0`` to ``rho_f``.
            Mutually exclusive with ``rho_tau``.
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
        >>> som.compile(rho_0=5.0, rho_f=0.3, rho_tau=20)
        >>> som.compile(rho_0=5.0, rho_f=0.3, target_epochs=50)
        """
        if anneal not in ('exponential',):
            raise ValueError(
                f"anneal must be 'exponential', got {anneal!r}. "
                f"Additional schedules may be added in future."
            )
        if rho_tau is None and target_epochs is None:
            raise ValueError(
                "Exactly one of rho_tau or target_epochs must be supplied."
            )
        if rho_tau is not None and target_epochs is not None:
            raise ValueError(
                "Supply either rho_tau or target_epochs, not both."
            )

        if anneal == 'exponential':
            if rho_tau is not None:
                self.rho_schedule = ExponentialAnneal(
                    initial=rho_0, final=rho_f, tau=rho_tau
                )
            else:
                self.rho_schedule = ExponentialAnneal.from_target(
                    initial=rho_0, final=rho_f, target_epochs=target_epochs
                )

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
                "Call som.compile(rho_0=..., rho_f=..., target_epochs=...) first."
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

        for local_epoch in range(n_epochs):
            rho = self.rho_schedule(self.age)
            self._compute_neighborhood(rho)
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

    def _compute_neighborhood(self, rho):
        """
        Compute neighbourhood activation weights for the current epoch.

        Calls ``kernel.compute(embed.dist, rho)`` and stores the result
        in ``self.nbr_W`` as a sparse CSR matrix of shape (M, M).

        Parameters
        ----------
        rho : float or np.ndarray, shape (M,)
            Neighbourhood bandwidth; scalar or per-prototype array.
        """
        self.nbr_W = self.kernel.compute(self.embed.dist, rho)

    def _update_prototypes(self, X):
        """
        Perform one batch SOM prototype update step.

        For each datum, scatters its contribution through the neighbourhood
        of its BMU, then normalises to compute new prototype positions.
        Prototypes with no data support in the current epoch are left
        unchanged.

        The update rule mirrors the C++ SOM_Prototype_Update worker:
        for each unique BMU b, all data points assigned to b contribute
        their neighbourhood-weighted values to every neuron j in b's
        neighbourhood::

            sum_HX[j] += nbr_W[b, j] * X[i]
            sum_H[j]  += nbr_W[b, j]

        After accumulating over all data, prototypes are updated as::

            W[j] = sum_HX[j] / sum_H[j]   (only where sum_H[j] > 0)

        Prototypes with zero support (sum_H[j] == 0) are left unchanged.

        Implementation notes
        --------------------
        Data points are grouped by BMU via a single argsort, giving each BMU
        a contiguous slice of the sorted index array. This costs O(N log N)
        once per epoch and avoids M separate O(N) boolean mask scans.

        Neighbourhood indices within each BMU row are always unique, so the
        scatter into sum_HX / sum_H uses direct fancy-index assignment
        (``sum_HX[js] +=``), which is buffered and faster than
        ``np.add.at`` (which is only required when indices may repeat).

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
        """
        M, d = self.M, self.d
        BMU = self.recaller.BMU[:, 0]   # (N,) first BMU for each datum

        sum_HX = np.zeros((M, d), dtype=np.float64)
        sum_H  = np.zeros(M,      dtype=np.float64)

        # --- Group data points by BMU via a single sort ---
        # order[start:end] gives the row indices of all points whose BMU is b.
        # This replaces M separate O(N) boolean mask scans with one O(N log N)
        # sort and O(M) boundary lookups.
        order      = np.argsort(BMU)           # O(N log N), unstable is fine
        sorted_bmu = BMU[order]                # BMU values in sorted order
        unique_bmus, first_occurrence = np.unique(sorted_bmu, return_index=True)
        # Slice boundaries: points for unique_bmus[i] live at
        # order[ first_occurrence[i] : first_occurrence[i+1] ]
        boundaries = np.empty(len(unique_bmus) + 1, dtype=np.intp)
        boundaries[:-1] = first_occurrence
        boundaries[-1]  = len(BMU)

        for i, b in enumerate(unique_bmus):
            # Contiguous slice of sorted indices whose BMU is b
            X_b = X[order[boundaries[i]:boundaries[i + 1]]]   # (n_b, d)

            # Neighbourhood weights for BMU b: sparse row -> arrays
            nbr_row = self.nbr_W.getrow(b)   # (1, M) sparse CSR
            js = nbr_row.indices              # neighbour prototype indices
            hs = nbr_row.data                 # corresponding weights

            if js.size == 0:
                continue

            # Aggregate all data in this BMU's RF, then scatter to neighbours.
            # sum_HX[j] += h_{b,j} * sum_{i: BMU_i=b}(X[i])
            # sum_H[j]  += h_{b,j} * n_b
            X_b_sum = X_b.sum(axis=0)    # (d,)
            n_b     = X_b.shape[0]

            # Fancy-index assignment is safe here because js contains unique
            # indices (no prototype appears twice in a neighbourhood row).
            sum_HX[js] += hs[:, None] * X_b_sum[None, :]
            sum_H[js]  += hs * n_b

        # Update only prototypes that received support
        active = sum_H > 0.0
        self.W[active] = (sum_HX[active] / sum_H[active, None]).astype(self.W.dtype)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self):
        compiled = self.rho_schedule is not None
        fitted   = self.age > 0
        n_snaps  = len(self.learn_history_)
        return (
            f"GTSOM(M={self.M}, d={self.d}, "
            f"embed_dim={self.embed.dim}, "
            f"compiled={compiled}, fitted={fitted}, "
            f"age={self.age}, snapshots={n_snaps})"
        )