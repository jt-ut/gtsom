"""
utils.py — Utility classes and functions for the gtsom package.
"""

import numpy as np


import warnings
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding


# ---------------------------------------------------------------------------
# DataValidator
# ---------------------------------------------------------------------------

class DataValidator:
    """
    Validates a data matrix X and stores a lightweight fingerprint for
    consistency checking across subsequent calls.

    Construction validates X (must be 2-D) and records its fingerprint.
    The :meth:`check` method warns if a later X does not match.

    Parameters
    ----------
    X : array-like
        Data matrix to validate and fingerprint.

    Raises
    ------
    ValueError
        If X is not 2-D after conversion to ndarray.

    Examples
    --------
    >>> import numpy as np
    >>> validator = DataValidator(np.ones((100, 5)))
    >>> validator.shape
    (100, 5)
    >>> validator.check(np.ones((100, 5)))   # silent — matches
    >>> validator.check(np.ones((200, 5)), context="fit")  # warns
    """

    def __init__(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        self._fingerprint = (
            X.shape,
            X.dtype,
            float(X.mean()),
            float(X.std()),
            float(X.min()),
            float(X.max()),
        )

    @property
    def shape(self):
        """Shape (N, d) of the validated dataset."""
        return self._fingerprint[0]

    @property
    def dtype(self):
        """dtype of the validated dataset."""
        return self._fingerprint[1]

    def check(self, X, context=""):
        """
        Warn if X does not match the stored fingerprint.

        Parameters
        ----------
        X : array-like
            Incoming data matrix to check.
        context : str, optional
            Caller name included in the warning message for traceability.
        """
        X = np.asarray(X)
        current = (
            X.shape,
            X.dtype,
            float(X.mean()),
            float(X.std()),
            float(X.min()),
            float(X.max()),
        )
        if current != self._fingerprint:
            fp = self._fingerprint
            prefix = f"{context}: " if context else ""
            warnings.warn(
                f"{prefix}X does not match the dataset used at initialisation. "
                f"Expected shape={fp[0]}, dtype={fp[1]}, "
                f"mean={fp[2]:.4g}, std={fp[3]:.4g}, "
                f"min={fp[4]:.4g}, max={fp[5]:.4g}. "
                f"Got shape={current[0]}, dtype={current[1]}, "
                f"mean={current[2]:.4g}, std={current[3]:.4g}, "
                f"min={current[4]:.4g}, max={current[5]:.4g}.",
                UserWarning,
                stacklevel=3,
            )

    def __repr__(self):
        return (
            f"DataValidator(shape={self._fingerprint[0]}, "
            f"dtype={self._fingerprint[1]})"
        )


# ---------------------------------------------------------------------------
# Dimensionality reduction helpers (used by GTSOM.from_data)
# ---------------------------------------------------------------------------

def reduce_coords_pca(W, coord_dim, random_state=None):
    """
    Reduce prototype matrix W to coord_dim dimensions via randomised PCA.

    Parameters
    ----------
    W : np.ndarray, shape (M, d)
        Prototype matrix.
    coord_dim : int
        Target dimensionality (2 or 3).
    random_state : int or None, default None

    Returns
    -------
    coords : np.ndarray, shape (M, coord_dim), float32
    """
    pca = PCA(n_components=coord_dim, svd_solver='randomized',
              random_state=random_state)
    coords = pca.fit_transform(W)
    return coords.astype(np.float32)


def reduce_coords_le(W, coord_dim, random_state=None):
    """
    Reduce prototype matrix W to coord_dim dimensions via Laplacian Eigenmaps.

    Uses sklearn SpectralEmbedding with a nearest-neighbors affinity graph
    built from the prototype vectors.

    Parameters
    ----------
    W : np.ndarray, shape (M, d)
        Prototype matrix.
    coord_dim : int
        Target dimensionality (2 or 3).
    random_state : int or None, default None

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


def reduce_coords_random(M, coord_dim, random_state=None):
    """
    Generate M neuron coordinates sampled uniformly from the unit hypercube.

    Produces a completely unstructured initial layout with no relationship to
    the prototype vectors. Useful for exposition — starting from a worst-case
    random state makes the organising effect of SOM training most visible.

    Parameters
    ----------
    M : int
        Number of prototypes (neurons).
    coord_dim : int
        Dimensionality of the output space (2 or 3).
    random_state : int or None, default None

    Returns
    -------
    coords : np.ndarray, shape (M, coord_dim), float32
        Coordinates drawn uniformly from [0, 1)^coord_dim.
    """
    rng = np.random.default_rng(random_state)
    return rng.random((M, coord_dim)).astype(np.float32)


def reduce_coords_random_proj(W, coord_dim, random_state=None):
    """
    Reduce prototype matrix W to coord_dim dimensions via random projection.

    Multiplies W by a random Gaussian matrix scaled by ``1 / sqrt(coord_dim)``
    (Johnson-Lindenstrauss style). Preserves approximate pairwise distances in
    expectation at zero computational cost — a structurally grounded starting
    layout that is nonetheless random and requires no decomposition.

    Parameters
    ----------
    W : np.ndarray, shape (M, d)
        Prototype matrix.
    coord_dim : int
        Target dimensionality (2 or 3).
    random_state : int or None, default None

    Returns
    -------
    coords : np.ndarray, shape (M, coord_dim), float32
    """
    rng = np.random.default_rng(random_state)
    d = W.shape[1]
    R = rng.standard_normal((d, coord_dim)).astype(np.float32) / np.sqrt(coord_dim)
    return (W.astype(np.float32) @ R)


_ANNEAL_EPS = 1e-8
# Small positive floor applied to initial and final when either is zero.
# Replaces the old additive-shift approach (which distorted the curve shape)
# with input clipping: zero values are replaced by _ANNEAL_EPS before the
# formula is applied, keeping the exponential a true exponential in value
# space rather than in (value + constant) space.


class ExponentialAnneal:
    """
    Exponential annealing schedule for a scalar parameter.

    Computes the annealed value at a given age (epoch count) as::

        value(age) = initial * (final / initial) ** (age / tau)

    The result is clipped at ``final`` once the schedule reaches it,
    ensuring the final value is respected exactly regardless of how many
    epochs have elapsed.

    Supports both decreasing schedules (``final < initial``, e.g. annealing
    a neighborhood bandwidth from broad to narrow) and increasing schedules
    (``final > initial``, e.g. shifting neighborhood influence from
    lattice-dominated toward CONN-dominated over training). The direction
    is inferred automatically from the relationship between ``initial``
    and ``final``; the formula and clipping behavior adapt accordingly.

    ``initial`` and ``final`` may be zero. If either is zero it is replaced
    internally by a small positive floor (``_ANNEAL_EPS = 1e-8``) before
    the formula is applied, keeping the exponential well-defined. The
    reported ``self.initial`` / ``self.final`` always reflect the values
    as supplied by the caller; only the internal computation is clipped.
    The resulting schedule is a true exponential in value space — not
    in a translated space — so the curve shape is never distorted by
    the zero-handling.

    Intended for use with any parameter that requires monotonic change
    during training — neighborhood bandwidth (rho), topology-blending
    weight (nbr_topo_alpha), learning rate, etc.

    Parameters
    ----------
    initial : float
        Starting value of the parameter (at age=0). Must be >= 0.
    final : float
        Target value of the parameter. Must be >= 0. If ``final <
        initial``, the schedule decays; if ``final > initial``, the
        schedule increases; if ``final == initial``, the schedule is
        flat and returns ``initial`` at every age.
    tau : float
        Time constant in epochs. Controls how quickly the parameter moves
        from ``initial`` toward ``final``. Smaller values change faster.
        Computed internally from ``halflife_epochs`` when that convenience
        constructor is used instead.

    Examples
    --------
    Decreasing schedule (decay):

    >>> schedule = ExponentialAnneal(initial=5.0, final=0.3, tau=20)
    >>> schedule(0)
    5.0
    >>> schedule(10)   # half a tau — midway through the decay
    1.2247...
    >>> schedule(20)   # one full tau — reaches final exactly
    0.3
    >>> schedule(1000)  # beyond tau — clipped at final
    0.3

    Decreasing to zero:

    >>> schedule = ExponentialAnneal(initial=1.0, final=0.0, tau=20)
    >>> schedule(0)
    1.0
    >>> schedule(1000)  # clipped at final
    0.0

    Increasing schedule (ramp):

    >>> schedule = ExponentialAnneal(initial=0.0, final=1.0, tau=20)
    >>> schedule(0)
    0.0
    >>> schedule(1000)  # clipped at final
    1.0

    Construction from a half-life epoch count:

    >>> schedule = ExponentialAnneal.from_halflife(initial=5.0, final=0.3,
    ...                                             halflife_epochs=50)
    >>> round(schedule(50), 6)  # geometric midpoint of 5.0 and 0.3
    1.2247...
    >>> round(schedule(100), 6)  # clipped at final
    0.3
    """

    def __init__(self, initial, final, tau):
        if initial < 0:
            raise ValueError(f"initial must be >= 0, got {initial}")
        if final < 0:
            raise ValueError(f"final must be >= 0, got {final}")
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")

        self.initial = float(initial)
        self.final   = float(final)
        self.tau     = float(tau)

        # Internal clipped values: replace zero with _ANNEAL_EPS so that
        # the ratio (sf/si) and log(sf/si) are always well-defined.
        # Unlike the old additive-shift approach, this does not distort
        # the curve shape for non-zero parameters.
        self._si = max(self.initial, _ANNEAL_EPS)
        self._sf = max(self.final,   _ANNEAL_EPS)

        # Detect flat schedule (initial == final)
        self._flat = self.initial == self.final
        if self._flat:
            self._increasing = False
        else:
            self._increasing = self.final > self.initial

    @classmethod
    def from_halflife(cls, initial, final, halflife_epochs):
        """
        Construct a schedule where ``halflife_epochs`` is the geometric half-life.

        ``halflife_epochs`` is the epoch at which the parameter reaches the
        **geometric mean** of ``initial`` and ``final``::

            value(halflife_epochs) = sqrt(initial * final)

        This is a half-life in **log space** (the schedule is linear in
        log space, and the geometric mean is the arithmetic midpoint of
        ``log(initial)`` and ``log(final)``), not in linear space.

        .. note:: **Geometric vs arithmetic midpoint**

            The geometric midpoint ``sqrt(initial * final)`` is always
            closer to ``final`` than the arithmetic midpoint
            ``(initial + final) / 2`` when ``initial != final``. For
            example, with ``initial=10, final=0.1``:

            - Arithmetic midpoint: 5.05  (barely moved from initial)
            - Geometric midpoint:  1.0   (already close to final)

            This means the schedule will feel "fast then slow" in linear
            space — the parameter drops quickly early on and then settles
            gradually. If you expect to be "halfway there" in the ordinary
            (arithmetic) sense at ``halflife_epochs``, you will find the
            parameter has already moved further than that.

        .. note:: **Relationship to tau and total training epochs**

            ``tau = 2 * halflife_epochs``, and the schedule is clipped at
            ``final`` at age ``2 * halflife_epochs``. This means:

            - Setting ``halflife_epochs ≈ n_epochs / 4`` places the
              geometric midpoint at the first quarter of training, giving
              a steep initial drop followed by a long flat tail — the
              classic exponential shape.
            - Setting ``halflife_epochs ≈ n_epochs / 2`` places the
              geometric midpoint at the halfway point, so the schedule is
              still actively decaying at the end of training and the curve
              looks nearly linear over the training window.

            As a rule of thumb, set ``halflife_epochs`` to roughly a
            quarter of your total training epochs to get a visually
            exponential decay with a clear knee and flat tail.

        Parameters
        ----------
        initial : float
            Starting value at age=0.
        final : float
            Floor value; clipped at this value from age
            ``2 * halflife_epochs`` onward.
        halflife_epochs : float
            The epoch at which the parameter reaches the geometric mean
            ``sqrt(initial * final)``. Must be positive.

        Returns
        -------
        ExponentialAnneal

        Examples
        --------
        >>> s = ExponentialAnneal.from_halflife(initial=3.0, final=0.3,
        ...                                      halflife_epochs=50)
        >>> round(s(0), 4)
        3.0
        >>> round(s(50), 4)   # geometric midpoint sqrt(3.0 * 0.3) ≈ 0.9487
        0.9487
        >>> round(s(100), 4)  # clipped at final
        0.3

        With halflife_epochs = n_epochs / 4 for a classic exponential shape:

        >>> s = ExponentialAnneal.from_halflife(initial=10.0, final=0.1,
        ...                                      halflife_epochs=25)
        >>> round(s(25), 4)   # geometric midpoint sqrt(10 * 0.1) = 1.0
        1.0
        >>> round(s(50), 4)   # clipped at final well before end of training
        0.1
        """
        if halflife_epochs <= 0:
            raise ValueError(
                f"halflife_epochs must be positive, got {halflife_epochs}"
            )
        # tau such that age=halflife_epochs gives the geometric midpoint:
        # initial * (final/initial)^(halflife_epochs/tau) = sqrt(initial*final)
        # => (final/initial)^(halflife_epochs/tau) = sqrt(final/initial)
        # => halflife_epochs/tau = 0.5
        # => tau = 2 * halflife_epochs
        tau = 2.0 * halflife_epochs
        return cls(initial=initial, final=final, tau=tau)

    def __call__(self, age):
        """
        Return the annealed parameter value at the given age.

        Parameters
        ----------
        age : int or float
            Current epoch count (0-based).

        Returns
        -------
        float
        """
        if self._flat:
            return self.final
        raw = self._si * (self._sf / self._si) ** (age / self.tau)
        if self._increasing:
            return float(min(raw, self.final))
        else:
            return float(max(raw, self.final))

    def values(self, n_epochs):
        """
        Return the annealed values for epochs 0 through n_epochs - 1.

        Useful for inspecting or plotting the planned schedule before
        training begins.

        Parameters
        ----------
        n_epochs : int

        Returns
        -------
        np.ndarray, shape (n_epochs,)
        """
        if self._flat:
            return np.full(n_epochs, self.final)
        ages = np.arange(n_epochs, dtype=float)
        raw  = self._si * (self._sf / self._si) ** (ages / self.tau)
        if self._increasing:
            return np.minimum(raw, self.final).astype(float)
        else:
            return np.maximum(raw, self.final).astype(float)

    def __repr__(self):
        if self._flat:
            return (
                f"ExponentialAnneal("
                f"initial={self.initial}, final={self.final}, "
                f"tau={self.tau:.4g}, flat=True)"
            )
        return (
            f"ExponentialAnneal("
            f"initial={self.initial}, "
            f"final={self.final}, "
            f"tau={self.tau:.4g})"
        )