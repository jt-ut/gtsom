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

    Uses sklearn SpectralEmbedding with a nearest-neighbours affinity graph
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


_ANNEAL_EPS = 1.0
# Additive shift applied to initial and final before the exponential formula,
# then subtracted from every output. This allows initial or final to be zero
# without causing division-by-zero or log(0) in the formula. Since the shift
# cancels exactly on output, its magnitude has no effect on the schedule shape.


class ExponentialAnneal:
    """
    Exponential annealing schedule for a scalar parameter.

    Computes the annealed value at a given age (epoch count) as::

        value(age) = initial * (final / initial) ** (age / tau)

    The result is clipped at ``final`` once the schedule reaches it,
    ensuring the final value is respected exactly regardless of how many
    epochs have elapsed.

    Supports both decreasing schedules (``final < initial``, e.g. annealing
    a neighbourhood bandwidth from broad to narrow) and increasing schedules
    (``final > initial``, e.g. shifting neighbourhood influence from
    lattice-dominated toward CONN-dominated over training). The direction
    is inferred automatically from the relationship between ``initial``
    and ``final``; the formula and clipping behaviour adapt accordingly.

    ``initial`` and ``final`` may be zero. Internally, both are shifted by
    a small positive constant before the exponential formula is applied, then
    the shift is subtracted from every output — so the schedule endpoints and
    shape are exactly as specified regardless of whether zero is used.

    Intended for use with any parameter that requires monotonic change
    during training — neighbourhood bandwidth (rho), topology-blending
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
    >>> schedule(20)   # one tau — decayed toward final
    0.549...
    >>> schedule(1000)  # far beyond target — clipped at final
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
        self.final = float(final)
        self.tau = float(tau)

        # Shifted values used in all formula evaluations. The shift cancels
        # exactly on output so has no effect on the schedule shape or endpoints.
        self._si = self.initial + _ANNEAL_EPS
        self._sf = self.final   + _ANNEAL_EPS

        # Detect flat schedule (initial == final); skip direction and
        # target_epochs calculations that would involve log(1) = 0.
        self._flat = self.initial == self.final
        if self._flat:
            self._increasing   = False   # unused, but defined for consistency
            self.target_epochs = 0.0
        else:
            # Infer direction from initial and final values
            self._increasing = self.final > self.initial
            # Age at which the raw exponential first reaches final.
            # abs() ensures positive regardless of direction.
            self.target_epochs = self.tau * abs(np.log(self._sf / self._si))

    @classmethod
    def from_halflife(cls, initial, final, halflife_epochs):
        """
        Construct a schedule where ``halflife_epochs`` is the half-life.

        ``halflife_epochs`` is the number of epochs at which the parameter
        reaches the geometric midpoint between ``initial`` and ``final`` on
        a log scale — i.e. ``sqrt(initial * final)``. This is the true
        half-life of the decay in log space.

        The schedule continues decaying beyond this point and is clipped at
        ``final`` at ``2 * halflife_epochs``, so the parameter has fully
        settled by that point.

        This gives an intuitive two-phase interpretation:
          - Phase 1 (0 → halflife_epochs): active decay, halfway there in log space.
          - Phase 2 (halflife_epochs → 2*halflife_epochs): second half of decay,
            settling toward ``final``.
          - Beyond 2*halflife_epochs: clipped at ``final``.

        Parameters
        ----------
        initial : float
            Starting value at age=0.
        final : float
            Floor value; clipped at this value from age ``2 *
            halflife_epochs`` onward.
        halflife_epochs : float
            The age at which the parameter reaches the geometric midpoint
            ``sqrt(initial * final)``.

        Returns
        -------
        ExponentialAnneal

        Examples
        --------
        >>> s = ExponentialAnneal.from_halflife(initial=3.0, final=0.3,
        ...                                      halflife_epochs=50)
        >>> round(s(0), 4)
        3.0
        >>> round(s(50), 4)   # geometric midpoint of 3.0 and 0.3
        0.9487
        >>> round(s(100), 4)  # clipped at final
        0.3
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

        Returns ``final`` immediately for flat schedules (``initial ==
        final``). Otherwise the raw exponential is clipped at ``final``
        once the schedule reaches it, with clip direction inferred from
        whether the schedule is increasing or decreasing.

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
        raw = self._si * (self._sf / self._si) ** (age / self.tau) - _ANNEAL_EPS
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
        raw = self._si * (self._sf / self._si) ** (ages / self.tau) - _ANNEAL_EPS
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
            f"tau={self.tau:.4g}, "
            f"target_epochs={self.target_epochs:.4g})"
        )