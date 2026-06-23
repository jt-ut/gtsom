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


class ExponentialAnneal:
    """
    Exponential annealing schedule for a scalar parameter.

    Computes the annealed value at a given age (epoch count) as::

        value(age) = initial * (final / initial) ** (age / tau)

    The result is clipped at ``final`` for all ``age >= target_epochs``,
    ensuring the final value is respected exactly regardless of how many
    epochs have elapsed.

    Intended for use with any parameter that requires monotonic decay
    during training — neighbourhood bandwidth (rho), learning rate, etc.

    Parameters
    ----------
    initial : float
        Starting value of the parameter (at age=0).
    final : float
        Minimum (floor) value of the parameter. The schedule is clipped
        at this value for all ages beyond ``target_epochs``.
    tau : float
        Decay time constant in epochs. Smaller values decay faster.
        Computed internally from ``target_epochs`` when that convenience
        argument is used instead.

    Examples
    --------
    Direct construction with tau:

    >>> schedule = ExponentialAnneal(initial=5.0, final=0.3, tau=20)
    >>> schedule(0)
    5.0
    >>> schedule(20)   # one tau — falls to ~37% of the initial range
    0.549...
    >>> schedule(1000)  # far beyond target — clipped at final
    0.3

    Construction from a target epoch count:

    >>> schedule = ExponentialAnneal.from_target(initial=5.0, final=0.3,
    ...                                           target_epochs=50)
    >>> round(schedule(50), 6)
    0.3
    """

    def __init__(self, initial, final, tau):
        if initial <= 0:
            raise ValueError(f"initial must be positive, got {initial}")
        if final <= 0:
            raise ValueError(f"final must be positive, got {final}")
        if final >= initial:
            raise ValueError(
                f"final ({final}) must be less than initial ({initial}) "
                f"for a decay schedule."
            )
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")

        self.initial = float(initial)
        self.final = float(final)
        self.tau = float(tau)

        # Precompute target_epochs: the age at which the raw exponential
        # first reaches final (used for clipping and __repr__)
        self.target_epochs = -self.tau * np.log(self.final / self.initial)

    @classmethod
    def from_target(cls, initial, final, target_epochs):
        """
        Construct a schedule where ``target_epochs`` is the half-life.

        ``target_epochs`` is the number of epochs at which rho reaches
        the geometric midpoint between ``initial`` and ``final`` on a
        log scale. The schedule continues decaying beyond this point
        and is clipped at ``final`` at ``2 * target_epochs``.

        This gives an intuitive interpretation: the parameter is still
        actively decaying throughout the first ``target_epochs`` epochs,
        and has fully settled by ``2 * target_epochs``.

        Parameters
        ----------
        initial : float
            Starting value at age=0.
        final : float
            Floor value; clipped at this value from age ``2 *
            target_epochs`` onward.
        target_epochs : float
            Half-life in epochs — the age at which the parameter reaches
            the geometric midpoint ``sqrt(initial * final)``.

        Returns
        -------
        ExponentialAnneal

        Examples
        --------
        >>> s = ExponentialAnneal.from_target(initial=3.0, final=0.3,
        ...                                    target_epochs=50)
        >>> round(s(0), 4)
        3.0
        >>> round(s(50), 4)   # geometric midpoint of 3.0 and 0.3
        0.9487
        >>> round(s(100), 4)  # clipped at final
        0.3
        """
        if target_epochs <= 0:
            raise ValueError(
                f"target_epochs must be positive, got {target_epochs}"
            )
        # tau such that age=target_epochs gives the geometric midpoint:
        # initial * (final/initial)^(target_epochs/tau) = sqrt(initial*final)
        # => (final/initial)^(target_epochs/tau) = sqrt(final/initial)
        # => target_epochs/tau = 0.5
        # => tau = 2 * target_epochs
        tau = 2.0 * target_epochs
        return cls(initial=initial, final=final, tau=tau)

    def __call__(self, age):
        """
        Return the annealed parameter value at the given age.

        The raw exponential is clipped at ``final`` for all ages beyond
        ``target_epochs``, ensuring the floor is respected exactly.

        Parameters
        ----------
        age : int or float
            Current epoch count (0-based).

        Returns
        -------
        float
        """
        raw = self.initial * (self.final / self.initial) ** (age / self.tau)
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
        ages = np.arange(n_epochs, dtype=float)
        raw = self.initial * (self.final / self.initial) ** (ages / self.tau)
        return np.maximum(raw, self.final).astype(float)

    def __repr__(self):
        return (
            f"ExponentialAnneal("
            f"initial={self.initial}, "
            f"final={self.final}, "
            f"tau={self.tau:.4g}, "
            f"target_epochs={self.target_epochs:.4g})"
        )