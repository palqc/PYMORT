from __future__ import annotations

"""Shared model utilities that avoid heavy imports.

These helpers stay small and isolated to prevent circular dependencies inside
``pymort.models``.

Note:
    Docstrings follow Google style for clarity and spec alignment.
"""


import numpy as np

from pymort._types import FloatArray


def estimate_rw_params(k: FloatArray) -> tuple[float, float]:
    """Estimate random-walk-with-drift parameters for a 1D series.

    The model is:
        k_t = k_{t-1} + mu + eps_t,  eps_t ~ N(0, sigma^2)

    Args:
        k: 1D array of factor values, shape (T,).

    Returns:
        Tuple of (mu, sigma) estimated from first differences.

    Raises:
        ValueError: If k is not 1D, too short, or contains non-finite values.

    Notes:
        If only two points are available, sigma is not identifiable with ddof=1
        and is returned as 0.0 (deterministic random walk).
    """
    k = np.asarray(k, dtype=float)
    if k.ndim != 1 or k.size < 2:
        raise ValueError("k must be 1D with at least 2 points.")
    if not np.isfinite(k).all():
        raise ValueError("k must contain finite values.")

    dk = np.diff(k)
    mu = float(dk.mean())

    # If dk has <2 samples, ddof=1 would produce NaN → return 0.0
    sigma = float(dk.std(ddof=1)) if dk.size >= 2 else 0.0

    if not np.isfinite(mu):
        raise ValueError("Estimated mu is not finite.")
    if not np.isfinite(sigma) or sigma < 0.0:
        sigma = 0.0

    return mu, sigma


def _estimate_rw_params(kappa: FloatArray) -> tuple[float, float]:
    """Estimate random-walk-with-drift parameters for a 1D CBD series.

    This helper is more permissive than ``estimate_rw_params``: if the
    estimated sigma is not finite or negative, it is set to 0.0 instead
    of raising.

    Args:
        kappa: 1D array of factor values, shape (T,).

    Returns:
        Tuple of (mu, sigma) estimated from first differences.

    Raises:
        ValueError: If kappa is not 1D or too short, or mu is not finite.
    """
    if kappa.ndim != 1 or kappa.size < 2:
        raise ValueError("kappa must be 1D with at least 2 points.")
    diffs = np.diff(kappa)
    mu = float(diffs.mean())
    sigma = float(diffs.std(ddof=1))
    if not np.isfinite(mu):
        raise ValueError("Estimated mu is not finite.")
    if not np.isfinite(sigma) or sigma < 0:
        sigma = 0.0
    return mu, sigma


__all__ = ["_estimate_rw_params", "estimate_rw_params"]
