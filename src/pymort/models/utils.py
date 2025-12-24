from __future__ import annotations

"""
Lightweight shared utilities that are free of model-level imports.

These helpers are intentionally kept small to avoid circular imports between
modules under ``pymort.models``.
"""

from typing import Tuple

import numpy as np


def estimate_rw_params(k: np.ndarray) -> Tuple[float, float]:
    """
    Estimate random-walk-with-drift parameters for a 1D time index:

        k_t = k_{t-1} + mu + eps_t,   eps_t ~ N(0, sigma^2)

    Returns (mu, sigma) based on differences of k_t.

    Notes
    -----
    If only 2 points are available, sigma is not identifiable with ddof=1.
    In that case we return sigma = 0.0 (deterministic RW).
    """
    k = np.asarray(k, dtype=float)
    if k.ndim != 1 or k.size < 2:
        raise ValueError("k must be 1D with at least 2 points.")
    if not np.isfinite(k).all():
        raise ValueError("k must contain finite values.")

    dk = np.diff(k)
    mu = float(dk.mean())

    # If dk has <2 samples, ddof=1 would produce NaN → return 0.0
    if dk.size >= 2:
        sigma = float(dk.std(ddof=1))
    else:
        sigma = 0.0

    if not np.isfinite(mu):
        raise ValueError("Estimated mu is not finite.")
    if not np.isfinite(sigma) or sigma < 0.0:
        sigma = 0.0

    return mu, sigma


def _estimate_rw_params(kappa: np.ndarray) -> Tuple[float, float]:
    """
    CBD helper: estimate RW+drift parameters for a 1D series.

    More permissive than ``estimate_rw_params``: if sigma is not finite or
    negative, it is set to 0.0 instead of raising.
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


__all__ = ["estimate_rw_params", "_estimate_rw_params"]
