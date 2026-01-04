"""Lee-Carter M1 mortality model (baseline).

This module implements the classic Lee-Carter decomposition and basic
random-walk dynamics for the time factor.

Note:
    Docstrings follow Google style for clarity and spec alignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymort._types import FloatArray
from pymort.models.utils import estimate_rw_params


@dataclass
class LCM1Params:
    a: FloatArray  # (A,)
    b: FloatArray  # (A,)
    k: FloatArray  # (T,)
    mu: float | None = None  # drift of k_t
    sigma: float | None = None  # volatility of k_t


def fit_lee_carter(m: FloatArray) -> LCM1Params:
    """Fit the Lee-Carter model to a mortality surface.

    Steps:
        1) Compute a_x = mean_t log(m_x,t)
        2) Apply SVD to the centered log-mortality matrix
        3) Extract the first singular vector pair (rank-1 LC)
        4) Enforce identifiability: sum(b_x)=1 and mean(k_t)=0

    Args:
        m: Mortality surface m_{x,t}, shape (A, T).

    Returns:
        LCM1Params with a, b, k fitted on the input surface.
    """
    # input validation
    if m.ndim != 2:
        raise ValueError("m must be a 2D array with shape (A, T).")
    if not np.isfinite(m).all() or (m <= 0).any():
        raise ValueError("m must be strictly positive and finite.")

    ln_m = np.log(m)  # (A, T)
    a = ln_m.mean(axis=1)  # (A,)
    Z = ln_m - a[:, None]  # center by age
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    # rank-1 LC
    b = U[:, 0]  # (A,)
    k = s[0] * Vt[0, :]  # (T,)

    # identifiability normalization
    b_sum = b.sum()
    if b_sum == 0:
        raise RuntimeError("SVD produced b with zero sum.")
    # make sum(b) = 1, prefer positive sum for interpretability
    if b_sum < 0:
        b = -b
        k = -k
        b_sum = -b_sum

    b = b / b_sum  # sum(b)=1
    k = k * b_sum

    # zero-mean k and absorb mean into a
    k_mean = k.mean()  # mean(k)=0
    k = k - k_mean
    a = a + b * k_mean

    return LCM1Params(a=a, b=b, k=k)


def reconstruct_log_m(params: LCM1Params) -> FloatArray:
    """Reconstruct the fitted log-mortality surface.

    Args:
        params: Fitted Lee-Carter parameters.

    Returns:
        Log-mortality surface log m_{x,t}, shape (A, T).
    """
    return np.asarray(params.a[:, None] + np.outer(params.b, params.k), dtype=float)


class LCM1:
    def __init__(self) -> None:
        self.params: LCM1Params | None = None

    def fit(self, m: FloatArray) -> LCM1:
        """Fit the Lee-Carter model and store parameters.

        Args:
            m: Mortality surface m_{x,t}, shape (A, T).

        Returns:
            Self, with fitted parameters stored in `self.params`.
        """
        self.params = fit_lee_carter(m)
        return self

    def estimate_rw(self) -> tuple[float, float]:
        if self.params is None:
            raise ValueError("Fit first.")
        mu, sigma = estimate_rw_params(self.params.k)
        self.params.mu, self.params.sigma = mu, sigma
        return mu, sigma

    def predict_log_m(self) -> FloatArray:
        """Reconstruct the log-mortality surface implied by fitted parameters."""
        if self.params is None:
            raise ValueError("Fit first.")
        return reconstruct_log_m(self.params)

    def simulate_k(
        self,
        horizon: int,
        n_sims: int = 1000,
        seed: int | None = None,
        include_last: bool = False,
    ) -> FloatArray:
        """Simulate random-walk paths for k_t.

        Args:
            horizon: Number of future steps.
            n_sims: Number of simulated paths.
            seed: RNG seed.
            include_last: If True, prepend the last observed k_t.

        Returns:
            Array of shape (n_sims, horizon) or (n_sims, horizon + 1).
        """
        if self.params is None or self.params.mu is None or self.params.sigma is None:
            raise ValueError("Fit & estimate_rw first.")

        rng = np.random.default_rng(seed)

        from pymort.analysis.projections import simulate_random_walk_paths

        return simulate_random_walk_paths(
            k_last=self.params.k[-1],
            mu=self.params.mu,
            sigma=self.params.sigma,
            horizon=int(horizon),
            n_sims=int(n_sims),
            rng=rng,
            include_last=include_last,
        )
