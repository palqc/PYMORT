from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LCParams:
    a: np.ndarray  # (A,)
    b: np.ndarray  # (A,)
    k: np.ndarray  # (T,)
    mu: Optional[float] = None  # drift of k_t
    sigma: Optional[float] = None  # volatility of k_t


def fit_lee_carter(m: np.ndarray) -> LCParams:
    """
    Fit the Lee–Carter model to a mortality surface m[age, year].
    Steps:
      1) Compute a_x = mean_t log(m_x,t),
      2) Apply SVD to the centered log-mortality matrix,
      3) Extract the first singular vector pair (rank-1 LC),
      4) Enforce identifiability: sum(b_x)=1 and mean(k_t)=0.
    Returns LCParams(a, b, k).
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

    return LCParams(a=a, b=b, k=k)


def reconstruct_log_m(params: LCParams) -> np.ndarray:
    """ "
    Reconstruct the fitted log-mortality surface via:
        log m_x,t = a_x + b_x * k_t
    Returns a matrix with shape (A, T).
    """
    return params.a[:, None] + np.outer(params.b, params.k)


def estimate_rw_params(k: np.ndarray) -> tuple[float, float]:
    """
    Estimate random-walk-with-drift parameters for the time index k_t:
        k_t = k_{t-1} + mu + eps_t,   eps_t ~ N(0, sigma^2)
    Returns (mu, sigma) based on differences of k_t.
    """
    if k.ndim != 1 or k.size < 2:
        raise ValueError("k must be 1D with at least 2 points.")
    dk = np.diff(k)
    mu = float(dk.mean())
    sigma = float(dk.std(ddof=1))
    if not np.isfinite(mu):
        raise ValueError("Estimated mu is not finite.")
    if not np.isfinite(sigma) or sigma < 0:
        # guard against numerical issues
        sigma = 0.0
    return mu, sigma


def simulate_k_paths(
    k_last: float,
    horizon: int,
    mu: float,
    sigma: float,
    n_sims: int = 1000,  # number of Monte Carlo paths (default: 1000; speed/accuracy trade-off)
    seed: int | None = None,
    include_last: bool = False,
) -> np.ndarray:
    """
    Simulate future trajectories of the Lee–Carter time index k_t using:
        k_t = k_{t-1} + mu + eps_t.
    Generates an array of shape (n_sims, horizon). If include_last=True,
    the initial value k_last is prepended as the first column.
    """
    try:
        horizon = int(horizon)
        n_sims = int(n_sims)
    except Exception as e:
        raise TypeError("horizon and n_sims must be integers.") from e
    if horizon <= 0:
        raise ValueError("horizon must be > 0.")
    if n_sims <= 0:
        raise ValueError("n_sims must be > 0.")

    mu = float(mu)
    sigma = float(sigma)
    if not np.isfinite(mu):
        raise ValueError("mu must be finite.")
    if not np.isfinite(sigma):
        raise ValueError("sigma must be finite.")
    # numpy requires scale >= 0; small epsilon avoids degenerate errors
    if sigma < 0:
        sigma = abs(sigma)
    sigma = max(sigma, 1e-12)

    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, size=(n_sims, horizon))
    steps = mu + eps
    k_paths = k_last + np.cumsum(steps, axis=1)
    if include_last:
        k_paths = np.concatenate([np.full((n_sims, 1), k_last), k_paths], axis=1)
    return k_paths


class LeeCarter:
    def __init__(self):
        self.params: Optional[LCParams] = None

    def fit(self, m: np.ndarray) -> "LeeCarter":
        """
        Fit the Lee–Carter model on a mortality surface m[age, year]
        and store the resulting LC parameters.
        """
        self.params = fit_lee_carter(m)
        return self

    def estimate_rw(self) -> tuple[float, float]:
        if self.params is None:
            raise ValueError("Fit first.")
        mu, sigma = estimate_rw_params(self.params.k)
        self.params.mu, self.params.sigma = mu, sigma
        return mu, sigma

    def predict_log_m(self) -> np.ndarray:
        """
        Reconstruct the log-mortality surface implied by the fitted LC parameters.
        """
        if self.params is None:
            raise ValueError("Fit first.")
        return reconstruct_log_m(self.params)

    def simulate_k(
        self,
        horizon: int,
        n_sims: int = 1000,
        seed: int | None = None,
        include_last: bool = False,
    ) -> np.ndarray:
        """
        Simulate random-walk forecasts of k_t using the fitted drift and volatility.
        Returns a matrix (n_sims, horizon).
        """
        if self.params is None or self.params.mu is None or self.params.sigma is None:
            raise ValueError("Fit & estimate_rw first.")
        horizon = int(horizon)
        n_sims = int(n_sims)
        if horizon <= 0 or n_sims <= 0:
            raise ValueError("horizon and n_sims must be positive integers.")

        return simulate_k_paths(
            k_last=self.params.k[-1],
            horizon=horizon,
            mu=self.params.mu,
            sigma=self.params.sigma,
            n_sims=n_sims,
            seed=seed,
            include_last=include_last,
        )
