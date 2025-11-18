from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CBDParams:
    """
    Parameters of the basic two–factor CBD model:

        logit(q_{x,t}) = kappa1_t + kappa2_t * (x - x_bar)

    kappa1_t controls the level, kappa2_t the age–slope.
    """

    kappa1: np.ndarray  # (T,)
    kappa2: np.ndarray  # (T,)
    ages: np.ndarray  # (A,)
    x_bar: float  # mean age used for centering

    # optional RW+drift parameters
    mu1: Optional[float] = None
    sigma1: Optional[float] = None
    mu2: Optional[float] = None
    sigma2: Optional[float] = None


def _logit(p: np.ndarray) -> np.ndarray:
    """Numerically stable logit transform log(p / (1 - p))."""
    p_clipped = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p_clipped / (1.0 - p_clipped))


def _build_cbd_design(ages: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Build the CBD design matrix X = [1, x - x_bar] for given ages.

    Parameters
    ----------
    ages : np.ndarray
        1D array of ages (A,).

    Returns
    -------
    X : np.ndarray
        Design matrix of shape (A, 2) with columns [1, x - x_bar].
    x_bar : float
        Mean age used for centering.
    """
    if ages.ndim != 1:
        raise ValueError("ages must be a 1D array.")
    x_bar = float(ages.mean())
    z = ages - x_bar
    X = np.column_stack([np.ones_like(z, dtype=float), z.astype(float)])  # (A, 2)
    return X, x_bar


def fit_cbd(q: np.ndarray, ages: np.ndarray) -> CBDParams:
    """
    Fit the Cairns–Blake–Dowd (CBD) model to a mortality surface q[age, year].

    We assume:
        logit(q_{x,t}) = kappa1_t + kappa2_t * (x - x_bar)

    Steps:
      1) Build design matrix X = [1, (x - x_bar)].
      2) For each year t, regress logit(q_{x,t}) on X to get kappa1_t, kappa2_t.
    """
    if q.ndim != 2:
        raise ValueError("q must be a 2D array with shape (A, T).")
    if ages.ndim != 1:
        raise ValueError("ages must be 1D.")
    A, T = q.shape
    if ages.shape[0] != A:
        raise ValueError("ages length must match q.shape[0].")
    if not np.isfinite(q).all():
        raise ValueError("q must contain finite values.")
    if (q <= 0).any() or (q >= 1).any():
        raise ValueError("q must lie strictly in (0, 1).")

    # Build design matrix once (same ages for all years)
    X, x_bar = _build_cbd_design(ages)  # X: (A, 2)
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("X'X is singular in CBD fit.") from exc

    # transform q to logits
    y = _logit(q)  # (A, T)

    # OLS for all years at once:
    # beta_hat_all = (X'X)^{-1} X' y  → shape (2, T)
    beta_hat_all = XtX_inv @ X.T @ y
    kappa1 = beta_hat_all[0, :]  # (T,)
    kappa2 = beta_hat_all[1, :]  # (T,)

    return CBDParams(
        kappa1=kappa1,
        kappa2=kappa2,
        ages=ages.astype(float),
        x_bar=x_bar,
    )


def reconstruct_logit_q(params: CBDParams) -> np.ndarray:
    """
    Reconstruct the logit mortality surface logit(q_{x,t}) from CBD parameters.
    Returns an array of shape (A, T).
    """
    k1 = params.kappa1  # (T,)
    k2 = params.kappa2  # (T,)
    ages = params.ages  # (A,)
    z = ages - params.x_bar  # (A,)

    # logit_q[x,t] = kappa1_t + kappa2_t * (x - x_bar)
    # → (A, T) = (1,T) + (A,1)*(1,T)
    return k1[None, :] + z[:, None] * k2[None, :]


def reconstruct_q(params: CBDParams) -> np.ndarray:
    """
    Reconstruct mortality probabilities q_{x,t} from CBD parameters.
    Returns an array with shape (A, T).
    """
    logit_q = reconstruct_logit_q(params)
    return 1.0 / (1.0 + np.exp(-logit_q))


def _estimate_rw_params(kappa: np.ndarray) -> Tuple[float, float]:
    """
    Estimate random–walk–with–drift parameters for a 1D time index:

        k_t = k_{t-1} + mu + eps_t,  eps_t ~ N(0, sigma^2).

    Returns (mu, sigma).
    """
    if kappa.ndim != 1 or kappa.size < 2:
        raise ValueError("kappa must be 1D with at least 2 points.")
    diffs = np.diff(kappa)
    mu = float(diffs.mean())
    sigma = float(diffs.std(ddof=1))
    return mu, sigma


def estimate_rw_params_cbd(params: CBDParams) -> CBDParams:
    """
    Estimate RW+drift parameters for (kappa1_t, kappa2_t) and store them
    in the CBDParams object.
    """
    mu1, sigma1 = _estimate_rw_params(params.kappa1)
    mu2, sigma2 = _estimate_rw_params(params.kappa2)
    params.mu1, params.sigma1 = mu1, sigma1
    params.mu2, params.sigma2 = mu2, sigma2
    return params


def simulate_kappa(
    k_last: float,
    mu: float,
    sigma: float,
    horizon: int,
    n_sims: int,
    seed: Optional[int] = None,
    include_last: bool = False,
) -> np.ndarray:
    """
    Simulate future trajectories of a CBD time index kappa_t using:

        kappa_t = kappa_{t-1} + mu + eps_t.

    Returns an array of shape (n_sims, horizon), or (n_sims, horizon+1)
    if include_last is True (first column = k_last).
    """
    try:
        horizon = int(horizon)
        n_sims = int(n_sims)
    except Exception as exc:
        raise TypeError("horizon and n_sims must be integers.") from exc
    if horizon <= 0:
        raise ValueError("horizon must be > 0.")
    if n_sims <= 0:
        raise ValueError("n_sims must be > 0.")

    mu = float(mu)
    sigma = float(sigma)
    if not np.isfinite(mu) or not np.isfinite(sigma):
        raise ValueError("mu and sigma must be finite.")
    if sigma < 0:
        sigma = abs(sigma)
    sigma = max(sigma, 1e-12)

    rng = np.random.default_rng(seed)
    increments = rng.normal(loc=mu, scale=sigma, size=(n_sims, horizon))
    kappa_paths = k_last + np.cumsum(increments, axis=1)

    if include_last:
        kappa_paths = np.hstack([np.full((n_sims, 1), k_last), kappa_paths])

    return kappa_paths


class CBDModel:
    def __init__(self) -> None:
        self.params: Optional[CBDParams] = None

    def fit(self, q: np.ndarray, ages: np.ndarray) -> "CBDModel":
        """
        Fit the CBD model on a mortality surface q[age, year]
        and store the resulting parameters in self.params.
        """
        self.params = fit_cbd(q, ages)
        return self

    def estimate_rw(self) -> Tuple[float, float, float, float]:
        """
        Estimate RW+drift parameters for (kappa1_t, kappa2_t)
        and store them in self.params.
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        self.params = estimate_rw_params_cbd(self.params)
        return (
            self.params.mu1,
            self.params.sigma1,
            self.params.mu2,
            self.params.sigma2,
        )

    def predict_logit_q(self) -> np.ndarray:
        """
        Reconstruct the logit-mortality surface from fitted CBD parameters.
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        return reconstruct_logit_q(self.params)

    def predict_q(self) -> np.ndarray:
        """
        Reconstruct mortality probabilities q_{x,t} from fitted CBD parameters.
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        return reconstruct_q(self.params)

    def simulate_kappa(
        self,
        kappa_index: str,
        horizon: int,
        n_sims: int = 1000,
        seed: Optional[int] = None,
        include_last: bool = False,
    ) -> np.ndarray:
        """
        Simulate random–walk forecasts of kappa1_t or kappa2_t using
        the fitted drift and volatility parameters.
        """
        if self.params is None:
            raise ValueError("Fit the model first.")

        if kappa_index == "kappa1":
            k_last = self.params.kappa1[-1]
            mu = self.params.mu1
            sigma = self.params.sigma1
        elif kappa_index == "kappa2":
            k_last = self.params.kappa2[-1]
            mu = self.params.mu2
            sigma = self.params.sigma2
        else:
            raise ValueError("kappa_index must be 'kappa1' or 'kappa2'.")

        if mu is None or sigma is None:
            raise ValueError("Call estimate_rw() before simulate_kappa().")

        return simulate_kappa(
            k_last=k_last,
            mu=mu,
            sigma=sigma,
            horizon=horizon,
            n_sims=n_sims,
            seed=seed,
            include_last=include_last,
        )
