"""Cairns-Blake-Dowd two-factor model (CBD M5).

This module fits the classic CBD model on logit mortality probabilities and
provides simple random-walk forecasts for the period factors.

Note:
    Docstrings follow Google style and type hints use NDArray for clarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from pymort.lifetables import validate_q
from pymort.models.utils import _estimate_rw_params

FloatArray = NDArray[np.floating]


@dataclass
class CBDM5Params:
    """Parameters for the basic two-factor CBD model.

    The model is:
        logit(q_{x,t}) = kappa1_t + kappa2_t * (x - x_bar),
    where kappa1_t controls the level and kappa2_t controls the age slope.
    """

    kappa1: FloatArray  # (T,)
    kappa2: FloatArray  # (T,)
    ages: FloatArray  # (A,)
    x_bar: float  # mean age used for centering

    # optional RW+drift parameters
    mu1: float | None = None
    sigma1: float | None = None
    mu2: float | None = None
    sigma2: float | None = None


def _logit(p: FloatArray) -> FloatArray:
    """Compute a numerically stable logit transform.

    Args:
        p: Probabilities. Shape (...,).

    Returns:
        log(p / (1 - p)) with clipping for numerical stability.
    """
    p_clipped = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p_clipped / (1.0 - p_clipped))


def _build_cbd_design(ages: FloatArray) -> tuple[FloatArray, float]:
    """Build the CBD design matrix X = [1, x - x_bar].

    Args:
        ages: Age grid. Shape (A,).

    Returns:
        Tuple (X, x_bar) where X has shape (A, 2) and x_bar is the mean age.
    """
    if ages.ndim != 1:
        raise ValueError("ages must be a 1D array.")
    x_bar = float(ages.mean())
    z = ages - x_bar
    X = np.column_stack([np.ones_like(z, dtype=float), z.astype(float)])  # (A, 2)
    return X, x_bar


def fit_cbd(q: FloatArray, ages: FloatArray) -> CBDM5Params:
    """Fit the two-factor CBD model to q[age, year].

    The model is:
        logit(q_{x,t}) = kappa1_t + kappa2_t * (x - x_bar).

    This model is typically used for higher ages (e.g., 60+).

    Args:
        q: Death probabilities. Shape (A, T).
        ages: Age grid. Shape (A,).

    Returns:
        CBDM5Params with fitted period factors and age grid.

    Raises:
        ValueError: If inputs are invalid or shapes do not match.
    """
    q = np.asarray(q, dtype=float)
    ages = np.asarray(ages, dtype=float)
    if q.ndim != 2:
        raise ValueError("q must be a 2D array with shape (A, T).")
    if ages.ndim != 1:
        raise ValueError("ages must be 1D.")
    A, _T = q.shape
    if ages.shape[0] != A:
        raise ValueError("ages length must match q.shape[0].")
    if not np.isfinite(q).all():
        raise ValueError("q must contain finite values.")
    if (q <= 0).any() or (q >= 1).any():
        raise ValueError("q must be strictly in (0,1).")
    if not np.isfinite(ages).all():
        raise ValueError("ages must be finite.")
    validate_q(q)

    # Build design matrix once (same ages for all years)
    X, x_bar = _build_cbd_design(ages)  # X: (A, 2)
    XtX = X.T @ X

    # transform q to logits
    y = _logit(q)  # (A, T)
    Xty = X.T @ y
    try:
        beta_hat_all = np.linalg.solve(XtX, Xty)  # (2, T)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("X'X is singular in CBD fit.") from exc

    kappa1 = beta_hat_all[0, :]  # (T,)
    kappa2 = beta_hat_all[1, :]  # (T,)

    return CBDM5Params(
        kappa1=kappa1,
        kappa2=kappa2,
        ages=ages,
        x_bar=x_bar,
    )


def reconstruct_logit_q(params: CBDM5Params) -> FloatArray:
    """Reconstruct the logit mortality surface from CBD parameters.

    Args:
        params: Fitted CBDM5 parameters.

    Returns:
        Logit mortality surface with shape (A, T).
    """
    k1 = params.kappa1  # (T,)
    k2 = params.kappa2  # (T,)
    ages = params.ages  # (A,)
    z = ages - params.x_bar  # (A,)

    # logit_q[x,t] = kappa1_t + kappa2_t * (x - x_bar)
    # → (A, T) = (1,T) + (A,1)*(1,T)
    return k1[None, :] + z[:, None] * k2[None, :]


def reconstruct_q(params: CBDM5Params) -> FloatArray:
    """Reconstruct mortality probabilities from CBD parameters.

    Args:
        params: Fitted CBDM5 parameters.

    Returns:
        Mortality probabilities with shape (A, T).
    """
    logit_q = reconstruct_logit_q(params)
    return 1.0 / (1.0 + np.exp(-logit_q))


def estimate_rw_params_cbd(params: CBDM5Params) -> CBDM5Params:
    """Estimate random-walk-with-drift parameters for kappa1_t and kappa2_t.

    Args:
        params: Fitted CBDM5 parameters.

    Returns:
        Updated CBDM5Params with mu/sigma values stored.
    """
    mu1, sigma1 = _estimate_rw_params(params.kappa1)
    mu2, sigma2 = _estimate_rw_params(params.kappa2)
    params.mu1, params.sigma1 = mu1, sigma1
    params.mu2, params.sigma2 = mu2, sigma2
    return params


class CBDM5:
    def __init__(self) -> None:
        self.params: CBDM5Params | None = None

    def fit(self, q: FloatArray, ages: FloatArray) -> CBDM5:
        """Fit the model and store parameters.

        Args:
            q: Death probabilities. Shape (A, T).
            ages: Age grid. Shape (A,).

        Returns:
            Self, with fitted parameters stored.
        """
        self.params = fit_cbd(q, ages)
        return self

    def estimate_rw(self) -> tuple[float, float, float, float]:
        """Estimate random-walk-with-drift parameters for kappa1_t and kappa2_t.

        Returns:
            Tuple of (mu1, sigma1, mu2, sigma2).
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        self.params = estimate_rw_params_cbd(self.params)
        mu1 = cast(float, self.params.mu1)
        sigma1 = cast(float, self.params.sigma1)
        mu2 = cast(float, self.params.mu2)
        sigma2 = cast(float, self.params.sigma2)
        return (mu1, sigma1, mu2, sigma2)

    def predict_logit_q(self) -> FloatArray:
        """Reconstruct the logit mortality surface from fitted parameters.

        Returns:
            Logit mortality surface with shape (A, T).
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        return reconstruct_logit_q(self.params)

    def predict_q(self) -> FloatArray:
        """Reconstruct mortality probabilities from fitted parameters.

        Returns:
            Mortality probabilities with shape (A, T).
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        return reconstruct_q(self.params)

    def simulate_kappa(
        self,
        kappa_index: str,
        horizon: int,
        n_sims: int = 1000,
        seed: int | None = None,
        include_last: bool = False,
    ) -> FloatArray:
        """Simulate random-walk forecasts for kappa1_t or kappa2_t.

        Args:
            kappa_index: "kappa1" or "kappa2".
            horizon: Number of years to simulate.
            n_sims: Number of scenarios.
            seed: Random seed for reproducibility.
            include_last: Whether to include the last observed kappa_t.

        Returns:
            Simulated paths with shape (N, H) or (N, H + 1) if include_last.
        """
        if self.params is None:
            raise ValueError("Fit the model first.")

        if kappa_index == "kappa1":
            k_last = float(self.params.kappa1[-1])
            mu = float(cast(float, self.params.mu1))
            sigma = float(cast(float, self.params.sigma1))
        elif kappa_index == "kappa2":
            k_last = float(self.params.kappa2[-1])
            mu = float(cast(float, self.params.mu2))
            sigma = float(cast(float, self.params.sigma2))
        else:
            raise ValueError("kappa_index must be 'kappa1' or 'kappa2'.")

        horizon = int(horizon)
        n_sims = int(n_sims)
        if horizon <= 0 or n_sims <= 0:
            raise ValueError("horizon and n_sims must be positive integers.")

        rng = np.random.default_rng(seed)

        from pymort.analysis.projections import simulate_random_walk_paths

        return simulate_random_walk_paths(
            k_last=k_last,
            mu=mu,
            sigma=sigma,
            horizon=horizon,
            n_sims=n_sims,
            rng=rng,
            include_last=include_last,
        )
