"""Lee-Carter model with cohort effect (LCM2).

This module implements a Lee-Carter variant that adds a cohort term to capture
year-of-birth effects.

Note:
    Docstrings follow Google style and type hints use NDArray for clarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from pymort.models.lc_m1 import fit_lee_carter
from pymort.models.utils import estimate_rw_params

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


@dataclass
class LCM2Params:
    """Parameters for the Lee-Carter model with a cohort effect.

    The model is:
        log m_{x,t} = a_x + b_x k_t + gamma_{t-x},
    where gamma_{t-x} captures the year-of-birth (cohort) effect.
    """

    # LC "classic" parameters
    a: FloatArray  # (A,)
    b: FloatArray  # (A,)
    k: FloatArray  # (T,)

    # cohort effect
    gamma: FloatArray  # (C,) values of gamma_c
    cohorts: FloatArray  # (C,) indices of cohort c = t - x

    # grids used
    ages: FloatArray  # (A,)
    years: IntArray  # (T,)

    # RW+drift on k_t
    mu: float | None = None
    sigma: float | None = None

    # convenience helper
    def gamma_for_age_at_last_year(self, age: float) -> float:
        """Return the cohort effect for a given age at the last calendar year.

        Args:
            age: Age at the last observed year.

        Returns:
            Cohort effect gamma_{t-x} for that age.

        Raises:
            ValueError: If the implied cohort is outside the stored range.
        """
        c = self.years[-1] - age
        # safety margin
        if c < self.cohorts[0] or c > self.cohorts[-1]:
            raise ValueError(f"Cohort {c} is outside stored cohort range.")

        # we project onto the nearest full cohort
        c_rounded = float(round(c))
        idx = int(np.searchsorted(self.cohorts, c_rounded))
        if idx >= len(self.cohorts):
            idx = len(self.cohorts) - 1

        return float(self.gamma[idx])


def _compute_cohort_index(ages: FloatArray, years: IntArray) -> FloatArray:
    """Compute the cohort index c = t - x on the age/year grid.

    Args:
        ages: Age grid. Shape (A,).
        years: Calendar year grid. Shape (T,).

    Returns:
        Cohort indices with shape (A, T) where C[x, t] = years[t] - ages[x].
    """
    ages = np.asarray(ages)
    years = np.asarray(years)
    return years[None, :] - ages[:, None]


def fit_lee_carter_cohort(
    m: FloatArray,
    ages: FloatArray,
    years: IntArray,
) -> LCM2Params:
    """Fit the Lee-Carter model with a cohort effect.

    The fitted model is:
        log m_{x,t} = a_x + b_x k_t + gamma_{t-x}.

    Estimation strategy:
    1) Fit the classic Lee-Carter parameters (a_x, b_x, k_t).
    2) Compute residuals on log m.
    3) Average residuals by cohort c = t - x to estimate gamma_c.
    4) Center gamma_c to enforce identifiability.

    Args:
        m: Central death rates. Shape (A, T).
        ages: Age grid. Shape (A,).
        years: Year grid. Shape (T,).

    Returns:
        LCM2Params with fitted parameters and grids.

    Raises:
        ValueError: If inputs have incompatible shapes or invalid values.
    """
    if m.ndim != 2:
        raise ValueError("m must be a 2D array (A, T).")
    if ages.ndim != 1:
        raise ValueError("ages must be 1D.")
    if years.ndim != 1:
        raise ValueError("years must be 1D.")
    A, T = m.shape
    if ages.shape[0] != A:
        raise ValueError("ages length must match m.shape[0].")
    if years.shape[0] != T:
        raise ValueError("years length must match m.shape[1].")
    if not np.isfinite(m).all() or (m <= 0).any():
        raise ValueError("m must be strictly positive and finite.")

    # 1) Fit the "classic" LC
    base = fit_lee_carter(m)  # LCParams(a, b, k)
    ln_m = np.log(m)  # (A, T)
    ln_hat_base = base.a[:, None] + np.outer(base.b, base.k)  # (A, T)

    # 2) Compute residuals on log m
    residuals = ln_m - ln_hat_base  # (A, T)

    # 3) Group by cohort c = t - x
    C = _compute_cohort_index(ages, years)  # (A, T)
    C_flat = C.ravel()
    r_flat = residuals.ravel()

    cohorts, inverse_idx = np.unique(C_flat, return_inverse=True)
    gamma = np.zeros_like(cohorts, dtype=float)
    counts = np.zeros_like(cohorts, dtype=float)

    # Sum of residuals and number of points per cohort
    np.add.at(gamma, inverse_idx, r_flat)
    np.add.at(counts, inverse_idx, 1.0)

    mask_nonzero = counts > 0
    gamma[mask_nonzero] /= counts[mask_nonzero]

    # 4) Center gamma (weighted mean = 0)
    if np.any(mask_nonzero):
        gamma = gamma - float(np.mean(gamma[mask_nonzero]))

    return LCM2Params(
        a=base.a,
        b=base.b,
        k=base.k,
        gamma=gamma,
        cohorts=cohorts,
        ages=ages.astype(int),
        years=years.astype(int),
    )


def reconstruct_log_m_cohort(params: LCM2Params) -> FloatArray:
    """Reconstruct log m_{x,t} for the LCM2 model on the original grid.

    Args:
        params: Fitted LCM2 parameters.

    Returns:
        Log mortality surface with shape (A, T).
    """
    a = params.a
    b = params.b
    k = params.k
    ages = params.ages
    years = params.years

    # classic LC part
    base_ln = a[:, None] + np.outer(b, k)  # (A, T)

    # cohort part gamma_{t-x}
    C = _compute_cohort_index(ages, years)  # (A, T)
    C_int = np.rint(C).astype(params.cohorts.dtype)
    idx = np.searchsorted(params.cohorts, C_int)
    # security margin
    idx = np.clip(idx, 0, len(params.cohorts) - 1)

    gamma_matrix = params.gamma[idx]
    out = np.asarray(base_ln + gamma_matrix, dtype=float)
    return cast(FloatArray, out)


def reconstruct_m_cohort(params: LCM2Params) -> FloatArray:
    """Reconstruct m_{x,t} for the LCM2 model.

    Args:
        params: Fitted LCM2 parameters.

    Returns:
        Central death rates with shape (A, T).
    """
    ln_m = reconstruct_log_m_cohort(params)
    return np.exp(ln_m)


class LCM2:
    """Lee-Carter with cohort effect.

    The model is:
        log m_{x,t} = a_x + b_x k_t + gamma_{t-x}.
    """

    def __init__(self) -> None:
        self.params: LCM2Params | None = None

    def fit(
        self,
        m: FloatArray,
        ages: FloatArray,
        years: IntArray,
    ) -> LCM2:
        """Fit the model and store parameters.

        Args:
            m: Central death rates. Shape (A, T).
            ages: Age grid. Shape (A,).
            years: Year grid. Shape (T,).

        Returns:
            Self, with fitted parameters stored.
        """
        self.params = fit_lee_carter_cohort(m, ages, years)
        return self

    def estimate_rw(self) -> tuple[float, float]:
        """Estimate random-walk-with-drift parameters for k_t.

        Gamma_c is treated as fixed (no dynamics).

        Returns:
            Tuple of (mu, sigma).

        Raises:
            ValueError: If the model is not fitted.
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        mu, sigma = estimate_rw_params(self.params.k)
        self.params.mu, self.params.sigma = mu, sigma
        return mu, sigma

    def predict_log_m(self) -> FloatArray:
        """Reconstruct log m_{x,t} from fitted parameters.

        Returns:
            Log mortality surface with shape (A, T).
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        return reconstruct_log_m_cohort(self.params)

    def predict_m(self) -> FloatArray:
        """Reconstruct m_{x,t} from fitted parameters.

        Returns:
            Central death rates with shape (A, T).
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        return reconstruct_m_cohort(self.params)

    def simulate_k(
        self,
        horizon: int,
        n_sims: int = 1000,
        seed: int | None = None,
        include_last: bool = False,
    ) -> FloatArray:
        """Simulate random-walk-with-drift paths of k_t.

        Gamma_c stays fixed.

        Args:
            horizon: Number of years to simulate.
            n_sims: Number of scenarios.
            seed: Random seed for reproducibility.
            include_last: Whether to include the last observed k_t.

        Returns:
            Simulated k_t paths with shape (N, H) or (N, H + 1) if include_last.
        """
        if self.params is None or self.params.mu is None or self.params.sigma is None:
            raise ValueError("Fit & estimate_rw first.")

        horizon = int(horizon)
        n_sims = int(n_sims)
        if horizon <= 0 or n_sims <= 0:
            raise ValueError("horizon and n_sims must be positive integers.")

        rng = np.random.default_rng(seed)

        k_last = float(self.params.k[-1])
        mu = float(self.params.mu)
        sigma = float(self.params.sigma)

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
