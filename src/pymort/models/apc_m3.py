"""Age-Period-Cohort mortality model (APC M3).

This module fits and simulates a simple age-period-cohort model on log mortality.

Note:
    Docstrings follow Google style and type hints use NDArray for clarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pymort.models.utils import estimate_rw_params

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


@dataclass
class APCM3Params:
    """Parameters for the Age-Period-Cohort model M3.

    The model is:
        log m_{x,t} = beta_x + kappa_t + gamma_{t-x}.

    In some CBD/APC notation, kappa_t and gamma_c can be normalized by a factor
    1 / n_a. Here we absorb that normalization into the parameters to keep the
    fitted surface unchanged.
    """

    beta_age: FloatArray  # (A,)  age effect beta_x
    kappa: FloatArray  # (T,)  period effect kappa_t
    gamma: FloatArray  # (C,)  cohort effect gamma_c
    cohorts: FloatArray  # (C,)  sorted cohort indices c = t - x

    ages: FloatArray  # (A,)
    years: IntArray  # (T,)

    # RW + drift parameters for kappa_t (used for forecasting)
    mu: float | None = None
    sigma: float | None = None

    def gamma_for_age_at_last_year(self, age: float) -> float:
        """Return the cohort effect for a given age at the last calendar year.

        Args:
            age: Age at the last observed year.

        Returns:
            Cohort effect gamma_{t-x} for that age.
        """
        c = self.years[-1] - age
        c_rounded = round(c)
        idx = np.searchsorted(self.cohorts, c_rounded)
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


def fit_apc_m3(
    m: FloatArray,
    ages: FloatArray,
    years: IntArray,
) -> APCM3Params:
    """Fit the APC M3 model on a mortality surface.

    The model is:
        log m_{x,t} = beta_x + kappa_t + gamma_{t-x} + eps_{x,t}.

    Estimation strategy (simple least squares):
    1) beta_x = mean_t log m_{x,t}
    2) Residuals R_{x,t} = log m_{x,t} - beta_x
    3) Regress R_{x,t} on period and cohort dummies (first levels as reference)
    4) Rebuild kappa_t and gamma_c with zero references

    Args:
        m: Central death rates. Shape (A, T).
        ages: Age grid. Shape (A,).
        years: Year grid. Shape (T,).

    Returns:
        APCM3Params with fitted effects and grids.

    Raises:
        ValueError: If inputs are invalid or shapes do not match.
    """
    m = np.asarray(m, dtype=float)
    ages = np.asarray(ages)
    years = np.asarray(years)

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

    # 1) Age effect β_x
    ln_m = np.log(m)  # (A, T)
    beta_age = ln_m.mean(axis=1)  # (A,)

    # 2) Residuals after removing age effect
    resid = ln_m - beta_age[:, None]  # (A, T)

    # 3) Linear model R_{x,t} ≈ κ_t + gamma_{t-x}
    # Flatten residuals
    y = resid.ravel()  # shape (N,) with N = A * T

    # Period index 0..T-1 for each cell
    _, J = np.indices((A, T))
    period_idx = J.ravel()  # length N

    # Cohort index for each cell (calendar birth year or offset)
    C_grid = _compute_cohort_index(ages, years)  # (A, T)
    C_flat = C_grid.ravel()
    cohorts, cohort_idx = np.unique(C_flat, return_inverse=True)
    C = cohorts.size

    # Design matrix X : N x P   (P = (T-1) + (C-1))
    # Columns:
    #   0 .. T-2             → period dummies for t = 1..T-1 (t = 0 is ref)
    #   T-1 .. T+C-3         → cohort dummies for c = 1..C-1 (c = 0 is ref)
    P = (T - 1) + (C - 1)
    X = np.zeros((y.size, P), dtype=float)

    # Period dummies
    # For each observation, if t > 0, put 1 in column (t-1)
    mask_t = period_idx > 0
    X[np.nonzero(mask_t)[0], period_idx[mask_t] - 1] = 1.0

    # Cohort dummies
    # For each observation, if cohort_idx > 0, put 1 in column (T-1 + cohort_idx - 1)
    mask_c = cohort_idx > 0
    X[
        np.nonzero(mask_c)[0],
        (T - 1) + cohort_idx[mask_c] - 1,
    ] = 1.0

    # Solve least squares X beta ≈ y
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)

    # Rebuild kappa_t and gamma_c with 0 reference for the first period and cohort
    kappa = np.zeros(T, dtype=float)
    gamma = np.zeros(C, dtype=float)

    kappa[1:] = beta_hat[0 : T - 1]
    gamma[1:] = beta_hat[T - 1 :]

    return APCM3Params(
        beta_age=beta_age,
        kappa=kappa,
        gamma=gamma,
        cohorts=cohorts,
        ages=ages,
        years=years,
    )


def reconstruct_log_m_apc(params: APCM3Params) -> FloatArray:
    """Reconstruct the fitted log-mortality surface.

    Args:
        params: Fitted APCM3 parameters.

    Returns:
        Log mortality surface with shape (A, T).
    """
    beta = params.beta_age
    kappa = params.kappa
    gamma = params.gamma

    ages = params.ages
    years = params.years

    ln_m_age = beta[:, None]  # (A, 1)
    ln_m_period = kappa[None, :]  # (1, T)

    # Cohort effect gamma_{t-x}
    C_grid = _compute_cohort_index(ages, years)
    C_int = np.rint(C_grid).astype(params.cohorts.dtype)
    idx = np.searchsorted(params.cohorts, C_int)
    idx = np.clip(idx, 0, len(params.cohorts) - 1)
    gamma_matrix = gamma[idx]

    return ln_m_age + ln_m_period + gamma_matrix


def reconstruct_m_apc(params: APCM3Params) -> FloatArray:
    """Reconstruct m_{x,t} from APC M3 parameters.

    Args:
        params: Fitted APCM3 parameters.

    Returns:
        Central death rates with shape (A, T).
    """
    ln_m = reconstruct_log_m_apc(params)
    return np.exp(ln_m)


class APCM3:
    """Age-Period-Cohort model M3.

    The model is:
        log m_{x,t} = beta_x + kappa_t + gamma_{t-x}.
    """

    def __init__(self) -> None:
        self.params: APCM3Params | None = None

    def fit(
        self,
        m: FloatArray,
        ages: FloatArray,
        years: IntArray,
    ) -> APCM3:
        """Fit the model and store parameters.

        Args:
            m: Central death rates. Shape (A, T).
            ages: Age grid. Shape (A,).
            years: Year grid. Shape (T,).

        Returns:
            Self, with fitted parameters stored.
        """
        self.params = fit_apc_m3(m, ages, years)
        return self

    def estimate_rw(self) -> tuple[float, float]:
        """Estimate random-walk-with-drift parameters for kappa_t.

        The cohort effect gamma_c is treated as static.

        Returns:
            Tuple of (mu, sigma).
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        mu, sigma = estimate_rw_params(self.params.kappa)
        self.params.mu, self.params.sigma = mu, sigma
        return mu, sigma

    def predict_log_m(self) -> FloatArray:
        """Reconstruct log m_{x,t} from fitted parameters.

        Returns:
            Log mortality surface with shape (A, T).
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        return reconstruct_log_m_apc(self.params)

    def predict_m(self) -> FloatArray:
        """Reconstruct m_{x,t} from fitted parameters.

        Returns:
            Central death rates with shape (A, T).
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        return reconstruct_m_apc(self.params)

    def simulate_kappa(
        self,
        horizon: int,
        n_sims: int = 1000,
        seed: int | None = None,
        include_last: bool = False,
    ) -> FloatArray:
        """Simulate random-walk-with-drift paths of kappa_t.

        Gamma_c remains fixed (cohort effect not simulated).

        Args:
            horizon: Number of years to simulate.
            n_sims: Number of scenarios.
            seed: Random seed for reproducibility.
            include_last: Whether to include the last observed kappa_t.

        Returns:
            Simulated kappa_t paths with shape (N, H) or (N, H + 1) if include_last.
        """
        if self.params is None or self.params.mu is None or self.params.sigma is None:
            raise ValueError("Fit & estimate_rw first.")

        horizon = int(horizon)
        n_sims = int(n_sims)
        if horizon <= 0 or n_sims <= 0:
            raise ValueError("horizon and n_sims must be positive integers.")

        rng = np.random.default_rng(seed)

        k_last = float(self.params.kappa[-1])
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
