from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pymort.analysis.projections import simulate_random_walk_paths
from pymort.models.lc_m1 import estimate_rw_params, fit_lee_carter


@dataclass
class LCM2Params:
    """
    Parameters of the Lee–Carter with cohort effect (LC M2) :

        log m_{x,t} = a_x + b_x k_t + gamma_{t-x}

    where gamma_{t-x} captures the cohort effect (year of birth).
    """

    # LC "classic" parameters
    a: np.ndarray  # (A,)
    b: np.ndarray  # (A,)
    k: np.ndarray  # (T,)

    # cohort effect
    gamma: np.ndarray  # (C,) values of gamma_c
    cohorts: np.ndarray  # (C,) indices of cohort c = t - x

    # grids used
    ages: np.ndarray  # (A,)
    years: np.ndarray  # (T,)

    # RW+drift on k_t
    mu: Optional[float] = None
    sigma: Optional[float] = None

    # convenience helper
    def gamma_for_age_at_last_year(self, age: float) -> float:
        c = self.years[-1] - age
        # safety margin
        if c < self.cohorts[0] or c > self.cohorts[-1]:
            raise ValueError(f"Cohort {c} is outside stored cohort range.")

        # we project onto the nearest full cohort
        c_rounded = float(round(c))
        idx = np.searchsorted(self.cohorts, c_rounded)
        if idx >= len(self.cohorts):
            idx = len(self.cohorts) - 1

        return float(self.gamma[idx])


def _compute_cohort_index(ages: np.ndarray, years: np.ndarray) -> np.ndarray:
    """
    Compute the cohort index c = t - x on the (age, year) grid.

    Returns an array C with shape (A, T) where C[x,t] = years[t] - ages[x].
    """
    ages = np.asarray(ages)
    years = np.asarray(years)
    return years[None, :] - ages[:, None]


def fit_lee_carter_cohort(
    m: np.ndarray,
    ages: np.ndarray,
    years: np.ndarray,
) -> LCM2Params:
    """
    Fit the Lee–Carter model with cohort effect:

        log m_{x,t} = a_x + b_x k_t + gamma_{t-x}.

    Simple strategy (analogous to your CBD+cohort):
      1) Fit the "classic" LC (a_x, b_x, k_t) over the entire surface.
      2) Compute residuals on log m.
      3) Aggregate residuals by cohort c = t - x -> gamma_c = mean(residuals).
      4) Center gamma_c so that its weighted mean is zero (identifiability).
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
        w_mean = np.average(gamma[mask_nonzero], weights=counts[mask_nonzero])
        gamma = gamma - w_mean

    return LCM2Params(
        a=base.a,
        b=base.b,
        k=base.k,
        gamma=gamma,
        cohorts=cohorts,
        ages=ages.astype(int),
        years=years.astype(int),
    )


def reconstruct_log_m_cohort(params: LCM2Params) -> np.ndarray:
    """
    Reconstruct log m_{x,t} for LC with cohort on the original grid (ages, years).
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
    return base_ln + gamma_matrix


def reconstruct_m_cohort(params: LCM2Params) -> np.ndarray:
    """
    Reconstruit m_{x,t} pour le modèle LC+cohorte.
    """
    ln_m = reconstruct_log_m_cohort(params)
    return np.exp(ln_m)


class LCM2:
    """
    Lee–Carter with cohort effect:

        log m_{x,t} = a_x + b_x k_t + gamma_{t-x}.
    """

    def __init__(self) -> None:
        self.params: Optional[LCM2Params] = None

    def fit(
        self,
        m: np.ndarray,
        ages: np.ndarray,
        years: np.ndarray,
    ) -> "LCM2":
        """
        Fit LC with cohort on a surface m[age, year] and store the parameters.
        """
        self.params = fit_lee_carter_cohort(m, ages, years)
        return self

    def estimate_rw(self) -> tuple[float, float]:
        """
        Estimate RW+drift parameters for k_t and store them in params.
        Gamma_c is treated as fixed (no dynamics).
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        mu, sigma = estimate_rw_params(self.params.k)
        self.params.mu, self.params.sigma = mu, sigma
        return mu, sigma

    def predict_log_m(self) -> np.ndarray:
        """
        Reconstruct log m_{x,t} from LC with cohort parameters.
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        return reconstruct_log_m_cohort(self.params)

    def predict_m(self) -> np.ndarray:
        """
        Reconstruct m_{x,t} from LC with cohort parameters.
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
    ) -> np.ndarray:
        """
        Simulate RW-drift paths of k_t using the central vectorized function.
        Gamma_c stays fixed.
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

        paths = simulate_random_walk_paths(
            k_last=k_last,
            mu=mu,
            sigma=sigma,
            horizon=horizon,
            n_sims=n_sims,
            rng=rng,
            include_last=include_last,
        )

        return paths
