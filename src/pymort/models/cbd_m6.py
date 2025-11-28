from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from pymort.analysis.projections import simulate_random_walk_paths
from pymort.lifetables import validate_q
from pymort.models.cbd_m5 import _estimate_rw_params, _logit, fit_cbd


@dataclass
class CBDM6Params:
    """
    Parameters of a CBD model with cohort effect (M6):

        logit(q_{x,t}) = kappa1_t + kappa2_t * (x - x_bar) + gamma_{t-x}

    where gamma_{t-x} captures cohort effects (year-of-birth effect).
    """

    # Period factors (as in basic CBD)
    kappa1: np.ndarray  # (T,)
    kappa2: np.ndarray  # (T,)

    # Cohort effect
    gamma: np.ndarray  # (C,) values of cohort effect
    cohorts: np.ndarray  # (C,) corresponding cohort indices c = t - x

    # Age / time grids used in the fit
    ages: np.ndarray  # (A,)
    years: np.ndarray  # (T,)
    x_bar: float  # mean age used for centering

    # Optional RW+drift parameters on (kappa1_t, kappa2_t)
    mu1: Optional[float] = None
    sigma1: Optional[float] = None
    mu2: Optional[float] = None
    sigma2: Optional[float] = None

    def gamma_for_age_at_last_year(self, age: float) -> float:
        """
        Return gamma_{t-x} for a given age at the last observed calendar year.
        Assumes age is (en pratique) sur la grille self.ages.
        """
        ages_grid = np.asarray(self.ages, dtype=float)
        age_eff = float(ages_grid[np.argmin(np.abs(ages_grid - float(age)))])

        c = float(self.years[-1] - age_eff)

        cohorts = np.asarray(self.cohorts, dtype=float)
        idx = np.searchsorted(cohorts, c)

        if idx == len(cohorts) or not np.isclose(cohorts[idx], c):
            raise ValueError(
                f"Cohort index {c} (age={age_eff}) not found in stored gamma grid."
            )

        return float(self.gamma[idx])


def _compute_cohort_index(ages: np.ndarray, years: np.ndarray) -> np.ndarray:
    """
    Compute the cohort index c = t - x on the (age, year) grid.

    Returns an array C with shape (A, T) where C[x,t] = years[t] - ages[x].
    """
    ages = np.asarray(ages)
    years = np.asarray(years)
    return years[None, :] - ages[:, None]


def fit_cbd_cohort(q: np.ndarray, ages: np.ndarray, years: np.ndarray) -> CBDM6Params:
    """
    Fit a CBD model with an additional cohort effect:

        logit(q_{x,t}) = kappa1_t + kappa2_t * (x - x_bar) + gamma_{t-x}.

    Strategy:
      1) Fit the basic CBD model (period factors only) on q_x,t.
      2) Compute residuals on logit scale.
      3) For each cohort c = t - x, average residuals over all (x,t) with that cohort.
      4) Center gamma_c so that its (weighted) mean is zero (identifiability).
    """
    if q.ndim != 2:
        raise ValueError("q must be a 2D array with shape (A, T).")
    if ages.ndim != 1:
        raise ValueError("ages must be 1D.")
    if years.ndim != 1:
        raise ValueError("years must be 1D.")
    A, T = q.shape
    if ages.shape[0] != A:
        raise ValueError("ages length must match q.shape[0].")
    if years.shape[0] != T:
        raise ValueError("years length must match q.shape[1].")
    validate_q(q)

    # 1) Fit basic CBD (period factors only)
    base = fit_cbd(q, ages)  # CBDParams(kappa1, kappa2, ages, x_bar)

    # 2) Compute logit residuals: res{x,t} = logit(q_{x,t}) - logit(q̂_{x,t}^{CBD})
    logit_q = _logit(q)  # (A, T)
    logit_q_hat_base = (
        base.kappa1[None, :] + (ages - base.x_bar)[:, None] * base.kappa2[None, :]
    )
    residuals = logit_q - logit_q_hat_base  # (A, T)

    # 3) Group residuals by cohort c = t - x
    C = _compute_cohort_index(ages, years)  # (A, T)
    C_flat = C.ravel()
    r_flat = residuals.ravel()

    cohorts, inverse_idx = np.unique(C_flat, return_inverse=True)
    gamma = np.zeros_like(cohorts, dtype=float)
    counts = np.zeros_like(cohorts, dtype=float)

    # Accumulate sum of residuals per cohort
    np.add.at(gamma, inverse_idx, r_flat)
    np.add.at(counts, inverse_idx, 1.0)

    # Avoid division by zero
    mask_nonzero = counts > 0
    gamma[mask_nonzero] /= counts[mask_nonzero]

    # 4) Center gamma so that weighted mean is zero (identifiability)
    w_mean = np.average(gamma[mask_nonzero], weights=counts[mask_nonzero])
    gamma = gamma - w_mean

    return CBDM6Params(
        kappa1=base.kappa1,
        kappa2=base.kappa2,
        gamma=gamma,
        cohorts=cohorts,
        ages=ages,
        years=years,
        x_bar=base.x_bar,
    )


def reconstruct_logit_q_cbd_cohort(params: CBDM6Params) -> np.ndarray:
    """
    Reconstruct logit(q_{x,t}) for the CBD cohort model on the original age/year grid.
    """
    ages = params.ages
    years = params.years
    z = ages - params.x_bar  # (A,)

    # baseline CBD part
    k1 = params.kappa1  # (T,)
    k2 = params.kappa2  # (T,)
    base_logit = k1[None, :] + z[:, None] * k2[None, :]  # (A, T)

    # cohort part: gamma_{t-x}
    C = _compute_cohort_index(ages, years)  # (A, T)
    idx = np.searchsorted(params.cohorts, C)

    # out-of-bounds indices or exact mismatch
    out_of_range = (idx == len(params.cohorts)) | (params.cohorts[idx] != C)
    if np.any(out_of_range):
        raise RuntimeError("Cohort index mismatch in CBDM6 reconstruction.")

    gamma_matrix = params.gamma[idx]
    return base_logit + gamma_matrix


def reconstruct_q_cbd_cohort(params: CBDM6Params) -> np.ndarray:
    """
    Reconstruct q_{x,t} for the CBD cohort model.
    """
    logit_q = reconstruct_logit_q_cbd_cohort(params)
    return 1.0 / (1.0 + np.exp(-logit_q))


def estimate_rw_params_cbd_cohort(params: CBDM6Params) -> CBDM6Params:
    """
    Estimate RW+drift parameters for (kappa1_t, kappa2_t) and store them
    into the CBDM6Params object. Gamma is cohort-specific and treated as static.
    """
    mu1, sigma1 = _estimate_rw_params(params.kappa1)
    mu2, sigma2 = _estimate_rw_params(params.kappa2)
    params.mu1, params.sigma1 = mu1, sigma1
    params.mu2, params.sigma2 = mu2, sigma2
    return params


class CBDM6:
    """
    CBD model with cohort effect:

        logit(q_{x,t}) = kappa1_t + kappa2_t * (x - x_bar) + gamma_{t-x}.
    """

    def __init__(self) -> None:
        self.params: Optional[CBDM6Params] = None

    def fit(self, q: np.ndarray, ages: np.ndarray, years: np.ndarray) -> "CBDM6":
        """
        Fit the CBD cohort model on q[age, year] and store parameters.
        """
        self.params = fit_cbd_cohort(q, ages, years)
        return self

    def estimate_rw(self) -> Tuple[float, float, float, float]:
        """
        Estimate RW+drift parameters for (kappa1_t, kappa2_t).
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        self.params = estimate_rw_params_cbd_cohort(self.params)
        return (
            self.params.mu1,
            self.params.sigma1,
            self.params.mu2,
            self.params.sigma2,
        )

    def predict_logit_q(self) -> np.ndarray:
        """
        Reconstruct the logit-mortality surface implied by the fitted CBD cohort model.
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        return reconstruct_logit_q_cbd_cohort(self.params)

    def predict_q(self) -> np.ndarray:
        """
        Reconstruct q_{x,t} implied by the fitted CBD cohort model.
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        return reconstruct_q_cbd_cohort(self.params)

    def simulate_kappa(
        self,
        kappa_index: str,
        horizon: int,
        n_sims: int = 1000,
        seed: Optional[int] = None,
        include_last: bool = False,
    ) -> np.ndarray:
        """
        Simulate random–walk forecasts of kappa1_t or kappa2_t
        using the fitted drift and volatility parameters (vectorized).
        """
        if self.params is None:
            raise ValueError("Fit the model first.")

        if kappa_index == "kappa1":
            k_last = float(self.params.kappa1[-1])
            mu = self.params.mu1
            sigma = self.params.sigma1
        elif kappa_index == "kappa2":
            k_last = float(self.params.kappa2[-1])
            mu = self.params.mu2
            sigma = self.params.sigma2
        else:
            raise ValueError("kappa_index must be 'kappa1' or 'kappa2'.")

        if mu is None or sigma is None:
            raise ValueError("Call estimate_rw() before simulate_kappa().")

        horizon = int(horizon)
        n_sims = int(n_sims)
        if horizon <= 0 or n_sims <= 0:
            raise ValueError("horizon and n_sims must be positive integers.")

        rng = np.random.default_rng(seed)

        paths = simulate_random_walk_paths(
            k_last=k_last,
            mu=float(mu),
            sigma=float(sigma),
            horizon=horizon,
            n_sims=n_sims,
            rng=rng,
            include_last=include_last,
        )

        return paths
