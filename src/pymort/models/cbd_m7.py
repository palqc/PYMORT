"""Cairns-Blake-Dowd quadratic model with cohort effect (CBD M7).

This module fits the CBD M7 model and provides random-walk forecasts for the
period factors.

Note:
    Docstrings follow Google style and type hints use NDArray for clarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from pymort.lifetables import validate_q
from pymort.models.cbd_m5 import _logit
from pymort.models.utils import _estimate_rw_params

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


@dataclass
class CBDM7Params:
    """Parameters for the CBD M7 model (quadratic + cohort).

    The model is:
        logit(q_{x,t}) =
            kappa1_t
          + kappa2_t (x - x_bar)
          + kappa3_t ((x - x_bar)^2 - sigma_x^2)
          + gamma_{t-x},
    where kappa1_t, kappa2_t, kappa3_t are period factors and gamma_{t-x}
    captures the cohort effect.
    """

    # Period factors
    kappa1: FloatArray  # (T,)
    kappa2: FloatArray  # (T,)
    kappa3: FloatArray  # (T,)

    # Cohort effect gamma_c and its grid of cohort indices c = t - x
    gamma: FloatArray  # (C,)
    cohorts: FloatArray  # (C,)

    # Age / time grids used in the fit
    ages: FloatArray  # (A,)
    years: IntArray  # (T,)
    x_bar: float  # mean age
    sigma2_x: float  # mean squared deviation of ages (for quadratic term)

    # Optional RW+drift parameters for (kappa1, kappa2, kappa3)
    mu1: float | None = None
    sigma1: float | None = None
    mu2: float | None = None
    sigma2: float | None = None
    mu3: float | None = None
    sigma3: float | None = None

    # Convenience helper
    def gamma_for_age_at_last_year(self, age: float) -> float:
        """Return gamma_{t-x} for a given age at the last observed year.

        Args:
            age: Age at the last observed year.

        Returns:
            Cohort effect gamma_{t-x}.
        """
        age = float(age)
        c = float(self.years[-1] - age)
        idx = np.searchsorted(self.cohorts, c)

        if idx >= len(self.cohorts) or self.cohorts[idx] != c:
            raise ValueError(
                f"Cohort index {c} not found in stored gamma grid. "
                "Make sure 'age' comes from params.ages."
            )
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


def fit_cbd_m7(q: FloatArray, ages: FloatArray, years: IntArray) -> CBDM7Params:
    """Fit the CBD M7 model (quadratic + cohort) to q[age, year].

    Estimation strategy (two-stage):
    1) Fit the quadratic CBD part via OLS on logit(q_{x,t}) ~ [1, z, z2c]
       with z = x - x_bar and z2c = z^2 - sigma_x^2.
    2) Compute residuals on logit scale and average them by cohort to
       estimate gamma_c.
    3) Center gamma for identifiability.

    Args:
        q: Death probabilities. Shape (A, T).
        ages: Age grid. Shape (A,).
        years: Year grid. Shape (T,).

    Returns:
        CBDM7Params with fitted period and cohort effects.

    Raises:
        ValueError: If inputs are invalid or shapes do not match.
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

    ages = ages.astype(float)
    years = years.astype(float)

    # Build age terms z and quadratic term z2c
    x_bar = float(ages.mean())
    z = ages - x_bar
    sigma2_x = float((z**2).mean())
    z2c = (z**2) - sigma2_x  # centered quadratic term

    # Design matrix X = [1, z, z2c]   shape: (A, 3)
    X = np.column_stack([np.ones_like(z), z, z2c])

    # Logit transform of q
    y = _logit(q)  # (A, T)

    # OLS for all years at once: beta_hat_all = (X'X)^(-1) X' y → shape (3, T)
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        raise ValueError(
            "CBD M7 fit failed: design matrix X'X is singular. "
            "Use at least two distinct ages with dispersion."
        )
    try:
        beta_hat_all = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError as exc:
        raise ValueError("CBD M7 fit failed: design matrix X'X is singular.") from exc

    kappa1 = beta_hat_all[0, :]  # (T,)
    kappa2 = beta_hat_all[1, :]  # (T,)
    kappa3 = beta_hat_all[2, :]  # (T,)

    # Baseline fitted logit without cohort effect
    base_logit = (
        kappa1[None, :] + z[:, None] * kappa2[None, :] + z2c[:, None] * kappa3[None, :]
    )  # (A, T)

    # Residuals to be explained by cohort effect gamma_{t-x}
    residuals = y - base_logit  # (A, T)

    # Estimate gamma_{t-x} by averaging residuals per cohort
    C = _compute_cohort_index(ages, years)  # (A, T)
    C_flat = C.ravel()
    r_flat = residuals.ravel()

    cohorts, inverse_idx = np.unique(C_flat, return_inverse=True)
    gamma = np.zeros_like(cohorts, dtype=float)
    counts = np.zeros_like(cohorts, dtype=float)

    np.add.at(gamma, inverse_idx, r_flat)
    np.add.at(counts, inverse_idx, 1.0)

    mask_nonzero = counts > 0
    gamma[mask_nonzero] /= counts[mask_nonzero]

    # Center gamma so that its weighted mean is zero (identifiability)
    if np.any(mask_nonzero):
        w_mean = np.average(gamma[mask_nonzero], weights=counts[mask_nonzero])
        gamma = gamma - w_mean

    return CBDM7Params(
        kappa1=kappa1,
        kappa2=kappa2,
        kappa3=kappa3,
        gamma=gamma,
        cohorts=cohorts,
        ages=ages,
        years=years,
        x_bar=x_bar,
        sigma2_x=sigma2_x,
    )


def reconstruct_logit_q_m7(params: CBDM7Params) -> FloatArray:
    """Reconstruct logit(q_{x,t}) from CBD M7 parameters.

    Args:
        params: Fitted CBDM7 parameters.

    Returns:
        Logit mortality surface with shape (A, T).
    """
    ages = params.ages
    years = params.years

    z = ages - params.x_bar
    z2c = (z**2) - params.sigma2_x

    k1 = params.kappa1
    k2 = params.kappa2
    k3 = params.kappa3

    # Baseline quadratic CBD part
    base_logit = k1[None, :] + z[:, None] * k2[None, :] + z2c[:, None] * k3[None, :]  # (A, T)

    # Cohort part: gamma_{t-x}
    C = _compute_cohort_index(ages, years)  # (A, T)
    idx = np.searchsorted(params.cohorts, C)
    if np.any(idx >= len(params.cohorts)):
        raise RuntimeError("Cohort indices out of range in reconstruction.")

    if np.any(params.cohorts[idx] != C):
        raise RuntimeError("Inconsistent cohort indices in reconstruction.")

    gamma_matrix = params.gamma[idx]  # (A, T)

    return base_logit + gamma_matrix


def reconstruct_q_m7(params: CBDM7Params) -> FloatArray:
    """Reconstruct q_{x,t} from CBD M7 parameters.

    Args:
        params: Fitted CBDM7 parameters.

    Returns:
        Mortality probabilities with shape (A, T).
    """
    logit_q = reconstruct_logit_q_m7(params)
    return 1.0 / (1.0 + np.exp(-logit_q))


def estimate_rw_params_m7(params: CBDM7Params) -> CBDM7Params:
    """Estimate random-walk-with-drift parameters for kappa1_t, kappa2_t, kappa3_t.

    Gamma (cohort effect) is treated as static.

    Args:
        params: Fitted CBDM7 parameters.

    Returns:
        Updated CBDM7Params with mu/sigma values stored.
    """
    mu1, sigma1 = _estimate_rw_params(params.kappa1)
    mu2, sigma2 = _estimate_rw_params(params.kappa2)
    mu3, sigma3 = _estimate_rw_params(params.kappa3)

    params.mu1, params.sigma1 = mu1, sigma1
    params.mu2, params.sigma2 = mu2, sigma2
    params.mu3, params.sigma3 = mu3, sigma3
    return params


class CBDM7:
    """CBD M7 model with quadratic age term and cohort effect."""

    def __init__(self) -> None:
        self.params: CBDM7Params | None = None

    def fit(self, q: FloatArray, ages: FloatArray, years: IntArray) -> CBDM7:
        """Fit the model and store parameters.

        Args:
            q: Death probabilities. Shape (A, T).
            ages: Age grid. Shape (A,).
            years: Year grid. Shape (T,).

        Returns:
            Self, with fitted parameters stored.
        """
        self.params = fit_cbd_m7(q, ages, years)
        return self

    def estimate_rw(self) -> tuple[float, float, float, float, float, float]:
        """Estimate random-walk-with-drift parameters for kappa1_t, kappa2_t, kappa3_t.

        Returns:
            Tuple of (mu1, sigma1, mu2, sigma2, mu3, sigma3).
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        self.params = estimate_rw_params_m7(self.params)
        mu1 = cast(float, self.params.mu1)
        sigma1 = cast(float, self.params.sigma1)
        mu2 = cast(float, self.params.mu2)
        sigma2 = cast(float, self.params.sigma2)
        mu3 = cast(float, self.params.mu3)
        sigma3 = cast(float, self.params.sigma3)
        return (mu1, sigma1, mu2, sigma2, mu3, sigma3)

    def predict_logit_q(self) -> FloatArray:
        """Reconstruct the logit mortality surface from fitted parameters.

        Returns:
            Logit mortality surface with shape (A, T).
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        return reconstruct_logit_q_m7(self.params)

    def predict_q(self) -> FloatArray:
        """Reconstruct mortality probabilities from fitted parameters.

        Returns:
            Mortality probabilities with shape (A, T).
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        return reconstruct_q_m7(self.params)

    def simulate_kappa(
        self,
        kappa_index: str,
        horizon: int,
        n_sims: int = 1000,
        seed: int | None = None,
        include_last: bool = False,
    ) -> FloatArray:
        """Simulate random-walk paths for kappa1_t, kappa2_t, or kappa3_t.

        Args:
            kappa_index: "kappa1", "kappa2", or "kappa3".
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
            mu = self.params.mu1
            sigma = self.params.sigma1
        elif kappa_index == "kappa2":
            k_last = float(self.params.kappa2[-1])
            mu = self.params.mu2
            sigma = self.params.sigma2
        elif kappa_index == "kappa3":
            k_last = float(self.params.kappa3[-1])
            mu = self.params.mu3
            sigma = self.params.sigma3
        else:
            raise ValueError("kappa_index must be 'kappa1', 'kappa2' or 'kappa3'.")

        if mu is None or sigma is None:
            raise ValueError("Call estimate_rw() before simulate_kappa().")

        horizon = int(horizon)
        n_sims = int(n_sims)
        if horizon <= 0 or n_sims <= 0:
            raise ValueError("horizon and n_sims must be positive integers.")

        rng = np.random.default_rng(seed)

        from pymort.analysis.projections import simulate_random_walk_paths

        return simulate_random_walk_paths(
            k_last=k_last,
            mu=float(mu),
            sigma=float(sigma),
            horizon=horizon,
            n_sims=n_sims,
            rng=rng,
            include_last=include_last,
        )
