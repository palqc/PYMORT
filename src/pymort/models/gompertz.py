"""Gompertz tail fitting helpers for mortality surfaces.

This module fits a per-year Gompertz curve to central death rates and can
extrapolate mortality beyond the observed age range.

Note:
    Docstrings follow Google style and type hints use NDArray for clarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


@dataclass
class GompertzFitResult:
    """Result of a per-year Gompertz fit on central death rates.

    The model is:
        m_{x,t} = exp(a_t + b_t * x).
    """

    ages: FloatArray  # (A,)
    years: IntArray  # (T,)
    a_t: FloatArray  # (T,)
    b_t: FloatArray  # (T,)
    age_fit_min: int
    age_fit_max: int
    m_floor: float
    m_fitted: FloatArray | None = None  # (A, T) if computed
    meta: dict[str, object] | None = None


def _safe_log_m(m: FloatArray, m_floor: float) -> FloatArray:
    """Compute log(m) after clipping to a positive floor.

    Args:
        m: Central death rates. Shape (...,).
        m_floor: Minimum allowed value before taking the log.

    Returns:
        Log of m with values clipped at m_floor.
    """
    m = np.asarray(m, dtype=float)
    m = np.clip(m, float(m_floor), None)
    return np.log(m)


def fit_gompertz_per_year(
    ages: FloatArray,
    years: IntArray,
    m: FloatArray,
    *,
    age_fit_min: int = 80,
    age_fit_max: int = 100,
    m_floor: float = 1e-12,
    compute_fitted_surface: bool = True,
) -> GompertzFitResult:
    """Fit a Gompertz model per calendar year using OLS on log(m).

    Args:
        ages: Age grid. Shape (A,).
        years: Year grid. Shape (T,).
        m: Central death rates. Shape (A, T).
        age_fit_min: Minimum age for the fit window (inclusive).
        age_fit_max: Maximum age for the fit window (inclusive).
        m_floor: Floor applied to m before taking logs.
        compute_fitted_surface: Whether to return the fitted surface on the full grid.

    Returns:
        GompertzFitResult with per-year parameters and optional fitted surface.

    Raises:
        ValueError: If inputs are invalid or fit window is empty.
    """
    ages = np.asarray(ages, dtype=float).reshape(-1)
    years = np.asarray(years, dtype=int).reshape(-1)
    m = np.asarray(m, dtype=float)

    if m.ndim != 2:
        raise ValueError(f"m must be 2D (A,T), got {m.shape}.")
    A, T = m.shape
    if ages.shape[0] != A:
        raise ValueError(f"ages length {ages.shape[0]} != A={A}.")
    if years.shape[0] != T:
        raise ValueError(f"years length {years.shape[0]} != T={T}.")

    # fit window mask
    a_min = int(age_fit_min)
    a_max = int(age_fit_max)
    if a_min >= a_max:
        raise ValueError("age_fit_min must be < age_fit_max.")
    mask = (ages >= a_min) & (ages <= a_max)
    if not np.any(mask):
        raise ValueError(
            f"No ages in fit window [{a_min},{a_max}] within ages range "
            f"[{int(ages.min())},{int(ages.max())}]."
        )

    x = ages[mask]  # (Af,)
    Af = x.shape[0]
    # Design matrix [1, x]
    X = np.column_stack([np.ones(Af, dtype=float), x])  # (Af,2)

    logm = _safe_log_m(m[mask, :], m_floor=m_floor)  # (Af,T)

    a_t = np.empty(T, dtype=float)
    b_t = np.empty(T, dtype=float)

    # OLS per year
    # beta_hat = (X'X)^{-1} X' y
    XtX_inv = np.linalg.inv(X.T @ X)  # (2,2)
    Xt = X.T  # (2,Af)

    for t in range(T):
        y = logm[:, t]  # (Af,)
        beta = XtX_inv @ (Xt @ y)  # (2,)
        a_t[t] = float(beta[0])
        b_t[t] = float(beta[1])

    m_fitted = None
    if compute_fitted_surface:
        # mhat[a,t] = exp(a_t + b_t * age_a)
        m_fitted = np.exp(a_t[None, :] + b_t[None, :] * ages[:, None])

    return GompertzFitResult(
        ages=ages,
        years=years,
        a_t=a_t,
        b_t=b_t,
        age_fit_min=a_min,
        age_fit_max=a_max,
        m_floor=float(m_floor),
        m_fitted=m_fitted,
        meta={
            "model": "gompertz_per_year",
            "fit_window": [a_min, a_max],
            "note": "Fitted per-year on log(m) via OLS.",
        },
    )


def extrapolate_gompertz_surface(
    fit: GompertzFitResult,
    *,
    age_max: int,
    age_min: int | None = None,
) -> tuple[FloatArray, IntArray, FloatArray]:
    """Extrapolate the fitted Gompertz surface to a new age grid.

    Args:
        fit: Fitted Gompertz parameters.
        age_max: Maximum age for the extended grid.
        age_min: Minimum age for the extended grid. Defaults to fit.ages.min().

    Returns:
        Tuple (ages_ext, years, m_ext) where m_ext is Gompertz-implied.
    """
    years = np.asarray(fit.years, dtype=int)
    a_t = np.asarray(fit.a_t, dtype=float)
    b_t = np.asarray(fit.b_t, dtype=float)

    if age_min is None:
        age_min = int(fit.ages.min())
    age_min = int(age_min)
    age_max = int(age_max)
    if age_max <= age_min:
        raise ValueError("age_max must be > age_min.")

    ages_ext = np.arange(age_min, age_max + 1, dtype=float)
    m_ext = np.exp(a_t[None, :] + b_t[None, :] * ages_ext[:, None])
    return ages_ext, years, m_ext
