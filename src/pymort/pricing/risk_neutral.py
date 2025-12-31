"""Risk-neutral calibration and scenario generation utilities.

Note:
    Docstrings follow Google style and type hints use NDArray for clarity.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy.optimize import least_squares

from pymort._types import AnyArray, FloatArray
from pymort.analysis import BootstrapResult, MortalityScenarioSet, bootstrap_from_m
from pymort.analysis.projections import project_mortality_from_bootstrap
from pymort.lifetables import m_to_q, survival_from_q, validate_q
from pymort.models.cbd_m7 import CBDM7
from pymort.models.lc_m2 import LCM2
from pymort.pricing.liabilities import CohortLifeAnnuitySpec, price_cohort_life_annuity
from pymort.pricing.longevity_bonds import (
    LongevityBondSpec,
    price_simple_longevity_bond,
)
from pymort.pricing.mortality_derivatives import (
    QForwardSpec,
    SForwardSpec,
    price_q_forward,
    price_s_forward,
)
from pymort.pricing.survivor_swaps import SurvivorSwapSpec, price_survivor_swap

Lambda = float | Sequence[float] | FloatArray
Shock = Sequence[float] | FloatArray


# ============================================================================
# 1) Esscher transform on Gaussian random walk factors
# ============================================================================


@dataclass(frozen=True)
class EsscherResult:
    """Container for Esscher-transformed random-walk parameters.

    Attributes:
        mu_P: Drift under P, shape (K,).
        sigma_P: Volatility under P, shape (K,).
        mu_Q: Drift under Q, shape (K,).
        lambda_esscher: Esscher tilt, shape (K,).
    """

    mu_P: FloatArray
    sigma_P: FloatArray
    mu_Q: FloatArray
    lambda_esscher: FloatArray


@dataclass(frozen=True)
class CalibrationCache:
    """Objects and common random numbers reused across lambda calls."""

    model: LCM2 | CBDM7
    bs_res: BootstrapResult
    eps: dict[str, FloatArray]
    ages: AnyArray
    years: AnyArray
    m: FloatArray
    model_name: str
    horizon: int
    n_process: int
    include_last: bool


def _to_1d_array(x: Iterable[float] | float, name: str) -> FloatArray:
    arr = np.atleast_1d(np.asarray(x, dtype=float)).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    return arr


def esscher_shift_normal_rw(
    mu_P: Iterable[float] | float,
    sigma_P: Iterable[float] | float,
    lambda_esscher: Iterable[float] | float,
) -> EsscherResult:
    """Apply a normal Esscher tilt to random-walk increments.

    Uses:
        mu_Q = mu_P + lambda * sigma^2
        sigma_Q = sigma_P

    Args:
        mu_P: Drift under P (scalar or vector).
        sigma_P: Volatility under P (scalar or vector).
        lambda_esscher: Esscher tilt (scalar or vector).

    Returns:
        EsscherResult with drift/volatility under P and Q.
    """
    mu_P_arr = _to_1d_array(mu_P, "mu_P")
    sigma_P_arr = _to_1d_array(sigma_P, "sigma_P")
    lambda_arr = _to_1d_array(lambda_esscher, "lambda_esscher")

    k = mu_P_arr.shape[0]
    if sigma_P_arr.shape[0] != k:
        raise ValueError(
            f"sigma_P must have same length as mu_P; got {sigma_P_arr.shape[0]} vs {k}."
        )

    if lambda_arr.shape[0] == 1 and k > 1:
        lambda_arr = np.full(k, float(lambda_arr[0]), dtype=float)
    elif lambda_arr.shape[0] != k:
        raise ValueError(f"lambda_esscher must have length 1 or {k}; got {lambda_arr.shape[0]}.")

    if np.any(sigma_P_arr < 0.0):
        raise ValueError("sigma_P entries must be non-negative.")

    mu_Q_arr = mu_P_arr + lambda_arr * (sigma_P_arr**2)

    return EsscherResult(
        mu_P=mu_P_arr,
        sigma_P=sigma_P_arr,
        mu_Q=mu_Q_arr,
        lambda_esscher=lambda_arr,
    )


# ============================================================================
# 2) Model-specific helpers: LCM2 and CBDM7
# ============================================================================


def risk_neutral_from_lcm2(model_lcm2: LCM2, lambda_esscher: Lambda) -> EsscherResult:
    mu_P, sigma_P = model_lcm2.estimate_rw()
    return esscher_shift_normal_rw(mu_P=mu_P, sigma_P=sigma_P, lambda_esscher=lambda_esscher)


def risk_neutral_from_cbdm7(model_cbdm7: CBDM7, lambda_esscher: Lambda) -> EsscherResult:
    rw = model_cbdm7.estimate_rw()
    if len(rw) != 6:
        raise RuntimeError(
            "CBDM7.estimate_rw() must return 6 values (mu1, sigma1, mu2, sigma2, mu3, sigma3)."
        )
    mu1, sigma1, mu2, sigma2, mu3, sigma3 = rw
    mu_P = np.array([mu1, mu2, mu3], dtype=float)
    sigma_P = np.array([sigma1, sigma2, sigma3], dtype=float)
    return esscher_shift_normal_rw(mu_P=mu_P, sigma_P=sigma_P, lambda_esscher=lambda_esscher)


# ============================================================================
# 3) Cache building (fit + bootstrap + common random numbers once)
# ============================================================================


def build_calibration_cache(
    *,
    ages: AnyArray,
    years: AnyArray,
    m: FloatArray,
    model_name: str,
    B_bootstrap: int,
    n_process: int,
    horizon: int,
    seed: int | None,
    include_last: bool = False,
) -> CalibrationCache:
    model_name_up = str(model_name).upper()

    model: LCM2 | CBDM7
    if model_name_up == "LCM2":
        model = LCM2().fit(m, ages, years)
    elif model_name_up == "CBDM7":
        q = m_to_q(m)
        model = CBDM7().fit(q, ages, years)
    else:
        raise ValueError("model_name must be 'LCM2' or 'CBDM7'.")

    bs_res = bootstrap_from_m(
        model.__class__,
        m,
        ages,
        years,
        B=B_bootstrap,
        seed=seed,
        resample="year_block",
    )

    b = len(bs_res.params_list)
    h = int(horizon)
    rng = np.random.default_rng(seed)

    eps: dict[str, FloatArray]
    if model_name_up == "LCM2":
        eps = {"eps_rw": cast(FloatArray, rng.normal(size=(b, n_process, h)))}
    else:
        eps = {
            "eps1": cast(FloatArray, rng.normal(size=(b, n_process, h))),
            "eps2": cast(FloatArray, rng.normal(size=(b, n_process, h))),
            "eps3": cast(FloatArray, rng.normal(size=(b, n_process, h))),
        }

    return CalibrationCache(
        model=model,
        bs_res=bs_res,
        eps=eps,
        ages=np.asarray(ages),
        years=np.asarray(years),
        m=np.asarray(m),
        model_name=model_name_up,
        horizon=h,
        n_process=int(n_process),
        include_last=bool(include_last),
    )


# ============================================================================
# 4) Optional shocks
# ============================================================================


def apply_kappa_drift_shock(
    mu_Q: FloatArray,
    *,
    shock: Iterable[float] | None = None,
    mode: str = "additive",
) -> FloatArray:
    mu_Q_arr = np.asarray(mu_Q, dtype=float).reshape(-1)
    k = mu_Q_arr.shape[0]

    if shock is None:
        return mu_Q_arr

    shock_arr = np.asarray(cast(Sequence[float] | FloatArray | float, shock), dtype=float)
    s = shock_arr.reshape(-1)
    if s.size == 1 and k > 1:
        s = np.full(k, float(s[0]), dtype=float)
    if s.size != k:
        raise ValueError(f"drift shock must have length 1 or {k}, got {s.size}.")

    mode_l = mode.lower()
    if mode_l == "additive":
        out = mu_Q_arr + s
    elif mode_l == "multiplicative":
        out = mu_Q_arr * (s + 1.0)
    else:
        raise ValueError("mode must be 'additive' or 'multiplicative'.")

    out_arr = np.asarray(out, dtype=float)
    if not np.all(np.isfinite(out_arr)):
        raise ValueError("Non-finite mu_Q after drift shock.")
    return out_arr


def apply_cohort_trend_shock_to_qpaths(
    q_paths: FloatArray,
    ages: AnyArray,
    years: AnyArray,
    *,
    shock_type: str = "cohort_improvement",
    magnitude: float = 0.01,
    pivot_cohort_year: int | None = None,
    cap: float = 0.50,
) -> FloatArray:
    q = np.asarray(q_paths, dtype=float)
    ages_arr = np.asarray(ages, dtype=float).reshape(-1)
    years_arr = np.asarray(years, dtype=int).reshape(-1)

    if q.ndim != 3:
        raise ValueError("q_paths must have shape (N, A, H).")
    _n, a, h = q.shape
    if ages_arr.shape[0] != a:
        raise ValueError("ages length must match q_paths second dimension.")
    if years_arr.shape[0] != h:
        raise ValueError("years length must match q_paths third dimension.")

    cohort_year = years_arr[None, :] - ages_arr[:, None]  # (A, H)

    if pivot_cohort_year is None:
        pivot_cohort_year = int(np.median(cohort_year))

    cohort_offset = cohort_year - float(pivot_cohort_year)
    cohort_offset = np.clip(cohort_offset, -100.0, 100.0)

    eps = float(magnitude)
    st = shock_type.lower()

    if st in {"cohort_improvement", "cohort_deterioration"}:
        sign = -1.0 if st == "cohort_improvement" else 1.0
        scaling = np.exp(sign * eps * cohort_offset)
        scaling = np.clip(scaling, 1.0 - cap, 1.0 + cap)
        q_new = q * scaling[None, :, :]
    elif st == "cohort_step":
        step = np.where(cohort_offset > 0, -eps, eps)
        step = np.clip(step, -cap, cap)
        q_new = q * (1.0 + step[None, :, :])
    else:
        raise ValueError(f"Unknown cohort shock_type='{shock_type}'.")

    q_new_arr = np.asarray(q_new, dtype=float)
    validate_q(q_new_arr)
    return q_new_arr


# ============================================================================
# 5) Fast scenario builder (depends on lambda only)
# ============================================================================


def build_scenarios_under_lambda_fast(
    cache: CalibrationCache,
    lambda_esscher: Lambda,
    *,
    scale_sigma: float = 1.0,
    kappa_drift_shock: Shock | None = None,
    kappa_drift_shock_mode: str = "additive",
    cohort_shock_type: str | None = None,
    cohort_shock_magnitude: float = 0.01,
    cohort_pivot_year: int | None = None,
) -> MortalityScenarioSet:
    if cache.model_name == "LCM2":
        esscher = risk_neutral_from_lcm2(cast(LCM2, cache.model), lambda_esscher=lambda_esscher)
    elif cache.model_name == "CBDM7":
        esscher = risk_neutral_from_cbdm7(cast(CBDM7, cache.model), lambda_esscher=lambda_esscher)
    else:
        raise ValueError("cache.model_name must be 'LCM2' or 'CBDM7'.")

    mu_Q = apply_kappa_drift_shock(
        mu_Q=esscher.mu_Q,
        shock=kappa_drift_shock,
        mode=kappa_drift_shock_mode,
    )

    proj_Q = project_mortality_from_bootstrap(
        model_cls=cache.model.__class__,
        ages=cache.ages,
        years=cache.years,
        m=cache.m,
        bootstrap_result=cache.bs_res,
        horizon=cache.horizon,
        n_process=cache.n_process,
        seed=None,  # IMPORTANT: do not re-seed here (CRN are injected)
        include_last=cache.include_last,
        drift_overrides=mu_Q,
        scale_sigma=scale_sigma,
        eps_rw=cache.eps.get("eps_rw"),
        eps1=cache.eps.get("eps1"),
        eps2=cache.eps.get("eps2"),
        eps3=cache.eps.get("eps3"),
    )

    q_paths_Q = proj_Q.q_paths
    years_Q = proj_Q.years

    if cohort_shock_type is not None:
        q_paths_Q = apply_cohort_trend_shock_to_qpaths(
            q_paths_Q,
            ages=cache.ages,
            years=years_Q,
            shock_type=cohort_shock_type,
            magnitude=cohort_shock_magnitude,
            pivot_cohort_year=cohort_pivot_year,
        )

    S_paths_Q = survival_from_q(q_paths_Q)

    return MortalityScenarioSet(
        years=years_Q,
        ages=cache.ages,
        q_paths=q_paths_Q,
        S_paths=S_paths_Q,
        discount_factors=None,
        metadata={"measure": "Q", "model": cache.model_name},
    )


# ============================================================================
# 6) Pricing helper (given scen_set)
# ============================================================================


def _price_from_scen_set(
    scen_set: MortalityScenarioSet,
    quote: MultiInstrumentQuote,
    *,
    short_rate: float,
) -> float:
    kind = quote.kind.lower()

    if kind == "longevity_bond":
        return float(
            price_simple_longevity_bond(
                scen_set=scen_set,
                spec=cast(LongevityBondSpec, quote.spec),
                short_rate=short_rate,
            )["price"]
        )

    if kind == "survivor_swap":
        return float(
            price_survivor_swap(
                scen_set=scen_set,
                spec=cast(SurvivorSwapSpec, quote.spec),
                short_rate=short_rate,
            )["price"]
        )

    if kind == "s_forward":
        return float(
            price_s_forward(
                scen_set=scen_set,
                spec=cast(SForwardSpec, quote.spec),
                short_rate=short_rate,
            )["price"]
        )

    if kind == "q_forward":
        return float(
            price_q_forward(
                scen_set=scen_set,
                spec=cast(QForwardSpec, quote.spec),
                short_rate=short_rate,
            )["price"]
        )

    if kind == "life_annuity":
        return float(
            price_cohort_life_annuity(
                scen_set=scen_set,
                spec=cast(CohortLifeAnnuitySpec, quote.spec),
                short_rate=short_rate,
            )["price"]
        )

    raise ValueError(
        "Unknown quote.kind. Expected one of "
        "['longevity_bond','survivor_swap','s_forward','q_forward','life_annuity']."
    )


# ============================================================================
# 7) Multi-instrument calibration of lambda via SciPy least_squares
# ============================================================================


@dataclass(frozen=True)
class MultiInstrumentQuote:
    """Market quote used to calibrate lambda.

    Attributes:
        kind: Instrument kind (e.g., "longevity_bond", "survivor_swap").
        spec: Instrument specification object.
        market_price: Observed market PV.
        weight: Objective weight.
    """

    kind: str
    spec: LongevityBondSpec | SurvivorSwapSpec | SForwardSpec | QForwardSpec | CohortLifeAnnuitySpec
    market_price: float
    weight: float = 1.0


def _infer_lambda_dim(model_name: str) -> int:
    name = str(model_name).upper()
    if name == "LCM2":
        return 1
    if name == "CBDM7":
        return 3
    raise ValueError("Unsupported model_name. Use 'LCM2' or 'CBDM7'.")


def _to_lambda_vec(lambda_esscher: Lambda, k: int) -> FloatArray:
    lam = np.atleast_1d(np.asarray(lambda_esscher, dtype=float)).reshape(-1)
    if lam.size == 1 and k > 1:
        lam = np.full(k, float(lam[0]), dtype=float)
    if lam.size != k:
        raise ValueError(f"lambda_esscher must have length 1 or {k}, got {lam.size}.")
    if not np.all(np.isfinite(lam)):
        raise ValueError("lambda_esscher must be finite.")
    return lam


def calibrate_lambda_least_squares(
    quotes: Iterable[MultiInstrumentQuote],
    ages: AnyArray,
    years: AnyArray,
    m: FloatArray,
    *,
    model_name: str = "CBDM7",
    lambda0: Lambda = 0.0,
    bounds: tuple[float, float] = (-5.0, 5.0),
    B_bootstrap: int = 50,
    n_process: int = 200,
    short_rate: float = 0.02,
    horizon: int | None = None,
    seed: int | None = None,
    scale_sigma: float = 1.0,
    include_last: bool = False,
    verbose: int = 0,
    # optional shocks forwarded into build_scenarios_under_lambda_fast
    kappa_drift_shock: Shock | None = None,
    kappa_drift_shock_mode: str = "additive",
    cohort_shock_type: str | None = None,
    cohort_shock_magnitude: float = 0.01,
    cohort_pivot_year: int | None = None,
) -> dict[str, Any]:
    quotes_list = list(quotes)
    if not quotes_list:
        raise ValueError("quotes must be a non-empty iterable.")

    k = _infer_lambda_dim(model_name)
    lam0 = _to_lambda_vec(lambda0, k)
    if np.allclose(lam0, 0.0):
        lam0 = lam0 + 1e-3

    lo, hi = float(bounds[0]), float(bounds[1])
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError("bounds must be finite with bounds[0] < bounds[1].")

    lb = np.full(k, lo, dtype=float)
    ub = np.full(k, hi, dtype=float)

    weights = np.array([float(q.weight) for q in quotes_list], dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("All quote weights must be finite and >= 0.")
    if np.all(weights == 0.0):
        raise ValueError("At least one quote must have a strictly positive weight.")

    market = np.array([float(q.market_price) for q in quotes_list], dtype=float)
    if np.any(~np.isfinite(market)):
        raise ValueError("All market_price values must be finite.")

    w_sqrt = np.sqrt(weights)
    h_eff = int(horizon) if horizon is not None else len(years)

    cache = build_calibration_cache(
        ages=ages,
        years=years,
        m=m,
        model_name=model_name,
        B_bootstrap=B_bootstrap,
        n_process=n_process,
        horizon=h_eff,
        seed=seed,
        include_last=include_last,
    )

    def residual_vec(lam_vec: FloatArray) -> FloatArray:
        scen_set_q = build_scenarios_under_lambda_fast(
            cache=cache,
            lambda_esscher=lam_vec,
            scale_sigma=scale_sigma,
            kappa_drift_shock=kappa_drift_shock,
            kappa_drift_shock_mode=kappa_drift_shock_mode,
            cohort_shock_type=cohort_shock_type,
            cohort_shock_magnitude=cohort_shock_magnitude,
            cohort_pivot_year=cohort_pivot_year,
        )
        model_prices = np.array(
            [_price_from_scen_set(scen_set_q, q, short_rate=short_rate) for q in quotes_list],
            dtype=float,
        )
        return cast(FloatArray, (model_prices - market) * w_sqrt)

    res = least_squares(
        residual_vec,
        x0=lam0,
        bounds=(lb, ub),
        verbose=int(verbose),
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=200,
    )

    lam_star = np.asarray(res.x, dtype=float).reshape(-1)

    scen_set_q_star = build_scenarios_under_lambda_fast(
        cache=cache,
        lambda_esscher=lam_star,
        scale_sigma=scale_sigma,
        kappa_drift_shock=kappa_drift_shock,
        kappa_drift_shock_mode=kappa_drift_shock_mode,
        cohort_shock_type=cohort_shock_type,
        cohort_shock_magnitude=cohort_shock_magnitude,
        cohort_pivot_year=cohort_pivot_year,
    )

    fitted = np.array(
        [_price_from_scen_set(scen_set_q_star, q, short_rate=short_rate) for q in quotes_list],
        dtype=float,
    )

    return {
        "lambda_star": lam_star,
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "cost": float(res.cost),
        "fun": np.asarray(res.fun, dtype=float),
        "jac": getattr(res, "jac", None),
        "nfev": int(res.nfev),
        "fitted_prices": fitted,
        "market_prices": market,
        "residuals_unweighted": fitted - market,
        "quotes": quotes_list,
        "weights": weights,
        "bounds": (lb, ub),
        "lambda0": lam0,
        "cache": cache,
    }


def calibrate_market_price_of_longevity_risk(
    quotes: Iterable[MultiInstrumentQuote],
    ages: AnyArray,
    years: AnyArray,
    m: FloatArray,
    *,
    model_name: str,
    lambda0: Lambda = 0.0,
    bounds: tuple[float, float] = (-5.0, 5.0),
    B_bootstrap: int = 50,
    n_process: int = 200,
    short_rate: float = 0.02,
    horizon: int | None = None,
    seed: int | None = None,
    scale_sigma: float = 1.0,
    include_last: bool = False,
    verbose: int = 0,
    kappa_drift_shock: Shock | None = None,
    kappa_drift_shock_mode: str = "additive",
    cohort_shock_type: str | None = None,
    cohort_shock_magnitude: float = 0.01,
    cohort_pivot_year: int | None = None,
) -> FloatArray:
    res = calibrate_lambda_least_squares(
        quotes=quotes,
        ages=ages,
        years=years,
        m=m,
        model_name=model_name,
        lambda0=lambda0,
        bounds=bounds,
        B_bootstrap=B_bootstrap,
        n_process=n_process,
        short_rate=short_rate,
        horizon=horizon,
        seed=seed,
        scale_sigma=scale_sigma,
        include_last=include_last,
        verbose=verbose,
        kappa_drift_shock=kappa_drift_shock,
        kappa_drift_shock_mode=kappa_drift_shock_mode,
        cohort_shock_type=cohort_shock_type,
        cohort_shock_magnitude=cohort_shock_magnitude,
        cohort_pivot_year=cohort_pivot_year,
    )
    return np.asarray(res["lambda_star"], dtype=float)
