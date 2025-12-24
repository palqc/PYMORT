"""
PYMORT pipelines
===========================

This module wires together the modelling, projection, risk-neutral, pricing,
sensitivities, hedging, and reporting layers into declarative end-to-end
workflows. It does not re-implement maths; it simply orchestrates existing
building blocks:

- Mortality Models / Smoothing / Backtests: fitting.py, smoothing.py
- Stochastic Projections: bootstrap.py, projections.py
- Risk-Neutral Valuation: pricing.risk_neutral
- Pricing: pricing.* products
- Sensitivities: analysis.sensivities
- Hedging: pricing.hedging
- Reporting: analysis.reporting
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Sequence

import numpy as np

from pymort.analysis import (
    MortalityScenarioSet,
    bootstrap_from_m,
    build_scenario_set_from_projection,
)
from pymort.analysis.fitting import (
    FittedModel,
    ModelName,
    select_and_fit_best_model_for_pricing,
)
from pymort.analysis.projections import (
    ProjectionResult,
    project_mortality_from_bootstrap,
)
from pymort.analysis.reporting import RiskReport, generate_risk_report
from pymort.analysis.scenario_analysis import ShockSpec, generate_stressed_scenarios
from pymort.analysis.sensitivities import (
    AllSensitivities,
    compute_all_sensitivities,
    make_single_product_pricer,
    price_all_products,
)
from pymort.interest_rates.hull_white import (
    InterestRateScenarioSet,
    build_interest_rate_scenarios,
)
from pymort.pricing.hedging import (
    GreekHedgeResult,
    HedgeResult,
    compute_duration_convexity_matching_hedge,
    compute_duration_matching_hedge,
    compute_greek_matching_hedge,
    compute_min_variance_hedge,
    compute_min_variance_hedge_constrained,
    compute_multihorizon_hedge,
)
from pymort.pricing.liabilities import CohortLifeAnnuitySpec
from pymort.pricing.longevity_bonds import LongevityBondSpec
from pymort.pricing.mortality_derivatives import QForwardSpec, SForwardSpec
from pymort.pricing.risk_neutral import (
    MultiInstrumentQuote,
    build_calibration_cache,
    build_scenarios_under_lambda_fast,
    calibrate_lambda_least_squares,
)
from pymort.pricing.survivor_swaps import SurvivorSwapSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_bootstrap_params(
    n_scenarios: int, bootstrap_kwargs: Optional[dict]
) -> tuple[int, int, str]:
    """
    Choose (B_bootstrap, n_process, resample) given a target number of scenarios.
    """
    if bootstrap_kwargs is None:
        bootstrap_kwargs = {}
    B = int(
        bootstrap_kwargs.get(
            "B", bootstrap_kwargs.get("B_bootstrap", max(1, n_scenarios))
        )
    )
    n_process = int(bootstrap_kwargs.get("n_process", max(1, n_scenarios // max(1, B))))
    resample = str(bootstrap_kwargs.get("resample", "year_block"))
    return B, n_process, resample


def _infer_kind(spec: object) -> str:
    if isinstance(spec, LongevityBondSpec):
        return "longevity_bond"
    # allow short alias used in some fixtures
    if isinstance(spec, dict) and str(spec.get("kind", "")).lower() == "bond":
        return "longevity_bond"
    if isinstance(spec, SurvivorSwapSpec):
        return "survivor_swap"
    if isinstance(spec, SForwardSpec):
        return "s_forward"
    if isinstance(spec, QForwardSpec):
        return "q_forward"
    if isinstance(spec, CohortLifeAnnuitySpec):
        return "life_annuity"
    if isinstance(spec, dict) and "kind" in spec:
        return str(spec["kind"])
    raise ValueError(
        "Cannot infer instrument kind; provide a spec dataclass or dict with 'kind'."
    )


def _normalize_spec(spec: object) -> object:
    """
    Accepts either already-instantiated specs or dict {"kind": ..., "spec": {...}}.
    """
    if isinstance(
        spec,
        (
            LongevityBondSpec,
            SurvivorSwapSpec,
            SForwardSpec,
            QForwardSpec,
            CohortLifeAnnuitySpec,
        ),
    ):
        return spec
    if isinstance(spec, dict):
        if "spec" in spec and "kind" in spec:
            kind = str(spec["kind"]).lower()
            data = spec["spec"]
        else:
            # assume dict maps directly to dataclass fields for the inferred kind
            kind = str(spec.get("kind", ""))
            data = {k: v for k, v in spec.items() if k != "kind"}
        if kind == "bond":
            kind = "longevity_bond"
        if kind == "longevity_bond":
            return LongevityBondSpec(**data)
        if kind == "survivor_swap":
            return SurvivorSwapSpec(**data)
        if kind == "s_forward":
            return SForwardSpec(**data)
        if kind == "q_forward":
            return QForwardSpec(**data)
        if kind == "life_annuity":
            return CohortLifeAnnuitySpec(**data)
    raise ValueError(
        "Unsupported spec format; expected dataclass or {'kind','spec'} dict."
    )


def _build_multi_instrument_quotes(
    instruments: Mapping[str, object],
    market_prices: Mapping[str, float],
) -> list[MultiInstrumentQuote]:
    quotes: list[MultiInstrumentQuote] = []
    for name, spec_obj in instruments.items():
        spec = _normalize_spec(spec_obj)
        kind = _infer_kind(spec)
        if name not in market_prices:
            raise ValueError(f"Missing market price for instrument '{name}'.")
        weight = 1.0
        if isinstance(spec_obj, dict) and "weight" in spec_obj:
            weight = float(spec_obj.get("weight", 1.0))
        quotes.append(
            MultiInstrumentQuote(
                kind=kind,
                spec=spec,  # type: ignore[arg-type]
                market_price=float(market_prices[name]),
                weight=weight,
            )
        )
    return quotes


def _calibration_summary(lam_res: dict[str, Any]) -> dict[str, Any]:
    """
    Build a structured calibration summary from calibrate_lambda_least_squares output.
    """
    prices_model = np.asarray(lam_res.get("fitted_prices", []), dtype=float)
    prices_market = np.asarray(lam_res.get("market_prices", []), dtype=float)
    errors = prices_model - prices_market
    obj = float(lam_res.get("cost", float(np.sum(errors**2))))
    rmse = float(np.sqrt(np.mean(errors**2))) if errors.size > 0 else float("nan")
    quotes = lam_res.get("quotes")
    instr_names = []
    if quotes is not None:
        try:
            instr_names = [getattr(q, "kind", f"q{i}") for i, q in enumerate(quotes)]
        except Exception:
            instr_names = []
    residual_table = []
    if instr_names and prices_model.size == prices_market.size:
        for name, pm, pobs in zip(
            instr_names, prices_model.tolist(), prices_market.tolist()
        ):
            residual_table.append(
                {
                    "instrument": name,
                    "model_price": pm,
                    "market_price": pobs,
                    "error": pm - pobs,
                }
            )
    return {
        "lambda_star": np.asarray(lam_res.get("lambda_star", []), dtype=float).tolist(),
        "objective_value": obj,
        "rmse_pricing_error": rmse,
        "prices_model": prices_model.tolist(),
        "prices_market": prices_market.tolist(),
        "pricing_errors": errors.tolist(),
        "instruments": instr_names,
        "residuals": residual_table,
        "success": bool(lam_res.get("success", True)),
        "n_iter": int(lam_res.get("nfev", 0)),
        "status": int(lam_res.get("status", 0)),
        "message": lam_res.get("message", ""),
    }


# ---------------------------------------------------------------------------
# Existing projection helpers retained for backwards compatibility
# ---------------------------------------------------------------------------


def build_mortality_scenarios_for_pricing(
    ages: np.ndarray,
    years: np.ndarray,
    m: np.ndarray,
    *,
    train_end: int,
    model_names: Iterable[ModelName] = (
        "LCM1",
        "LCM2",
        "APCM3",
        "CBDM5",
        "CBDM6",
        "CBDM7",
    ),
    selection_metric: Literal["log_m", "logit_q"] = "logit_q",
    cpsplines_kwargs: Optional[Dict[str, Any]] = None,
    B_bootstrap: int = 1000,
    horizon: int = 50,
    n_process: int = 200,
    seed: Optional[int] = None,
    include_last: bool = True,
) -> tuple[FittedModel, ProjectionResult, MortalityScenarioSet]:
    """
    Legacy end-to-end fit + projection pipeline used by earlier CLI code.
    """
    ages = np.asarray(ages, dtype=float)
    years = np.asarray(years, dtype=int)
    m = np.asarray(m, dtype=float)

    _selected_df, fitted_best = select_and_fit_best_model_for_pricing(
        ages=ages,
        years=years,
        m=m,
        train_end=train_end,
        model_names=model_names,
        metric=selection_metric,
        cpsplines_kwargs=cpsplines_kwargs,
    )

    if fitted_best.m_fit_surface is None:
        raise RuntimeError(
            "FittedModel.m_fit_surface is None; expected CPsplines-smoothed m."
        )
    m_smooth = fitted_best.m_fit_surface
    model_cls = type(fitted_best.model)

    bs_res = bootstrap_from_m(
        model_cls,
        m_smooth,
        ages,
        years,
        B=B_bootstrap,
        seed=seed,
        resample="year_block",
    )

    proj = project_mortality_from_bootstrap(
        model_cls=model_cls,
        ages=ages,
        years=years,
        m=m_smooth,
        bootstrap_result=bs_res,
        horizon=horizon,
        n_process=n_process,
        seed=None if seed is None else seed + 123,
        include_last=include_last,
    )

    metadata: Dict[str, Any] = {
        "selected_model": fitted_best.name,
        "selection_metric": selection_metric,
        "train_end": int(train_end),
        "B_bootstrap": int(B_bootstrap),
        "n_process": int(n_process),
        "N_scenarios": int(proj.q_paths.shape[0]),
        "projection_horizon": int(horizon),
        "include_last": bool(include_last),
        "model_class": model_cls.__name__,
        "data_source": fitted_best.metadata.get("data_source"),
        "smoothing": fitted_best.metadata.get("smoothing"),
    }

    scen_set = build_scenario_set_from_projection(
        proj=proj,
        ages=ages,
        discount_factors=None,
        metadata=metadata,
    )

    return fitted_best, proj, scen_set


def project_from_fitted_model(
    fitted: FittedModel,
    *,
    B_bootstrap: int = 1000,
    horizon: int = 50,
    n_process: int = 200,
    seed: Optional[int] = None,
    include_last: bool = True,
    resample: Literal["cell", "year_block"] = "year_block",
) -> tuple[ProjectionResult, MortalityScenarioSet]:
    """
    Bootstrap + stochastic projection starting from an existing fitted model.
    """
    ages = np.asarray(fitted.ages, dtype=float)
    years = np.asarray(fitted.years, dtype=int)

    if fitted.m_fit_surface is None:
        raise RuntimeError("FittedModel.m_fit_surface is None; cannot project.")
    m_smooth = np.asarray(fitted.m_fit_surface, dtype=float)

    model_cls = type(fitted.model)

    bs_res = bootstrap_from_m(
        model_cls,
        m_smooth,
        ages,
        years,
        B=B_bootstrap,
        seed=seed,
        resample=resample,
    )

    proj = project_mortality_from_bootstrap(
        model_cls=model_cls,
        ages=ages,
        years=years,
        m=m_smooth,
        bootstrap_result=bs_res,
        horizon=horizon,
        n_process=n_process,
        seed=None if seed is None else seed + 123,
        include_last=include_last,
    )

    metadata: Dict[str, Any] = {
        "selected_model": fitted.name,
        "selection_metric": fitted.metadata.get("selection_metric"),
        "train_end": fitted.metadata.get("selection_train_end"),
        "B_bootstrap": int(B_bootstrap),
        "n_process": int(n_process),
        "N_scenarios": int(proj.q_paths.shape[0]),
        "projection_horizon": int(horizon),
        "include_last": bool(include_last),
        "model_class": model_cls.__name__,
        "data_source": fitted.metadata.get("data_source"),
        "smoothing": fitted.metadata.get("smoothing"),
    }

    scen_set = build_scenario_set_from_projection(
        proj=proj,
        ages=ages,
        discount_factors=None,
        metadata=metadata,
    )

    return proj, scen_set


# ---------------------------------------------------------------------------
# New orchestration pipelines (spec-mandated)
# ---------------------------------------------------------------------------


def build_projection_pipeline(
    *,
    ages: np.ndarray,
    years: np.ndarray,
    m: np.ndarray,
    train_end: int,
    horizon: int,
    n_scenarios: int,
    model_names: Iterable[str] = (
        "LCM1",
        "LCM2",
        "APCM3",
        "CBDM5",
        "CBDM6",
        "CBDM7",
    ),
    cpsplines_kwargs: Optional[dict] = None,
    bootstrap_kwargs: dict | None = None,
    seed: int | None = None,
) -> MortalityScenarioSet:
    """
    End-to-end P-measure mortality projection pipeline.

    Steps (spec: Mortality models → Stochastic projections):
      1) Model selection via forecast RMSE on raw m.
      2) CPsplines smoothing of m.
      3) Final fit on smoothed surface.
      4) Parameter bootstrap.
      5) Stochastic projection (RW + param uncertainty).
      6) Return MortalityScenarioSet with q_paths/S_paths.
    """
    B_bootstrap, n_process, resample = _derive_bootstrap_params(
        n_scenarios=n_scenarios, bootstrap_kwargs=bootstrap_kwargs
    )
    include_last = bool(
        bootstrap_kwargs.get("include_last", True) if bootstrap_kwargs else True
    )
    fitted, proj, scen = build_mortality_scenarios_for_pricing(
        ages=ages,
        years=years,
        m=m,
        train_end=train_end,
        model_names=tuple(model_names),  # type: ignore[arg-type]
        selection_metric="logit_q",
        cpsplines_kwargs=cpsplines_kwargs,
        B_bootstrap=B_bootstrap,
        horizon=horizon,
        n_process=n_process,
        seed=seed,
        include_last=include_last,
    )

    q = np.asarray(scen.q_paths)
    N = q.shape[0]
    target = int(n_scenarios)

    if N != target:
        rng = np.random.default_rng(seed)
        if N > target:
            idx = np.arange(N)[:target]

        else:

            idx = rng.choice(N, size=target, replace=True)

        scen = MortalityScenarioSet(
            years=scen.years,
            ages=scen.ages,
            q_paths=scen.q_paths[idx],
            S_paths=scen.S_paths[idx],
            m_paths=None if scen.m_paths is None else scen.m_paths[idx],
            discount_factors=(
                None if scen.discount_factors is None else scen.discount_factors[idx]
            ),
            metadata=dict(scen.metadata),
        )
        scen.metadata["N_scenarios"] = int(target)
    # annotate desired scenario count for downstream awareness
    scen.metadata.setdefault("target_n_scenarios", int(n_scenarios))
    scen.metadata["resample"] = resample
    scen.metadata["fitted_model_name"] = fitted.name
    return scen


def build_risk_neutral_pipeline(
    scen_P: MortalityScenarioSet | None,
    *,
    instruments: Mapping[str, object],
    market_prices: Mapping[str, float],
    short_rate: float,
    calibration_kwargs: Dict[str, Any],
) -> tuple[MortalityScenarioSet, dict[str, Any], Any]:
    """
    Calibrate market price of longevity risk and build Q-measure scenarios.

    Spec: Risk-neutral valuation layer.
      - Calibrate lambda via observed instrument prices.
      - Reuse bootstrap/CRN via CalibrationCache if provided.
      - Transform P-scenarios to Q (Esscher-tilted RW drifts).
      - Returns (scen_Q, calibration_summary, calibration_cache).

    Note: Requires calibration inputs (ages, years, m, model_name, etc.) via
    calibration_kwargs if a CalibrationCache is not supplied.
    """
    cache = calibration_kwargs.get("cache")
    if cache is None:
        required = [
            "ages",
            "years",
            "m",
            "model_name",
            "B_bootstrap",
            "n_process",
            "horizon",
        ]
        missing = [k for k in required if k not in calibration_kwargs]
        if missing and scen_P is not None:
            # attempt to auto-fill from P-scenarios if possible
            if scen_P.m_paths is not None:
                calibration_kwargs = dict(calibration_kwargs)
                calibration_kwargs.setdefault("ages", scen_P.ages)
                calibration_kwargs.setdefault("years", scen_P.years)
                calibration_kwargs.setdefault(
                    "m", np.asarray(scen_P.m_paths).mean(axis=0)
                )
                missing = [k for k in required if k not in calibration_kwargs]
        if missing:
            raise ValueError(
                "calibration_kwargs must provide a CalibrationCache or keys: "
                + ", ".join(missing)
            )
        cache = build_calibration_cache(
            ages=np.asarray(calibration_kwargs["ages"], dtype=float),
            years=np.asarray(calibration_kwargs["years"], dtype=int),
            m=np.asarray(calibration_kwargs["m"], dtype=float),
            model_name=str(calibration_kwargs["model_name"]),
            B_bootstrap=int(calibration_kwargs["B_bootstrap"]),
            n_process=int(calibration_kwargs["n_process"]),
            horizon=int(calibration_kwargs["horizon"]),
            seed=calibration_kwargs.get("seed"),
            include_last=bool(calibration_kwargs.get("include_last", False)),
        )

    quotes = _build_multi_instrument_quotes(instruments, market_prices)
    lam_res = calibrate_lambda_least_squares(
        quotes=quotes,
        ages=cache.ages,
        years=cache.years,
        m=cache.m,
        model_name=cache.model_name,
        lambda0=calibration_kwargs.get("lambda0", 0.0),
        bounds=calibration_kwargs.get("bounds", (-5.0, 5.0)),
        B_bootstrap=len(cache.bs_res.params_list),
        n_process=cache.n_process,
        short_rate=short_rate,
        horizon=calibration_kwargs.get("horizon", cache.horizon),
        seed=calibration_kwargs.get("seed"),
        scale_sigma=calibration_kwargs.get("scale_sigma", 1.0),
        include_last=cache.include_last,
    )
    lambda_star = lam_res["lambda_star"]
    calib_summary = _calibration_summary(lam_res)
    calib_summary["metadata"] = {
        "model_name": cache.model_name,
        "horizon": cache.horizon,
        "B_bootstrap": (
            cache.bs_res.mu_sigma.shape[0] if hasattr(cache, "bs_res") else None
        ),
        "n_process": cache.n_process,
    }

    scen_Q = build_scenarios_under_lambda_fast(
        cache=cache,
        lambda_esscher=lambda_star,
        scale_sigma=calibration_kwargs.get("scale_sigma", 1.0),
        kappa_drift_shock=calibration_kwargs.get("kappa_drift_shock"),
        kappa_drift_shock_mode=calibration_kwargs.get(
            "kappa_drift_shock_mode", "additive"
        ),
        cohort_shock_type=calibration_kwargs.get("cohort_shock_type"),
        cohort_shock_magnitude=calibration_kwargs.get("cohort_shock_magnitude", 0.01),
        cohort_pivot_year=calibration_kwargs.get("cohort_pivot_year"),
    )
    scen_Q.metadata.setdefault("measure", "Q")
    scen_Q.metadata["lambda_star"] = lambda_star
    scen_Q.metadata["short_rate_for_calibration"] = float(short_rate)
    scen_Q.metadata["calibration_success"] = bool(lam_res.get("success", True))
    scen_Q.metadata["calibration_summary"] = calib_summary
    return scen_Q, calib_summary, cache


def pricing_pipeline(
    scen_Q: MortalityScenarioSet,
    *,
    specs: Mapping[str, object],
    short_rate: float,
) -> dict[str, float]:
    """
    Price a set of longevity instruments on Q-measure scenarios.

    Spec: Pricing engine layer.
      - Supports longevity bonds, survivor swaps, q-/s-forwards, life annuities.
      - Returns {instrument_name: price}.
    """
    normalized_specs: dict[str, object] = {}
    for name, spec_obj in specs.items():
        normalized_specs[name] = _normalize_spec(spec_obj)
    prices: dict[str, float] = {}
    for name, spec in normalized_specs.items():
        kind = _infer_kind(spec)
        pricer = make_single_product_pricer(kind=kind, spec=spec, short_rate=short_rate)
        prices[name] = float(pricer(scen_Q))
    return prices


def risk_analysis_pipeline(
    scen_Q: MortalityScenarioSet,
    *,
    specs: Mapping[str, object],
    short_rate: float,
    bumps: dict,
) -> AllSensitivities:
    """
    Compute mortality/rate sensitivities and convexity for a set of instruments.

    Spec: Sensitivity analysis layer.
      - Mortality delta-by-age
      - Mortality vega via sigma scaling (requires a scenario builder)
      - Rate DV01/duration and convexity
    """
    normalized_specs: dict[str, object] = {}
    for name, spec_obj in specs.items():
        normalized_specs[name] = _normalize_spec(spec_obj)

    if "build_scenarios_func" in bumps:
        build_scen_func = bumps["build_scenarios_func"]
    elif "calibration_cache" in bumps and "lambda_esscher" in bumps:
        cache = bumps["calibration_cache"]
        lam = bumps["lambda_esscher"]

        def build_scen_func(scale_sigma: float) -> MortalityScenarioSet:
            return build_scenarios_under_lambda_fast(
                cache=cache,
                lambda_esscher=lam,
                scale_sigma=scale_sigma,
            )

    else:
        raise ValueError(
            "risk_analysis_pipeline requires 'build_scenarios_func' or "
            "('calibration_cache' and 'lambda_esscher') in bumps to compute vega."
        )

    return compute_all_sensitivities(
        build_scenarios_func=build_scen_func,
        specs=normalized_specs,
        base_short_rate=float(short_rate),
        short_rate_for_pricing=bumps.get("short_rate_for_pricing"),
        sigma_rel_bump=bumps.get("sigma_rel_bump", 0.05),
        q_rel_bump=bumps.get("q_rel_bump", 0.01),
        rate_bump=bumps.get("rate_bump", 1e-4),
        ages_for_delta=bumps.get("ages_for_delta"),
    )


def stress_testing_pipeline(
    scen_Q: MortalityScenarioSet,
    *,
    shock_specs: list[ShockSpec] | list[dict[str, Any]],
) -> dict[str, MortalityScenarioSet]:
    """
    Generate stressed scenario sets (base/optimistic/pessimistic/custom shocks).

    Spec: Scenario analysis layer.
      - Supports pandemic, plateau, accel_improvement, cohort, life_expectancy, etc.
    """
    specs_norm: list[ShockSpec] = []
    for spec in shock_specs:
        if isinstance(spec, ShockSpec):
            specs_norm.append(spec)
        elif isinstance(spec, dict):
            specs_norm.append(
                ShockSpec(
                    name=str(spec.get("name", spec.get("shock_type", "shock"))),
                    shock_type=str(spec["shock_type"]),
                    params=spec.get("params", {}),
                )
            )
        else:
            raise ValueError("shock_specs entries must be ShockSpec or dict.")

    return generate_stressed_scenarios(scen_Q, shock_list=specs_norm)


# ---------------------------------------------------------------------------
# Interest rates and joint scenarios
# ---------------------------------------------------------------------------


def build_interest_rate_pipeline(
    *,
    times: Optional[np.ndarray] = None,
    zero_rates: Optional[np.ndarray] = None,
    zero_curve: Optional[np.ndarray] = None,
    horizon: Optional[int] = None,
    a: float,
    sigma: float,
    n_scenarios: int,
    r0: Optional[float] = None,
    seed: Optional[int] = None,
) -> InterestRateScenarioSet:
    """
    Build Hull–White (1F) interest-rate scenarios calibrated to a zero curve.
    """
    if zero_rates is None and zero_curve is not None:
        zero_rates = zero_curve
    if times is None:
        if horizon is None or zero_rates is None:
            raise ValueError(
                "Provide either times or both horizon and zero_rates/zero_curve."
            )
        times = np.arange(1, int(horizon) + 1, dtype=float)
    if zero_rates is None:
        raise ValueError("zero_rates (or zero_curve) must be provided.")
    return build_interest_rate_scenarios(
        times=np.asarray(times, dtype=float),
        zero_rates=np.asarray(zero_rates, dtype=float),
        a=a,
        sigma=sigma,
        n_scenarios=n_scenarios,
        r0=r0,
        seed=seed,
    )


def build_joint_scenarios(
    mort_scen: MortalityScenarioSet,
    rate_scen: InterestRateScenarioSet,
) -> MortalityScenarioSet:
    """
    Combine mortality scenarios with rate scenarios by attaching discount factors.

    Assumes independence (no correlation). If rate_scen has N=1, it is broadcast
    across mortality scenarios; otherwise N must match.

    Robust to discount_factors coming as (T, N) instead of (N, T).
    """
    q_paths = np.asarray(mort_scen.q_paths, dtype=float)
    S_paths = np.asarray(mort_scen.S_paths, dtype=float)
    N_m, A, H_m = q_paths.shape

    df_rates = np.asarray(rate_scen.discount_factors, dtype=float)

    if df_rates.ndim != 2:
        raise ValueError("rate_scen.discount_factors must have shape (N_rates, T).")

    n0, n1 = df_rates.shape  # could be (N,T) or (T,N)

    # ---- fix common orientation mistake: (T, N) ----
    # If first dim doesn't look like scenario count, but second does, transpose.
    if (n0 not in (1, N_m)) and (n1 in (1, N_m)):
        df_rates = df_rates.T
        n0, n1 = df_rates.shape

    # Now enforce (N_rates, T)
    N_r, H_r = n0, n1

    H = min(H_m, H_r)

    if N_r == 1:
        df = np.repeat(df_rates[:, :H], N_m, axis=0)
    elif N_r == N_m:
        df = df_rates[:, :H]
    else:
        raise ValueError(
            "Number of rate scenarios must be 1 or equal to mortality scenarios."
        )

    q_paths = q_paths[:, :, :H]
    S_paths = S_paths[:, :, :H]
    years = mort_scen.years[:H]

    metadata = dict(mort_scen.metadata)
    metadata["rate_model"] = rate_scen.metadata
    metadata["has_stochastic_rates"] = True

    m_paths = None
    if mort_scen.m_paths is not None:
        m_paths = mort_scen.m_paths[:, :, :H]

    return MortalityScenarioSet(
        years=years,
        ages=mort_scen.ages,
        q_paths=q_paths,
        S_paths=S_paths,
        m_paths=m_paths,
        discount_factors=df,
        metadata=metadata,
    )


def hedging_pipeline(
    *,
    liability_pv_paths: np.ndarray,
    hedge_pv_paths: np.ndarray,
    hedge_greeks: dict | None = None,
    method: str = "min_variance",
    constraints: dict | None = None,
) -> HedgeResult | GreekHedgeResult:
    """
    Compute hedge weights using various strategies.

    Spec: Hedging layer.
      - Variance-minimising hedge (OLS or bounded).
      - Multi-horizon hedge on cashflows.
      - Greek matching (duration / duration+convexity).
    """
    m = method.lower()
    if constraints is None:
        constraints = {}

    if m == "min_variance":
        return compute_min_variance_hedge(liability_pv_paths, hedge_pv_paths)

    if m == "min_variance_constrained":
        return compute_min_variance_hedge_constrained(
            liability_pv_paths,
            hedge_pv_paths,
            lb=float(constraints.get("lb", 0.0)),
            ub=float(constraints.get("ub", np.inf)),
        )

    if m == "multihorizon":
        return compute_multihorizon_hedge(
            liability_cf_paths=liability_pv_paths,
            instruments_cf_paths=hedge_pv_paths,
            discount_factors=constraints.get("discount_factors"),
            time_weights=constraints.get("time_weights"),
        )

    if m == "greek":
        if (
            hedge_greeks is None
            or "liability" not in hedge_greeks
            or "instruments" not in hedge_greeks
        ):
            raise ValueError(
                "hedge_greeks must provide 'liability' and 'instruments' arrays."
            )
        return compute_greek_matching_hedge(
            liability_greeks=hedge_greeks["liability"],
            instruments_greeks=np.asarray(hedge_greeks["instruments"], dtype=float),
            method=str(constraints.get("solver", "ols")),
            alpha=float(constraints.get("alpha", 1.0)),
        )

    if m == "duration":
        if hedge_greeks is None:
            raise ValueError(
                "hedge_greeks must provide 'liability_dPdr' and 'instruments_dPdr'."
            )
        return compute_duration_matching_hedge(
            liability_dPdr=float(hedge_greeks["liability_dPdr"]),
            instruments_dPdr=hedge_greeks["instruments_dPdr"],
            method=str(constraints.get("solver", "ols")),
            alpha=float(constraints.get("alpha", 1.0)),
        )

    if m == "duration_convexity":
        if hedge_greeks is None:
            raise ValueError(
                "hedge_greeks must provide liability_dPdr, liability_d2Pdr2, "
                "instruments_dPdr, instruments_d2Pdr2."
            )
        return compute_duration_convexity_matching_hedge(
            liability_dPdr=float(hedge_greeks["liability_dPdr"]),
            liability_d2Pdr2=float(hedge_greeks["liability_d2Pdr2"]),
            instruments_dPdr=hedge_greeks["instruments_dPdr"],
            instruments_d2Pdr2=hedge_greeks["instruments_d2Pdr2"],
            method=str(constraints.get("solver", "ols")),
            alpha=float(constraints.get("alpha", 1.0)),
        )

    raise ValueError(f"Unknown hedging method '{method}'.")


def reporting_pipeline(
    *,
    pv_paths: np.ndarray,
    ref_pv_paths: np.ndarray | None = None,
    name: str,
    var_level: float = 0.99,
) -> RiskReport:
    """
    Generate a RiskReport for PV paths (before/after hedge).

    Spec: Reporting layer.
    """
    return generate_risk_report(
        pv_paths=pv_paths,
        name=name,
        var_level=var_level,
        ref_pv_paths=ref_pv_paths,
    )
