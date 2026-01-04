"""Pipeline orchestration for PYMORT.

This module wires together modeling, projection, risk-neutral valuation,
pricing, sensitivities, hedging, and reporting into end-to-end workflows. It
orchestrates existing building blocks without re-implementing the math.

Note:
    Docstrings follow Google style to align with project standards.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

import numpy as np

from pymort._types import AnyArray, FloatArray
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
from pymort.analysis.scenario_analysis import (
    ShockSpec,
    clone_scen_set_with,
    generate_stressed_scenarios,
)
from pymort.analysis.sensitivities import (
    AllSensitivities,
    compute_all_sensitivities,
    make_single_product_pricer,
    mortality_delta_by_age_all_products,
    mortality_vega_all_products,
    price_all_products,
    rate_convexity_all_products,
    rate_sensitivity_all_products,
)
from pymort.interest_rates.hull_white import (
    InterestRateScenarioSet,
    build_interest_rate_scenarios,
)
from pymort.lifetables import m_to_q
from pymort.models.gompertz import extrapolate_gompertz_surface, fit_gompertz_per_year
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
from pymort.pricing.mortality_derivatives import (
    QForwardSpec,
    SForwardSpec,
    price_q_forward,
    price_s_forward,
)
from pymort.pricing.risk_neutral import (
    CalibrationCache,
    MultiInstrumentQuote,
    build_calibration_cache,
    build_scenarios_under_lambda_fast,
    calibrate_lambda_least_squares,
)
from pymort.pricing.survivor_swaps import SurvivorSwapSpec, price_survivor_swap

InstrumentSpec = (
    LongevityBondSpec | SurvivorSwapSpec | SForwardSpec | QForwardSpec | CohortLifeAnnuitySpec
)


class BootstrapKwargs(TypedDict, total=False):
    """Overrides for bootstrap scenario generation.

    Attributes:
        B: Number of bootstrap replications.
        B_bootstrap: Bootstrap sample size.
        n_process: Parallel worker count.
        resample: Resampling mode string.
        include_last: Whether to include the last observed year.
    """

    B: int
    B_bootstrap: int
    n_process: int
    resample: str
    include_last: bool


class BumpsConfig(TypedDict, total=False):
    """Configuration for sensitivity bump sizes and scenario builders.

    Attributes:
        build_scenarios_func: Factory to rebuild scenarios given sigma scale.
        calibration_cache: Precomputed calibration cache for fast rebuilds.
        lambda_esscher: Esscher tilt parameter(s).
        short_rate_for_pricing: Override short rate for pricing.
        sigma_rel_bump: Relative bump for sigma scaling.
        q_rel_bump: Relative bump for q in delta-by-age.
        rate_bump: Additive bump for rate sensitivities.
        ages_for_delta: Optional ages subset for delta-by-age.
    """

    build_scenarios_func: Callable[[float], MortalityScenarioSet]
    calibration_cache: CalibrationCache
    lambda_esscher: float | Sequence[float] | FloatArray
    short_rate_for_pricing: float | None
    sigma_rel_bump: float
    q_rel_bump: float
    rate_bump: float
    ages_for_delta: Iterable[float] | None


class HedgeConstraints(TypedDict, total=False):
    """Constraints for hedging optimizers.

    Attributes:
        lb: Lower bound on weights.
        ub: Upper bound on weights.
        mode: Constraint mode identifier.
        discount_factors: Discount factors for time weighting.
        time_weights: Explicit time weights for objectives.
        instrument_names: Optional instrument names for reporting.
        solver: Solver name or backend identifier.
        alpha: Regularization or penalty strength.
    """

    lb: float
    ub: float
    mode: str
    discount_factors: FloatArray
    time_weights: FloatArray
    instrument_names: list[str] | None
    solver: str
    alpha: float


class HedgeGreeks(TypedDict, total=False):
    """Greek targets used by hedge construction routines.

    Attributes:
        liability: Liability PV paths.
        instruments: Instrument PV paths.
        liability_dPdr: Liability first-order rate sensitivity.
        instruments_dPdr: Instrument first-order rate sensitivities.
        liability_d2Pdr2: Liability second-order rate sensitivity.
        instruments_d2Pdr2: Instrument second-order rate sensitivities.
    """

    liability: Iterable[float]
    instruments: FloatArray
    liability_dPdr: float
    instruments_dPdr: Iterable[float]
    liability_d2Pdr2: float
    instruments_d2Pdr2: Iterable[float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_bootstrap_params(
    n_scenarios: int, bootstrap_kwargs: BootstrapKwargs | None
) -> tuple[int, int, str]:
    """Choose bootstrap parameters given a target number of scenarios.

    Args:
        n_scenarios: Target number of scenarios.
        bootstrap_kwargs: Optional overrides (B, n_process, resample).

    Returns:
        Tuple of (B_bootstrap, n_process, resample).
    """
    if bootstrap_kwargs is None:
        bootstrap_kwargs = {}
    B = int(bootstrap_kwargs.get("B", bootstrap_kwargs.get("B_bootstrap", max(1, n_scenarios))))
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
    raise ValueError("Cannot infer instrument kind; provide a spec dataclass or dict with 'kind'.")


def _normalize_spec(spec: object) -> InstrumentSpec:
    """Normalize an instrument spec to its dataclass form.

    Args:
        spec: Spec dataclass or dict with keys {"kind", "spec"}.

    Returns:
        Normalized spec dataclass.

    Raises:
        ValueError: If the spec format is unsupported.
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
    raise ValueError("Unsupported spec format; expected dataclass or {'kind','spec'} dict.")


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
                spec=spec,
                market_price=float(market_prices[name]),
                weight=weight,
            )
        )
    return quotes


def _calibration_summary(lam_res: dict[str, Any]) -> dict[str, Any]:
    """Build a structured calibration summary.

    Args:
        lam_res: Output dict from calibrate_lambda_least_squares.

    Returns:
        Summary dict with prices, errors, and metadata.
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
            instr_names,
            prices_model.tolist(),
            prices_market.tolist(),
            strict=True,
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
    ages: AnyArray,
    years: AnyArray,
    m: FloatArray,
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
    cpsplines_kwargs: dict[str, Any] | None = None,
    B_bootstrap: int = 1000,
    horizon: int = 50,
    n_process: int = 200,
    seed: int | None = None,
    include_last: bool = True,
) -> tuple[FittedModel, ProjectionResult, MortalityScenarioSet, CalibrationCache]:
    """Legacy end-to-end fit + projection pipeline.

    Args:
        ages: Age grid. Shape (A,).
        years: Year grid. Shape (T,).
        m: Central death rates. Shape (A, T).
        train_end: Last year included in training.
        model_names: Candidate model names for selection.
        selection_metric: Metric used for model selection.
        cpsplines_kwargs: Optional smoothing parameters.
        B_bootstrap: Number of bootstrap resamples.
        horizon: Projection horizon in years.
        n_process: Number of process-risk simulations per bootstrap draw.
        seed: Random seed for reproducibility.
        include_last: Whether to include the last observed year.

    Returns:
        Tuple of (fitted_model, projection_result, scenario_set, calibration_cache).
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
        raise RuntimeError("FittedModel.m_fit_surface is None; expected CPsplines-smoothed m.")
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

    metadata: dict[str, Any] = {
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

    cache = build_calibration_cache(
        ages=ages,
        years=years,
        m=m_smooth,
        model_name=str(fitted_best.name),
        B_bootstrap=int(B_bootstrap),
        n_process=int(n_process),
        horizon=int(horizon),
        seed=seed,  # seed used HERE (good)
        include_last=bool(include_last),
    )
    scen_set.metadata["calibration_cache"] = cache

    return fitted_best, proj, scen_set, cache


def project_from_fitted_model(
    fitted: FittedModel,
    *,
    B_bootstrap: int = 1000,
    horizon: int = 50,
    n_process: int = 200,
    seed: int | None = None,
    include_last: bool = True,
    resample: Literal["cell", "year_block"] = "year_block",
    ages_raw: AnyArray | None = None,
    years_raw: AnyArray | None = None,
    m_raw: FloatArray | None = None,
    plot_age_start: int = 95,
    plot_age_max: int = 200,
) -> tuple[ProjectionResult, MortalityScenarioSet, CalibrationCache]:
    """Bootstrap and project mortality starting from a fitted model.

    Args:
        fitted: Fitted model object with smoothed surface.
        B_bootstrap: Number of bootstrap resamples.
        horizon: Projection horizon in years.
        n_process: Number of process-risk simulations per bootstrap draw.
        seed: Random seed for reproducibility.
        include_last: Whether to include the last observed year.
        resample: Bootstrap resampling mode ("cell" or "year_block").
        ages_raw: Optional raw ages for plot-only Gompertz extension.
        years_raw: Optional raw years for plot-only Gompertz extension.
        m_raw: Optional raw mortality surface for plot-only extension.
        plot_age_start: Start age for plot-only extension.
        plot_age_max: Maximum age for plot-only extension.

    Returns:
        Tuple of (projection_result, scenario_set, calibration_cache).
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

    metadata: dict[str, Any] = {
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
    try:
        _attach_plot_extension_gompertz(
            scen_set,
            ages_raw=ages_raw,
            years_raw=years_raw,
            m_raw=m_raw,
            plot_age_start=plot_age_start,
            plot_age_max=plot_age_max,
            age_fit_min=80,
            age_fit_max=100,
        )
    except Exception as exc:
        scen_set.metadata["plot_extension_error"] = str(exc)
    cache = build_calibration_cache(
        ages=ages,
        years=years,
        m=m_smooth,
        model_name=str(fitted.name),
        B_bootstrap=int(B_bootstrap),
        n_process=int(n_process),
        horizon=int(horizon),
        seed=seed,
        include_last=bool(include_last),
    )
    scen_set.metadata["calibration_cache"] = cache

    return proj, scen_set, cache


# ---------------------------------------------------------------------------
# New orchestration pipelines (spec-mandated)
# ---------------------------------------------------------------------------


def build_projection_pipeline(
    *,
    ages: AnyArray,
    years: AnyArray,
    m: FloatArray,
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
    cpsplines_kwargs: dict[str, object] | None = None,
    bootstrap_kwargs: BootstrapKwargs | None = None,
    seed: int | None = None,
    ages_raw: AnyArray | None = None,
    years_raw: AnyArray | None = None,
    m_raw: FloatArray | None = None,
    plot_age_start: int = 95,
    plot_age_max: int = 200,
) -> tuple[MortalityScenarioSet, CalibrationCache]:
    """End-to-end P-measure mortality projection pipeline.

    Steps:
    1) Model selection via forecast RMSE on raw m.
    2) CPsplines smoothing of m.
    3) Final fit on the smoothed surface.
    4) Parameter bootstrap.
    5) Stochastic projection (random walk + parameter uncertainty).

    Args:
        ages: Age grid. Shape (A,).
        years: Year grid. Shape (T,).
        m: Central death rates. Shape (A, T).
        train_end: Last year included in training.
        horizon: Projection horizon in years.
        n_scenarios: Target number of scenarios.
        model_names: Candidate model names for selection.
        cpsplines_kwargs: Optional smoothing parameters.
        bootstrap_kwargs: Optional bootstrap overrides.
        seed: Random seed for reproducibility.
        ages_raw: Optional raw ages for plot-only Gompertz extension.
        years_raw: Optional raw years for plot-only Gompertz extension.
        m_raw: Optional raw mortality surface for plot-only extension.
        plot_age_start: Start age for plot-only extension.
        plot_age_max: Maximum age for plot-only extension.

    Returns:
        Tuple of (scenario_set, calibration_cache).
    """
    B_bootstrap, n_process, resample = _derive_bootstrap_params(
        n_scenarios=n_scenarios, bootstrap_kwargs=bootstrap_kwargs
    )
    include_last = bool(bootstrap_kwargs.get("include_last", True) if bootstrap_kwargs else True)
    fitted, _proj, scen, cache = build_mortality_scenarios_for_pricing(
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

    if target != N:
        rng = np.random.default_rng(seed)
        if target < N:
            idx = rng.choice(N, size=target, replace=False)
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

    # ------------------------------------------------------------------
    # Plot-only extension beyond plot_age_start using Gompertz on RAW data
    # ------------------------------------------------------------------
    try:
        _attach_plot_extension_gompertz(
            scen,
            ages_raw=ages_raw,
            years_raw=years_raw,
            m_raw=m_raw,
            plot_age_start=plot_age_start,
            plot_age_max=plot_age_max,
            age_fit_min=80,
            age_fit_max=100,
        )
    except Exception as _exc:
        # Don't kill the pipeline for plot-only features.
        scen.metadata["plot_extension_error"] = str(_exc)

    return scen, cache


def build_risk_neutral_pipeline(
    scen_P: MortalityScenarioSet | None,
    *,
    instruments: Mapping[str, object],
    market_prices: Mapping[str, float],
    short_rate: float,
    calibration_kwargs: dict[str, Any],
) -> tuple[MortalityScenarioSet, dict[str, Any], Any]:
    """Calibrate market price of longevity risk and build Q-measure scenarios.

    Steps:
    - Calibrate lambda via observed instrument prices.
    - Reuse bootstrap/CRN via CalibrationCache if provided.
    - Transform P-scenarios to Q (Esscher-tilted RW drifts).

    Args:
        scen_P: Optional P-measure scenarios.
        instruments: Instrument specs keyed by name.
        market_prices: Observed market prices keyed by instrument name.
        short_rate: Flat short rate used for pricing during calibration.
        calibration_kwargs: Inputs for calibration or cache creation.

    Returns:
        Tuple of (scen_Q, calibration_summary, calibration_cache).

    Raises:
        ValueError: If required calibration inputs are missing.
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
        if missing and scen_P is not None and scen_P.m_paths is not None:
            calibration_kwargs = dict(calibration_kwargs)
            calibration_kwargs.setdefault("ages", scen_P.ages)
            calibration_kwargs.setdefault("years", scen_P.years)
            calibration_kwargs.setdefault("m", np.asarray(scen_P.m_paths).mean(axis=0))
            missing = [k for k in required if k not in calibration_kwargs]
        if missing:
            raise ValueError(
                "calibration_kwargs must provide a CalibrationCache or keys: " + ", ".join(missing)
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

    cache = cast(CalibrationCache, cache)
    quotes = _build_multi_instrument_quotes(instruments, market_prices)
    B_bootstrap = getattr(cache.bs_res, "B", None)
    if B_bootstrap is None:
        if hasattr(cache.bs_res, "params_list"):
            B_bootstrap = len(cache.bs_res.params_list)
        else:
            B_bootstrap = int(cache.bs_res.mu_sigma.shape[0])
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
        "B_bootstrap": (cache.bs_res.mu_sigma.shape[0] if hasattr(cache, "bs_res") else None),
        "n_process": cache.n_process,
    }

    scen_Q = build_scenarios_under_lambda_fast(
        cache=cache,
        lambda_esscher=lambda_star,
        scale_sigma=calibration_kwargs.get("scale_sigma", 1.0),
        kappa_drift_shock=calibration_kwargs.get("kappa_drift_shock"),
        kappa_drift_shock_mode=calibration_kwargs.get("kappa_drift_shock_mode", "additive"),
        cohort_shock_type=calibration_kwargs.get("cohort_shock_type"),
        cohort_shock_magnitude=calibration_kwargs.get("cohort_shock_magnitude", 0.01),
        cohort_pivot_year=calibration_kwargs.get("cohort_pivot_year"),
    )
    scen_Q.metadata.setdefault("measure", "Q")
    scen_Q.metadata["lambda_star"] = lambda_star
    scen_Q.metadata["short_rate_for_calibration"] = float(short_rate)
    scen_Q.metadata["calibration_success"] = bool(lam_res.get("success", True))
    scen_Q.metadata["calibration_summary"] = calib_summary
    scen_Q.metadata["calibration_cache"] = cache
    return scen_Q, calib_summary, cache


def pricing_pipeline(
    scen_Q: MortalityScenarioSet,
    *,
    specs: Mapping[str, object],
    short_rate: float | None = 0.02,
    hull_white: HullWhiteConfig | None = None,
) -> dict[str, float]:
    """Price a set of longevity instruments on Q-measure scenarios.

    Args:
        scen_Q: Q-measure scenario set.
        specs: Instrument specs keyed by name.
        short_rate: Flat short rate used for pricing.
        hull_white: Optional Hull-White discounting configuration.

    Returns:
        Mapping of instrument name to price.
    """
    if scen_Q is None:
        raise ValueError("pricing_pipeline: scen_Q is None (no scenario set provided).")
    hw = hull_white or HullWhiteConfig(enabled=False)
    scen_used = apply_hull_white_discounting(scen_Q, hw=hw, short_rate=short_rate)
    normalized_specs: dict[str, InstrumentSpec] = {}
    for name, spec_obj in specs.items():
        normalized_specs[name] = _normalize_spec(spec_obj)
    prices: dict[str, float] = {}
    short_rate_pricer = cast(float, short_rate)
    for name, spec in normalized_specs.items():
        kind = _infer_kind(spec)
        pricer = make_single_product_pricer(kind=kind, spec=spec, short_rate=short_rate_pricer)
        prices[name] = float(pricer(scen_used))
    return prices


def risk_analysis_pipeline(
    scen_Q: MortalityScenarioSet,
    *,
    specs: Mapping[str, object],
    short_rate: float,
    bumps: BumpsConfig,
) -> AllSensitivities:
    """Compute mortality and rate sensitivities for a set of instruments.

    Args:
        scen_Q: Q-measure scenario set.
        specs: Instrument specs keyed by name.
        short_rate: Base short rate for pricing.
        bumps: Configuration for sensitivity bumps and scenario builders.

    Returns:
        AllSensitivities object with rate and mortality measures.
    """
    _ = scen_Q  # currently unused, but could be used for base pricing
    normalized_specs: dict[str, InstrumentSpec] = {}
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


def sensitivities_pipeline(
    scen: MortalityScenarioSet,
    *,
    specs: Mapping[str, object],
    short_rate: float,
    compute_rate: bool = True,
    compute_convexity: bool = False,
    compute_delta_by_age: bool = False,
    compute_vega: bool = False,
    bumps: BumpsConfig | None = None,
) -> dict[str, Any]:
    """UI-friendly sensitivities pipeline.

    Args:
        scen: Scenario set used for base pricing.
        specs: Instrument specs keyed by name.
        short_rate: Base short rate for pricing.
        compute_rate: Whether to compute rate sensitivity.
        compute_convexity: Whether to compute rate convexity.
        compute_delta_by_age: Whether to compute mortality delta by age.
        compute_vega: Whether to compute mortality vega via sigma scaling.
        bumps: Optional bump configuration.

    Returns:
        Dictionary with base prices and any requested sensitivities.
    """
    if bumps is None:
        bumps = {}

    # normalize specs
    normalized_specs: dict[str, InstrumentSpec] = {}
    for name, spec_obj in specs.items():
        normalized_specs[name] = _normalize_spec(spec_obj)

    # ---- Freeze strikes for ATM derivatives so bumps don't re-ATM every time ----
    # (Only if your specs support "strike" and you set strike=None for ATM)
    try:
        from copy import deepcopy

        normalized_specs = deepcopy(normalized_specs)

        # Freeze Q-forward strike if missing
        if "q_forward" in normalized_specs:
            sp = normalized_specs["q_forward"]
            if isinstance(sp, QForwardSpec) and sp.strike is None:
                out_qf = price_q_forward(scen_set=scen, spec=sp, short_rate=float(short_rate))
                if "strike" in out_qf:
                    sp.strike = float(out_qf["strike"])

        # Freeze S-forward strike if missing
        if "s_forward" in normalized_specs:
            sp = normalized_specs["s_forward"]
            if isinstance(sp, SForwardSpec) and sp.strike is None:
                out_sf = price_s_forward(scen_set=scen, spec=sp, short_rate=float(short_rate))
                if "strike" in out_sf:
                    sp.strike = float(out_sf["strike"])

        # Freeze survivor swap strike if missing (depends on your API)
        if "survivor_swap" in normalized_specs:
            sp = normalized_specs["survivor_swap"]
            if isinstance(sp, SurvivorSwapSpec) and sp.strike is None:
                out_ss = price_survivor_swap(scen_set=scen, spec=sp, short_rate=float(short_rate))
                if "strike" in out_ss:
                    sp.strike = float(out_ss["strike"])

    except Exception:
        # If a spec/pricer doesn't expose strike, skip freezing.
        pass

    out: dict[str, Any] = {}
    out["prices_base"] = price_all_products(
        scen, specs=normalized_specs, short_rate=float(short_rate)
    )

    # rate sens
    if compute_rate:
        out["rate_sensitivity"] = rate_sensitivity_all_products(
            scen,
            specs=normalized_specs,
            base_short_rate=float(short_rate),
            bump=float(bumps.get("rate_bump", 1e-4)),
        )

    # convexity
    if compute_convexity:
        out["rate_convexity"] = rate_convexity_all_products(
            scen,
            specs=normalized_specs,
            base_short_rate=float(short_rate),
            bump=float(bumps.get("rate_bump", 1e-4)),
        )

    # delta by age
    if compute_delta_by_age:
        out["delta_by_age"] = mortality_delta_by_age_all_products(
            scen,
            specs=normalized_specs,
            short_rate=float(short_rate),
            ages=bumps.get("ages_for_delta"),
            rel_bump=float(bumps.get("q_rel_bump", 0.01)),
        )

    # vega
    if compute_vega:
        # 1) direct builder
        build_scen_func = bumps.get("build_scenarios_func")

        # 2) or infer builder from cache + lambda
        if build_scen_func is None and ("calibration_cache" in bumps and "lambda_esscher" in bumps):
            cache = bumps["calibration_cache"]
            lam = bumps["lambda_esscher"]

            def build_scen_func(scale_sigma: float) -> MortalityScenarioSet:
                return build_scenarios_under_lambda_fast(
                    cache=cache,
                    lambda_esscher=lam,
                    scale_sigma=scale_sigma,
                )

        if build_scen_func is None:
            raise ValueError(
                "compute_vega=True but no builder provided. "
                "Pass bumps['build_scenarios_func'] or (calibration_cache + lambda_esscher)."
            )

        out["vega_sigma_scale"] = mortality_vega_all_products(
            build_scen_func,
            specs=normalized_specs,
            short_rate=float(short_rate),
            rel_bump=float(bumps.get("sigma_rel_bump", 0.05)),
        )

    out["meta"] = {
        "short_rate": float(short_rate),
        "rate_bump": float(bumps.get("rate_bump", 1e-4)),
        "q_rel_bump": float(bumps.get("q_rel_bump", 0.01)),
        "sigma_rel_bump": float(bumps.get("sigma_rel_bump", 0.05)),
        "ages_for_delta": bumps.get("ages_for_delta"),
    }
    return out


def stress_testing_pipeline(
    scen_Q: MortalityScenarioSet,
    *,
    shock_specs: list[ShockSpec] | list[dict[str, Any]],
) -> dict[str, MortalityScenarioSet]:
    """Generate stressed scenario sets from shock specifications.

    Args:
        scen_Q: Base scenario set.
        shock_specs: List of ShockSpec or dict specifications.

    Returns:
        Mapping of shock name to stressed scenario set.
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
    times: FloatArray | None = None,
    zero_rates: FloatArray | None = None,
    zero_curve: FloatArray | None = None,
    horizon: int | None = None,
    a: float,
    sigma: float,
    n_scenarios: int,
    r0: float | None = None,
    seed: int | None = None,
) -> InterestRateScenarioSet:
    """Build Hull-White (1F) interest-rate scenarios calibrated to a zero curve.

    Args:
        times: Time grid in years. If None, built from horizon.
        zero_rates: Zero curve values. Shape (T,).
        zero_curve: Alias for zero_rates.
        horizon: Number of years if times is not provided.
        a: Mean-reversion parameter.
        sigma: Volatility parameter.
        n_scenarios: Number of scenarios.
        r0: Initial short rate. Defaults to zero_rates[0].
        seed: Random seed for reproducibility.

    Returns:
        InterestRateScenarioSet with short-rate paths and discount factors.
    """
    if zero_rates is None and zero_curve is not None:
        zero_rates = zero_curve
    if times is None:
        if horizon is None or zero_rates is None:
            raise ValueError("Provide either times or both horizon and zero_rates/zero_curve.")
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
    """Combine mortality scenarios with rate scenarios by attaching discount factors.

    Assumes independence. If rate_scen has N=1, it is broadcast across mortality
    scenarios; otherwise N must match.

    Args:
        mort_scen: Mortality scenario set (q_paths, S_paths).
        rate_scen: Interest rate scenario set (discount_factors).

    Returns:
        MortalityScenarioSet with discount_factors attached.
    """
    q_paths = np.asarray(mort_scen.q_paths, dtype=float)
    S_paths = np.asarray(mort_scen.S_paths, dtype=float)
    N_m, _A, H_m = q_paths.shape

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
        raise ValueError("Number of rate scenarios must be 1 or equal to mortality scenarios.")

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
    liability_pv_paths: FloatArray,
    hedge_pv_paths: FloatArray,
    liability_cf_paths: FloatArray | None = None,
    hedge_cf_paths: FloatArray | None = None,
    hedge_greeks: HedgeGreeks | None = None,
    method: str = "min_variance",
    constraints: HedgeConstraints | None = None,
    discount_factors: FloatArray | None = None,
) -> HedgeResult | GreekHedgeResult:
    """Compute hedge weights using various strategies.

    Args:
        liability_pv_paths: Liability PV paths. Shape (N,).
        hedge_pv_paths: Hedge instrument PV paths. Shape (N, M).
        liability_cf_paths: Optional liability cashflows. Shape (N, T).
        hedge_cf_paths: Optional hedge cashflows. Shape (N, M, T).
        hedge_greeks: Optional greek inputs for duration/convexity matching.
        method: Hedging method ("min_variance", "multihorizon", "greek", etc.).
        constraints: Optional constraints and solver settings.
        discount_factors: Optional discount factors for multihorizon mode.

    Returns:
        HedgeResult or GreekHedgeResult depending on method.
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
        if liability_cf_paths is None or hedge_cf_paths is None:
            raise ValueError(
                "For method='multihorizon', provide liability_cf_paths (N,T) and hedge_cf_paths (N,M,T)."
            )

        mode = str(constraints.get("mode", "pv_by_horizon"))
        df = constraints.get("discount_factors", discount_factors)
        if mode == "pv_by_horizon" and df is None:
            raise ValueError(
                "mode='pv_by_horizon' requires constraints['discount_factors'] "
                "(shape (T,) or (N,T))."
            )

        return compute_multihorizon_hedge(
            liability_cf_paths=liability_cf_paths,
            instruments_cf_paths=hedge_cf_paths,
            discount_factors=df,
            time_weights=constraints.get("time_weights"),
            instrument_names=constraints.get("instrument_names"),
            mode=mode,
        )

    if m == "greek":
        if (
            hedge_greeks is None
            or "liability" not in hedge_greeks
            or "instruments" not in hedge_greeks
        ):
            raise ValueError("hedge_greeks must provide 'liability' and 'instruments' arrays.")
        return compute_greek_matching_hedge(
            liability_greeks=hedge_greeks["liability"],
            instruments_greeks=np.asarray(hedge_greeks["instruments"], dtype=float),
            method=str(constraints.get("solver", "ols")),
            alpha=float(constraints.get("alpha", 1.0)),
        )

    if m == "duration":
        if hedge_greeks is None:
            raise ValueError("hedge_greeks must provide 'liability_dPdr' and 'instruments_dPdr'.")
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
    pv_paths: FloatArray,
    ref_pv_paths: FloatArray | None = None,
    name: str,
    var_level: float = 0.99,
) -> RiskReport:
    """Generate a RiskReport for PV paths (before/after hedge).

    Args:
        pv_paths: PV paths to summarize.
        ref_pv_paths: Optional reference PV paths (e.g., pre-hedge).
        name: Report name.
        var_level: VaR confidence level.

    Returns:
        RiskReport with summary statistics.
    """
    return generate_risk_report(
        pv_paths=pv_paths,
        name=name,
        var_level=var_level,
        ref_pv_paths=ref_pv_paths,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _attach_plot_extension_gompertz(
    scen: MortalityScenarioSet,
    *,
    ages_raw: AnyArray | None,
    years_raw: AnyArray | None,
    m_raw: FloatArray | None,
    plot_age_start: int,
    plot_age_max: int,
    age_fit_min: int = 80,
    age_fit_max: int = 100,
) -> None:
    """Attach plot-only Gompertz extension beyond plot_age_start.

    This writes into scen.metadata["plot_extension"] and never raises; errors
    are captured in scen.metadata["plot_extension_error"].

    Args:
        scen: Scenario set to annotate.
        ages_raw: Raw ages for fitting.
        years_raw: Raw years for fitting.
        m_raw: Raw mortality surface for fitting.
        plot_age_start: Start age for plotting.
        plot_age_max: Maximum age for plotting.
        age_fit_min: Minimum age for Gompertz fit.
        age_fit_max: Maximum age for Gompertz fit.
    """
    try:
        if ages_raw is None or years_raw is None or m_raw is None:
            return

        ages_raw = np.asarray(ages_raw, dtype=float)
        years_raw = np.asarray(years_raw, dtype=int)
        m_raw = np.asarray(m_raw, dtype=float)

        gfit = fit_gompertz_per_year(
            ages=ages_raw,
            years=years_raw,
            m=m_raw,
            age_fit_min=age_fit_min,
            age_fit_max=age_fit_max,
            m_floor=1e-12,
            compute_fitted_surface=False,
        )

        ages_ext_raw, years_ext_raw, m_ext_raw = extrapolate_gompertz_surface(
            gfit,
            age_min=int(np.min(ages_raw)),
            age_max=int(plot_age_max),
        )

        q_ext_raw = m_to_q(m_ext_raw)  # (A_ext, T_raw)

        years_proj = np.asarray(scen.years, dtype=int)  # (H_out,)
        y0, y1 = int(years_ext_raw[0]), int(years_ext_raw[-1])
        years_clamped = np.clip(years_proj, y0, y1)
        year_index = (years_clamped - y0).astype(int)

        q_ext_on_proj = q_ext_raw[:, year_index]  # (A_ext, H_out)

        A_ext, H_out = q_ext_on_proj.shape
        S_cohort = np.full((A_ext, H_out), np.nan, dtype=float)

        for a_idx in range(A_ext):
            K = min(H_out, A_ext - a_idx)
            if K <= 0:
                continue
            j = np.arange(K)
            q_diag = q_ext_on_proj[a_idx + j, j]
            S_cohort[a_idx, :K] = np.cumprod(1.0 - q_diag)

        astart = max(int(plot_age_start), int(ages_ext_raw.min()))
        start_idx = int(astart - int(ages_ext_raw.min()))

        scen.metadata["plot_extension"] = {
            "method": "gompertz_per_year_on_raw",
            "fit_window": [int(age_fit_min), int(age_fit_max)],
            "plot_age_start": int(plot_age_start),
            "plot_age_max": int(plot_age_max),
            "ages_ext": ages_ext_raw[start_idx:].astype(float).tolist(),
            "years": years_proj.astype(int).tolist(),
            "q_ext": q_ext_on_proj[start_idx:, :].astype(float).tolist(),
            "S_ext": S_cohort[start_idx:, :].astype(float).tolist(),
        }

    except Exception as exc:
        scen.metadata["plot_extension_error"] = str(exc)


@dataclass(frozen=True)
class HullWhiteConfig:
    """Configuration for optional Hull-White discounting.

    Attributes:
        enabled: Whether Hull-White discounting is active.
        a: Mean-reversion parameter.
        sigma: Volatility parameter.
        seed: Random seed for reproducibility.
        zero_rates: Zero curve as array (H,), flat scalar, or None to use short_rate.
        r0: Initial short rate. Defaults to zero_rates[0] if provided.
    """

    enabled: bool = False
    a: float = 0.05
    sigma: float = 0.01
    seed: int | None = 0
    # Zero curve: array (H,), flat scalar, or None -> fallback to short_rate.
    zero_rates: FloatArray | None = None
    r0: float | None = None  # If None, uses zero_rates[0].


def apply_hull_white_discounting(
    scen: MortalityScenarioSet,
    *,
    hw: HullWhiteConfig,
    short_rate: float | None,
) -> MortalityScenarioSet:
    """Apply Hull-White discounting to a scenario set.

    Args:
        scen: Scenario set to discount.
        hw: Hull-White configuration.
        short_rate: Fallback flat short rate if zero_rates is None.

    Returns:
        Scenario set with discount_factors attached when enabled.
    """
    if not hw.enabled:
        return scen

    H = len(scen.years)
    times = np.arange(1, H + 1, dtype=float)

    # --- build the zero curve ---
    zero_rates: FloatArray
    if hw.zero_rates is None:
        if short_rate is None:
            raise ValueError(
                "Hull–White enabled but no zero curve provided and short_rate is None."
            )
        zero_rates = cast(FloatArray, np.full(H, float(short_rate), dtype=float))
    else:
        zr = np.asarray(hw.zero_rates, dtype=float).reshape(-1)
        if zr.size == 1:
            zero_rates = cast(FloatArray, np.full(H, float(zr[0]), dtype=float))
        else:
            if zr.size < H:
                raise ValueError(f"zero_rates must have length >= H={H}, got {zr.size}.")
            zero_rates = cast(FloatArray, zr[:H])

    # --- simulate rate paths + DF paths ---
    ir_set = build_interest_rate_scenarios(
        times=times,
        zero_rates=zero_rates,
        a=float(hw.a),
        sigma=float(hw.sigma),
        n_scenarios=int(np.asarray(scen.q_paths).shape[0]),
        r0=hw.r0,
        seed=hw.seed,
    )

    scen_hw = clone_scen_set_with(scen, discount_factors=ir_set.discount_factors)

    md = dict(getattr(scen_hw, "metadata", {}) or {})
    md["interest_rate_model"] = ir_set.metadata

    return clone_scen_set_with(scen_hw, metadata=md)
