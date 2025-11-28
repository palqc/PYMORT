from __future__ import annotations

from typing import Any, Dict, Iterable, Literal, Optional, Type

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
) -> tuple[
    FittedModel,  # final model (on CPsplines)
    ProjectionResult,  # project_mortality_from_bootstrap result
    MortalityScenarioSet,  # standard object for pricing
]:
    """
    Full PYMORT pipeline up to mortality scenarios:

        raw m
        → model selection (forecast RMSE, via backtests)
        → CPsplines smoothing + final fit on smoothed m
        → bootstrap on smoothed surface
        → stochastic projections (RW + param uncertainty)
        → MortalityScenarioSet (q_paths + S_paths)

    Cette fonction est le point d'entrée “officiel” avant le pricing.
    """

    ages = np.asarray(ages, dtype=float)
    years = np.asarray(years, dtype=float)
    m = np.asarray(m, dtype=float)

    # best model selection + final fit on CPsplines
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
    model_cls: Type = type(fitted_best.model)

    # 2) Parameters bootstrap on CPsplines
    bs_res = bootstrap_from_m(
        model_cls,
        m_smooth,
        ages,
        years,
        B=B_bootstrap,
        seed=seed,
        resample="year_block",
    )

    # 3) Stochastic projections (param + process uncertainty)
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

    # 4) MortalityScenarioSet with scenario.py
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
        discount_factors=None,  # ⚠️⚠️ à brancher plus tard avec une courbe de taux
        metadata=metadata,
    )

    return fitted_best, proj, scen_set
