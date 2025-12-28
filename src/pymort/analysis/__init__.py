"""Lazy analysis namespace with lightweight imports.

This module avoids importing heavy submodules at import time. Public symbols
are loaded on demand via ``__getattr__`` to reduce startup cost and prevent
circular imports.

Note:
    Docstrings follow Google style for clarity and spec alignment.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__all__ = [
    "BootstrapResult",
    "MortalityScenarioSet",
    "_freeze_gamma_last_per_age",
    "_rmse",
    "_rmse_logit_q",
    "_rw_drift_forecast",
    "bootstrap_from_m",
    "bootstrap_logitq_model",
    "bootstrap_logm_model",
    "build_scenario_set_from_projection",
    "load_scenario_set_npz",
    "rmse_aic_bic",
    "save_scenario_set_npz",
    "smooth_mortality_with_cpsplines",
    "time_split_backtest_apc_m3",
    "time_split_backtest_cbd_m5",
    "time_split_backtest_cbd_m6",
    "time_split_backtest_cbd_m7",
    "time_split_backtest_lc_m1",
    "time_split_backtest_lc_m2",
    "validate_scenario_set",
]

_ATTR_MAP: dict[str, tuple[str, str]] = {
    "BootstrapResult": ("pymort.analysis.bootstrap", "BootstrapResult"),
    "bootstrap_from_m": ("pymort.analysis.bootstrap", "bootstrap_from_m"),
    "bootstrap_logm_model": ("pymort.analysis.bootstrap", "bootstrap_logm_model"),
    "bootstrap_logitq_model": ("pymort.analysis.bootstrap", "bootstrap_logitq_model"),
    "smooth_mortality_with_cpsplines": (
        "pymort.analysis.smoothing",
        "smooth_mortality_with_cpsplines",
    ),
    "rmse_aic_bic": ("pymort.analysis.validation", "rmse_aic_bic"),
    "time_split_backtest_apc_m3": ("pymort.analysis.validation", "time_split_backtest_apc_m3"),
    "time_split_backtest_cbd_m5": ("pymort.analysis.validation", "time_split_backtest_cbd_m5"),
    "time_split_backtest_cbd_m6": ("pymort.analysis.validation", "time_split_backtest_cbd_m6"),
    "time_split_backtest_cbd_m7": ("pymort.analysis.validation", "time_split_backtest_cbd_m7"),
    "time_split_backtest_lc_m1": ("pymort.analysis.validation", "time_split_backtest_lc_m1"),
    "time_split_backtest_lc_m2": ("pymort.analysis.validation", "time_split_backtest_lc_m2"),
    "_rmse": ("pymort.analysis.validation", "_rmse"),
    "_rmse_logit_q": ("pymort.analysis.validation", "_rmse_logit_q"),
    "_rw_drift_forecast": ("pymort.analysis.validation", "_rw_drift_forecast"),
    "_freeze_gamma_last_per_age": ("pymort.analysis.validation", "_freeze_gamma_last_per_age"),
    "MortalityScenarioSet": ("pymort.analysis.scenario", "MortalityScenarioSet"),
    "build_scenario_set_from_projection": (
        "pymort.analysis.scenario",
        "build_scenario_set_from_projection",
    ),
    "validate_scenario_set": ("pymort.analysis.scenario", "validate_scenario_set"),
    "save_scenario_set_npz": ("pymort.analysis.scenario", "save_scenario_set_npz"),
    "load_scenario_set_npz": ("pymort.analysis.scenario", "load_scenario_set_npz"),
}


def __getattr__(name: str) -> Any:
    if name not in _ATTR_MAP:
        raise AttributeError(f"module 'pymort.analysis' has no attribute '{name}'")
    module_name, attr_name = _ATTR_MAP[name]
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover
    from pymort.analysis.bootstrap import (
        BootstrapResult,
        bootstrap_from_m,
        bootstrap_logitq_model,
        bootstrap_logm_model,
    )
    from pymort.analysis.scenario import (
        MortalityScenarioSet,
        build_scenario_set_from_projection,
        load_scenario_set_npz,
        save_scenario_set_npz,
        validate_scenario_set,
    )
    from pymort.analysis.smoothing import smooth_mortality_with_cpsplines
    from pymort.analysis.validation import (
        _freeze_gamma_last_per_age,
        _rmse,
        _rmse_logit_q,
        _rw_drift_forecast,
        rmse_aic_bic,
        time_split_backtest_apc_m3,
        time_split_backtest_cbd_m5,
        time_split_backtest_cbd_m6,
        time_split_backtest_cbd_m7,
        time_split_backtest_lc_m1,
        time_split_backtest_lc_m2,
    )
