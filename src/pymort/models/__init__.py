"""Lazy import shim for mortality models.

This module avoids importing model implementations at import time to prevent
circular imports between packages. Symbols are loaded lazily via __getattr__.

Note:
    Docstrings follow Google style to align with project standards.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__all__ = [
    "APCM3",
    "CBDM5",
    "CBDM6",
    "CBDM7",
    "LCM1",
    "LCM2",
    "APCM3Params",
    "CBDM5Params",
    "CBDM6Params",
    "CBDM7Params",
    "LCM1Params",
    "LCM2Params",
    "LeeCarterM1",
    "_logit",
    "estimate_rw_params",
    "estimate_rw_params_cbd",
    "fit_cbd",
    "fit_cbd_cohort",
    "fit_cbd_m7",
    "fit_lee_carter",
    "reconstruct_log_m",
    "reconstruct_logit_q",
    "reconstruct_logit_q_cbd_cohort",
    "reconstruct_logit_q_m7",
    "reconstruct_q",
]

_ATTR_MAP: dict[str, tuple[str, str]] = {
    # Lee-Carter family
    "LCM1Params": ("pymort.models.lc_m1", "LCM1Params"),
    "LCM1": ("pymort.models.lc_m1", "LCM1"),
    "fit_lee_carter": ("pymort.models.lc_m1", "fit_lee_carter"),
    "reconstruct_log_m": ("pymort.models.lc_m1", "reconstruct_log_m"),
    "estimate_rw_params": ("pymort.models.utils", "estimate_rw_params"),
    "LeeCarterM1": ("pymort.models.lc_m1", "LCM1"),  # backwards compat alias
    "LCM2Params": ("pymort.models.lc_m2", "LCM2Params"),
    "LCM2": ("pymort.models.lc_m2", "LCM2"),
    # APC
    "APCM3": ("pymort.models.apc_m3", "APCM3"),
    "APCM3Params": ("pymort.models.apc_m3", "APCM3Params"),
    # CBD family
    "CBDM5": ("pymort.models.cbd_m5", "CBDM5"),
    "CBDM5Params": ("pymort.models.cbd_m5", "CBDM5Params"),
    "fit_cbd": ("pymort.models.cbd_m5", "fit_cbd"),
    "reconstruct_logit_q": ("pymort.models.cbd_m5", "reconstruct_logit_q"),
    "reconstruct_q": ("pymort.models.cbd_m5", "reconstruct_q"),
    "estimate_rw_params_cbd": ("pymort.models.cbd_m5", "estimate_rw_params_cbd"),
    "_logit": ("pymort.models.cbd_m5", "_logit"),
    "CBDM6": ("pymort.models.cbd_m6", "CBDM6"),
    "CBDM6Params": ("pymort.models.cbd_m6", "CBDM6Params"),
    "fit_cbd_cohort": ("pymort.models.cbd_m6", "fit_cbd_cohort"),
    "reconstruct_logit_q_cbd_cohort": ("pymort.models.cbd_m6", "reconstruct_logit_q_cbd_cohort"),
    "CBDM7": ("pymort.models.cbd_m7", "CBDM7"),
    "CBDM7Params": ("pymort.models.cbd_m7", "CBDM7Params"),
    "fit_cbd_m7": ("pymort.models.cbd_m7", "fit_cbd_m7"),
    "reconstruct_logit_q_m7": ("pymort.models.cbd_m7", "reconstruct_logit_q_m7"),
}


def __getattr__(name: str) -> Any:
    if name not in _ATTR_MAP:
        raise AttributeError(f"module 'pymort.models' has no attribute '{name}'")
    module_name, attr_name = _ATTR_MAP[name]
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover
    from pymort.models.apc_m3 import APCM3, APCM3Params
    from pymort.models.cbd_m5 import (
        CBDM5,
        CBDM5Params,
        _logit,
        estimate_rw_params_cbd,
        fit_cbd,
        reconstruct_logit_q,
        reconstruct_q,
    )
    from pymort.models.cbd_m6 import (
        CBDM6,
        CBDM6Params,
        fit_cbd_cohort,
        reconstruct_logit_q_cbd_cohort,
    )
    from pymort.models.cbd_m7 import CBDM7, CBDM7Params, fit_cbd_m7, reconstruct_logit_q_m7
    from pymort.models.lc_m1 import LCM1, LCM1Params, fit_lee_carter, reconstruct_log_m
    from pymort.models.lc_m2 import LCM2, LCM2Params
    from pymort.models.utils import estimate_rw_params
