"""Risk summary utilities for scenario outputs.

This module provides simple summary statistics for mortality scenarios
and present-value paths.

Note:
    Docstrings follow Google style for clarity and spec alignment.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from pymort._types import FloatArray, IntArray
from pymort.analysis import MortalityScenarioSet


@dataclass
class SurvivalScenarioSummary:
    """Summary statistics of survival and mortality scenarios.

    This condenses large 3D arrays (N scenarios, A ages, H years) into
    interpretable statistics by age and year.

    Attributes:
        ages (np.ndarray): Age grid, shape (A,).
        years (np.ndarray): Year grid, shape (H,).
        S_mean (np.ndarray): Mean survival probabilities, shape (A, H).
        S_std (np.ndarray): Std. dev. of survival probabilities, shape (A, H).
        S_quantiles (dict[int, np.ndarray]): Percentile-to-surface mapping for
            survival, each surface shape (A, H).
        q_mean (np.ndarray): Mean death probabilities, shape (A, H).
        q_std (np.ndarray): Std. dev. of death probabilities, shape (A, H).
        q_quantiles (dict[int, np.ndarray]): Percentile-to-surface mapping for
            death probabilities, each surface shape (A, H).
    """

    ages: FloatArray
    years: IntArray
    S_mean: FloatArray
    S_std: FloatArray
    S_quantiles: dict[int, FloatArray]
    q_mean: FloatArray
    q_std: FloatArray
    q_quantiles: dict[int, FloatArray]


@dataclass
class PVSummary:
    """Summary statistics of present-value paths.

    Attributes:
        mean (float): Mean PV over scenarios.
        std (float): Std. dev. of PV over scenarios.
        p5 (float): 5th percentile of the PV distribution.
        p50 (float): Median PV.
        p95 (float): 95th percentile.
        min (float): Minimum PV.
        max (float): Maximum PV.
        n_scenarios (int): Number of scenarios.
    """

    mean: float
    std: float
    p5: float
    p50: float
    p95: float
    min: float
    max: float
    n_scenarios: int


def summarize_survival_scenarios(
    scen_set: MortalityScenarioSet,
    *,
    percentiles: Iterable[int] = (5, 50, 95),
) -> SurvivalScenarioSummary:
    """Compute summary statistics for a mortality scenario set.

    Args:
        scen_set: Scenario container with q_paths, S_paths, ages, years.
        percentiles: Percentiles to compute across scenarios.

    Returns:
        SurvivalScenarioSummary with mean/std and quantiles for S and q.
    """
    q_paths = np.asarray(scen_set.q_paths, dtype=float)
    S_paths = np.asarray(scen_set.S_paths, dtype=float)

    percentiles = sorted({int(p) for p in percentiles})
    if len(percentiles) == 0:
        raise ValueError("percentiles must be non-empty.")
    if any(p < 0 or p > 100 for p in percentiles):
        raise ValueError("percentiles must be between 0 and 100.")

    if q_paths.shape != S_paths.shape:
        raise ValueError(
            f"q_paths and S_paths must have same shape; got {q_paths.shape} vs {S_paths.shape}."
        )

    # Mean / std across scenarios (axis=0)
    S_mean = S_paths.mean(axis=0)  # (A, H)
    S_std = S_paths.std(axis=0)  # (A, H)
    q_mean = q_paths.mean(axis=0)
    q_std = q_paths.std(axis=0)

    # Quantiles across scenarios
    S_quantiles: dict[int, FloatArray] = {}
    q_quantiles: dict[int, FloatArray] = {}

    for p in percentiles:
        S_quantiles[p] = np.percentile(S_paths, p, axis=0)  # (A, H)
        q_quantiles[p] = np.percentile(q_paths, p, axis=0)  # (A, H)

    return SurvivalScenarioSummary(
        ages=np.asarray(scen_set.ages, dtype=float),
        years=np.asarray(scen_set.years, dtype=int),
        S_mean=S_mean,
        S_std=S_std,
        S_quantiles=S_quantiles,
        q_mean=q_mean,
        q_std=q_std,
        q_quantiles=q_quantiles,
    )


def summarize_pv_paths(pv_paths: FloatArray) -> PVSummary:
    """Summarize a vector of PV paths.

    This helper works for longevity bonds, swaps, forwards, and liabilities.

    Args:
        pv_paths: PVs per scenario, shape (N,) or (N, 1).

    Returns:
        PVSummary with mean/std and a few quantiles.
    """
    x = np.asarray(pv_paths, dtype=float).reshape(-1)
    if x.ndim != 1:
        raise ValueError("pv_paths must be 1D or (N,1).")
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("pv_paths contains no finite values.")

    n = x.shape[0]
    mean = float(x.mean())
    std = float(x.std(ddof=0))
    p5, p50, p95 = np.percentile(x, [5, 50, 95])
    x_min = float(x.min())
    x_max = float(x.max())

    return PVSummary(
        mean=mean,
        std=std,
        p5=float(p5),
        p50=float(p50),
        p95=float(p95),
        min=x_min,
        max=x_max,
        n_scenarios=int(n),
    )
