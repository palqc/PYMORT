from __future__ import annotations

from typing import Iterable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np

from pymort.analysis import MortalityScenarioSet


def _agg_stat(arr: np.ndarray, statistic: Literal["mean", "median"]) -> np.ndarray:
    if statistic == "mean":
        return np.mean(arr, axis=0)
    if statistic == "median":
        return np.median(arr, axis=0)
    raise ValueError("statistic must be 'mean' or 'median'.")


def plot_lexis(
    scen_set: MortalityScenarioSet,
    value: Literal["m", "q", "S"] = "q",
    statistic: Literal["mean", "median"] = "median",
    cohorts: Optional[Iterable[int]] = None,
    ax=None,
):
    """
    Lexis-style heatmap for mortality scenario summaries.

    Parameters
    ----------
    scen_set : MortalityScenarioSet
        Scenario container with q_paths/S_paths/m_paths and grids.
    value : {'m','q','S'}
        Which surface to display. If 'm' and m_paths is None, falls back to q.
    statistic : {'mean','median'}
        Aggregation across scenarios.
    cohorts : iterable of int, optional
        Calendar years of birth to highlight as diagonal lines.
    ax : matplotlib axis, optional
        Axis to draw on; creates a new one if None.
    """
    q = np.asarray(scen_set.q_paths, dtype=float)
    S = np.asarray(scen_set.S_paths, dtype=float)
    m = scen_set.m_paths

    if value == "m":
        if m is None:
            val = _agg_stat(q, statistic)
        else:
            val = _agg_stat(np.asarray(m, dtype=float), statistic)
    elif value == "q":
        val = _agg_stat(q, statistic)
    elif value == "S":
        val = _agg_stat(S, statistic)
    else:
        raise ValueError("value must be one of {'m','q','S'}.")

    ages = np.asarray(scen_set.ages, dtype=float)
    years = np.asarray(scen_set.years, dtype=int)

    X, Y = np.meshgrid(years, ages)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    im = ax.pcolormesh(X, Y, val, shading="auto", cmap="magma")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(value)

    # cohort lines: year - age = birth year
    if cohorts:
        for coh in cohorts:
            ax.plot(years, years - coh, ls="--", lw=1.0, color="cyan", alpha=0.8, label=f"cohort {coh}")
        # Avoid duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            uniq = dict(zip(labels, handles))
            ax.legend(uniq.values(), uniq.keys(), loc="upper right")

    ax.set_xlabel("Calendar year")
    ax.set_ylabel("Age")
    ax.set_title(f"Lexis diagram ({value}, {statistic})")
    return ax
