"""Lexis diagram plotting helpers.

Note:
    Docstrings follow Google style and type hints use NDArray for clarity.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from pymort.analysis import MortalityScenarioSet

FloatArray = NDArray[np.floating]


def _agg_stat(arr: FloatArray, statistic: Literal["mean", "median"]) -> FloatArray:
    """Aggregate a scenario array across the first axis.

    Args:
        arr: Scenario array. Shape (N, A, T).
        statistic: Aggregation statistic, "mean" or "median".

    Returns:
        Aggregated surface with shape (A, T).
    """
    if statistic == "mean":
        return cast(FloatArray, np.mean(arr, axis=0))
    if statistic == "median":
        return cast(FloatArray, np.median(arr, axis=0))
    raise ValueError("statistic must be 'mean' or 'median'.")


def plot_lexis(
    scen_set: MortalityScenarioSet,
    value: Literal["m", "q", "S"] = "q",
    statistic: Literal["mean", "median"] = "median",
    cohorts: Iterable[int] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot a Lexis-style heatmap for scenario summaries.

    Args:
        scen_set: Scenario container with q_paths/S_paths/m_paths and grids.
        value: Which surface to display ("m", "q", or "S").
        statistic: Aggregation across scenarios ("mean" or "median").
        cohorts: Optional calendar years of birth to highlight.
        ax: Optional matplotlib axes.

    Returns:
        Matplotlib axes with the plot.
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
            ax.plot(
                years,
                years - coh,
                ls="--",
                lw=1.0,
                color="cyan",
                alpha=0.8,
                label=f"cohort {coh}",
            )
        # Avoid duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            uniq = dict(zip(labels, handles, strict=True))
            ax.legend(uniq.values(), uniq.keys(), loc="upper right")

    ax.set_xlabel("Calendar year")
    ax.set_ylabel("Age")
    ax.set_title(f"Lexis diagram ({value}, {statistic})", pad=15)
    return ax
