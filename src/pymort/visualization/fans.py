"""Fan chart plotting utilities for mortality scenarios.

Note:
    Docstrings follow Google style and type hints use NDArray for clarity.
"""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from pymort.analysis import MortalityScenarioSet

_DEFAULT_QUANTILES: tuple[int, ...] = (5, 25, 50, 75, 95)
FloatArray = NDArray[np.floating]
GridArray = NDArray[np.integer] | NDArray[np.floating]


def _fan(
    paths: FloatArray,
    grid: GridArray,
    *,
    quantiles: Iterable[int] = _DEFAULT_QUANTILES,
    ax: Axes | None = None,
    label: str = "",
) -> Axes:
    qs = sorted({int(q) for q in quantiles})
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    for q in qs:
        ax.plot(
            grid,
            np.percentile(paths, q, axis=0),
            label=f"P{q}" if q in (50,) else None,
            lw=1.5 if q == 50 else 1.0,
        )

    ax.set_title(label)
    return ax


def plot_survival_fan(
    scen_set: MortalityScenarioSet,
    *,
    age: float,
    quantiles: Iterable[int] = _DEFAULT_QUANTILES,
    ax: Axes | None = None,
) -> None:
    """Plot a survival fan chart for a given cohort age.

    Args:
        scen_set: Scenario set with S_paths of shape (N, A, H).
        age: Cohort age at valuation.
        quantiles: Percentiles to plot.
        ax: Optional matplotlib axes.
    """
    ages = np.asarray(scen_set.ages, dtype=float)
    years = np.asarray(scen_set.years, dtype=int)
    idx = int(np.argmin(np.abs(ages - age)))
    S_age = np.asarray(scen_set.S_paths)[:, idx, :]  # (N, H)
    ax = _fan(S_age, years, quantiles=quantiles, ax=ax, label=f"Survival fan (age≈{ages[idx]:.1f})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Survival probability")
    ax.legend()


def plot_mortality_fan(
    scen_set: MortalityScenarioSet,
    *,
    age: float,
    quantiles: Iterable[int] = _DEFAULT_QUANTILES,
    ax: Axes | None = None,
) -> None:
    """Plot a mortality fan chart for a given cohort age.

    Args:
        scen_set: Scenario set with q_paths of shape (N, A, H).
        age: Cohort age at valuation.
        quantiles: Percentiles to plot.
        ax: Optional matplotlib axes.
    """
    ages = np.asarray(scen_set.ages, dtype=float)
    years = np.asarray(scen_set.years, dtype=int)
    idx = int(np.argmin(np.abs(ages - age)))
    q_age = np.asarray(scen_set.q_paths)[:, idx, :]  # (N, H)
    ax = _fan(
        q_age, years, quantiles=quantiles, ax=ax, label=f"Mortality fan (age≈{ages[idx]:.1f})"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("q")
    ax.legend()
