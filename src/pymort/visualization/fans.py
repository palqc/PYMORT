from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

from pymort.analysis import MortalityScenarioSet

_DEFAULT_QUANTILES: tuple[int, ...] = (5, 25, 50, 75, 95)


def _fan(
    paths: np.ndarray,
    grid: np.ndarray,
    *,
    quantiles: Iterable[int] = _DEFAULT_QUANTILES,
    ax=None,
    label: str = "",
):
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
    ax=None,
) -> None:
    """Plot a fan chart of survival probabilities for a given age over projection years."""
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
    ax=None,
) -> None:
    """Plot a fan chart of mortality rates q for a given age over projection years."""
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
