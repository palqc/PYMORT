from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pymort.analysis.scenario import MortalityScenarioSet
from pymort.visualization.fans import plot_mortality_fan, plot_survival_fan
from pymort.visualization.lexis import plot_lexis


def _toy_scenarios(N: int = 3, A: int = 2, T: int = 5) -> MortalityScenarioSet:
    ages = np.linspace(60, 60 + A - 1, A, dtype=float)
    years = np.arange(2020, 2020 + T, dtype=int)
    q_paths = np.full((N, A, T), 0.01, dtype=float)
    q_paths += 0.0005 * np.arange(N)[:, None, None]
    S_paths = np.ones_like(q_paths)
    for t in range(T):
        if t == 0:
            S_paths[:, :, t] = 1.0 - q_paths[:, :, t]
        else:
            S_paths[:, :, t] = S_paths[:, :, t - 1] * (1.0 - q_paths[:, :, t])
    return MortalityScenarioSet(
        years=years,
        ages=ages,
        q_paths=q_paths,
        S_paths=S_paths,
        metadata={"name": "toy"},
    )


def test_plot_survival_fan_runs_and_creates_axes():
    scen = _toy_scenarios()
    plt.close("all")
    plot_survival_fan(scen, age=60.0, quantiles=[5, 50, 95])
    fig = plt.gcf()
    assert len(fig.axes) >= 1
    ax = fig.axes[0]
    assert len(ax.lines) > 0
    plt.close("all")


def test_plot_mortality_fan_runs_and_creates_axes():
    scen = _toy_scenarios()
    plt.close("all")
    plot_mortality_fan(scen, age=60.0, quantiles=[5, 50, 95])
    fig = plt.gcf()
    assert len(fig.axes) >= 1
    ax = fig.axes[0]
    assert len(ax.lines) > 0
    plt.close("all")


def test_plot_lexis_runs_for_q_and_S():
    scen = _toy_scenarios()
    plt.close("all")
    ax_q = plot_lexis(scen, value="q", statistic="median", cohorts=None)
    assert ax_q is not None
    ax_s = plot_lexis(scen, value="S", statistic="mean", cohorts=[1960])
    assert ax_s is not None
    plt.close("all")


def test_plot_functions_raise_on_unknown_value_or_statistic():
    scen = _toy_scenarios()
    with pytest.raises(ValueError):
        plot_lexis(scen, value="bad", statistic="median")
    with pytest.raises(ValueError):
        plot_lexis(scen, value="q", statistic="bad")
    plt.close("all")
