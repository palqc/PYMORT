from __future__ import annotations

import numpy as np
import pytest

from pymort.analysis.reporting import (
    generate_risk_report,
    plot_hedge_performance,
    plot_price_distribution,
    plot_survival_fan,
)
from pymort.analysis.scenario import MortalityScenarioSet

# ----------------------------
# Core: generate_risk_report
# ----------------------------


def test_generate_risk_report_constant_paths_zero_variance():
    pv = np.array([1.0, 1.0, 1.0])
    report = generate_risk_report(pv, var_level=0.5)
    assert report.std_pv == 0.0
    assert report.mean_pv == 1.0
    # default pv_kind="cost" => loss = pv
    assert report.var == report.cvar == 1.0
    assert report.mean_loss == 1.0
    assert report.std_loss == 0.0


def test_generate_risk_report_single_scenario_and_quantiles_exist():
    pv = np.array([2.5])
    report = generate_risk_report(pv, var_level=0.9)
    assert report.n_scenarios == 1
    assert report.mean_pv == 2.5
    assert report.std_pv == 0.0
    assert report.var <= report.cvar
    assert 0.01 in report.quantiles_pv
    assert 0.9 in report.quantiles_pv
    assert 0.9 in report.quantiles_loss


def test_generate_risk_report_empty_raises():
    with pytest.raises(ValueError):
        generate_risk_report(np.array([]))


def test_generate_risk_report_non_finite_raises():
    with pytest.raises(ValueError):
        generate_risk_report(np.array([1.0, np.inf]))
    with pytest.raises(ValueError):
        generate_risk_report(np.array([1.0, np.nan]))


def test_generate_risk_report_var_level_out_of_bounds_raises():
    pv = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        generate_risk_report(pv, var_level=0.0)
    with pytest.raises(ValueError):
        generate_risk_report(pv, var_level=1.0)


def test_generate_risk_report_quantile_grid_invalid_raises():
    pv = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        generate_risk_report(pv, quantile_grid=(0.01, 0.0, 0.5))
    with pytest.raises(ValueError):
        generate_risk_report(pv, quantile_grid=(0.01, 1.0, 0.5))


def test_generate_risk_report_pv_kind_value_flips_loss_sign():
    # PV is "value" => loss = -PV
    pv = np.array([10.0, 12.0, 8.0])
    rep_cost = generate_risk_report(pv, pv_kind="cost", var_level=0.5)
    rep_value = generate_risk_report(pv, pv_kind="value", var_level=0.5)

    assert rep_cost.mean_pv == rep_value.mean_pv
    assert np.isclose(rep_value.mean_loss, -rep_cost.mean_pv)
    # losses should be flipped distribution-wise
    assert np.isclose(rep_value.quantiles_loss[0.5], -rep_cost.quantiles_pv[0.5])


def test_generate_risk_report_unknown_pv_kind_raises():
    pv = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        generate_risk_report(pv, pv_kind="xxx")  # type: ignore[arg-type]


def test_generate_risk_report_loss_threshold_adds_metric():
    pv = np.array([1.0, 2.0, 3.0, 4.0])
    rep = generate_risk_report(pv, pv_kind="cost", loss_threshold=3.0)
    assert "prob_loss_exceeds_threshold" in rep.extra
    assert 0.0 <= rep.extra["prob_loss_exceeds_threshold"] <= 1.0


def test_generate_risk_report_with_reference_metrics_and_mismatch_error():
    rng = np.random.default_rng(0)
    ref = rng.normal(10.0, 2.0, size=500)
    pv = ref * 0.7 + rng.normal(0.0, 0.2, size=500)  # correlated but less var

    rep = generate_risk_report(pv, ref_pv_paths=ref, var_level=0.95)

    # variance reduction metrics should exist (ref var > 0)
    assert rep.hedge_var_reduction is not None
    assert rep.hedge_var_reduction_loss is not None
    assert rep.hedge_var_reduction_var is not None
    assert rep.hedge_var_reduction_cvar is not None

    # corr/beta/tracking error computed
    assert rep.corr_with_ref is not None
    assert np.isfinite(rep.corr_with_ref)
    assert rep.beta_vs_ref is not None
    assert np.isfinite(rep.beta_vs_ref)
    assert rep.tracking_error is not None
    assert np.isfinite(rep.tracking_error)

    with pytest.raises(ValueError):
        generate_risk_report(pv, ref_pv_paths=ref[:10])


# ----------------------------
# Plots (smoke + errors)
# ----------------------------


@pytest.fixture
def tiny_scen_set():
    years = np.array([2020, 2021, 2022], dtype=int)
    ages = np.array([60.0, 61.0], dtype=float)
    q_paths = np.full((5, 2, 3), 0.01, dtype=float)
    S_paths = np.cumprod(1.0 - q_paths, axis=-1)
    return MortalityScenarioSet(
        years=years,
        ages=ages,
        q_paths=q_paths,
        S_paths=S_paths,
        discount_factors=None,
        metadata={},
    )


def test_plot_survival_fan_smoke(tiny_scen_set, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    ax = plot_survival_fan(tiny_scen_set, age=60.0)
    assert ax is not None


def test_plot_survival_fan_quantiles_errors(tiny_scen_set, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")

    with pytest.raises(ValueError):
        plot_survival_fan(
            tiny_scen_set, age=60.0, quantiles=(0.05, 0.25, 0.75, 0.95)
        )  # missing 0.50
    with pytest.raises(ValueError):
        plot_survival_fan(
            tiny_scen_set, age=60.0, quantiles=(0.05, 0.25, 0.50, 0.50, 0.95)
        )  # duplicate
    with pytest.raises(ValueError):
        plot_survival_fan(
            tiny_scen_set, age=60.0, quantiles=(0.0, 0.25, 0.50, 0.75, 0.95)
        )  # out of bounds


def test_plot_price_distribution_smoke_and_errors(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")

    x = np.random.default_rng(1).normal(size=200)
    ax = plot_price_distribution(x, bins=20, density=False)
    assert ax is not None

    with pytest.raises(ValueError):
        plot_price_distribution(np.array([1.0, np.nan]))


def test_plot_hedge_performance_smoke_and_errors(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")

    rng = np.random.default_rng(2)
    L = rng.normal(size=50)
    N = L * 0.5
    ax = plot_hedge_performance(L, N)
    assert ax is not None

    with pytest.raises(ValueError):
        plot_hedge_performance(np.array([1.0, 2.0]), np.array([1.0]))  # length mismatch
