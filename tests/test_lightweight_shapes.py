from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pymort.analysis.projections import (
    project_mortality_from_bootstrap,
    simulate_random_walk_paths,
    simulate_random_walk_paths_with_eps,
)
from pymort.analysis.scenario import MortalityScenarioSet
from pymort.analysis.scenario_analysis import apply_mortality_shock, generate_stressed_bundle
from pymort.lifetables import survival_from_q
from pymort.models.lc_m1 import LCM1, fit_lee_carter
from pymort.models.lc_m2 import LCM2
from pymort.pipeline import _derive_bootstrap_params, hedging_pipeline
from pymort.pricing.risk_neutral import (
    CalibrationCache,
    apply_cohort_trend_shock_to_qpaths,
    build_scenarios_under_lambda_fast,
    esscher_shift_normal_rw,
)


def test_random_walk_helpers_shapes():
    rng = np.random.default_rng(0)
    paths = simulate_random_walk_paths(
        k_last=0.1, mu=0.0, sigma=0.1, horizon=3, n_sims=2, rng=rng, include_last=True
    )
    assert paths.shape == (2, 4)
    eps = np.zeros((2, 3))
    paths_eps = simulate_random_walk_paths_with_eps(
        k_last=0.1, mu=0.0, sigma=0.05, eps=eps, include_last=False
    )
    assert np.allclose(paths_eps, np.full((2, 3), 0.1))


def test_projection_output_shapes_and_finiteness():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2000, 2001], dtype=int)
    m = np.array([[0.01, 0.011], [0.012, 0.013]], dtype=float)
    params = fit_lee_carter(m)
    bs = SimpleNamespace(params_list=[params], mu_sigma=np.array([[0.0, 0.05]]))
    eps_rw = np.zeros((1, 2, 2))
    proj = project_mortality_from_bootstrap(
        model_cls=LCM1,
        ages=ages,
        years=years,
        m=m,
        bootstrap_result=bs,
        horizon=2,
        n_process=2,
        seed=123,
        include_last=True,
        eps_rw=eps_rw,
    )
    assert proj.q_paths.shape == (2, ages.size, 3)
    assert proj.m_paths is not None and proj.m_paths.shape == proj.q_paths.shape
    assert proj.k_paths is not None and proj.k_paths.shape == (2, 3)
    assert np.isfinite(proj.q_paths).all()
    assert np.all((proj.q_paths >= 0.0) & (proj.q_paths <= 1.0))


def test_scenario_analysis_shocks_preserve_shapes():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2020, 2021, 2022], dtype=int)
    q = np.full((2, ages.size, years.size), 0.01, dtype=float)
    S = survival_from_q(q)
    base = MortalityScenarioSet(
        years=years, ages=ages, q_paths=q, S_paths=S, metadata={"name": "base"}
    )
    stressed = apply_mortality_shock(base, shock_type="short_life", magnitude=0.05)
    assert stressed.q_paths.shape == q.shape
    assert stressed.q_paths.mean() > q.mean()
    bundle = generate_stressed_bundle(
        base,
        long_life_bump=0.05,
        short_life_bump=0.05,
        pandemic_year=2021,
        pandemic_severity=0.1,
        pandemic_duration=1,
    )
    assert bundle.optimistic.q_paths.shape == q.shape
    assert bundle.pessimistic.q_paths.mean() > bundle.optimistic.q_paths.mean()


def test_risk_neutral_helpers_and_cohort_shock():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2000, 2001], dtype=int)
    m = np.clip(np.array([[0.01, 0.011], [0.012, 0.013]], dtype=float), 1e-6, None)
    model = LCM2().fit(m, ages, years)
    model.estimate_rw()
    bs_res = SimpleNamespace(params_list=[model.params], mu_sigma=np.array([[0.0, 0.05]]))
    eps = {"eps_rw": np.zeros((1, 1, 1))}
    cache = CalibrationCache(
        model=model,
        bs_res=bs_res,
        eps=eps,
        ages=ages,
        years=years,
        m=m,
        model_name="LCM2",
        horizon=1,
        n_process=1,
        include_last=False,
    )
    scen = build_scenarios_under_lambda_fast(
        cache=cache,
        lambda_esscher=np.array([0.0]),
        scale_sigma=1.0,
    )
    assert scen.q_paths.shape == (1, ages.size, 1)
    assert np.isfinite(scen.q_paths).all()
    shocked = apply_cohort_trend_shock_to_qpaths(
        scen.q_paths,
        ages=ages,
        years=scen.years,
        shock_type="cohort_step",
        magnitude=0.01,
    )
    assert shocked.shape == scen.q_paths.shape
    esscher = esscher_shift_normal_rw(mu_P=[0.0], sigma_P=[0.1], lambda_esscher=[0.2])
    assert np.isclose(esscher.mu_Q[0], 0.0 + 0.2 * 0.1**2)


def test_pipeline_helpers_greek_branch_and_bootstrap_params():
    B, n_proc, resample = _derive_bootstrap_params(
        n_scenarios=10, bootstrap_kwargs={"B": 3, "n_process": 4, "resample": "cell"}
    )
    assert (B, n_proc, resample) == (3, 4, "cell")

    res = hedging_pipeline(
        liability_pv_paths=np.array([1.0, 1.1]),
        hedge_pv_paths=np.array([[0.5, 0.6], [0.4, 0.5]]),
        hedge_greeks={"liability_dPdr": -5.0, "instruments_dPdr": [-2.0, -1.5]},
        method="duration",
        constraints={"solver": "ols"},
    )
    assert res.weights.shape == (2,)
    assert np.isfinite(res.weights).all()
