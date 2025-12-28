from __future__ import annotations

import numpy as np

from pymort.analysis.scenario import MortalityScenarioSet, validate_scenario_set
from pymort.pipeline import (
    build_interest_rate_pipeline,
    build_joint_scenarios,
    build_projection_pipeline,
    build_risk_neutral_pipeline,
    pricing_pipeline,
    risk_analysis_pipeline,
    stress_testing_pipeline,
)
from pymort.pricing.longevity_bonds import LongevityBondSpec
from pymort.pricing.survivor_swaps import SurvivorSwapSpec


def _toy_m_surface():
    ages = np.array([60.0, 61.0, 62.0], dtype=float)
    years = np.arange(2000, 2008, dtype=int)  # T=8
    base = np.array([0.01, 0.011, 0.012], dtype=float)[:, None]
    trend = np.linspace(1.0, 0.8, years.size)[None, :]
    m = base * trend
    return ages, years, m


def test_build_projection_pipeline_smoke_valid_scenarios():
    ages, years, m = _toy_m_surface()
    scen = build_projection_pipeline(
        ages=ages,
        years=years,
        m=m,
        train_end=2004,
        horizon=5,
        n_scenarios=10,
        model_names=("LCM1",),
        cpsplines_kwargs={"k": None, "horizon": 0, "verbose": False},
        bootstrap_kwargs={"B": 3, "n_process": 5},
        seed=123,
    )
    assert isinstance(scen, MortalityScenarioSet)
    validate_scenario_set(scen)
    assert scen.q_paths.shape[0] > 0
    assert scen.q_paths.shape[1] == ages.size
    assert scen.q_paths.shape[2] == scen.years.shape[0]
    assert isinstance(scen.metadata, dict)


def test_build_risk_neutral_pipeline_smoke_outputs():
    ages, years, m = _toy_m_surface()
    # simple instruments
    bond = LongevityBondSpec(issue_age=60.0, maturity_years=3, include_principal=True)
    swap = SurvivorSwapSpec(age=60.0, maturity_years=3, payer="fixed")
    instruments = {"bond": bond, "swap": swap}
    # derive market prices from lambda=0
    scen_P = build_projection_pipeline(
        ages=ages,
        years=years,
        m=m,
        train_end=2004,
        horizon=4,
        n_scenarios=6,
        model_names=("LCM1",),
        bootstrap_kwargs={"B": 2, "n_process": 4},
        seed=7,
    )
    market_prices = pricing_pipeline(scen_Q=scen_P, specs=instruments, short_rate=0.02)
    scen_Q, calib_summary, cache = build_risk_neutral_pipeline(
        scen_P,
        instruments=instruments,
        market_prices=market_prices,
        short_rate=0.02,
        calibration_kwargs={
            "model_name": "LCM2",
            "ages": ages,
            "years": years,
            "m": m,
            "B_bootstrap": 4,
            "n_process": 8,
            "horizon": 4,
            "seed": 42,
        },
    )
    validate_scenario_set(scen_Q)
    assert calib_summary is not None
    assert "lambda_star" in calib_summary
    assert np.isfinite(calib_summary["lambda_star"]).all()
    assert cache is not None


def test_build_joint_scenarios_smoke_discount_factors_attached():
    ages, years, m = _toy_m_surface()
    scen_mort = build_projection_pipeline(
        ages=ages,
        years=years,
        m=m,
        train_end=2004,
        horizon=3,
        n_scenarios=5,
        model_names=("LCM1",),
        bootstrap_kwargs={"B": 2, "n_process": 3},
        seed=10,
    )
    rate_scen = build_interest_rate_pipeline(
        n_scenarios=5,
        horizon=3,
        a=0.1,
        sigma=0.01,
        zero_curve=np.array([0.02, 0.021, 0.022]),
        seed=10,
    )
    joint = build_joint_scenarios(scen_mort, rate_scen)
    assert joint.discount_factors is not None
    assert joint.discount_factors.shape[-1] == joint.years.shape[0]
    validate_scenario_set(joint)


def test_stress_testing_pipeline_smoke_produces_stressed_sets():
    ages, years, m = _toy_m_surface()
    scen = build_projection_pipeline(
        ages=ages,
        years=years,
        m=m,
        train_end=2004,
        horizon=3,
        n_scenarios=4,
        model_names=("LCM1",),
        bootstrap_kwargs={"B": 2, "n_process": 3},
        seed=5,
    )
    res = stress_testing_pipeline(
        scen,
        shock_specs=[
            {
                "name": "long_life",
                "shock_type": "long_life",
                "params": {"magnitude": 0.1},
            }
        ],
    )
    assert "long_life" in res
    stressed = res["long_life"]
    validate_scenario_set(stressed)
    assert stressed.q_paths.shape == scen.q_paths.shape
    assert not np.allclose(stressed.q_paths, scen.q_paths)


def test_end_to_end_mini_run_projection_to_pricing():
    ages, years, m = _toy_m_surface()
    scen = build_projection_pipeline(
        ages=ages,
        years=years,
        m=m,
        train_end=2004,
        horizon=3,
        n_scenarios=5,
        model_names=("LCM1",),
        bootstrap_kwargs={"B": 2, "n_process": 3},
        seed=11,
    )
    spec = {"bond": LongevityBondSpec(issue_age=60.0, maturity_years=3, include_principal=True)}
    price = pricing_pipeline(scen_Q=scen, specs=spec, short_rate=0.02)["bond"]
    assert np.isfinite(price)
    assert price > 0.0


def test_risk_analysis_pipeline_smoke():
    ages, years, m = _toy_m_surface()
    scen = build_projection_pipeline(
        ages=ages,
        years=years,
        m=m,
        train_end=2004,
        horizon=3,
        n_scenarios=4,
        model_names=("LCM1",),
        bootstrap_kwargs={"B": 2, "n_process": 3},
        seed=12,
    )
    specs = {"bond": LongevityBondSpec(issue_age=60.0, maturity_years=3)}

    def build_scen(scale_sigma: float):
        return scen

    res = risk_analysis_pipeline(
        scen_Q=scen,
        specs=specs,
        short_rate=0.02,
        bumps={
            "build_scenarios_func": build_scen,
            "sigma_rel_bump": 0.05,
            "q_rel_bump": 0.01,
            "rate_bump": 1e-4,
        },
    )
    assert "bond" in res.prices_base
    assert np.isfinite(res.prices_base["bond"])
    assert np.isfinite(res.rate_sensitivity["bond"].dP_dr)
