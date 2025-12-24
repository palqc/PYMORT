from __future__ import annotations

import numpy as np
import pytest

from pymort.analysis.scenario import MortalityScenarioSet
from pymort.lifetables import survival_from_q
from pymort.pipeline import build_joint_scenarios, pricing_pipeline
from pymort.pricing.liabilities import CohortLifeAnnuitySpec
from pymort.pricing.longevity_bonds import LongevityBondSpec


def _simple_mort_scenarios(n: int = 2):
    ages = np.array([60.0], dtype=float)
    years = np.array([2020, 2021], dtype=int)
    q = np.stack([np.array([[0.01, 0.011]]), np.array([[0.012, 0.013]])])[:n]
    S = survival_from_q(q)
    return MortalityScenarioSet(years=years, ages=ages, q_paths=q, S_paths=S, metadata={})


def _simple_rate_scenarios(n: int = 2):
    df = np.array([[0.99, 0.98]] * n, dtype=float)
    return type(
        "DummyRate",
        (),
        {
            "discount_factors": df,
            "metadata": {"dummy": True},
        },
    )()


def test_build_joint_scenarios_attaches_discount_factors():
    mort = _simple_mort_scenarios()
    rate = _simple_rate_scenarios()
    joint = build_joint_scenarios(mort, rate)
    assert joint.discount_factors.shape == (mort.q_paths.shape[0], mort.q_paths.shape[2])
    assert joint.metadata.get("has_stochastic_rates") is True


def test_build_joint_scenarios_bad_inputs():
    mort = _simple_mort_scenarios(n=1)
    rate = _simple_rate_scenarios(n=3)
    with pytest.raises(ValueError):
        build_joint_scenarios(mort, rate)


def test_pricing_pipeline_minimal_specs():
    mort = _simple_mort_scenarios()
    specs = {"bond": LongevityBondSpec(issue_age=60.0, maturity_years=2, include_principal=True)}
    prices = pricing_pipeline(scen_Q=mort, specs=specs, short_rate=0.01)
    assert "bond" in prices and np.isfinite(prices["bond"])
    ann_spec = {"ann": CohortLifeAnnuitySpec(issue_age=60.0, maturity_years=2)}
    prices_ann = pricing_pipeline(scen_Q=mort, specs=ann_spec, short_rate=0.01)
    assert "ann" in prices_ann and np.isfinite(prices_ann["ann"])
