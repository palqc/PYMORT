from __future__ import annotations

import numpy as np
import pytest

from pymort.analysis.scenario import MortalityScenarioSet
from pymort.lifetables import survival_from_q
from pymort.pricing.survivor_swaps import SurvivorSwapSpec, price_survivor_swap


def _toy_scenarios():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2020, 2021, 2022], dtype=int)
    q_paths = np.array(
        [
            [[0.01, 0.011, 0.012], [0.012, 0.013, 0.014]],
            [[0.009, 0.0105, 0.011], [0.011, 0.012, 0.013]],
        ],
        dtype=float,
    )
    S_paths = survival_from_q(q_paths)
    return MortalityScenarioSet(
        years=years, ages=ages, q_paths=q_paths, S_paths=S_paths, metadata={}
    )


def test_survivor_swap_price_finite_and_payment_times():
    scen = _toy_scenarios()
    spec = SurvivorSwapSpec(age=60.0, maturity_years=2, notional=1.0, strike=None, payer="fixed")
    res = price_survivor_swap(scen, spec, short_rate=0.01)
    assert np.isfinite(res["price"])
    assert res["pv_paths"].shape[0] == scen.q_paths.shape[0]
    spec_custom = SurvivorSwapSpec(
        age=60.0,
        maturity_years=3,
        notional=1.0,
        strike=None,
        payer="fixed",
        payment_times=np.array([2]),
    )
    res_custom = price_survivor_swap(scen, spec_custom, short_rate=0.01)
    assert np.isfinite(res_custom["price"])
    assert res_custom["metadata"]["n_payments"] == 1


def test_survivor_swap_monotonic_with_survival():
    scen_low = _toy_scenarios()
    spec = SurvivorSwapSpec(age=60.0, maturity_years=2, payer="fixed")
    price_low = price_survivor_swap(scen_low, spec, short_rate=0.01)["price"]

    scen_high = _toy_scenarios()
    scen_high.q_paths = scen_high.q_paths * 0.8
    scen_high.S_paths = survival_from_q(scen_high.q_paths)
    price_high = price_survivor_swap(scen_high, spec, short_rate=0.01)["price"]
    assert price_high > price_low


def test_survivor_swap_payment_times_errors():
    scen = _toy_scenarios()
    spec_bad = SurvivorSwapSpec(age=60.0, maturity_years=2, payment_times=np.array([]))
    with pytest.raises(ValueError):
        price_survivor_swap(scen, spec_bad, short_rate=0.01)
    spec_bad2 = SurvivorSwapSpec(age=60.0, maturity_years=2, payment_times=np.array([0, 1]))
    with pytest.raises(ValueError):
        price_survivor_swap(scen, spec_bad2, short_rate=0.01)
