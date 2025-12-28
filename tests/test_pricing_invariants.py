from __future__ import annotations

import numpy as np
import pytest

from pymort.analysis.scenario import MortalityScenarioSet
from pymort.pipeline import pricing_pipeline
from pymort.pricing.liabilities import CohortLifeAnnuitySpec
from pymort.pricing.longevity_bonds import LongevityBondSpec, price_simple_longevity_bond
from pymort.pricing.mortality_derivatives import QForwardSpec, SForwardSpec
from pymort.pricing.survivor_swaps import SurvivorSwapSpec


def _make_scenario_set(
    N: int = 3,
    T: int = 6,
    base_q: float = 0.01,
    variation: float = 0.001,
    discount_factors: np.ndarray | None = None,
) -> MortalityScenarioSet:
    ages = np.array([60.0], dtype=float)
    years = np.arange(2020, 2020 + T, dtype=int)
    q_paths = np.full((N, 1, T), base_q, dtype=float)
    for i in range(N):
        q_paths[i, 0, :] += variation * i
    # Survival paths: S(t) = prod_{j<=t} (1 - q_j)
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
        discount_factors=discount_factors,
        metadata={},
    )


def _flat_discount_factors(r: float, T: int) -> np.ndarray:
    # annual continuous compounding
    return np.exp(-r * np.arange(1, T + 1, dtype=float))


def test_positive_prices_with_discount_factors():
    T = 6
    df = _flat_discount_factors(0.02, T)
    scen = _make_scenario_set(discount_factors=df)

    specs = {
        "bond": LongevityBondSpec(issue_age=60.0, notional=10.0, maturity_years=T),
        "swap": SurvivorSwapSpec(
            age=60.0, maturity_years=T, strike=0.2, notional=5.0, payer="fixed"
        ),
        "q_fwd": QForwardSpec(age=60.0, maturity_years=2, strike=0.0, notional=1.0),
        "s_fwd": SForwardSpec(age=60.0, maturity_years=3, strike=0.5, notional=1.0),
        "annuity": CohortLifeAnnuitySpec(
            issue_age=60.0, maturity_years=T, payment_per_survivor=1.0
        ),
    }

    prices = pricing_pipeline(scen_Q=scen, specs=specs, short_rate=0.0)
    for key, val in prices.items():
        assert np.isfinite(val), f"{key} price not finite"
        assert abs(val) < 1e9
        if key in {"bond", "annuity", "q_fwd", "s_fwd"}:
            assert val >= 0.0


def test_discounting_monotonicity_for_positive_cashflows():
    scen = _make_scenario_set(discount_factors=None)
    bond_spec = {
        "bond": LongevityBondSpec(issue_age=60.0, maturity_years=5, include_principal=True)
    }
    ann_spec = {"annuity": CohortLifeAnnuitySpec(issue_age=60.0, maturity_years=5)}

    price_low = pricing_pipeline(scen_Q=scen, specs=bond_spec, short_rate=0.01)["bond"]
    price_high = pricing_pipeline(scen_Q=scen, specs=bond_spec, short_rate=0.05)["bond"]
    assert price_high <= price_low

    price_low_ann = pricing_pipeline(scen_Q=scen, specs=ann_spec, short_rate=0.01)["annuity"]
    price_high_ann = pricing_pipeline(scen_Q=scen, specs=ann_spec, short_rate=0.05)["annuity"]
    assert price_high_ann <= price_low_ann


def test_discount_factor_equivalence_between_1d_and_2d():
    T = 5
    df1d = _flat_discount_factors(0.03, T)
    scen_1d = _make_scenario_set(N=3, T=T, discount_factors=df1d)
    df2d = np.tile(df1d, (scen_1d.n_scenarios(), 1))
    scen_2d = _make_scenario_set(N=3, T=T, discount_factors=df2d)

    spec = {"bond": LongevityBondSpec(issue_age=60.0, maturity_years=T)}
    price_1d = pricing_pipeline(scen_Q=scen_1d, specs=spec, short_rate=0.0)["bond"]
    price_2d = pricing_pipeline(scen_Q=scen_2d, specs=spec, short_rate=0.0)["bond"]
    assert np.allclose(price_1d, price_2d, atol=1e-10)


def test_single_scenario_matches_manual_pv():
    T = 4
    df = _flat_discount_factors(0.025, T)
    scen = _make_scenario_set(N=1, T=T, discount_factors=df)
    spec = {"bond": LongevityBondSpec(issue_age=60.0, maturity_years=T, include_principal=True)}

    price = pricing_pipeline(scen_Q=scen, specs=spec, short_rate=0.0)["bond"]

    S = scen.S_paths[0, 0, :T]
    expected = float(np.sum(S * df) + S[-1] * df[-1])
    assert np.isclose(price, expected, atol=1e-12)


def test_discount_factor_shape_error():
    scen = _make_scenario_set(N=2, T=6, discount_factors=None)
    spec = LongevityBondSpec(issue_age=60.0, maturity_years=6)
    bad_df = np.array([0.9, 0.8], dtype=float)  # too short
    with pytest.raises(ValueError):
        price_simple_longevity_bond(scen, spec, short_rate=None, discount_factors=bad_df)
