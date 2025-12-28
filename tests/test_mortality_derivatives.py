from __future__ import annotations

import numpy as np
import pytest

from pymort.analysis.scenario import MortalityScenarioSet
from pymort.pricing.mortality_derivatives import (
    QForwardSpec,
    SForwardSpec,
    price_q_forward,
    price_s_forward,
)


def _make_scenario_set(N: int = 3, T: int = 6) -> MortalityScenarioSet:
    ages = np.array([60.0], dtype=float)
    years = np.arange(2020, 2020 + T, dtype=int)
    q_paths = np.full((N, 1, T), 0.01, dtype=float)
    for i in range(N):
        q_paths[i, 0, :] += 0.001 * i
    # S = cumprod(1-q) along time
    S_paths = np.cumprod(1.0 - q_paths, axis=-1)
    return MortalityScenarioSet(
        years=years, ages=ages, q_paths=q_paths, S_paths=S_paths, discount_factors=None, metadata={}
    )


def _flat_df(r: float, T: int) -> np.ndarray:
    return np.exp(-r * np.arange(1, T + 1, dtype=float))


def test_q_forward_settlement_differs_from_measurement_uses_df_at_settlement():
    # measure at Tm=2 but settle at Ts=5 => discount factor at Ts must apply
    T = 6
    scen = _make_scenario_set(N=3, T=T)
    df = _flat_df(0.03, T)

    spec = QForwardSpec(age=60.0, maturity_years=2, settlement_years=5, notional=10.0, strike=0.0)
    out = price_q_forward(scen, spec, short_rate=None, discount_factors=df)

    assert out["measurement_index"] == 1
    assert out["settlement_index"] == 4
    # payoff uses q at Tm only, discounted at Ts
    q_Tm = scen.q_paths[:, 0, 1]
    expected = float(np.mean(spec.notional * (q_Tm - 0.0) * df[4]))
    assert np.isclose(out["price"], expected, atol=1e-12)


def test_s_forward_atm_strike_gives_zero_price():
    scen = _make_scenario_set(N=5, T=6)
    df = _flat_df(0.02, 6)
    # ATM: strike = mean(S_Tm) => expected price ~ 0
    spec = SForwardSpec(age=60.0, maturity_years=3, settlement_years=3, notional=7.0, strike=None)
    out = price_s_forward(scen, spec, short_rate=None, discount_factors=df)
    assert abs(out["price"]) < 1e-12


def test_discount_factors_2d_with_first_dim_1_repeats_across_scenarios():
    scen = _make_scenario_set(N=4, T=5)
    df1d = _flat_df(0.01, 5)
    df2d = df1d[None, :]  # shape (1,T) -> triggers repeat to (N,T)

    spec = QForwardSpec(age=60.0, maturity_years=2, notional=1.0, strike=0.0)
    out_1d = price_q_forward(scen, spec, short_rate=None, discount_factors=df1d)
    out_2d = price_q_forward(scen, spec, short_rate=None, discount_factors=df2d)

    assert np.isclose(out_1d["price"], out_2d["price"], atol=1e-12)


def test_q_forward_input_validation_errors():
    scen = _make_scenario_set(N=2, T=4)

    with pytest.raises(ValueError):
        price_q_forward(scen, QForwardSpec(age=60.0, maturity_years=0), short_rate=0.0)

    with pytest.raises(ValueError):
        price_q_forward(scen, QForwardSpec(age=60.0, maturity_years=10), short_rate=0.0)

    with pytest.raises(ValueError):
        price_q_forward(
            scen, QForwardSpec(age=60.0, maturity_years=3, settlement_years=2), short_rate=0.0
        )

    with pytest.raises(ValueError):
        price_q_forward(
            scen, QForwardSpec(age=60.0, maturity_years=3, settlement_years=10), short_rate=0.0
        )


def test_s_forward_discount_factor_bad_shape_raises():
    scen = _make_scenario_set(N=3, T=5)
    bad = np.ones((2, 5))  # first dim neither 1 nor N
    with pytest.raises(ValueError):
        price_s_forward(
            scen, SForwardSpec(age=60.0, maturity_years=2), short_rate=None, discount_factors=bad
        )

    bad3d = np.ones((1, 1, 5))
    with pytest.raises(ValueError):
        price_s_forward(
            scen, SForwardSpec(age=60.0, maturity_years=2), short_rate=None, discount_factors=bad3d
        )


def test_q_forward_non_finite_measurement_values_raise():
    scen = _make_scenario_set(N=2, T=4)
    scen.q_paths[0, 0, 1] = np.nan
    with pytest.raises(ValueError):
        price_q_forward(scen, QForwardSpec(age=60.0, maturity_years=2, strike=0.0), short_rate=0.0)


def test_s_forward_non_finite_measurement_values_raise():
    scen = _make_scenario_set(N=2, T=4)
    scen.S_paths[0, 0, 2] = np.inf
    with pytest.raises(ValueError):
        price_s_forward(scen, SForwardSpec(age=60.0, maturity_years=3, strike=0.0), short_rate=0.0)
