from __future__ import annotations

import numpy as np
import pytest

from pymort.analysis.scenario import MortalityScenarioSet, validate_scenario_set
from pymort.lifetables import survival_from_q
from pymort.pricing.liabilities import CohortLifeAnnuitySpec, price_cohort_life_annuity
from pymort.pricing.utils import build_discount_factors, find_nearest_age_index


def _simple_scenario():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2020, 2021, 2022], dtype=int)
    q = np.array(
        [
            [0.01, 0.011, 0.012],
            [0.012, 0.013, 0.014],
        ],
        dtype=float,
    )
    S = survival_from_q(q)
    return MortalityScenarioSet(
        years=years,
        ages=ages,
        q_paths=q[None, :, :],
        S_paths=S[None, :, :],
        metadata={"source": "test"},
    )


def test_pricing_utils_discount_factors_priority_and_errors():
    scen = _simple_scenario()
    # explicit discount factors
    df = build_discount_factors(scen, short_rate=None, discount_factors=np.array([1.0, 0.9]), H=2)
    assert np.allclose(df, np.array([1.0, 0.9]))
    # scenario-provided discount factors
    scen_df = scen
    scen_df.discount_factors = np.array([1.0, 0.97, 0.94])
    df2 = build_discount_factors(scen_df, short_rate=None, discount_factors=None, H=3)
    assert np.allclose(df2, scen_df.discount_factors)
    # flat short rate fallback
    df_flat = build_discount_factors(scen, short_rate=0.01, discount_factors=None, H=3)
    assert np.all(df_flat > 0) and df_flat.shape == (3,)
    # invalid inputs
    with pytest.raises(ValueError):
        build_discount_factors(scen, short_rate=None, discount_factors=np.array([-1.0, 0.9]), H=2)
    with pytest.raises(ValueError):
        build_discount_factors(scen, short_rate=None, discount_factors=None, H=5)


def test_find_nearest_age_index():
    ages = np.array([50.0, 55.0, 60.0])
    assert find_nearest_age_index(ages, 57.0) == 1
    assert find_nearest_age_index(ages, 61.0) == 2


def test_life_annuity_price_monotonic_with_survival():
    scen_low = _simple_scenario()
    spec = CohortLifeAnnuitySpec(issue_age=60.0, maturity_years=3, payment_per_survivor=1.0)
    res_low = price_cohort_life_annuity(scen_low, spec, short_rate=0.01)
    # create higher survival by scaling down q
    scen_high = _simple_scenario()
    scen_high.q_paths = scen_high.q_paths * 0.8
    scen_high.S_paths = survival_from_q(scen_high.q_paths)
    validate_scenario_set(scen_high)
    res_high = price_cohort_life_annuity(scen_high, spec, short_rate=0.01)
    assert res_high["price"] > res_low["price"]
    assert res_low["pv_paths"].shape == (1,)
    assert np.isfinite(res_low["price"])

    # error on invalid maturity
    bad_spec = CohortLifeAnnuitySpec(issue_age=60.0, maturity_years=0)
    with pytest.raises(ValueError):
        price_cohort_life_annuity(scen_low, bad_spec, short_rate=0.01)


def test_utils_shim_exports():
    import pymort.utils as u
    from pymort.models.utils import estimate_rw_params as real

    assert u.estimate_rw_params is real
    assert callable(u._estimate_rw_params)
    assert "estimate_rw_params" in getattr(u, "__all__", [])


def _simple_scenario_N(N: int = 3, H: int = 3):
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.arange(2020, 2020 + H, dtype=int)

    q = np.array(
        [
            [0.01, 0.011, 0.012],
            [0.012, 0.013, 0.014],
        ],
        dtype=float,
    )[:, :H]
    S = survival_from_q(q)

    return MortalityScenarioSet(
        years=years,
        ages=ages,
        q_paths=np.repeat(q[None, :, :], N, axis=0),
        S_paths=np.repeat(S[None, :, :], N, axis=0),
        metadata={"source": "test"},
    )


def test_life_annuity_raises_on_q_S_shape_mismatch():
    scen = _simple_scenario()
    spec = CohortLifeAnnuitySpec(issue_age=60.0, maturity_years=3)

    scen_bad = MortalityScenarioSet(
        years=scen.years,
        ages=scen.ages,
        q_paths=scen.q_paths,
        S_paths=scen.S_paths[:, :, :2],  # mismatch
        metadata={},
    )
    with pytest.raises(ValueError):
        price_cohort_life_annuity(scen_bad, spec, short_rate=0.01)


def test_life_annuity_maturity_none_uses_full_horizon_and_errors_on_too_long():
    scen = _simple_scenario_N(N=2, H=3)

    spec_none = CohortLifeAnnuitySpec(issue_age=60.0, maturity_years=None)
    out = price_cohort_life_annuity(scen, spec_none, short_rate=0.01)
    assert out["expected_cashflows"].shape == (3,)
    assert out["pv_paths"].shape == (2,)

    spec_too_long = CohortLifeAnnuitySpec(issue_age=60.0, maturity_years=99)
    with pytest.raises(ValueError):
        price_cohort_life_annuity(scen, spec_too_long, short_rate=0.01)


def test_life_annuity_deferral_errors():
    scen = _simple_scenario_N(N=2, H=3)

    with pytest.raises(ValueError):
        price_cohort_life_annuity(
            scen,
            CohortLifeAnnuitySpec(issue_age=60.0, maturity_years=3, defer_years=-1),
            short_rate=0.01,
        )

    # defer >= H => no payments, forbidden by design
    with pytest.raises(ValueError):
        price_cohort_life_annuity(
            scen,
            CohortLifeAnnuitySpec(issue_age=60.0, maturity_years=3, defer_years=3),
            short_rate=0.01,
        )


def test_life_annuity_terminal_and_exposure_scaling_increases_price():
    scen = _simple_scenario_N(N=5, H=3)
    r = 0.01

    base = price_cohort_life_annuity(
        scen,
        CohortLifeAnnuitySpec(issue_age=60.0, maturity_years=3, payment_per_survivor=1.0),
        short_rate=r,
    )["price"]

    with_terminal = price_cohort_life_annuity(
        scen,
        CohortLifeAnnuitySpec(
            issue_age=60.0,
            maturity_years=3,
            payment_per_survivor=1.0,
            include_terminal=True,
            terminal_notional=10.0,
        ),
        short_rate=r,
    )["price"]
    assert with_terminal > base

    scaled = price_cohort_life_annuity(
        scen,
        CohortLifeAnnuitySpec(
            issue_age=60.0,
            maturity_years=3,
            payment_per_survivor=1.0,
            exposure_at_issue=2.5,
        ),
        short_rate=r,
    )["price"]
    assert np.isclose(scaled, base * 2.5, rtol=1e-12, atol=1e-12)


def test_life_annuity_discount_factors_2d_first_dim_1_repeats_and_NxH_ok():
    scen = _simple_scenario_N(N=4, H=3)
    spec = CohortLifeAnnuitySpec(issue_age=60.0, maturity_years=3, payment_per_survivor=1.0)

    df_1xH = np.array([[1.0, 0.97, 0.94]], dtype=float)
    out1 = price_cohort_life_annuity(scen, spec, discount_factors=df_1xH)
    assert out1["pv_paths"].shape == (4,)
    assert np.isfinite(out1["price"])

    df_NxH = np.tile(df_1xH, (4, 1))
    out2 = price_cohort_life_annuity(scen, spec, discount_factors=df_NxH)
    assert np.isclose(out1["price"], out2["price"], rtol=1e-12, atol=1e-12)


def test_life_annuity_discount_factors_bad_shapes_raise():
    scen = _simple_scenario_N(N=4, H=3)
    spec = CohortLifeAnnuitySpec(issue_age=60.0, maturity_years=3)

    # 2D but first dim neither 1 nor N
    bad = np.ones((2, 3), dtype=float)
    with pytest.raises(ValueError):
        price_cohort_life_annuity(scen, spec, discount_factors=bad)

    # 3D df -> should raise
    bad3 = np.ones((1, 1, 3), dtype=float)
    with pytest.raises(ValueError):
        price_cohort_life_annuity(scen, spec, discount_factors=bad3)


def test_life_annuity_non_monotonic_survival_raises():
    spec = CohortLifeAnnuitySpec(issue_age=60.0, maturity_years=3)

    # Make survival INCREASING over time for the selected age => violates monotonicity check
    scen_bad = _simple_scenario_N(N=2, H=3)
    scen_bad.S_paths[:, 0, :] = np.array([0.8, 0.85, 0.9])  # increasing
    scen_bad.q_paths = scen_bad.q_paths.copy()

    with pytest.raises(AssertionError):
        price_cohort_life_annuity(scen_bad, spec, short_rate=0.01)
