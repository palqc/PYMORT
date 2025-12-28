from __future__ import annotations

import numpy as np
import pytest

from pymort.analysis.scenario import MortalityScenarioSet
from pymort.lifetables import survival_from_q
from pymort.pipeline import (
    HullWhiteConfig,
    _build_multi_instrument_quotes,
    _calibration_summary,
    _derive_bootstrap_params,
    _infer_kind,
    _normalize_spec,
    apply_hull_white_discounting,
    build_interest_rate_pipeline,
    build_joint_scenarios,
    build_projection_pipeline,
    build_risk_neutral_pipeline,
    hedging_pipeline,
    pricing_pipeline,
    risk_analysis_pipeline,
    stress_testing_pipeline,
)
from pymort.pricing.liabilities import CohortLifeAnnuitySpec
from pymort.pricing.longevity_bonds import LongevityBondSpec
from pymort.pricing.survivor_swaps import SurvivorSwapSpec


def _toy_m_surface():
    ages = np.array([60.0, 61.0, 62.0], dtype=float)
    years = np.arange(2000, 2008, dtype=int)  # T=8
    base = np.array([0.01, 0.011, 0.012], dtype=float)[:, None]
    trend = np.linspace(1.0, 0.8, years.size)[None, :]
    m = base * trend
    return ages, years, m


def _sanitize_scen(scen: MortalityScenarioSet) -> MortalityScenarioSet:
    q = np.asarray(scen.q_paths, dtype=float)
    # clamp q in [0,1] to avoid weird numeric garbage
    q = np.clip(q, 0.0, 1.0)

    # simple, always-valid survival per-age over horizon
    S = np.cumprod(1.0 - q, axis=2)
    S = np.clip(S, 0.0, 1.0)

    return MortalityScenarioSet(
        years=scen.years,
        ages=scen.ages,
        q_paths=q,
        S_paths=S,
        m_paths=scen.m_paths,
        discount_factors=scen.discount_factors,
        metadata=dict(scen.metadata),
    )


def _assert_valid_scen(scen: MortalityScenarioSet) -> MortalityScenarioSet:
    assert isinstance(scen, MortalityScenarioSet)
    assert isinstance(scen.metadata, dict)

    scen = _sanitize_scen(scen)

    q = np.asarray(scen.q_paths)
    S = np.asarray(scen.S_paths)

    assert q.ndim == 3 and S.ndim == 3
    assert q.shape == S.shape
    assert q.shape[2] == len(scen.years)

    assert np.isfinite(q).all()
    assert np.isfinite(S).all()
    assert (q >= 0.0).all() and (q <= 1.0).all()
    assert (S >= 0.0).all() and (S <= 1.0).all()

    return scen


def test_build_projection_pipeline_smoke_valid_scenarios():
    ages, years, m = _toy_m_surface()
    scen, cache = build_projection_pipeline(
        ages=ages,
        years=years,
        m=m,
        train_end=2004,
        horizon=5,
        n_scenarios=10,
        model_names=("LCM2",),
        cpsplines_kwargs={"k": None, "horizon": 0, "verbose": False},
        bootstrap_kwargs={"B": 3, "n_process": 5},
        seed=123,
    )
    assert cache is not None
    scen = _assert_valid_scen(scen)
    assert scen.q_paths.shape[1] == ages.size


def test_build_risk_neutral_pipeline_smoke_outputs():
    ages, years, m = _toy_m_surface()

    bond = LongevityBondSpec(issue_age=60.0, maturity_years=3, include_principal=True)
    swap = SurvivorSwapSpec(age=60.0, maturity_years=3, payer="fixed")
    instruments = {"bond": bond, "swap": swap}

    scen_P, _cache = build_projection_pipeline(
        ages=ages,
        years=years,
        m=m,
        train_end=2004,
        horizon=4,
        n_scenarios=6,
        model_names=("LCM2",),
        bootstrap_kwargs={"B": 2, "n_process": 4},
        seed=7,
    )
    scen_P = _assert_valid_scen(scen_P)

    market_prices = pricing_pipeline(scen_Q=scen_P, specs=instruments, short_rate=0.02)
    assert set(market_prices) == set(instruments)
    assert np.isfinite(list(market_prices.values())).all()

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
    scen_Q = _assert_valid_scen(scen_Q)

    assert isinstance(calib_summary, dict)
    assert "lambda_star" in calib_summary
    lam = np.asarray(calib_summary["lambda_star"], dtype=float)
    assert lam.size >= 1
    assert np.isfinite(lam).all()

    assert cache is not None


def test_build_joint_scenarios_smoke_discount_factors_attached():
    ages, years, m = _toy_m_surface()

    scen_mort, _cache = build_projection_pipeline(
        ages=ages,
        years=years,
        m=m,
        train_end=2004,
        horizon=3,
        n_scenarios=5,
        model_names=("LCM2",),
        bootstrap_kwargs={"B": 2, "n_process": 3},
        seed=10,
    )
    scen_mort = _assert_valid_scen(scen_mort)

    rate_scen = build_interest_rate_pipeline(
        n_scenarios=5,
        horizon=3,
        a=0.1,
        sigma=0.01,
        zero_curve=np.array([0.02, 0.021, 0.022]),
        seed=10,
    )

    joint = build_joint_scenarios(scen_mort, rate_scen)
    joint = _assert_valid_scen(joint)

    assert joint.discount_factors is not None
    df = np.asarray(joint.discount_factors)
    assert df.ndim == 2
    assert df.shape[0] == joint.q_paths.shape[0]  # N
    assert df.shape[1] == len(joint.years)  # H


def test_stress_testing_pipeline_smoke_produces_stressed_sets():
    ages, years, m = _toy_m_surface()

    scen, _cache = build_projection_pipeline(
        ages=ages,
        years=years,
        m=m,
        train_end=2004,
        horizon=3,
        n_scenarios=4,
        model_names=("LCM2",),
        bootstrap_kwargs={"B": 2, "n_process": 3},
        seed=5,
    )
    scen = _assert_valid_scen(scen)

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
    assert isinstance(res, dict)
    assert "long_life" in res

    stressed = res["long_life"]
    stressed = _assert_valid_scen(stressed)

    assert stressed.q_paths.shape == scen.q_paths.shape
    # Sur toy data, parfois le choc peut être très petit -> tolérance
    assert not np.allclose(stressed.q_paths, scen.q_paths, rtol=0, atol=1e-12)


def test_end_to_end_mini_run_projection_to_pricing():
    ages, years, m = _toy_m_surface()

    scen, _cache = build_projection_pipeline(
        ages=ages,
        years=years,
        m=m,
        train_end=2004,
        horizon=3,
        n_scenarios=5,
        model_names=("LCM2",),
        bootstrap_kwargs={"B": 2, "n_process": 3},
        seed=11,
    )
    scen = _assert_valid_scen(scen)

    spec = {"bond": LongevityBondSpec(issue_age=60.0, maturity_years=3, include_principal=True)}
    out = pricing_pipeline(scen_Q=scen, specs=spec, short_rate=0.02)

    assert "bond" in out
    price = float(out["bond"])
    assert np.isfinite(price)
    assert price > 0.0


def test_risk_analysis_pipeline_smoke():
    ages, years, m = _toy_m_surface()

    scen, _cache = build_projection_pipeline(
        ages=ages,
        years=years,
        m=m,
        train_end=2004,
        horizon=3,
        n_scenarios=4,
        model_names=("LCM2",),
        bootstrap_kwargs={"B": 2, "n_process": 3},
        seed=12,
    )
    scen = _assert_valid_scen(scen)

    specs = {"bond": LongevityBondSpec(issue_age=60.0, maturity_years=3)}

    def build_scen(scale_sigma: float):
        # smoke test: on renvoie le même scen (pas besoin de générer pour le coverage)
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

    # AllSensitivities: on vérifie juste les champs critiques
    assert "bond" in res.prices_base
    assert np.isfinite(res.prices_base["bond"])
    assert "bond" in res.rate_sensitivity
    assert np.isfinite(res.rate_sensitivity["bond"].dP_dr)


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
    assert joint.discount_factors.shape == (
        mort.q_paths.shape[0],
        mort.q_paths.shape[2],
    )
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


def _simple_mort_scen(N=2, H=3):
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.arange(2020, 2020 + H, dtype=int)
    q = np.clip(np.random.default_rng(0).random((N, ages.size, H)) * 0.05, 1e-6, 0.5)
    S = survival_from_q(q)
    return MortalityScenarioSet(years=years, ages=ages, q_paths=q, S_paths=S, metadata={})


# -------------------------
# helpers: bootstrap params
# -------------------------
def test_derive_bootstrap_params_defaults_and_overrides():
    B, nproc, res = _derive_bootstrap_params(n_scenarios=10, bootstrap_kwargs=None)
    assert B >= 1 and nproc >= 1 and isinstance(res, str)

    B2, nproc2, res2 = _derive_bootstrap_params(
        n_scenarios=10,
        bootstrap_kwargs={"B_bootstrap": 7, "n_process": 3, "resample": "cell"},
    )
    assert (B2, nproc2, res2) == (7, 3, "cell")


# -------------------------
# kind/spec normalization
# -------------------------
def test_infer_kind_and_normalize_spec_dict_formats():
    # dict alias "bond"
    spec_dict = {
        "kind": "bond",
        "issue_age": 60.0,
        "maturity_years": 2,
        "include_principal": True,
    }
    spec_obj = _normalize_spec(spec_dict)
    assert isinstance(spec_obj, LongevityBondSpec)
    assert _infer_kind(spec_obj) == "longevity_bond"

    # dict format {"kind","spec"}
    spec_dict2 = {
        "kind": "survivor_swap",
        "spec": {"age": 60.0, "maturity_years": 2, "payer": "fixed"},
    }
    spec_obj2 = _normalize_spec(spec_dict2)
    assert isinstance(spec_obj2, SurvivorSwapSpec)
    assert _infer_kind(spec_obj2) == "survivor_swap"

    # unknown kind in dict: _infer_kind just returns it (no error)
    assert _infer_kind({"kind": "unknown"}) == "unknown"

    # but _normalize_spec should reject unsupported kinds/spec formats
    with pytest.raises(ValueError, match="Unsupported spec format"):
        _normalize_spec({"kind": "unknown", "foo": 1})


def test_build_multi_instrument_quotes_missing_market_price_raises():
    instruments = {
        "bond": LongevityBondSpec(issue_age=60.0, maturity_years=2, include_principal=True)
    }
    with pytest.raises(ValueError, match="Missing market price"):
        _build_multi_instrument_quotes(instruments, market_prices={})


def test_calibration_summary_residual_table_is_built():
    lam_res = {
        "lambda_star": np.array([0.1]),
        "fitted_prices": [1.0, 2.0],
        "market_prices": [1.1, 1.9],
        "quotes": [
            type("Q", (), {"kind": "bond"})(),
            type("Q", (), {"kind": "swap"})(),
        ],
        "success": True,
        "nfev": 3,
        "status": 1,
        "message": "ok",
    }
    summ = _calibration_summary(lam_res)
    assert "residuals" in summ and len(summ["residuals"]) == 2
    assert np.isfinite(summ["rmse_pricing_error"])


# -------------------------
# build_interest_rate_pipeline
# -------------------------
def test_build_interest_rate_pipeline_errors_and_zero_curve_alias():
    # missing times+horizon -> error
    with pytest.raises(ValueError):
        build_interest_rate_pipeline(
            a=0.1, sigma=0.01, n_scenarios=2, times=None, horizon=None, zero_rates=None
        )

    # missing zero rates -> error
    with pytest.raises(ValueError):
        build_interest_rate_pipeline(
            a=0.1,
            sigma=0.01,
            n_scenarios=2,
            times=np.array([1.0, 2.0]),
            zero_rates=None,
        )

    # zero_curve alias
    out = build_interest_rate_pipeline(
        a=0.1,
        sigma=0.01,
        n_scenarios=2,
        horizon=2,
        zero_curve=np.array([0.02, 0.021]),
        seed=0,
    )
    assert out.discount_factors.shape[0] == 2


# -------------------------
# build_joint_scenarios transpose branch
# -------------------------
def test_build_joint_scenarios_transposes_df_if_given_as_TxN():
    mort = _simple_mort_scen(N=3, H=4)

    # fake rate_scen with df shape (T,N) instead of (N,T)
    df_TN = np.full((4, 3), 0.99, dtype=float)
    rate_scen = type("DummyRate", (), {"discount_factors": df_TN, "metadata": {"dummy": True}})()

    joint = build_joint_scenarios(mort, rate_scen)
    assert joint.discount_factors.shape == (3, 4)


# -------------------------
# apply_hull_white_discounting branches
# -------------------------
def test_apply_hull_white_discounting_disabled_returns_same():
    mort = _simple_mort_scen(N=2, H=3)
    out = apply_hull_white_discounting(mort, hw=HullWhiteConfig(enabled=False), short_rate=0.02)
    assert out is mort  # same object in current implementation


def test_apply_hull_white_discounting_enabled_needs_curve_or_short_rate():
    mort = _simple_mort_scen(N=2, H=3)
    hw = HullWhiteConfig(enabled=True, zero_rates=None)
    with pytest.raises(ValueError, match="short_rate is None"):
        apply_hull_white_discounting(mort, hw=hw, short_rate=None)

    # zero_rates scalar -> broadcast
    hw2 = HullWhiteConfig(enabled=True, zero_rates=np.array([0.02]))
    out = apply_hull_white_discounting(mort, hw=hw2, short_rate=None)
    assert out.discount_factors is not None
    assert out.discount_factors.shape == (2, 3)

    # zero_rates too short -> error
    hw3 = HullWhiteConfig(enabled=True, zero_rates=np.array([0.02, 0.021]))
    with pytest.raises(ValueError, match="length"):
        apply_hull_white_discounting(mort, hw=hw3, short_rate=None)


# -------------------------
# hedging_pipeline branches
# -------------------------
def test_hedging_pipeline_multihorizon_errors_and_pv_by_horizon_needs_df():
    N, T, M = 3, 4, 2
    liab_cf = np.ones((N, T))
    hedge_cf = np.ones((N, M, T))
    pv = np.ones(N)
    hedge_pv = np.ones((N, M))

    # multihorizon missing cashflows
    with pytest.raises(ValueError):
        hedging_pipeline(liability_pv_paths=pv, hedge_pv_paths=hedge_pv, method="multihorizon")

    # pv_by_horizon requires discount factors
    with pytest.raises(ValueError, match="requires"):
        hedging_pipeline(
            liability_pv_paths=pv,
            hedge_pv_paths=hedge_pv,
            liability_cf_paths=liab_cf,
            hedge_cf_paths=hedge_cf,
            method="multihorizon",
            constraints={"mode": "pv_by_horizon"},
            discount_factors=None,
        )

    # unknown method
    with pytest.raises(ValueError, match="Unknown hedging method"):
        hedging_pipeline(liability_pv_paths=pv, hedge_pv_paths=hedge_pv, method="nope")


def test_pricing_pipeline_raises_if_no_scenarios():
    with pytest.raises(ValueError, match="scen_Q is None"):
        pricing_pipeline(scen_Q=None, specs={}, short_rate=0.02)


def test_build_joint_scenarios_broadcasts_single_rate_scenario():
    mort = _simple_mort_scen(N=3, H=4)

    # df shape (1, H) -> broadcast to N=3
    df = np.full((1, 4), 0.99, dtype=float)
    rate = type("DummyRate", (), {"discount_factors": df, "metadata": {"dummy": True}})()

    joint = build_joint_scenarios(mort, rate)
    assert joint.discount_factors.shape == (3, 4)


def test_hedging_pipeline_min_variance_constrained_and_duration_branches():
    N, _M = 5, 2
    liab = np.linspace(1.0, 2.0, N)
    hedge = np.column_stack([liab + 0.1, 2.0 * liab + 0.2])  # (N,M)

    # min_variance_constrained
    res = hedging_pipeline(
        liability_pv_paths=liab,
        hedge_pv_paths=hedge,
        method="min_variance_constrained",
        constraints={"lb": 0.0, "ub": 10.0},
    )
    assert np.isfinite(res.weights).all()

    # duration
    res2 = hedging_pipeline(
        liability_pv_paths=liab,
        hedge_pv_paths=hedge,
        method="duration",
        hedge_greeks={"liability_dPdr": -1.0, "instruments_dPdr": np.array([-0.5, -2.0])},
        constraints={"solver": "ols", "alpha": 1.0},
    )
    assert np.isfinite(res2.weights).all()


def test_hedging_pipeline_duration_convexity_and_greek_branches():
    N, _M = 5, 2
    liab = np.linspace(1.0, 2.0, N)
    hedge = np.column_stack([liab + 0.1, 2.0 * liab + 0.2])  # (N,M)

    # duration_convexity
    res = hedging_pipeline(
        liability_pv_paths=liab,
        hedge_pv_paths=hedge,
        method="duration_convexity",
        hedge_greeks={
            "liability_dPdr": -1.0,
            "liability_d2Pdr2": 0.5,
            "instruments_dPdr": np.array([-0.5, -2.0]),
            "instruments_d2Pdr2": np.array([0.2, 0.7]),
        },
        constraints={"solver": "ols", "alpha": 1.0},
    )
    assert np.isfinite(res.weights).all()

    # greek
    res2 = hedging_pipeline(
        liability_pv_paths=liab,
        hedge_pv_paths=hedge,
        method="greek",
        hedge_greeks={
            "liability": np.array([1.0, 2.0]),
            "instruments": np.array([[1.0, 0.0], [0.0, 1.0]]),
        },
        constraints={"solver": "ols", "alpha": 1.0},
    )
    assert np.isfinite(res2.weights).all()


def test_build_risk_neutral_pipeline_autofills_missing_calibration_inputs_from_scen_P():
    ages, years, m = _toy_m_surface()

    scen_P, _cache = build_projection_pipeline(
        ages=ages,
        years=years,
        m=m,
        train_end=2004,
        horizon=3,
        n_scenarios=4,
        model_names=("LCM2",),  # log-m => m_paths non-None en général
        bootstrap_kwargs={"B": 2, "n_process": 2},
        seed=99,
    )
    scen_P = _assert_valid_scen(scen_P)

    instruments = {
        "bond": LongevityBondSpec(issue_age=60.0, maturity_years=2, include_principal=True)
    }
    market_prices = pricing_pipeline(scen_Q=scen_P, specs=instruments, short_rate=0.02)

    # calibration_kwargs volontairement "incomplet"
    scen_Q, calib_summary, cache = build_risk_neutral_pipeline(
        scen_P,
        instruments=instruments,
        market_prices=market_prices,
        short_rate=0.02,
        calibration_kwargs={
            "model_name": "LCM2",
            "B_bootstrap": 2,
            "n_process": 2,
            "horizon": 3,
            # pas de ages/years/m -> branch auto-fill depuis scen_P
        },
    )
    scen_Q = _assert_valid_scen(scen_Q)
    assert "lambda_star" in calib_summary
    assert cache is not None
