from __future__ import annotations

import numpy as np
import pytest

import pymort.pricing.risk_neutral as rn
from pymort.analysis.scenario import MortalityScenarioSet
from pymort.pipeline import pricing_pipeline
from pymort.pricing.liabilities import CohortLifeAnnuitySpec
from pymort.pricing.longevity_bonds import LongevityBondSpec
from pymort.pricing.mortality_derivatives import QForwardSpec, SForwardSpec
from pymort.pricing.risk_neutral import (
    MultiInstrumentQuote,
    build_calibration_cache,
    build_scenarios_under_lambda_fast,
    calibrate_lambda_least_squares,
)
from pymort.pricing.survivor_swaps import SurvivorSwapSpec


def _toy_surface():
    ages = np.array([60.0, 61.0, 62.0], dtype=float)
    years = np.arange(2000, 2010, dtype=int)  # T=10
    A, T = ages.size, years.size
    base = 0.01
    trend = np.linspace(1.0, 0.8, T)  # mild improvement over time
    m = np.zeros((A, T), dtype=float)
    for i, age in enumerate(ages):
        m[i, :] = base * (1.0 + 0.01 * (age - ages[0])) * trend
    return ages, years, m


def _pricing_specs(maturity: int = 5):
    bond = LongevityBondSpec(
        issue_age=60.0, maturity_years=maturity, include_principal=True, notional=1.0
    )
    swap = SurvivorSwapSpec(
        age=60.0, maturity_years=maturity, payer="fixed", strike=None, notional=1.0
    )
    return bond, swap


def _make_quotes_from_prices(
    prices: dict[str, float], bond_spec: LongevityBondSpec, swap_spec: SurvivorSwapSpec
):
    return [
        MultiInstrumentQuote(
            kind="longevity_bond",
            spec=bond_spec,
            market_price=prices["bond"],
            weight=1.0,
        ),
        MultiInstrumentQuote(
            kind="survivor_swap",
            spec=swap_spec,
            market_price=prices["swap"],
            weight=1.0,
        ),
    ]


def _price_with_lambda(cache, lam, specs, short_rate: float = 0.02):
    scen_q = build_scenarios_under_lambda_fast(
        cache=cache,
        lambda_esscher=lam,
        scale_sigma=1.0,
    )
    return pricing_pipeline(
        scen_Q=scen_q, specs={"bond": specs[0], "swap": specs[1]}, short_rate=short_rate
    )


def test_self_consistency_lambda_zero():
    ages, years, m = _toy_surface()
    bond_spec, swap_spec = _pricing_specs(maturity=5)
    cache = build_calibration_cache(
        ages=ages,
        years=years,
        m=m,
        model_name="LCM2",
        B_bootstrap=4,
        n_process=12,
        horizon=6,
        seed=123,
        include_last=True,
    )
    # market generated at lambda=0
    market_prices = _price_with_lambda(cache, lam=0.0, specs=(bond_spec, swap_spec))
    quotes = _make_quotes_from_prices(market_prices, bond_spec, swap_spec)

    res = calibrate_lambda_least_squares(
        quotes=quotes,
        ages=ages,
        years=years,
        m=m,
        model_name="LCM2",
        lambda0=0.0,
        bounds=(-1.0, 1.0),
        B_bootstrap=4,
        n_process=12,
        short_rate=0.02,
        horizon=6,
        seed=123,
        include_last=True,
        verbose=0,
    )
    lam_star = float(np.asarray(res["lambda_star"]).reshape(-1)[0])
    assert abs(lam_star) < 1e-2
    assert res["success"]


def test_calibration_recovers_lambda_true_and_reduces_error():
    ages, years, m = _toy_surface()
    bond_spec, swap_spec = _pricing_specs(maturity=5)
    cache = build_calibration_cache(
        ages=ages,
        years=years,
        m=m,
        model_name="LCM2",
        B_bootstrap=6,
        n_process=20,
        horizon=6,
        seed=321,
        include_last=True,
    )
    lambda_true = 0.5
    market_prices = _price_with_lambda(cache, lam=lambda_true, specs=(bond_spec, swap_spec))
    quotes = _make_quotes_from_prices(market_prices, bond_spec, swap_spec)

    res = calibrate_lambda_least_squares(
        quotes=quotes,
        ages=ages,
        years=years,
        m=m,
        model_name="LCM2",
        lambda0=0.0,
        bounds=(-1.0, 1.0),
        B_bootstrap=6,
        n_process=20,
        short_rate=0.02,
        horizon=6,
        seed=321,
        include_last=True,
        verbose=0,
    )
    lam_star = float(np.asarray(res["lambda_star"]).reshape(-1)[0])
    assert np.isfinite(lam_star)
    # error reduction vs lambda0 baseline
    baseline_prices = _price_with_lambda(cache, lam=0.0, specs=(bond_spec, swap_spec))
    baseline_err = np.sqrt(
        np.mean(
            [
                (baseline_prices["bond"] - market_prices["bond"]) ** 2,
                (baseline_prices["swap"] - market_prices["swap"]) ** 2,
            ]
        )
    )
    fitted = res["fitted_prices"]
    market = res["market_prices"]
    calibrated_err = np.sqrt(np.mean((fitted - market) ** 2))
    assert calibrated_err < baseline_err
    assert calibrated_err < baseline_err * 0.75  # meaningful improvement
    assert res["success"]


def test_lambda_bounds_respected():
    ages, years, m = _toy_surface()
    bond_spec, swap_spec = _pricing_specs(maturity=4)
    cache = build_calibration_cache(
        ages=ages,
        years=years,
        m=m,
        model_name="LCM2",
        B_bootstrap=4,
        n_process=15,
        horizon=5,
        seed=99,
        include_last=True,
    )
    lambda_true = 0.5
    market_prices = _price_with_lambda(cache, lam=lambda_true, specs=(bond_spec, swap_spec))
    quotes = _make_quotes_from_prices(market_prices, bond_spec, swap_spec)
    bounds = (-0.1, 0.1)
    res = calibrate_lambda_least_squares(
        quotes=quotes,
        ages=ages,
        years=years,
        m=m,
        model_name="LCM2",
        lambda0=0.0,
        bounds=bounds,
        B_bootstrap=4,
        n_process=15,
        short_rate=0.02,
        horizon=5,
        seed=99,
        include_last=True,
        verbose=0,
    )
    lam_star = float(np.asarray(res["lambda_star"]).reshape(-1)[0])
    assert bounds[0] - 1e-12 <= lam_star <= bounds[1] + 1e-12
    assert res["success"]


# ============================================================
# Coverage add-ons for risk_neutral: validation branches & kind dispatch
# ============================================================


def _simple_scen_for_pricing():
    # Tiny scenario set (N=2, A=1, H=2)
    years = np.array([2020, 2021], dtype=int)
    ages = np.array([60.0], dtype=float)
    q = np.array([[[0.01, 0.011]], [[0.012, 0.013]]], dtype=float)
    S = np.cumprod(1.0 - q, axis=2)
    return MortalityScenarioSet(years=years, ages=ages, q_paths=q, S_paths=S, metadata={})


# -------------------------
# Part 1: esscher_shift_normal_rw shape/neg checks
# -------------------------
def test_esscher_shift_normal_rw_sigma_length_mismatch_raises():
    # k=3 from mu_P, sigma_P length=2 -> should raise
    mu_P = np.array([0.1, 0.2, 0.3], dtype=float)
    sigma_P = np.array([0.1, 0.2], dtype=float)
    with pytest.raises(ValueError, match=r"sigma_P must have same length as mu_P"):
        rn.esscher_shift_normal_rw(mu_P=mu_P, sigma_P=sigma_P, lambda_esscher=0.0)


def test_esscher_shift_normal_rw_lambda_broadcast_and_bad_length_raises():
    mu_P = np.array([0.1, 0.2, 0.3], dtype=float)
    sigma_P = np.array([0.1, 0.2, 0.3], dtype=float)

    # lambda scalar -> broadcast to k
    out = rn.esscher_shift_normal_rw(mu_P=mu_P, sigma_P=sigma_P, lambda_esscher=0.5)
    assert np.asarray(out.mu_Q).shape == (3,)

    # bad lambda length
    with pytest.raises(ValueError, match=r"lambda_esscher must have length 1 or 3"):
        rn.esscher_shift_normal_rw(mu_P=mu_P, sigma_P=sigma_P, lambda_esscher=[0.1, 0.2])


def test_esscher_shift_normal_rw_negative_sigma_raises():
    mu_P = np.array([0.1, 0.2, 0.3], dtype=float)
    sigma_P = np.array([0.1, -0.2, 0.3], dtype=float)
    with pytest.raises(ValueError, match=r"sigma_P entries must be non-negative"):
        rn.esscher_shift_normal_rw(mu_P=mu_P, sigma_P=sigma_P, lambda_esscher=0.0)


# -------------------------
# Part 2: risk_neutral_from_cbdm7 expects estimate_rw length 6
# -------------------------
def test_risk_neutral_from_cbdm7_raises_if_estimate_rw_wrong_length():
    class DummyCBDM7:
        def estimate_rw(self):
            return (0.1, 0.2)  # wrong length

    with pytest.raises(RuntimeError, match=r"must return 6 values"):
        rn.risk_neutral_from_cbdm7(DummyCBDM7(), lambda_esscher=0.0)  # type: ignore[arg-type]


def test_risk_neutral_from_cbdm7_calls_esscher_with_3dim_vectors(
    monkeypatch: pytest.MonkeyPatch,
):
    class DummyCBDM7:
        def estimate_rw(self):
            # mu1,s1, mu2,s2, mu3,s3
            return (0.1, 0.01, 0.2, 0.02, 0.3, 0.03)

    captured = {}

    def fake_esscher_shift_normal_rw(mu_P, sigma_P, lambda_esscher):
        captured["mu_P"] = np.asarray(mu_P)
        captured["sigma_P"] = np.asarray(sigma_P)
        captured["lambda"] = lambda_esscher
        return type(
            "EsscherRes",
            (),
            {"mu_Q": mu_P, "sigma_Q": sigma_P, "lambda_esscher": lambda_esscher},
        )()

    monkeypatch.setattr(rn, "esscher_shift_normal_rw", fake_esscher_shift_normal_rw)

    res = rn.risk_neutral_from_cbdm7(DummyCBDM7(), lambda_esscher=0.5)  # type: ignore[arg-type]
    assert np.asarray(captured["mu_P"]).shape == (3,)
    assert np.asarray(captured["sigma_P"]).shape == (3,)
    assert captured["lambda"] == 0.5
    assert hasattr(res, "mu_Q")


# -------------------------
# Part 3: build_calibration_cache branches (LCM2/CBDM7/else)
# -------------------------
def test_build_calibration_cache_rejects_unknown_model_name():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2000, 2001], dtype=int)
    m = np.full((2, 2), 0.01, dtype=float)
    with pytest.raises(ValueError, match="model_name must be 'LCM2' or 'CBDM7'"):
        build_calibration_cache(
            ages=ages,
            years=years,
            m=m,
            model_name="NOPE",
            B_bootstrap=2,
            n_process=2,
            horizon=2,
            seed=0,
            include_last=False,
        )


def test_build_calibration_cache_calls_correct_fit_for_lcm2_and_cbdm7(
    monkeypatch: pytest.MonkeyPatch,
):
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2000, 2001, 2002], dtype=int)
    m = np.full((2, 3), 0.01, dtype=float)

    class StopAfterFit(RuntimeError):
        pass

    called = {"m_to_q": 0, "lcm2_fit": 0, "cbdm7_fit": 0}

    # Patch m_to_q to verify it's called for CBDM7 branch
    def fake_m_to_q(x):
        called["m_to_q"] += 1
        return x * 0.9

    monkeypatch.setattr(rn, "m_to_q", fake_m_to_q)

    class DummyLCM2:
        def fit(self, m_in, ages_in, years_in):
            called["lcm2_fit"] += 1
            # stop here so we don't reach bootstrap/model-class checks
            raise StopAfterFit("LCM2 fit reached")

    class DummyCBDM7:
        def fit(self, q_in, ages_in, years_in):
            called["cbdm7_fit"] += 1
            # stop here so we don't reach downstream logic
            raise StopAfterFit("CBDM7 fit reached")

    monkeypatch.setattr(rn, "LCM2", DummyLCM2)
    monkeypatch.setattr(rn, "CBDM7", DummyCBDM7)

    # ---- LCM2 branch ----
    with pytest.raises(StopAfterFit, match="LCM2 fit reached"):
        build_calibration_cache(
            ages=ages,
            years=years,
            m=m,
            model_name="LCM2",
            B_bootstrap=2,
            n_process=2,
            horizon=2,
            seed=0,
            include_last=False,
        )
    assert called["lcm2_fit"] == 1
    assert called["m_to_q"] == 0  # not used in LCM2 branch

    # ---- CBDM7 branch ----
    with pytest.raises(StopAfterFit, match="CBDM7 fit reached"):
        build_calibration_cache(
            ages=ages,
            years=years,
            m=m,
            model_name="CBDM7",
            B_bootstrap=2,
            n_process=2,
            horizon=2,
            seed=0,
            include_last=False,
        )
    assert called["m_to_q"] == 1  # CBDM7 branch converts m -> q
    assert called["cbdm7_fit"] == 1


# -------------------------
# Part 4: apply_kappa_drift_shock branches
# -------------------------
def test_apply_kappa_drift_shock_none_returns_same_values():
    mu = np.array([0.1, 0.2], dtype=float)
    out = rn.apply_kappa_drift_shock(mu, shock=None)
    assert np.allclose(out, mu)


def test_apply_kappa_drift_shock_broadcast_and_modes_and_errors():
    mu = np.array([0.1, 0.2, 0.3], dtype=float)

    # scalar shock broadcast
    out_add = rn.apply_kappa_drift_shock(mu, shock=0.1, mode="additive")
    assert out_add.shape == (3,)

    out_mul = rn.apply_kappa_drift_shock(mu, shock=0.1, mode="multiplicative")
    assert np.allclose(out_mul, mu * 1.1)

    # bad shock length
    with pytest.raises(ValueError, match=r"drift shock must have length 1 or 3"):
        rn.apply_kappa_drift_shock(mu, shock=[0.1, 0.2], mode="additive")

    # bad mode
    with pytest.raises(ValueError, match=r"mode must be 'additive' or 'multiplicative'"):
        rn.apply_kappa_drift_shock(mu, shock=0.1, mode="nope")

    # non-finite after shock -> error
    with pytest.raises(ValueError, match="Non-finite mu_Q"):
        rn.apply_kappa_drift_shock(mu, shock=np.nan, mode="additive")


# -------------------------
# Part 5: apply_cohort_trend_shock_to_qpaths branches
# -------------------------
def test_apply_cohort_trend_shock_to_qpaths_input_checks_and_unknown_type():
    scen = _simple_scen_for_pricing()
    q = np.asarray(scen.q_paths)

    # q_paths must be 3D
    with pytest.raises(ValueError, match="q_paths must have shape"):
        rn.apply_cohort_trend_shock_to_qpaths(q[0], ages=scen.ages, years=scen.years)  # 2D

    # ages length mismatch
    with pytest.raises(ValueError, match="ages length must match"):
        rn.apply_cohort_trend_shock_to_qpaths(q, ages=np.array([60.0, 61.0]), years=scen.years)

    # years length mismatch
    with pytest.raises(ValueError, match="years length must match"):
        rn.apply_cohort_trend_shock_to_qpaths(q, ages=scen.ages, years=np.array([2020, 2021, 2022]))

    # unknown shock type
    with pytest.raises(ValueError, match="Unknown cohort shock_type"):
        rn.apply_cohort_trend_shock_to_qpaths(
            q, ages=scen.ages, years=scen.years, shock_type="nope"
        )


def test_apply_cohort_trend_shock_to_qpaths_all_branches_and_validate_q_called(
    monkeypatch: pytest.MonkeyPatch,
):
    scen = _simple_scen_for_pricing()
    q = np.asarray(scen.q_paths, dtype=float)

    called = {"validate": 0}
    monkeypatch.setattr(
        rn,
        "validate_q",
        lambda x: called.__setitem__("validate", called["validate"] + 1),
    )

    out1 = rn.apply_cohort_trend_shock_to_qpaths(
        q,
        ages=scen.ages,
        years=scen.years,
        shock_type="cohort_improvement",
        magnitude=0.02,
        cap=0.5,
    )
    assert out1.shape == q.shape
    assert called["validate"] == 1

    out2 = rn.apply_cohort_trend_shock_to_qpaths(
        q,
        ages=scen.ages,
        years=scen.years,
        shock_type="cohort_deterioration",
        magnitude=0.02,
        cap=0.5,
    )
    assert out2.shape == q.shape
    assert called["validate"] == 2

    out3 = rn.apply_cohort_trend_shock_to_qpaths(
        q,
        ages=scen.ages,
        years=scen.years,
        shock_type="cohort_step",
        magnitude=0.02,
        cap=0.5,
    )
    assert out3.shape == q.shape
    assert called["validate"] == 3


# -------------------------
# Part 6: _price_from_scen_set all kind branches + unknown
# -------------------------
def test_price_from_scen_set_dispatches_all_kinds(monkeypatch: pytest.MonkeyPatch):
    scen = _simple_scen_for_pricing()

    # Patch each pricer to return {"price": ...}
    monkeypatch.setattr(rn, "price_simple_longevity_bond", lambda **_k: {"price": 1.1})
    monkeypatch.setattr(rn, "price_survivor_swap", lambda **_k: {"price": 2.2})
    monkeypatch.setattr(rn, "price_s_forward", lambda **_k: {"price": 3.3})
    monkeypatch.setattr(rn, "price_q_forward", lambda **_k: {"price": 4.4})
    monkeypatch.setattr(rn, "price_cohort_life_annuity", lambda **_k: {"price": 5.5})

    bond = LongevityBondSpec(issue_age=60.0, maturity_years=2, include_principal=True)
    swap = SurvivorSwapSpec(age=60.0, maturity_years=2, payer="fixed")
    sf = SForwardSpec(age=60.0, maturity_years=2, strike=0.1)
    qf = QForwardSpec(age=60.0, maturity_years=2, strike=0.1)
    ann = CohortLifeAnnuitySpec(issue_age=60.0, maturity_years=2)

    quotes = [
        rn.MultiInstrumentQuote(kind="longevity_bond", spec=bond, market_price=1.0, weight=1.0),
        rn.MultiInstrumentQuote(kind="survivor_swap", spec=swap, market_price=1.0, weight=1.0),
        rn.MultiInstrumentQuote(kind="s_forward", spec=sf, market_price=1.0, weight=1.0),
        rn.MultiInstrumentQuote(kind="q_forward", spec=qf, market_price=1.0, weight=1.0),
        rn.MultiInstrumentQuote(kind="life_annuity", spec=ann, market_price=1.0, weight=1.0),
    ]
    got = [rn._price_from_scen_set(scen, q, short_rate=0.02) for q in quotes]
    assert got == [1.1, 2.2, 3.3, 4.4, 5.5]

    with pytest.raises(ValueError, match=r"Unknown quote.kind"):
        rn._price_from_scen_set(
            scen,
            rn.MultiInstrumentQuote(kind="nope", spec=bond, market_price=1.0, weight=1.0),
            short_rate=0.02,
        )


# -------------------------
# Part 7: calibrate_lambda_least_squares input validations (early branches)
# -------------------------
def test_calibrate_lambda_least_squares_validations(monkeypatch: pytest.MonkeyPatch):
    ages = np.array([60.0], dtype=float)
    years = np.array([2000, 2001], dtype=int)
    m = np.full((1, 2), 0.01, dtype=float)
    bond = LongevityBondSpec(issue_age=60.0, maturity_years=1, include_principal=True)

    # empty quotes
    with pytest.raises(ValueError, match="quotes must be a non-empty iterable"):
        calibrate_lambda_least_squares(
            quotes=[],
            ages=ages,
            years=years,
            m=m,
            model_name="LCM2",
        )

    # invalid bounds
    q_ok = [rn.MultiInstrumentQuote(kind="longevity_bond", spec=bond, market_price=1.0, weight=1.0)]
    with pytest.raises(ValueError, match="bounds must be finite"):
        calibrate_lambda_least_squares(
            quotes=q_ok,
            ages=ages,
            years=years,
            m=m,
            model_name="LCM2",
            bounds=(1.0, 1.0),
        )

    # invalid weights (negative)
    q_bad_w = [
        rn.MultiInstrumentQuote(kind="longevity_bond", spec=bond, market_price=1.0, weight=-1.0)
    ]
    with pytest.raises(ValueError, match="All quote weights must be finite and >= 0"):
        calibrate_lambda_least_squares(
            quotes=q_bad_w,
            ages=ages,
            years=years,
            m=m,
            model_name="LCM2",
        )

    # all weights zero
    q_zero_w = [
        rn.MultiInstrumentQuote(kind="longevity_bond", spec=bond, market_price=1.0, weight=0.0)
    ]
    with pytest.raises(ValueError, match="At least one quote must have a strictly positive weight"):
        calibrate_lambda_least_squares(
            quotes=q_zero_w,
            ages=ages,
            years=years,
            m=m,
            model_name="LCM2",
        )

    # non-finite market price
    q_bad_mkt = [
        rn.MultiInstrumentQuote(kind="longevity_bond", spec=bond, market_price=np.nan, weight=1.0)
    ]
    with pytest.raises(ValueError, match="All market_price values must be finite"):
        calibrate_lambda_least_squares(
            quotes=q_bad_mkt,
            ages=ages,
            years=years,
            m=m,
            model_name="LCM2",
        )

    # Cover the branch where horizon is None -> h_eff=len(years), and build_calibration_cache called with that.
    captured = {}

    def fake_build_calibration_cache(**kwargs):
        captured["horizon"] = kwargs["horizon"]
        return {"cache": True}

    # Then stop execution early after cache is created (to avoid heavy optimizer)
    def fake_least_squares(*_a, **_k):
        raise RuntimeError("STOP_AFTER_CACHE")

    monkeypatch.setattr(rn, "build_calibration_cache", fake_build_calibration_cache)
    monkeypatch.setattr(rn, "least_squares", fake_least_squares)

    with pytest.raises(RuntimeError, match="STOP_AFTER_CACHE"):
        calibrate_lambda_least_squares(
            quotes=q_ok,
            ages=ages,
            years=years,
            m=m,
            model_name="LCM2",
            horizon=None,  # triggers h_eff = len(years)
            bounds=(-1.0, 1.0),
        )

    assert captured["horizon"] == len(years)


# -------------------------
# Part 8: calibrate_market_price_of_longevity_risk returns lambda_star as ndarray
# -------------------------
def test_calibrate_market_price_of_longevity_risk_returns_lambda_star(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_calibrate_lambda_least_squares(**_k):
        return {"lambda_star": [0.12, 0.34]}

    monkeypatch.setattr(rn, "calibrate_lambda_least_squares", fake_calibrate_lambda_least_squares)

    ages = np.array([60.0], dtype=float)
    years = np.array([2000, 2001], dtype=int)
    m = np.full((1, 2), 0.01, dtype=float)
    bond = LongevityBondSpec(issue_age=60.0, maturity_years=1, include_principal=True)

    out = rn.calibrate_market_price_of_longevity_risk(
        quotes=[
            rn.MultiInstrumentQuote(kind="longevity_bond", spec=bond, market_price=1.0, weight=1.0)
        ],
        ages=ages,
        years=years,
        m=m,
        model_name="LCM2",
    )
    out = np.asarray(out, dtype=float)
    assert out.shape == (2,)
    assert np.allclose(out, np.array([0.12, 0.34], dtype=float))
