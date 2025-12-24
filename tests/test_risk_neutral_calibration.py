from __future__ import annotations

import numpy as np
import pytest

from pymort.pipeline import pricing_pipeline
from pymort.pricing.longevity_bonds import LongevityBondSpec
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
    market_prices = _price_with_lambda(
        cache, lam=lambda_true, specs=(bond_spec, swap_spec)
    )
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
    market_prices = _price_with_lambda(
        cache, lam=lambda_true, specs=(bond_spec, swap_spec)
    )
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
