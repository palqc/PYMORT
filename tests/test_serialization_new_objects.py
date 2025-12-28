from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from pymort.interest_rates.hull_white import (
    build_interest_rate_scenarios,
    load_ir_scenarios_npz,
    save_ir_scenarios_npz,
)
from pymort.pricing.liabilities import CohortLifeAnnuitySpec
from pymort.pricing.longevity_bonds import LongevityBondSpec
from pymort.pricing.mortality_derivatives import QForwardSpec, SForwardSpec
from pymort.pricing.risk_neutral import (
    MultiInstrumentQuote,
    build_calibration_cache,
    build_scenarios_under_lambda_fast,
)
from pymort.pricing.survivor_swaps import SurvivorSwapSpec


def _toy_surface():
    ages = np.array([60.0, 61.0, 62.0], dtype=float)
    years = np.arange(2000, 2005, dtype=int)
    base = np.array([0.01, 0.011, 0.012], dtype=float)[:, None]
    trend = np.linspace(1.0, 0.9, years.size)[None, :]
    m = base * trend
    return ages, years, m


def test_interest_rate_scenario_round_trip(tmp_path: Path):
    times = np.array([1.0, 2.0, 3.0], dtype=float)
    zero = np.array([0.02, 0.021, 0.022], dtype=float)
    ir_set = build_interest_rate_scenarios(
        times=times, zero_rates=zero, a=0.1, sigma=0.01, n_scenarios=4, seed=123
    )
    path = tmp_path / "ir.npz"
    save_ir_scenarios_npz(ir_set, path)
    loaded = load_ir_scenarios_npz(path)
    assert np.allclose(loaded.r_paths, ir_set.r_paths)
    assert np.allclose(loaded.discount_factors, ir_set.discount_factors)
    assert np.allclose(loaded.times, ir_set.times)
    assert loaded.metadata == ir_set.metadata
    assert loaded.n_scenarios() == ir_set.n_scenarios()
    assert loaded.horizon() == ir_set.horizon()


def test_calibration_cache_round_trip(tmp_path: Path):
    ages, years, m = _toy_surface()
    cache = build_calibration_cache(
        ages=ages,
        years=years,
        m=m,
        model_name="LCM2",
        B_bootstrap=4,
        n_process=5,
        horizon=4,
        seed=42,
        include_last=True,
    )
    path = tmp_path / "cache.pkl"
    with path.open("wb") as f:
        pickle.dump(cache, f)
    with path.open("rb") as f:
        cache_loaded = pickle.load(f)
    # Light downstream usage: build scenarios
    scen_q = build_scenarios_under_lambda_fast(cache_loaded, lambda_esscher=0.0, scale_sigma=1.0)
    expected_n = cache_loaded.bs_res.mu_sigma.shape[0] * cache_loaded.n_process
    assert scen_q.q_paths.shape[0] == expected_n


def test_spec_dataclasses_json_round_trip():
    specs = [
        LongevityBondSpec(issue_age=60.0, notional=1.0, include_principal=True, maturity_years=5),
        SurvivorSwapSpec(age=60.0, maturity_years=5, payer="fixed", strike=0.2, notional=1.0),
        QForwardSpec(age=60.0, maturity_years=2, strike=0.01, settlement_years=2, notional=1.0),
        SForwardSpec(age=60.0, maturity_years=2, strike=0.8, settlement_years=2, notional=1.0),
        CohortLifeAnnuitySpec(
            issue_age=60.0,
            maturity_years=5,
            payment_per_survivor=1.0,
            defer_years=0,
            exposure_at_issue=1.0,
            include_terminal=False,
            terminal_notional=0.0,
        ),
    ]
    for spec in specs:
        data = spec.__dict__
        payload = json.loads(json.dumps(data))
        spec_re = spec.__class__(**payload)
        assert spec_re == spec


def test_multi_instrument_quote_pickle_round_trip(tmp_path: Path):
    bond = LongevityBondSpec(issue_age=60.0, maturity_years=3, include_principal=True)
    q = MultiInstrumentQuote(kind="longevity_bond", spec=bond, market_price=12.3, weight=1.0)
    path = tmp_path / "quote.pkl"
    with path.open("wb") as f:
        pickle.dump(q, f)
    with path.open("rb") as f:
        q_loaded = pickle.load(f)
    assert q_loaded.kind == q.kind
    assert q_loaded.market_price == q.market_price
    assert isinstance(q_loaded.spec, LongevityBondSpec)
