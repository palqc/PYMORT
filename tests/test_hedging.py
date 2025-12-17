from __future__ import annotations

import numpy as np
import pytest

from pymort.pipeline import hedging_pipeline
from pymort.pricing.hedging import compute_min_variance_hedge, compute_multihorizon_hedge


def test_min_variance_exact_replication_single_hedge():
    rng = np.random.default_rng(42)
    N = 200
    L = rng.normal(loc=10.0, scale=1.0, size=N)
    H = L.copy().reshape(-1, 1)  # perfect hedge
    res = hedging_pipeline(liability_pv_paths=L, hedge_pv_paths=H, method="min_variance")
    assert res.weights.shape == (1,)
    assert np.isclose(res.weights[0], -1.0, atol=1e-6)  # sign to offset liability
    assert res.summary["var_net"] < 1e-8


def test_min_variance_two_hedges_known_solution():
    rng = np.random.default_rng(123)
    N = 300
    H1 = rng.normal(5.0, 0.5, size=N)
    H2 = rng.normal(2.0, 0.3, size=N)
    noise = rng.normal(0.0, 0.01, size=N)
    L = 2.0 * H1 - 0.5 * H2 + noise
    H = np.stack([H1, H2], axis=1)
    res = hedging_pipeline(liability_pv_paths=L, hedge_pv_paths=H, method="min_variance")
    assert res.weights.shape == (2,)
    assert np.allclose(res.weights, np.array([-2.0, 0.5]), atol=0.02)
    assert res.summary["var_net"] < res.summary["var_liability"] * 0.05


def test_min_variance_scaling_invariance():
    rng = np.random.default_rng(7)
    N = 150
    L = rng.normal(1.0, 0.2, size=N)
    H = rng.normal(0.5, 0.1, size=(N, 2))
    res_base = hedging_pipeline(liability_pv_paths=L, hedge_pv_paths=H, method="min_variance")
    scale = 3.5
    res_scaled = hedging_pipeline(liability_pv_paths=L * scale, hedge_pv_paths=H, method="min_variance")
    assert np.allclose(res_scaled.weights, res_base.weights * scale, atol=1e-6)


def test_multihorizon_outputs_shapes_and_finiteness():
    rng = np.random.default_rng(99)
    N, T, M = 120, 5, 2
    L_cf = rng.normal(1.0, 0.2, size=(N, T))
    H_cf = rng.normal(0.5, 0.1, size=(N, M, T))
    df = np.exp(-0.02 * np.arange(T))
    res = hedging_pipeline(
        liability_pv_paths=L_cf,
        hedge_pv_paths=H_cf,
        method="multihorizon",
        constraints={"discount_factors": df},
    )
    assert res.weights.shape == (M,)
    assert np.isfinite(res.weights).all()
    assert res.summary["var_net"] < res.summary["var_liability"]


def test_hedging_raises_on_bad_shapes():
    rng = np.random.default_rng(1)
    L = rng.normal(size=100)
    H = rng.normal(size=(50, 2))  # mismatched N
    with pytest.raises(ValueError):
        compute_min_variance_hedge(liability_pv_paths=L, instruments_pv_paths=H)

    L_cf = rng.normal(size=(10, 3))
    H_cf_bad = rng.normal(size=(10, 2))  # not 3D
    with pytest.raises(ValueError):
        compute_multihorizon_hedge(liability_cf_paths=L_cf, instruments_cf_paths=H_cf_bad)

    H_cf_bad2 = rng.normal(size=(5, 2, 4))  # wrong N
    with pytest.raises(ValueError):
        compute_multihorizon_hedge(liability_cf_paths=L_cf, instruments_cf_paths=H_cf_bad2)

    df_bad = np.array([0.9, 0.8])  # length mismatch T=3
    H_cf = rng.normal(size=(10, 2, 3))
    with pytest.raises(ValueError):
        compute_multihorizon_hedge(liability_cf_paths=L_cf, instruments_cf_paths=H_cf, discount_factors=df_bad)
