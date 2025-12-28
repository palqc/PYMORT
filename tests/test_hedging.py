from __future__ import annotations

import numpy as np
import pytest

from pymort.pipeline import hedging_pipeline
from pymort.pricing.hedging import (
    compute_duration_convexity_matching_hedge,
    compute_duration_matching_hedge,
    compute_greek_matching_hedge,
    compute_min_variance_hedge,
    compute_min_variance_hedge_constrained,
    compute_multihorizon_hedge,
)

# ======================================================================================
# Min-variance hedge (scenario OLS)
# ======================================================================================


def test_min_variance_exact_replication_single_hedge():
    rng = np.random.default_rng(42)
    N = 200
    L = rng.normal(loc=10.0, scale=1.0, size=N)
    H = L.copy().reshape(-1, 1)  # perfect hedge

    res = hedging_pipeline(liability_pv_paths=L, hedge_pv_paths=H, method="min_variance")

    assert res.weights.shape == (1,)
    assert np.isfinite(res.weights[0])
    assert np.isclose(res.weights[0], -1.0, rtol=1e-2, atol=1e-3)
    assert res.summary["var_net"] < res.summary["var_liability"] * 1e-4


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
    assert np.allclose(res.weights, np.array([-2.0, 0.5]), rtol=0.05, atol=0.05)
    assert res.summary["var_net"] < res.summary["var_liability"] * 0.1


def test_min_variance_scaling_invariance():
    rng = np.random.default_rng(7)
    N = 150
    L = rng.normal(1.0, 0.2, size=N)
    H = rng.normal(0.5, 0.1, size=(N, 2))

    res_base = hedging_pipeline(liability_pv_paths=L, hedge_pv_paths=H, method="min_variance")

    scale = 3.5
    res_scaled = hedging_pipeline(
        liability_pv_paths=L * scale, hedge_pv_paths=H, method="min_variance"
    )

    assert np.allclose(res_scaled.weights, res_base.weights * scale, rtol=1e-2, atol=1e-3)


def test_min_variance_accepts_transposed_H_and_custom_names():
    rng = np.random.default_rng(0)
    N, M = 120, 3
    H = rng.normal(size=(N, M))
    w_true = np.array([0.4, -0.2, 0.1])
    L = -(H @ w_true) + rng.normal(scale=1e-6, size=N)  # ~ exact

    names = ["A", "B", "C"]
    res = compute_min_variance_hedge(L, H.T, instrument_names=names)  # H is (M,N)

    assert res.instrument_names == names
    assert res.weights.shape == (M,)
    assert np.allclose(res.weights, w_true, atol=1e-2, rtol=1e-2)


def test_min_variance_raises_when_N_less_than_M():
    rng = np.random.default_rng(1)
    N, M = 5, 6
    L = rng.normal(size=N)
    H = rng.normal(size=(N, M))
    with pytest.raises(ValueError):
        compute_min_variance_hedge(L, H)


def test_min_variance_instrument_names_length_mismatch_raises():
    rng = np.random.default_rng(2)
    N, M = 50, 2
    L = rng.normal(size=N)
    H = rng.normal(size=(N, M))
    with pytest.raises(ValueError):
        compute_min_variance_hedge(L, H, instrument_names=["only_one"])


def test_min_variance_constant_liability_gives_nan_corr_and_var_reduction():
    # var_L == 0 -> var_reduction nan; corr nan if std_L==0 or std_net==0
    N = 50
    L = np.ones(N) * 3.0
    H = np.ones((N, 1))  # constant hedge too
    res = compute_min_variance_hedge(L, H)
    assert np.isnan(res.summary["var_reduction"])
    assert np.isnan(res.summary["corr_L_net"])


# ======================================================================================
# Multi-horizon hedge (cashflows)
# ======================================================================================


def test_multihorizon_shapes_and_finiteness_smoke():
    rng = np.random.default_rng(99)
    N, T, M = 120, 5, 2
    L_cf = rng.normal(1.0, 0.2, size=(N, T))
    H_cf = rng.normal(0.5, 0.1, size=(N, M, T))
    df = np.exp(-0.02 * np.arange(T))

    L_pv = (L_cf * df[None, :]).sum(axis=1)  # (N,)
    H_pv = (H_cf * df[None, None, :]).sum(axis=2)  # (N, M)

    res = hedging_pipeline(
        liability_pv_paths=L_pv,
        hedge_pv_paths=H_pv,
        liability_cf_paths=L_cf,
        hedge_cf_paths=H_cf,
        method="multihorizon",
        constraints={"discount_factors": df},
    )

    assert res.weights.shape == (M,)
    assert np.isfinite(res.weights).all()
    assert np.isfinite(res.summary["var_liability"])
    assert np.isfinite(res.summary["var_net"])
    assert res.summary["var_liability"] >= 0.0
    assert res.summary["var_net"] >= 0.0


def test_multihorizon_reduces_variance_when_hedge_spans_liability():
    rng = np.random.default_rng(202)
    N, T, M = 300, 6, 2

    H_cf = rng.normal(0.5, 0.1, size=(N, M, T))
    noise = rng.normal(0.0, 0.01, size=(N, T))

    a, b = 1.7, -0.8
    L_cf = a * H_cf[:, 0, :] + b * H_cf[:, 1, :] + noise

    df = np.exp(-0.02 * np.arange(T))

    L_pv = (L_cf * df[None, :]).sum(axis=1)  # (N,)
    H_pv = (H_cf * df[None, None, :]).sum(axis=2)  # (N, M)

    res = hedging_pipeline(
        liability_pv_paths=L_pv,
        hedge_pv_paths=H_pv,
        liability_cf_paths=L_cf,
        hedge_cf_paths=H_cf,
        method="multihorizon",
        constraints={"discount_factors": df},
    )

    assert res.weights.shape == (M,)
    assert np.isfinite(res.weights).all()
    assert res.summary["var_net"] < res.summary["var_liability"] * 0.5


def test_multihorizon_accepts_transposed_H_and_2d_discount_factors_and_time_weights():
    rng = np.random.default_rng(3)
    N, T, M = 80, 4, 2
    L_cf = rng.normal(size=(N, T))
    H_cf = rng.normal(size=(N, M, T))

    # provide H as (M,N,T)
    H_cf_t = np.transpose(H_cf, (1, 0, 2))

    # 2D discount factors (N,T)
    df = np.exp(-0.01 * np.arange(T))[None, :] * np.ones((N, 1))
    tw = np.array([1.0, 0.5, 0.25, 0.1])

    res = compute_multihorizon_hedge(L_cf, H_cf_t, discount_factors=df, time_weights=tw)

    assert res.weights.shape == (M,)
    assert np.isfinite(res.weights).all()
    assert res.summary["rank_H"] >= 0


def test_multihorizon_time_weights_invalid_raise():
    rng = np.random.default_rng(4)
    N, T, M = 20, 3, 2
    L_cf = rng.normal(size=(N, T))
    H_cf = rng.normal(size=(N, M, T))

    with pytest.raises(ValueError):
        compute_multihorizon_hedge(L_cf, H_cf, time_weights=np.array([1.0, -1.0, 1.0]))

    with pytest.raises(ValueError):
        compute_multihorizon_hedge(L_cf, H_cf, time_weights=np.array([1.0, 2.0]))


def test_multihorizon_discount_factors_invalid_raise():
    rng = np.random.default_rng(5)
    N, T, M = 20, 3, 2
    L_cf = rng.normal(size=(N, T))
    H_cf = rng.normal(size=(N, M, T))

    with pytest.raises(ValueError):
        compute_multihorizon_hedge(L_cf, H_cf, discount_factors=np.array([0.9, 0.8]))  # wrong len

    with pytest.raises(ValueError):
        compute_multihorizon_hedge(L_cf, H_cf, discount_factors=np.zeros(T))  # non-positive

    with pytest.raises(ValueError):
        compute_multihorizon_hedge(L_cf, H_cf, discount_factors=np.ones((N, T + 1)))  # wrong shape


# ======================================================================================
# Greek matching / duration / convexity
# ======================================================================================


def test_greek_matching_ols_ridge_lasso_and_transposed_G():
    g = [1.0, -0.5]
    G = np.array([[1.0, 0.0], [0.0, 1.0]])  # (K,M) = (2,2)

    res_ols = compute_greek_matching_hedge(g, G, method="ols")
    assert res_ols.method == "ols"
    assert res_ols.weights.shape == (2,)
    assert np.allclose(res_ols.residuals, np.zeros(2), atol=1e-12)

    # ridge path
    res_ridge = compute_greek_matching_hedge(g, G, method="ridge", alpha=0.1)
    assert res_ridge.method == "ridge"
    assert res_ridge.weights.shape == (2,)
    assert np.isfinite(res_ridge.weights).all()

    # lasso path (may shrink but should run)
    res_lasso = compute_greek_matching_hedge(g, G, method="lasso", alpha=0.01)
    assert res_lasso.method == "lasso"
    assert res_lasso.weights.shape == (2,)
    assert np.isfinite(res_lasso.weights).all()

    # transposed instruments_greeks accepted: (M,K)
    res_T = compute_greek_matching_hedge(g, G.T, method="ols")
    assert res_T.instruments_greeks.shape == (2, 2)


def test_greek_matching_invalid_method_raises():
    with pytest.raises(ValueError):
        compute_greek_matching_hedge([1.0], np.array([[1.0]]), method="nope")


def test_greek_matching_instrument_names_mismatch_raises():
    with pytest.raises(ValueError):
        compute_greek_matching_hedge([1.0, 2.0], np.eye(2), instrument_names=["only_one"])


def test_duration_and_duration_convexity_hedges_shapes():
    res_dur = compute_duration_matching_hedge(liability_dPdr=-5.0, instruments_dPdr=[-2.0, -1.0])
    assert res_dur.weights.shape == (2,)
    assert np.isfinite(res_dur.weights).all()

    res_dc = compute_duration_convexity_matching_hedge(
        liability_dPdr=-4.0,
        liability_d2Pdr2=0.5,
        instruments_dPdr=[-1.5, -0.8],
        instruments_d2Pdr2=[0.2, 0.1],
    )
    assert res_dc.weights.shape == (2,)
    assert np.isfinite(res_dc.weights).all()


def test_duration_convexity_invalid_shapes_raise():
    with pytest.raises(ValueError):
        compute_duration_convexity_matching_hedge(
            liability_dPdr=-1.0,
            liability_d2Pdr2=0.1,
            instruments_dPdr=[-1.0],
            instruments_d2Pdr2=[0.1, 0.2],
        )


# ======================================================================================
# Constrained min-variance hedge (bounded LSQ)
# ======================================================================================


def test_min_variance_hedge_constrained_bounds_and_summary_keys():
    rng = np.random.default_rng(0)
    L = rng.normal(0.0, 1.0, size=120)
    H = rng.normal(0.0, 1.0, size=(120, 2))

    lb, ub = -0.5, 0.5
    res = compute_min_variance_hedge_constrained(L, H, lb=lb, ub=ub)

    assert res.weights.shape == (2,)
    assert np.all(res.weights <= ub + 1e-9)
    assert np.all(res.weights >= lb - 1e-9)

    # should not increase variance in LS sense (can be equal)
    assert res.summary["var_net"] <= res.summary["var_liability"] + 1e-12

    # extra coverage: constrained-specific summary keys
    assert res.summary["constrained"] is True
    assert "bounds" in res.summary
    assert "success" in res.summary
    assert "status" in res.summary
    assert "message" in res.summary
    assert "cost" in res.summary


def test_min_variance_constrained_transposed_H_supported():
    rng = np.random.default_rng(10)
    N, M = 60, 2
    H = rng.normal(size=(N, M))
    L = rng.normal(size=N)
    res = compute_min_variance_hedge_constrained(L, H.T, lb=-1.0, ub=1.0)
    assert res.weights.shape == (M,)


def test_min_variance_constrained_instrument_names_length_mismatch_raises():
    rng = np.random.default_rng(11)
    L = rng.normal(size=50)
    H = rng.normal(size=(50, 2))
    with pytest.raises(ValueError):
        compute_min_variance_hedge_constrained(L, H, instrument_names=["x"])


# ======================================================================================
# Shape validation (existing coverage)
# ======================================================================================


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
        compute_multihorizon_hedge(
            liability_cf_paths=L_cf, instruments_cf_paths=H_cf, discount_factors=df_bad
        )
