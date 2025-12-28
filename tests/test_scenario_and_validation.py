from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pymort.analysis.scenario import (
    MortalityScenarioSet,
    build_scenario_set_from_projection,
    load_scenario_set_npz,
    save_scenario_set_npz,
    validate_scenario_set,
)
from pymort.analysis.validation import (
    _check_surface_time_inputs,
    _freeze_gamma_last_per_age,
    _rmse,
    _rmse_logit_q,
    _rw_drift_forecast,
    _time_split,
    rmse_aic_bic,
    time_split_backtest_cbd_m5,
)
from pymort.lifetables import survival_from_q


def test_validate_scenario_set_and_metadata():
    years = np.array([2020, 2021], dtype=int)
    ages = np.array([60.0, 61.0], dtype=float)
    q = np.array([[[0.01, 0.011], [0.012, 0.013]]], dtype=float)
    S = survival_from_q(q)
    scen = MortalityScenarioSet(years=years, ages=ages, q_paths=q, S_paths=S, metadata={"k": "v"})
    validate_scenario_set(scen)
    assert scen.metadata["k"] == "v"

    # shape mismatch
    scen_bad = MortalityScenarioSet(
        years=years, ages=ages, q_paths=q, S_paths=S[:, :1, :], metadata={}
    )
    with pytest.raises(ValueError):
        validate_scenario_set(scen_bad)

    # non-monotone survival
    S_bad = S.copy()
    S_bad[0, 0, 1] = S_bad[0, 0, 0] + 0.1
    scen_bad2 = MortalityScenarioSet(years=years, ages=ages, q_paths=q, S_paths=S_bad, metadata={})
    with pytest.raises(ValueError):
        validate_scenario_set(scen_bad2)


def test_validation_helpers_time_split_and_rmse():
    years = np.array([2000, 2001, 2002], dtype=int)
    tr_mask, te_mask, _yrs_tr, _yrs_te = _time_split(years, train_end=2001)
    assert tr_mask.sum() == 2 and te_mask.sum() == 1
    with pytest.raises(ValueError):
        _time_split(years, train_end=1999)

    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.5])
    assert _rmse(a, b) > 0.0
    with pytest.raises(ValueError):
        _rmse(a, b[:-1])

    q_true = np.array([0.1, 0.2])
    q_hat = np.array([0.1, 0.21])
    assert _rmse_logit_q(q_true, q_hat) >= 0


def test_validation_input_checks_and_gamma_freeze():
    years = np.array([2000, 2001], dtype=int)
    m = np.array([[0.01, 0.011]])
    _check_surface_time_inputs(years, m, "m")  # should pass
    with pytest.raises(ValueError):
        _check_surface_time_inputs(years, m.reshape(1, 1, 2), "m")
    with pytest.raises(ValueError):
        _check_surface_time_inputs(np.array([[2000, 2001]]), m, "m")
    q = np.array([[0.1, 0.11]])
    _check_surface_time_inputs(years, q, "q")
    q_bad = q.copy()
    q_bad[0, 0] = 0.0
    with pytest.raises(ValueError):
        _check_surface_time_inputs(years, q_bad, "q")

    gamma = np.array([-0.1, 0.0, 0.1])
    cohorts = np.array([1930, 1931, 1932], dtype=float)
    ages = np.array([68.0, 69.0], dtype=float)
    frozen = _freeze_gamma_last_per_age(ages, cohorts, gamma, train_end=2000)
    assert frozen.shape == ages.shape

    forecast = _rw_drift_forecast(last=1.0, mu=0.1, H=3)
    assert np.allclose(forecast, np.array([1.1, 1.2, 1.3]))
    assert _rw_drift_forecast(last=1.0, mu=0.1, H=0).size == 0


def test_freeze_gamma_last_per_age_hits_edges_and_tie_break():
    cohorts = np.array([1930.0, 1931.0, 1932.0])
    gamma = np.array([-0.1, 0.0, 0.2])
    # force idx == 0 (very low c_last) and idx >= len (very high c_last)
    ages = np.array([5000.0, -5000.0])
    out = _freeze_gamma_last_per_age(ages, cohorts, gamma, train_end=2000)
    assert np.allclose(out[0], gamma[0])  # idx==0 branch
    assert np.allclose(out[1], gamma[-1])  # idx>=len branch

    # tie-break: c_last exactly mid between cohorts[0] and cohorts[1] => choose left
    ages2 = np.array([2000.0 - 1930.5])
    out2 = _freeze_gamma_last_per_age(ages2, cohorts, gamma, train_end=2000)
    assert np.allclose(out2[0], gamma[0])


def test_rmse_aic_bic_happy_path_and_errors():
    true = np.array([0.0, 0.5, -0.5])
    hat = np.array([0.1, 0.4, -0.6])
    rmse, aic, bic = rmse_aic_bic(true, hat, n_params=2)
    assert rmse >= 0.0
    assert np.isfinite([rmse, aic, bic]).all()

    with pytest.raises(ValueError):
        rmse_aic_bic(true, hat[:-1], n_params=2)

    bad = hat.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="Residuals contain"):
        rmse_aic_bic(true, bad, n_params=2)


def test_time_split_backtest_cbd_m5_smoke_small_surface():
    # keep it tiny so it's fast
    ages = np.array([60.0, 61.0, 62.0])
    years = np.array([2000, 2001, 2002, 2003], dtype=int)
    train_end = 2001
    # simple q in (0,1)
    q = np.full((ages.size, years.size), 0.02, dtype=float)
    q[:, 1] = 0.021

    out = time_split_backtest_cbd_m5(ages=ages, years=years, q=q, train_end=train_end)
    assert (out["train_years"] == np.array([2000, 2001])).all()
    assert (out["test_years"] == np.array([2002, 2003])).all()
    assert out["rmse_logit_train"] >= 0.0
    assert out["rmse_logit_forecast"] >= 0.0


def _build_scenarios(N: int = 3, T: int = 5, discount_factors=None, with_m: bool = False):
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.arange(2020, 2020 + T, dtype=int)
    q_paths = np.full((N, ages.size, T), 0.01, dtype=float)
    # small variation per scenario/age/time
    for n in range(N):
        q_paths[n] += (n + 1) * 0.0005
    S_paths = np.ones_like(q_paths)
    for t in range(T):
        if t == 0:
            S_paths[:, :, t] = 1.0 - q_paths[:, :, t]
        else:
            S_paths[:, :, t] = S_paths[:, :, t - 1] * (1.0 - q_paths[:, :, t])
    m_paths = np.log1p(q_paths) if with_m else None
    metadata = {"name": "toy", "nested": {"k": 1}, "scalar": 3.14}
    return MortalityScenarioSet(
        years=years,
        ages=ages,
        q_paths=q_paths,
        S_paths=S_paths,
        m_paths=m_paths,
        discount_factors=discount_factors,
        metadata=metadata,
    )


def test_invariants_shapes_bounds_and_monotonicity():
    scen = _build_scenarios()
    validate_scenario_set(scen)  # should not raise
    assert scen.q_paths.shape == (3, 2, 5)
    assert scen.S_paths.shape == (3, 2, 5)
    assert scen.ages.shape == (2,)
    assert scen.years.shape == (5,)
    # Monotonic survival
    diffs = np.diff(scen.S_paths, axis=-1)
    assert np.all(diffs <= 1e-12)
    assert np.allclose(scen.S_paths[..., 0], 1.0 - scen.q_paths[..., 0])
    # Bounds
    assert np.all((scen.q_paths > 0) & (scen.q_paths < 1))
    assert np.all((scen.S_paths >= 0) & (scen.S_paths <= 1))


def test_save_load_round_trip_preserves_arrays_and_metadata(tmp_path: Path):
    T = 5
    df = np.exp(-0.02 * np.arange(1, T + 1, dtype=float))
    scen = _build_scenarios(discount_factors=df, with_m=True)
    out = tmp_path / "scen.npz"
    save_scenario_set_npz(scen, out)
    loaded = load_scenario_set_npz(out)
    validate_scenario_set(loaded)
    assert np.allclose(loaded.q_paths, scen.q_paths)
    assert np.allclose(loaded.S_paths, scen.S_paths)
    assert np.allclose(loaded.m_paths, scen.m_paths)
    assert np.allclose(loaded.ages, scen.ages)
    assert np.allclose(loaded.years, scen.years)
    assert np.allclose(loaded.discount_factors, scen.discount_factors)
    # metadata round-trip through JSON
    assert loaded.metadata == scen.metadata


def test_save_load_round_trip_with_2d_discount_factors(tmp_path: Path):
    scen = _build_scenarios()
    df2d = np.tile(np.exp(-0.01 * np.arange(1, scen.horizon() + 1)), (scen.n_scenarios(), 1))
    scen = MortalityScenarioSet(
        years=scen.years,
        ages=scen.ages,
        q_paths=scen.q_paths,
        S_paths=scen.S_paths,
        discount_factors=df2d,
        metadata={"shape": "2d"},
    )
    out = tmp_path / "scen2.npz"
    save_scenario_set_npz(scen, out)
    loaded = load_scenario_set_npz(out)
    assert np.allclose(loaded.discount_factors, df2d)
    assert loaded.metadata["shape"] == "2d"


def test_validation_errors_on_bad_inputs():
    # invalid q (contains 0)
    bad = _build_scenarios()
    bad.q_paths[0, 0, 0] = 0.0
    with pytest.raises(ValueError):
        validate_scenario_set(bad)

    # survival increasing
    bad2 = _build_scenarios()
    bad2.S_paths[0, 0, 1] = bad2.S_paths[0, 0, 0] + 0.1
    with pytest.raises(ValueError):
        validate_scenario_set(bad2)

    # discount factors wrong shape
    bad3 = _build_scenarios(discount_factors=np.array([0.9, 0.8]))
    with pytest.raises(ValueError):
        validate_scenario_set(bad3)

    # mismatched ages dimension
    bad4 = _build_scenarios()
    bad4.ages = np.array([60.0])
    with pytest.raises(ValueError):
        validate_scenario_set(bad4)


def test_helper_methods_n_scenarios_and_horizon():
    scen = _build_scenarios(N=4, T=7)
    assert scen.n_scenarios() == 4
    assert scen.horizon() == 7


def _toy_scen(N=2, A=3, T=4, df=None):
    years = np.arange(2020, 2020 + T, dtype=int)
    ages = np.arange(60, 60 + A, dtype=float)
    q = np.full((N, A, T), 0.01, dtype=float)
    S = survival_from_q(q)
    return MortalityScenarioSet(
        years=years, ages=ages, q_paths=q, S_paths=S, discount_factors=df, metadata={}
    )


# ----------------------------
# validate_scenario_set: S_paths edge cases
# ----------------------------


def test_validate_scenario_set_rejects_S_nonfinite_and_out_of_bounds():
    scen_nan = _toy_scen()
    scen_nan.S_paths[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="S_paths must lie in \\[0,1\\] and be finite"):
        validate_scenario_set(scen_nan)

    scen_oob = _toy_scen()
    scen_oob.S_paths[0, 0, 0] = 1.5
    with pytest.raises(ValueError, match="S_paths must lie in \\[0,1\\] and be finite"):
        validate_scenario_set(scen_oob)


# ----------------------------
# validate_scenario_set: discount_factors branches
# ----------------------------


def test_validate_scenario_set_discount_factors_2d_first_dim_must_be_1_or_N():
    N, A, T = 3, 2, 4
    base = _toy_scen(N=N, A=A, T=T)

    df_bad0 = np.ones((2, T), dtype=float)  # first dim neither 1 nor N(=3)
    base.discount_factors = df_bad0
    with pytest.raises(ValueError, match="first dim must be 1 or N"):
        validate_scenario_set(base)


def test_validate_scenario_set_discount_factors_bad_ndim_and_nonpositive():
    scen = _toy_scen()

    scen.discount_factors = np.ones((1, 1, 4), dtype=float)
    with pytest.raises(ValueError, match="must be 1D or 2D"):
        validate_scenario_set(scen)

    scen2 = _toy_scen()
    scen2.discount_factors = np.array([1.0, 0.0, 0.9, 0.8], dtype=float)  # nonpositive
    with pytest.raises(ValueError, match="positive and finite"):
        validate_scenario_set(scen2)

    scen3 = _toy_scen()
    scen3.discount_factors = np.array([1.0, np.nan, 0.9, 0.8], dtype=float)
    with pytest.raises(ValueError, match="positive and finite"):
        validate_scenario_set(scen3)


# ----------------------------
# load_scenario_set_npz: invalid JSON metadata fallback
# ----------------------------


def test_load_scenario_set_npz_bad_metadata_json_falls_back_to_empty(tmp_path: Path):
    scen = _toy_scen()
    p = tmp_path / "badmeta.npz"
    save_scenario_set_npz(scen, p)

    # rewrite the file with corrupted metadata (keep arrays)
    d = np.load(p, allow_pickle=True)
    np.savez_compressed(
        p,
        q_paths=d["q_paths"],
        S_paths=d["S_paths"],
        ages=d["ages"],
        years=d["years"],
        metadata="NOT JSON {]",
    )

    loaded = load_scenario_set_npz(p)
    assert isinstance(loaded.metadata, dict)
    assert loaded.metadata == {}  # fallback path hit


# ----------------------------
# build_scenario_set_from_projection: input validations + df branches
# ----------------------------


def test_build_scenario_set_from_projection_rejects_bad_q_shape_and_mismatches():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2020, 2021, 2022], dtype=int)

    # q_paths not 3D
    proj_bad = SimpleNamespace(
        years=years,
        q_paths=np.full((2, 3), 0.01, dtype=float),  # 2D mais valide -> validate_q passe
        m_paths=None,
    )
    with pytest.raises(ValueError, match=r"proj\.q_paths must have shape"):
        build_scenario_set_from_projection(proj_bad, ages)

    # age mismatch
    proj = SimpleNamespace(years=years, q_paths=np.full((2, 2, 3), 0.01), m_paths=None)
    with pytest.raises(ValueError, match="Age dimension mismatch"):
        build_scenario_set_from_projection(proj, np.array([60.0, 61.0, 62.0]))

    # year mismatch
    proj2 = SimpleNamespace(
        years=np.array([2020, 2021], dtype=int),
        q_paths=np.full((2, 2, 3), 0.01),
        m_paths=None,
    )
    with pytest.raises(ValueError, match="Time dimension mismatch"):
        build_scenario_set_from_projection(proj2, ages)


def test_build_scenario_set_from_projection_discount_factors_validation():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2020, 2021, 2022], dtype=int)
    q_paths = np.full((2, 2, 3), 0.01, dtype=float)
    proj = SimpleNamespace(years=years, q_paths=q_paths, m_paths=None)

    # 1D length mismatch
    with pytest.raises(ValueError, match=r"same length as proj.years"):
        build_scenario_set_from_projection(proj, ages, discount_factors=np.array([0.99, 0.98]))

    # 2D shape mismatch
    with pytest.raises(ValueError, match="shape \\(N,H\\)"):
        build_scenario_set_from_projection(proj, ages, discount_factors=np.ones((2, 2)))

    # bad ndim
    with pytest.raises(ValueError, match="must be 1D or 2D"):
        build_scenario_set_from_projection(proj, ages, discount_factors=np.ones((1, 1, 3)))

    # nonpositive / nonfinite
    with pytest.raises(ValueError, match="positive and finite"):
        build_scenario_set_from_projection(proj, ages, discount_factors=np.array([1.0, 0.0, 0.9]))

    with pytest.raises(ValueError, match="positive and finite"):
        build_scenario_set_from_projection(
            proj, ages, discount_factors=np.array([1.0, np.nan, 0.9])
        )


def test_build_scenario_set_from_projection_metadata_none_and_dict():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2020, 2021, 2022], dtype=int)
    q_paths = np.full((2, 2, 3), 0.01, dtype=float)
    proj = SimpleNamespace(years=years, q_paths=q_paths, m_paths=None)

    s1 = build_scenario_set_from_projection(proj, ages, metadata=None)
    assert s1.metadata == {}

    s2 = build_scenario_set_from_projection(proj, ages, metadata={"a": 1})
    assert s2.metadata["a"] == 1
