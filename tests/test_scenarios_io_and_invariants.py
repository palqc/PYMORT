from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pymort.analysis.scenario import (
    MortalityScenarioSet,
    load_scenario_set_npz,
    save_scenario_set_npz,
    validate_scenario_set,
)


def _build_scenarios(
    N: int = 3, T: int = 5, discount_factors=None, with_m: bool = False
):
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
    df2d = np.tile(
        np.exp(-0.01 * np.arange(1, scen.horizon() + 1)), (scen.n_scenarios(), 1)
    )
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
