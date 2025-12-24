from __future__ import annotations

import numpy as np
import pytest

from pymort.analysis.scenario import MortalityScenarioSet, validate_scenario_set
from pymort.analysis.validation import (
    _check_surface_time_inputs,
    _freeze_gamma_last_per_age,
    _rmse,
    _rmse_logit_q,
    _rw_drift_forecast,
    _time_split,
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
    scen_bad = MortalityScenarioSet(years=years, ages=ages, q_paths=q, S_paths=S[:, :1, :], metadata={})
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
    tr_mask, te_mask, yrs_tr, yrs_te = _time_split(years, train_end=2001)
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
