from __future__ import annotations

import numpy as np
import pytest

from pymort.models.apc_m3 import (
    APCM3,
    APCM3Params,
    _compute_cohort_index,
    fit_apc_m3,
    reconstruct_log_m_apc,
    reconstruct_m_apc,
)


def _toy_surface():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2000, 2001, 2002], dtype=int)
    m = np.array([[0.01, 0.011, 0.012], [0.015, 0.016, 0.017]], dtype=float)
    return ages, years, m


def test_compute_cohort_index():
    ages = np.array([60.0, 61.0])
    years = np.array([2000, 2001])
    C = _compute_cohort_index(ages, years)
    assert C.shape == (2, 2)
    assert np.allclose(C, [[1940, 1941], [1939, 1940]])


def test_fit_apc_m3_reconstruction_matches_input():
    ages, years, m = _toy_surface()
    params = fit_apc_m3(m, ages, years)
    assert isinstance(params, APCM3Params)
    assert params.beta_age.shape == (len(ages),)
    assert params.kappa.shape == (len(years),)
    assert params.gamma.shape == params.cohorts.shape
    # reconstruction matches input
    ln_rec = reconstruct_log_m_apc(params)
    assert ln_rec.shape == m.shape
    assert np.allclose(np.exp(ln_rec), m, atol=1e-8)
    # gamma helper works within range
    val = params.gamma_for_age_at_last_year(age=ages[0])
    assert np.isfinite(val)


def test_fit_apc_m3_invalid_inputs():
    ages, years, m = _toy_surface()
    with pytest.raises(ValueError):
        fit_apc_m3(m[:, :2], ages, years)
    with pytest.raises(ValueError):
        fit_apc_m3(np.array([1.0, 2.0]), ages, years)
    bad_m = m.copy()
    bad_m[0, 0] = -1.0
    with pytest.raises(ValueError):
        fit_apc_m3(bad_m, ages, years)


def test_reconstruct_m_apc():
    ages, years, m = _toy_surface()
    params = fit_apc_m3(m, ages, years)
    m_rec = reconstruct_m_apc(params)
    assert m_rec.shape == m.shape
    assert np.allclose(m_rec, m, atol=1e-8)


def test_apc_class_fit_predict_and_rw_simulation():
    ages, years, m = _toy_surface()
    model = APCM3().fit(m, ages, years)
    ln_hat = model.predict_log_m()
    assert ln_hat.shape == m.shape
    assert np.allclose(np.exp(ln_hat), m, atol=1e-8)
    mu, sigma = model.estimate_rw()
    assert model.params is not None
    assert np.isclose(model.params.mu, mu)
    assert np.isclose(model.params.sigma, sigma)
    sims = model.simulate_kappa(horizon=4, n_sims=3, seed=123, include_last=False)
    assert sims.shape == (3, 4)
    # deterministic with seed
    sims2 = model.simulate_kappa(horizon=4, n_sims=3, seed=123, include_last=False)
    assert np.allclose(sims, sims2)
    sims_inc = model.simulate_kappa(horizon=4, n_sims=1, seed=123, include_last=True)
    assert sims_inc.shape == (1, 5)


def test_apc_errors_without_fit_or_rw():
    model = APCM3()
    with pytest.raises(ValueError):
        model.predict_log_m()
    with pytest.raises(ValueError):
        model.predict_m()
    with pytest.raises(ValueError):
        model.estimate_rw()
    ages, years, m = _toy_surface()
    model.fit(m, ages, years)
    with pytest.raises(ValueError):
        model.simulate_kappa(horizon=2, n_sims=1)
