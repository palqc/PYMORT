from __future__ import annotations

import numpy as np
import pytest

from pymort.models.lc_m2 import (
    LCM2,
    LCM2Params,
    _compute_cohort_index,
    fit_lee_carter_cohort,
    reconstruct_log_m_cohort,
    reconstruct_m_cohort,
)


def _toy_surface():
    # 2 ages, 3 years with mild trend
    ages = np.array([60, 61], dtype=float)
    years = np.array([2000, 2001, 2002], dtype=int)
    m = np.array([[0.01, 0.011, 0.012], [0.015, 0.016, 0.017]], dtype=float)
    return ages, years, m


def test_compute_cohort_index_basic():
    ages = np.array([60, 61])
    years = np.array([2000, 2001, 2002])
    C = _compute_cohort_index(ages, years)
    assert C.shape == (2, 3)
    # cohort indices are year - age
    assert np.allclose(C[0], [1940, 1941, 1942])
    assert np.allclose(C[1], [1939, 1940, 1941])


def test_fit_lee_carter_cohort_reconstruction_matches_input():
    ages, years, m = _toy_surface()
    params = fit_lee_carter_cohort(m, ages, years)
    assert isinstance(params, LCM2Params)
    # shapes
    assert params.a.shape == (len(ages),)
    assert params.b.shape == (len(ages),)
    assert params.k.shape == (len(years),)
    assert params.gamma.shape == params.cohorts.shape
    # reconstructed log m matches original
    ln_rec = reconstruct_log_m_cohort(params)
    assert ln_rec.shape == m.shape
    assert np.allclose(np.exp(ln_rec), m, atol=1e-8)
    # gamma should have zero mean (identifiability)
    assert np.isclose(np.average(params.gamma), 0.0)


def test_fit_lee_carter_cohort_single_point():
    ages = np.array([60.0])
    years = np.array([2000])
    m = np.array([[0.01]])
    params = fit_lee_carter_cohort(m, ages, years)
    assert params.gamma.shape[0] == 1
    ln_rec = reconstruct_log_m_cohort(params)
    assert ln_rec.shape == (1, 1)
    assert np.allclose(np.exp(ln_rec), m, atol=1e-12)


def test_fit_lee_carter_cohort_invalid_inputs():
    ages, years, m = _toy_surface()
    with pytest.raises(ValueError):
        fit_lee_carter_cohort(m[:, :2], ages, years)  # shape mismatch
    with pytest.raises(ValueError):
        fit_lee_carter_cohort(np.array([1.0, 2.0]), ages, years)  # not 2D
    bad_m = m.copy()
    bad_m[0, 0] = -1.0
    with pytest.raises(ValueError):
        fit_lee_carter_cohort(bad_m, ages, years)


def test_gamma_for_age_at_last_year_and_bounds():
    ages, years, m = _toy_surface()
    params = fit_lee_carter_cohort(m, ages, years)
    # cohort corresponding to age 60 at last year 2002 -> cohort 1942
    val = params.gamma_for_age_at_last_year(age=60.0)
    assert np.isfinite(val)
    with pytest.raises(ValueError):
        params.gamma_for_age_at_last_year(age=10.0)  # out of cohort range


def test_reconstruct_m_cohort_matches_log_version():
    ages, years, m = _toy_surface()
    params = fit_lee_carter_cohort(m, ages, years)
    m_rec = reconstruct_m_cohort(params)
    assert m_rec.shape == m.shape
    assert np.allclose(m_rec, m, atol=1e-8)


def test_lcm2_fit_predict_and_rw_simulation():
    ages, years, m = _toy_surface()
    model = LCM2().fit(m, ages, years)
    # predict log m equals reconstruction
    ln_hat = model.predict_log_m()
    assert ln_hat.shape == m.shape
    assert np.allclose(np.exp(ln_hat), m, atol=1e-8)
    # estimate_rw stores params
    mu, sigma = model.estimate_rw()
    assert model.params is not None
    assert np.isclose(model.params.mu, mu)
    assert np.isclose(model.params.sigma, sigma)
    # simulate_k deterministic with seed
    sims = model.simulate_k(horizon=4, n_sims=2, seed=123, include_last=False)
    assert sims.shape == (2, 4)
    sims2 = model.simulate_k(horizon=4, n_sims=2, seed=123, include_last=False)
    assert np.allclose(sims, sims2)
    sims_inc = model.simulate_k(horizon=4, n_sims=1, seed=123, include_last=True)
    assert sims_inc.shape == (1, 5)


def test_lcm2_errors_without_fit_or_rw():
    model = LCM2()
    with pytest.raises(ValueError):
        model.predict_log_m()
    with pytest.raises(ValueError):
        model.predict_m()
    with pytest.raises(ValueError):
        model.estimate_rw()
    # fit but no estimate_rw before simulate
    ages, years, m = _toy_surface()
    model.fit(m, ages, years)
    with pytest.raises(ValueError):
        model.simulate_k(horizon=2, n_sims=1)
