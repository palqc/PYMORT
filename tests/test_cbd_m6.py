from __future__ import annotations

import numpy as np
import pytest

from pymort.models.cbd_m6 import (
    CBDM6,
    CBDM6Params,
    _compute_cohort_index,
    estimate_rw_params_cbd_cohort,
    fit_cbd_cohort,
    reconstruct_logit_q_cbd_cohort,
    reconstruct_q_cbd_cohort,
)


def _toy_q():
    ages = np.array([70.0, 71.0], dtype=float)
    years = np.array([2000, 2001, 2002], dtype=int)
    q = np.array([[0.1, 0.11, 0.12], [0.2, 0.21, 0.22]], dtype=float)
    return ages, years, q


def test_compute_cohort_index_basic():
    ages = np.array([70.0, 71.0])
    years = np.array([2000, 2001])
    C = _compute_cohort_index(ages, years)
    assert C.shape == (2, 2)
    assert np.allclose(C, [[1930.0, 1931.0], [1929.0, 1930.0]])


def test_fit_cbd_cohort_and_reconstruct():
    ages, years, q = _toy_q()
    params = fit_cbd_cohort(q, ages, years)
    assert isinstance(params, CBDM6Params)
    assert params.kappa1.shape == (q.shape[1],)
    assert params.kappa2.shape == (q.shape[1],)
    assert params.gamma.shape == params.cohorts.shape
    # reconstruction matches input q
    logit_rec = reconstruct_logit_q_cbd_cohort(params)
    q_rec = reconstruct_q_cbd_cohort(params)
    assert logit_rec.shape == q.shape
    assert q_rec.shape == q.shape
    assert np.allclose(q_rec, q, atol=1e-8)
    # gamma helper for last year / age in grid
    val = params.gamma_for_age_at_last_year(age=ages[0])
    assert np.isfinite(val)


def test_fit_cbd_cohort_invalid_inputs():
    ages, years, q = _toy_q()
    with pytest.raises(ValueError):
        fit_cbd_cohort(q[:, :2], ages, years)  # shape mismatch
    with pytest.raises(ValueError):
        fit_cbd_cohort(np.array([0.1, 0.2]), ages, years)  # not 2D
    bad_q = q.copy()
    bad_q[0, 0] = -0.1
    with pytest.raises(ValueError):
        fit_cbd_cohort(bad_q, ages, years)


def test_reconstruct_raises_on_cohort_mismatch():
    ages, years, q = _toy_q()
    params = fit_cbd_cohort(q, ages, years)
    # corrupt cohorts to trigger mismatch
    params.cohorts = params.cohorts + 1
    with pytest.raises(RuntimeError):
        reconstruct_q_cbd_cohort(params)


def test_estimate_rw_params_cbd_cohort():
    ages, years, q = _toy_q()
    params = fit_cbd_cohort(q, ages, years)
    params = estimate_rw_params_cbd_cohort(params)
    assert params.mu1 is not None and params.sigma1 is not None
    assert params.mu2 is not None and params.sigma2 is not None


def test_cbdm6_class_fit_predict_simulate():
    ages, years, q = _toy_q()
    model = CBDM6().fit(q, ages, years)
    logit_hat = model.predict_logit_q()
    q_hat = model.predict_q()
    assert logit_hat.shape == q.shape
    assert q_hat.shape == q.shape
    assert np.allclose(q_hat, q, atol=1e-8)

    mu1, sigma1, mu2, sigma2 = model.estimate_rw()
    assert np.isfinite(mu1) and np.isfinite(sigma1)
    assert np.isfinite(mu2) and np.isfinite(sigma2)

    sims1 = model.simulate_kappa("kappa1", horizon=3, n_sims=2, seed=42, include_last=False)
    assert sims1.shape == (2, 3)
    sims1_bis = model.simulate_kappa("kappa1", horizon=3, n_sims=2, seed=42, include_last=False)
    assert np.allclose(sims1, sims1_bis)

    sims2 = model.simulate_kappa("kappa2", horizon=2, n_sims=1, seed=42, include_last=True)
    assert sims2.shape == (1, 3)


def test_cbdm6_errors():
    model = CBDM6()
    with pytest.raises(ValueError):
        model.predict_q()
    with pytest.raises(ValueError):
        model.predict_logit_q()
    with pytest.raises(ValueError):
        model.estimate_rw()
    ages, years, q = _toy_q()
    model.fit(q, ages, years)
    with pytest.raises(ValueError):
        model.simulate_kappa("unknown", horizon=1, n_sims=1)
    # simulate before RW estimate
    with pytest.raises(ValueError):
        model.simulate_kappa("kappa1", horizon=1, n_sims=1)
