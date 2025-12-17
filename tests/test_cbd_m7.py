from __future__ import annotations

import numpy as np
import pytest

from pymort.models.cbd_m7 import (
    CBDM7,
    CBDM7Params,
    _compute_cohort_index,
    estimate_rw_params_m7,
    fit_cbd_m7,
    reconstruct_logit_q_m7,
    reconstruct_q_m7,
)


def _toy_q():
    ages = np.array([70.0, 71.0], dtype=float)
    years = np.array([2000, 2001, 2002], dtype=int)
    q = np.array([[0.1, 0.11, 0.12], [0.2, 0.21, 0.22]], dtype=float)
    return ages, years, q


def test_compute_cohort_index_basic():
    ages = np.array([70.0, 72.0])
    years = np.array([2000, 2001])
    C = _compute_cohort_index(ages, years)
    assert C.shape == (2, 2)
    assert np.allclose(C, [[1930.0, 1931.0], [1928.0, 1929.0]])


def test_fit_cbd_m7_and_reconstruct():
    ages, years, q = _toy_q()
    params = fit_cbd_m7(q, ages, years)
    assert isinstance(params, CBDM7Params)
    assert params.kappa1.shape == (q.shape[1],)
    assert params.kappa2.shape == (q.shape[1],)
    assert params.kappa3.shape == (q.shape[1],)
    assert params.gamma.shape == params.cohorts.shape
    # reconstruction matches input q
    logit_rec = reconstruct_logit_q_m7(params)
    q_rec = reconstruct_q_m7(params)
    assert logit_rec.shape == q.shape
    assert q_rec.shape == q.shape
    assert np.allclose(q_rec, q, atol=1e-8)
    # gamma helper
    val = params.gamma_for_age_at_last_year(age=ages[0])
    assert np.isfinite(val)


def test_fit_cbd_m7_invalid_inputs():
    ages, years, q = _toy_q()
    with pytest.raises(ValueError):
        fit_cbd_m7(q[:, :2], ages, years)  # shape mismatch
    with pytest.raises(ValueError):
        fit_cbd_m7(np.array([0.1, 0.2]), ages, years)  # not 2D
    bad_q = q.copy()
    bad_q[0, 0] = -0.1
    with pytest.raises(ValueError):
        fit_cbd_m7(bad_q, ages, years)


def test_reconstruct_errors_on_cohort_mismatch():
    ages, years, q = _toy_q()
    params = fit_cbd_m7(q, ages, years)
    params.cohorts = params.cohorts + 1  # force mismatch
    with pytest.raises(RuntimeError):
        reconstruct_q_m7(params)


def test_estimate_rw_params_m7():
    ages, years, q = _toy_q()
    params = fit_cbd_m7(q, ages, years)
    params = estimate_rw_params_m7(params)
    assert params.mu1 is not None and params.sigma1 is not None
    assert params.mu2 is not None and params.sigma2 is not None
    assert params.mu3 is not None and params.sigma3 is not None


def test_cbdm7_class_fit_predict_simulate():
    ages, years, q = _toy_q()
    model = CBDM7().fit(q, ages, years)
    logit_hat = model.predict_logit_q()
    q_hat = model.predict_q()
    assert logit_hat.shape == q.shape
    assert q_hat.shape == q.shape
    assert np.allclose(q_hat, q, atol=1e-8)

    mu1, sigma1, mu2, sigma2, mu3, sigma3 = model.estimate_rw()
    assert np.isfinite(mu1) and np.isfinite(sigma1)
    assert np.isfinite(mu2) and np.isfinite(sigma2)
    assert np.isfinite(mu3) and np.isfinite(sigma3)

    sims1 = model.simulate_kappa("kappa1", horizon=3, n_sims=2, seed=7, include_last=False)
    assert sims1.shape == (2, 3)
    sims1_bis = model.simulate_kappa("kappa1", horizon=3, n_sims=2, seed=7, include_last=False)
    assert np.allclose(sims1, sims1_bis)

    sims3 = model.simulate_kappa("kappa3", horizon=2, n_sims=1, seed=7, include_last=True)
    assert sims3.shape == (1, 3)


def test_cbdm7_errors():
    model = CBDM7()
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
    with pytest.raises(ValueError):
        model.simulate_kappa("kappa1", horizon=1, n_sims=1)
