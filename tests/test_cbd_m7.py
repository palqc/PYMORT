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
    ages = np.array([68.0, 70.0, 72.0], dtype=float)
    years = np.array([2000, 2001, 2002, 2003], dtype=int)
    base = np.array([0.09, 0.1, 0.11], dtype=float)[:, None]
    trend = np.linspace(1.0, 1.05, years.size)[None, :]
    q = np.clip(base * trend, 1e-4, 0.3)
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
    assert np.sqrt(np.mean((q_rec - q) ** 2)) < 1e-4
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


def test_fit_cbd_m7_raises_on_singular_design():
    ages = np.array([70.0], dtype=float)  # single age -> X'X singular
    years = np.array([2000, 2001], dtype=int)
    q = np.full((1, 2), 0.1, dtype=float)
    with pytest.raises(ValueError, match="singular"):
        fit_cbd_m7(q, ages, years)


def test_fit_cbd_m7_well_conditioned_dataset():
    ages = np.array([65.0, 70.0, 75.0], dtype=float)
    years = np.array([2000, 2001, 2002, 2003], dtype=int)
    base = np.linspace(0.08, 0.12, ages.size)[:, None]
    trend = np.linspace(1.0, 1.1, years.size)[None, :]
    q = np.clip(base * trend, 1e-4, 0.25)
    params = fit_cbd_m7(q, ages, years)
    q_rec = reconstruct_q_m7(params)
    assert q_rec.shape == q.shape
    assert np.sqrt(np.mean((q_rec - q) ** 2)) < 1e-3


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
    assert np.sqrt(np.mean((q_hat - q) ** 2)) < 1e-4

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
