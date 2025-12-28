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

_BETA_TRUE = np.log(np.array([0.0105, 0.0155], dtype=float))
_KAPPA_TRUE = np.array([-0.05, 0.0, 0.06], dtype=float)


def _toy_surface():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2000, 2001, 2002], dtype=int)
    cohorts = _compute_cohort_index(ages, years)
    unique_cohorts = np.unique(cohorts)
    gamma_true = 0.015 * (unique_cohorts - unique_cohorts.mean())
    gamma_true -= np.average(gamma_true, weights=np.ones_like(gamma_true))
    gamma_lookup = dict(zip(unique_cohorts, gamma_true, strict=True))
    gamma_matrix = np.vectorize(gamma_lookup.get)(cohorts)
    ln_m = _BETA_TRUE[:, None] + _KAPPA_TRUE[None, :] + gamma_matrix
    m = np.exp(ln_m)
    return ages, years, m, gamma_true


def _normalize_vec(x: np.ndarray) -> np.ndarray:
    """Center and return 1D vector for sign/corr comparisons."""
    x = np.asarray(x, dtype=float)
    return x - x.mean()


def test_compute_cohort_index():
    ages = np.array([60.0, 61.0])
    years = np.array([2000, 2001])
    C = _compute_cohort_index(ages, years)
    assert C.shape == (2, 2)
    assert np.allclose(C, [[1940, 1941], [1939, 1940]])


def test_fit_apc_m3_reconstruction_matches_input():
    ages, years, m, _gamma_true = _toy_surface()
    params = fit_apc_m3(m, ages, years)
    assert isinstance(params, APCM3Params)
    assert params.beta_age.shape == (len(ages),)
    assert params.kappa.shape == (len(years),)
    assert params.gamma.shape == params.cohorts.shape
    # reconstruction matches input
    ln_rec = reconstruct_log_m_apc(params)
    assert ln_rec.shape == m.shape
    rmse = np.sqrt(np.mean((ln_rec - np.log(m)) ** 2))
    assert (rmse < 0.03) or (rmse / np.std(np.log(m)) < 0.1)
    # identifiability-aware correlation checks
    k_fit = _normalize_vec(params.kappa)
    k_true = _normalize_vec(_KAPPA_TRUE)
    if np.corrcoef(k_fit, k_true)[0, 1] < 0:
        k_fit = -k_fit
    assert np.corrcoef(k_fit, k_true)[0, 1] > 0.99
    # gamma helper works within range
    val = params.gamma_for_age_at_last_year(age=ages[0])
    assert np.isfinite(val)


def test_fit_apc_m3_invalid_inputs():
    ages, years, m, _ = _toy_surface()
    with pytest.raises(ValueError):
        fit_apc_m3(m[:, :2], ages, years)
    with pytest.raises(ValueError):
        fit_apc_m3(np.array([1.0, 2.0]), ages, years)
    bad_m = m.copy()
    bad_m[0, 0] = -1.0
    with pytest.raises(ValueError):
        fit_apc_m3(bad_m, ages, years)


def test_reconstruct_m_apc():
    ages, years, m, _ = _toy_surface()
    params = fit_apc_m3(m, ages, years)
    m_rec = reconstruct_m_apc(params)
    assert m_rec.shape == m.shape
    rmse = np.sqrt(np.mean((np.log(m_rec) - np.log(m)) ** 2))
    assert (rmse < 0.03) or (rmse / np.std(np.log(m)) < 0.1)


def test_apc_class_fit_predict_and_rw_simulation():
    ages, years, m, _ = _toy_surface()
    model = APCM3().fit(m, ages, years)
    ln_hat = model.predict_log_m()
    assert ln_hat.shape == m.shape
    rmse = np.sqrt(np.mean((ln_hat - np.log(m)) ** 2))
    assert (rmse < 0.03) or (rmse / np.std(np.log(m)) < 0.1)
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
    ages, years, m, _ = _toy_surface()
    model.fit(m, ages, years)
    with pytest.raises(ValueError):
        model.simulate_kappa(horizon=2, n_sims=1)
