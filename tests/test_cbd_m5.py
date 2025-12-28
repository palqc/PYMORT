from __future__ import annotations

import numpy as np
import pytest

from pymort.models.cbd_m5 import (
    CBDM5,
    CBDM5Params,
    _build_cbd_design,
    _logit,
    estimate_rw_params_cbd,
    fit_cbd,
    reconstruct_logit_q,
    reconstruct_q,
)
from pymort.models.utils import _estimate_rw_params


def _toy_q() -> tuple[np.ndarray, np.ndarray]:
    # 2 ages, 3 years, strictly between 0 and 1
    ages = np.array([70.0, 71.0], dtype=float)
    q = np.array([[0.1, 0.11, 0.12], [0.2, 0.21, 0.22]], dtype=float)
    return ages, q


def test_logit_stability_and_bounds() -> None:
    p = np.array([1e-15, 0.5, 1 - 1e-15], dtype=float)
    res = _logit(p)
    assert res.shape == p.shape
    assert np.isfinite(res).all()


def test_build_cbd_design() -> None:
    ages = np.array([70, 71, 72], dtype=float)
    X, x_bar = _build_cbd_design(ages)
    assert X.shape == (3, 2)
    assert np.isclose(x_bar, ages.mean())
    # second column centered ages
    assert np.allclose(X[:, 1], ages - ages.mean())

    with pytest.raises(ValueError):
        _build_cbd_design(np.array([[70, 71]], dtype=float))


def test_fit_cbd_and_reconstruct() -> None:
    ages, q = _toy_q()
    params = fit_cbd(q, ages)
    assert isinstance(params, CBDM5Params)

    assert params.kappa1.shape == (q.shape[1],)
    assert params.kappa2.shape == (q.shape[1],)
    assert np.isfinite(params.kappa1).all()
    assert np.isfinite(params.kappa2).all()

    logit_rec = reconstruct_logit_q(params)
    assert logit_rec.shape == q.shape

    q_rec = reconstruct_q(params)
    assert q_rec.shape == q.shape
    # reconstruction should match original q (within numerical precision)
    assert np.allclose(q_rec, q, atol=1e-8)


def test_fit_cbd_invalid_inputs() -> None:
    ages, q = _toy_q()

    # true age mismatch: q has A=1 but ages has len=2
    with pytest.raises(ValueError):
        fit_cbd(q[:1, :], ages)

    # not 2D
    with pytest.raises(ValueError):
        fit_cbd(np.array([0.1, 0.2], dtype=float), ages)

    # q out of bounds
    bad_q = q.copy()
    bad_q[0, 0] = -0.1
    with pytest.raises(ValueError):
        fit_cbd(bad_q, ages)

    # q contains NaN
    nan_q = q.copy()
    nan_q[0, 1] = np.nan
    with pytest.raises(ValueError):
        fit_cbd(nan_q, ages)

    # ages contains NaN
    bad_age = ages.copy()
    bad_age[0] = np.nan
    with pytest.raises(ValueError):
        fit_cbd(q, bad_age)


def test_estimate_rw_params_cbd_and_helper() -> None:
    kappa = np.array([0.0, 0.1, 0.2, 0.25], dtype=float)
    mu, sigma = _estimate_rw_params(kappa)
    assert np.isfinite(mu) and np.isfinite(sigma)

    with pytest.raises(ValueError):
        _estimate_rw_params(np.array([0.1], dtype=float))

    ages, q = _toy_q()
    params = fit_cbd(q, ages)
    params = estimate_rw_params_cbd(params)
    assert params.mu1 is not None and params.sigma1 is not None
    assert params.mu2 is not None and params.sigma2 is not None


def test_cbdm5_class_fit_predict_simulate() -> None:
    ages, q = _toy_q()
    model = CBDM5().fit(q, ages)

    logit_hat = model.predict_logit_q()
    q_hat = model.predict_q()
    assert logit_hat.shape == q.shape
    assert q_hat.shape == q.shape
    assert np.allclose(q_hat, q, atol=1e-8)

    mu1, sigma1, mu2, sigma2 = model.estimate_rw()
    assert np.isfinite(mu1) and np.isfinite(sigma1)
    assert np.isfinite(mu2) and np.isfinite(sigma2)

    sims1 = model.simulate_kappa("kappa1", horizon=3, n_sims=2, seed=123, include_last=False)
    assert sims1.shape == (2, 3)

    sims1_bis = model.simulate_kappa("kappa1", horizon=3, n_sims=2, seed=123, include_last=False)
    assert np.allclose(sims1, sims1_bis)

    sims2 = model.simulate_kappa("kappa2", horizon=2, n_sims=1, seed=123, include_last=True)
    assert sims2.shape == (1, 3)


def test_cbdm5_simulate_errors() -> None:
    model = CBDM5()

    with pytest.raises(ValueError):
        model.predict_q()
    with pytest.raises(ValueError):
        model.predict_logit_q()
    with pytest.raises(ValueError):
        model.estimate_rw()

    ages, q = _toy_q()
    model.fit(q, ages)

    with pytest.raises(ValueError):
        model.simulate_kappa("unknown", horizon=2, n_sims=1)
