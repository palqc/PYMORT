from __future__ import annotations

import numpy as np
import pytest

from pymort.models.lc_m1 import LCM1, LCM1Params, fit_lee_carter, reconstruct_log_m
from pymort.models.utils import estimate_rw_params

_A_TRUE = np.array([-4.7, -4.3], dtype=float)
_B_TRUE = np.array([0.4, 0.6], dtype=float)  # sum=1
_K_TRUE = np.array([-0.08, 0.0, 0.09], dtype=float)  # mean≈0


def _toy_surface() -> np.ndarray:
    # Synthetic LC surface with identifiable (a, b, k) up to sign
    ln_m = _A_TRUE[:, None] + np.outer(_B_TRUE, _K_TRUE)
    return np.exp(ln_m)


def _normalize_bk(b: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enforce sum(b)=1 and mean(k)=0 for stable comparisons."""
    scale = float(b.sum())
    b_norm = b / scale
    k_norm = k * scale
    k_norm = k_norm - k_norm.mean()
    return b_norm, k_norm


def test_fit_lee_carter_shapes_and_properties():
    m = _toy_surface()
    params = fit_lee_carter(m)
    assert isinstance(params, LCM1Params)
    # Shapes
    assert params.a.shape == (m.shape[0],)
    assert params.b.shape == (m.shape[0],)
    assert params.k.shape == (m.shape[1],)
    # b sums to 1 (identifiability)
    assert np.isclose(params.b.sum(), 1.0)
    # reconstruct matches predict_log_m
    ln_hat = reconstruct_log_m(params)
    assert ln_hat.shape == m.shape
    # reconstruction should be close to log m used for fit (tolerant)
    rmse = np.sqrt(np.mean((ln_hat - np.log(m)) ** 2))
    assert rmse < 0.02
    # parameter direction invariance: align signs via correlation
    b_fit, k_fit = _normalize_bk(params.b, params.k)
    b_true, k_true = _normalize_bk(_B_TRUE, _K_TRUE)
    if np.corrcoef(k_fit, k_true)[0, 1] < 0:
        b_fit, k_fit = -b_fit, -k_fit
    assert np.corrcoef(k_fit, k_true)[0, 1] > 0.99
    assert np.corrcoef(b_fit, b_true)[0, 1] > 0.99


def test_fit_lee_carter_invalid_inputs():
    with pytest.raises(ValueError):
        fit_lee_carter(np.array([1.0, 2.0]))  # not 2D
    with pytest.raises(ValueError):
        fit_lee_carter(np.array([[0.0, -1.0], [0.1, 0.2]]))  # non-positive


def test_estimate_rw_params_valid_and_errors():
    k = np.array([0.0, 0.1, 0.2])
    mu, sigma = estimate_rw_params(k)
    assert np.isfinite(mu) and np.isfinite(sigma)
    assert sigma >= 0.0
    with pytest.raises(ValueError):
        estimate_rw_params(np.array([0.1]))  # too short


def test_lcm1_class_fit_predict_and_rw():
    m = _toy_surface()
    model = LCM1().fit(m)
    # predict log m equals reconstruction from params
    ln_hat = model.predict_log_m()
    assert ln_hat.shape == m.shape
    assert np.sqrt(np.mean((ln_hat - np.log(m)) ** 2)) < 0.02
    # RW estimates stored in params and deterministic
    mu, sigma = model.estimate_rw()
    assert model.params is not None
    assert np.isclose(model.params.mu, mu)
    assert np.isclose(model.params.sigma, sigma)


def test_lcm1_simulate_k_shapes_and_seed():
    m = _toy_surface()
    model = LCM1().fit(m)
    model.estimate_rw()
    sims = model.simulate_k(horizon=5, n_sims=3, seed=123, include_last=False)
    assert sims.shape == (3, 5)
    # seed determinism
    sims2 = model.simulate_k(horizon=5, n_sims=3, seed=123, include_last=False)
    assert np.allclose(sims, sims2)
    # include_last adds one step
    sims_inc = model.simulate_k(horizon=5, n_sims=2, seed=123, include_last=True)
    assert sims_inc.shape == (2, 6)


def test_lcm1_simulate_k_errors_without_params():
    model = LCM1()
    with pytest.raises(ValueError):
        model.simulate_k(horizon=3, n_sims=1)
