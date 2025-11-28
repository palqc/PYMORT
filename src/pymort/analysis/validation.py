from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from pymort.lifetables import m_to_q
from pymort.models import APCM3, CBDM5, CBDM6, CBDM7, LCM1, LCM2, _logit


def _time_split(
    years: np.ndarray, train_end: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if years.ndim != 1:
        raise ValueError("years must be 1D.")
    if train_end < years[0] or train_end >= years[-1]:
        raise ValueError(f"train_end must be in [{years[0]}, {years[-1]-1}].")
    tr_mask = years <= train_end
    te_mask = years > train_end
    return tr_mask, te_mask, years[tr_mask], years[te_mask]


def _check_surface_time_inputs(years: np.ndarray, mat: np.ndarray, name: str):
    years = np.asarray(years, dtype=int)
    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"{name} must be 2D (A, T).")
    if years.ndim != 1 or years.shape[0] != mat.shape[1]:
        raise ValueError("years must be 1D and match surface time dimension.")
    if not np.isfinite(mat).all():
        raise ValueError(f"{name} must contain finite values.")
    if name == "m" and (mat <= 0).any():
        raise ValueError("m must be strictly positive.")
    if name == "q" and ((mat <= 0).any() or (mat >= 1).any()):
        raise ValueError("q must lie strictly in (0,1).")


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"RMSE: shapes mismatch {a.shape} vs {b.shape}.")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _rmse_logit_q(q_true: np.ndarray, q_hat: np.ndarray) -> float:
    return _rmse(_logit(q_true), _logit(q_hat))


def _rmse_logit_q_from_m(m_true: np.ndarray, m_hat: np.ndarray) -> float:
    return _rmse_logit_q(m_to_q(m_true), m_to_q(m_hat))


def _rmse_logit_forecast_from_logit_hat(
    q_te: np.ndarray, logit_hat_te: np.ndarray
) -> float:
    return _rmse(_logit(q_te), logit_hat_te)


def _rw_drift_forecast(last: float, mu: float, H: int) -> np.ndarray:
    steps = np.arange(1, H + 1)
    return last + mu * steps


def _freeze_gamma_last_per_age(
    ages: np.ndarray,
    cohorts: np.ndarray,
    gamma: np.ndarray,
    train_end: int,
) -> np.ndarray:
    """
    Return gamma_last_per_age[i] = gamma_{c_last = train_end - ages[i]},
    choosing the closest cohort index if there is no exact match.
    """
    out = np.zeros_like(ages, dtype=float)
    for i, age_x in enumerate(ages):
        c_last = float(train_end - age_x)
        idx = np.searchsorted(cohorts, c_last)

        if idx == 0:
            idx_use = 0
        elif idx >= len(cohorts):
            idx_use = len(cohorts) - 1
        else:
            left = cohorts[idx - 1]
            right = cohorts[idx]
            idx_use = idx - 1 if abs(left - c_last) <= abs(right - c_last) else idx

        out[i] = gamma[idx_use]
    return out


def time_split_backtest_lc_m1(
    years: np.ndarray,
    m: np.ndarray,
    train_end: int,
) -> Dict[str, np.ndarray | float]:
    """
    Backtest Lee–Carter M1 with an explicit time split.

    Fits LC on years <= train_end (training set), then produces a
    deterministic forecast of log m on years > train_end (test set)
    and computes:
      - out-of-sample RMSE on log m (forecast window),
      - out-of-sample RMSE on logit(q) (forecast window).
    """
    _check_surface_time_inputs(years, m, "m")
    tr_mask, te_mask, yrs_tr, yrs_te = _time_split(years, train_end)
    m_tr = m[:, tr_mask]
    m_te = m[:, te_mask]

    model = LCM1().fit(m_tr)

    params = model.params
    if params is None:
        raise RuntimeError("LCM1.fit() failed")
    mu, _sigma = model.estimate_rw()
    H = m_te.shape[1]
    k_for = _rw_drift_forecast(params.k[-1], mu, H)

    ln_pred = params.a[:, None] + np.outer(params.b, k_for)

    ln_true = np.log(m_te)
    rmse_log = _rmse(ln_true, ln_pred)

    m_te_hat = np.exp(ln_pred)
    rmse_logit_forecast = _rmse_logit_q_from_m(m_te, m_te_hat)

    return {
        "train_years": yrs_tr,
        "test_years": yrs_te,
        "rmse_log_forecast": rmse_log,
        "rmse_logit_forecast": rmse_logit_forecast,
    }


def time_split_backtest_lc_m2(
    ages: np.ndarray,
    years: np.ndarray,
    m: np.ndarray,
    train_end: int,
) -> Dict[str, float | np.ndarray]:
    """
    Backtest for Lee–Carter with cohort effect (LC M2):

        log m_{x,t} = a_x + b_x k_t + gamma_{t-x}

    Strategy:
      1) Fit LC M2 on train set
      2) RW+drift forecast for k_t
      3) Freeze gamma_{t-x} at last train year for each age
      4) Compute:
             - RMSE logit(q) train
             - RMSE logit(q) forecast
             - RMSE log(m) forecast
    """
    _check_surface_time_inputs(years, m, "m")
    tr_mask, te_mask, yrs_tr, yrs_te = _time_split(years, train_end)
    m_tr = m[:, tr_mask]
    m_te = m[:, te_mask]

    model = LCM2().fit(m_tr, ages, yrs_tr)
    params = model.params
    if params is None:
        raise RuntimeError("LCM2Model.fit() failed")

    m_tr_hat = model.predict_m()
    rmse_logit_train = _rmse_logit_q_from_m(m_tr, m_tr_hat)

    mu, _sigma = model.estimate_rw()
    H = m_te.shape[1]
    k_for = _rw_drift_forecast(params.k[-1], mu, H)

    C = params.cohorts
    gamma = params.gamma
    gamma_last_per_age = _freeze_gamma_last_per_age(ages, C, gamma, train_end)
    gamma_grid = gamma_last_per_age[:, None]

    ln_te_hat = params.a[:, None] + np.outer(params.b, k_for) + gamma_grid
    m_te_hat = np.exp(ln_te_hat)

    ln_te_true = np.log(m_te)
    rmse_log_forecast = _rmse(ln_te_true, ln_te_hat)

    rmse_logit_forecast = _rmse_logit_q_from_m(m_te, m_te_hat)

    return {
        "train_years": yrs_tr,
        "test_years": yrs_te,
        "rmse_logit_train": rmse_logit_train,
        "rmse_logit_forecast": rmse_logit_forecast,
        "rmse_log_forecast": rmse_log_forecast,
    }


def time_split_backtest_cbd_m5(
    ages: np.ndarray,
    years: np.ndarray,
    q: np.ndarray,
    train_end: int,
) -> Dict[str, np.ndarray | float]:
    """
    Backtest CBD (M5) with an explicit time split on q_x,t (logit scale).

    - Fit CBD on years <= train_end,
    - Use RW+drift on (kappa1_t, kappa2_t) to produce a deterministic
      forecast on years > train_end,
    - Compute:
        * in-sample RMSE on train (logit q),
        * out-of-sample RMSE on test (logit q).
    """
    _check_surface_time_inputs(years, q, "q")
    tr_mask, te_mask, yrs_tr, yrs_te = _time_split(years, train_end)
    q_tr = q[:, tr_mask]
    q_te = q[:, te_mask]

    model = CBDM5().fit(q_tr, ages)
    params = model.params
    if params is None:
        raise RuntimeError("CBDModel has no fitted parameters.")

    q_hat_tr = model.predict_q()
    rmse_logit_train = _rmse_logit_q(q_tr, q_hat_tr)

    mu1, _sigma1, mu2, _sigma2 = model.estimate_rw()

    H = q_te.shape[1]

    k1_for = _rw_drift_forecast(params.kappa1[-1], mu1, H)
    k2_for = _rw_drift_forecast(params.kappa2[-1], mu2, H)

    z = ages - params.x_bar  # (A,)
    logit_hat_te = k1_for[None, :] + z[:, None] * k2_for[None, :]  # (A, H)

    rmse_logit_forecast = _rmse_logit_forecast_from_logit_hat(q_te, logit_hat_te)

    return {
        "train_years": yrs_tr,
        "test_years": yrs_te,
        "rmse_logit_train": rmse_logit_train,
        "rmse_logit_forecast": rmse_logit_forecast,
    }


def time_split_backtest_cbd_m6(
    ages: np.ndarray,
    years: np.ndarray,
    q: np.ndarray,
    train_end: int,
) -> Dict[str, np.ndarray | float]:
    """
    Backtest CBD+cohort (M6) with explicit time split on q_x,t (logit scale).
    """
    _check_surface_time_inputs(years, q, "q")
    tr_mask, te_mask, yrs_tr, yrs_te = _time_split(years, train_end)
    q_tr = q[:, tr_mask]
    q_te = q[:, te_mask]

    model = CBDM6().fit(q_tr, ages, yrs_tr)
    params = model.params
    if params is None:
        raise RuntimeError("CBDM6Model has no fitted parameters.")

    q_hat_tr = model.predict_q()
    rmse_logit_train = _rmse_logit_q(q_tr, q_hat_tr)

    mu1, _sigma1, mu2, _sigma2 = model.estimate_rw()

    H = q_te.shape[1]
    k1_for = _rw_drift_forecast(params.kappa1[-1], mu1, H)
    k2_for = _rw_drift_forecast(params.kappa2[-1], mu2, H)

    z = ages - params.x_bar

    gamma_last_per_age = _freeze_gamma_last_per_age(
        ages, params.cohorts, params.gamma, train_end
    )
    gamma_grid = gamma_last_per_age[:, None]

    logit_hat_te = k1_for[None, :] + z[:, None] * k2_for[None, :] + gamma_grid

    rmse_logit_forecast = _rmse_logit_forecast_from_logit_hat(q_te, logit_hat_te)

    return {
        "train_years": yrs_tr,
        "test_years": yrs_te,
        "rmse_logit_train": rmse_logit_train,
        "rmse_logit_forecast": rmse_logit_forecast,
    }


def time_split_backtest_cbd_m7(
    ages: np.ndarray,
    years: np.ndarray,
    q: np.ndarray,
    train_end: int,
) -> Dict[str, np.ndarray | float]:
    """
    Backtest CBD M7 (quadratic + cohort) with explicit time split on q_x,t (logit scale).
    """
    _check_surface_time_inputs(years, q, "q")
    tr_mask, te_mask, yrs_tr, yrs_te = _time_split(years, train_end)
    q_tr = q[:, tr_mask]
    q_te = q[:, te_mask]

    model = CBDM7().fit(q_tr, ages, yrs_tr)
    params = model.params
    if params is None:
        raise RuntimeError("CBDM7Model has no fitted parameters.")

    q_hat_tr = model.predict_q()
    rmse_logit_train = _rmse_logit_q(q_tr, q_hat_tr)

    mu1, _sigma1, mu2, _sigma2, mu3, _sigma3 = model.estimate_rw()

    H = q_te.shape[1]
    k1_for = _rw_drift_forecast(params.kappa1[-1], mu1, H)
    k2_for = _rw_drift_forecast(params.kappa2[-1], mu2, H)
    k3_for = _rw_drift_forecast(params.kappa3[-1], mu3, H)

    z = ages - params.x_bar
    z2c = z**2 - params.sigma2_x

    gamma_last_per_age = _freeze_gamma_last_per_age(
        ages, params.cohorts, params.gamma, train_end
    )
    gamma_grid = gamma_last_per_age[:, None]

    logit_hat_te = (
        k1_for[None, :]
        + z[:, None] * k2_for[None, :]
        + z2c[:, None] * k3_for[None, :]
        + gamma_grid
    )

    rmse_logit_forecast = _rmse_logit_forecast_from_logit_hat(q_te, logit_hat_te)

    return {
        "train_years": yrs_tr,
        "test_years": yrs_te,
        "rmse_logit_train": rmse_logit_train,
        "rmse_logit_forecast": rmse_logit_forecast,
    }


def time_split_backtest_apc_m3(
    ages: np.ndarray,
    years: np.ndarray,
    m: np.ndarray,
    train_end: int,
) -> Dict[str, float | np.ndarray]:
    """
    Backtest for APC Model M3:

        ln m_{x,t} = beta_x + kappa_t + gamma_{t-x}

    Strategy:
      1) Fit APC M3 on train set (years <= train_end)
      2) Estimate RW+drift on kappa_t
      3) Deterministic forecast kappa_t on test years
      4) Freeze cohort effect gamma_{t-x} at its value for the last train year
         for each age (conservative choice, like LC M2 backtest)
      5) Compute:
           - RMSE forecast on log m
           - RMSE forecast on logit(q)
    """
    _check_surface_time_inputs(years, m, "m")
    tr_mask, te_mask, yrs_tr, yrs_te = _time_split(years, train_end)
    m_tr = m[:, tr_mask]
    m_te = m[:, te_mask]

    model = APCM3().fit(m_tr, ages, yrs_tr)
    params = model.params
    if params is None:
        raise RuntimeError("APCM3Model.fit() failed")

    m_tr_hat = model.predict_m()
    rmse_logit_train = _rmse_logit_q_from_m(m_tr, m_tr_hat)

    mu, _sigma = model.estimate_rw()
    H = m_te.shape[1]

    k_for = _rw_drift_forecast(params.kappa[-1], mu, H)

    C = params.cohorts
    gamma = params.gamma

    gamma_last_per_age = _freeze_gamma_last_per_age(ages, C, gamma, train_end)
    gamma_grid = gamma_last_per_age[:, None]

    ln_te_hat = params.beta_age[:, None] + k_for[None, :] + gamma_grid
    m_te_hat = np.exp(ln_te_hat)

    ln_te_true = np.log(m_te)
    rmse_log_forecast = _rmse(ln_te_true, ln_te_hat)

    rmse_logit_forecast = _rmse_logit_q_from_m(m_te, m_te_hat)

    return {
        "train_years": yrs_tr,
        "test_years": yrs_te,
        "rmse_logit_train": rmse_logit_train,
        "rmse_logit_forecast": rmse_logit_forecast,
        "rmse_log_forecast": rmse_log_forecast,
    }


def rmse_aic_bic(
    logit_true: np.ndarray, logit_hat: np.ndarray, n_params: int
) -> tuple[float, float, float]:
    """
    Compute RMSE, AIC and BIC under a Gaussian error assumption
    on the logit scale.
    """
    logit_true = np.asarray(logit_true).ravel()
    logit_hat = np.asarray(logit_hat).ravel()
    if logit_true.shape != logit_hat.shape:
        raise ValueError("logit_true and logit_hat must have the same shape.")

    resid = logit_true - logit_hat
    if not np.all(np.isfinite(resid)):
        raise ValueError("Residuals contain NaN or non-finite values.")
    n = resid.size
    rss = float(np.sum(resid**2))
    rmse = float(np.sqrt(rss / n))

    sigma2_hat = rss / n
    sigma2_hat = max(sigma2_hat, 1e-12)
    loglik = -0.5 * n * (np.log(2.0 * np.pi * sigma2_hat) + 1.0)

    aic = 2 * n_params - 2 * loglik
    bic = n_params * np.log(n) - 2 * loglik
    return rmse, aic, bic
