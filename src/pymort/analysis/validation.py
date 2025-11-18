from __future__ import annotations

from typing import Dict

import numpy as np

from pymort.models.lc import fit_lee_carter, reconstruct_log_m, estimate_rw_params


def reconstruction_rmse_log(m: np.ndarray) -> float:
    params = fit_lee_carter(m)
    ln_hat = reconstruct_log_m(params)
    ln_true = np.log(m)
    rms_in = float(np.sqrt(np.mean((ln_true - ln_hat) ** 2)))
    print(f"In-sample RMSE (log): {rms_in:.6f}")
    print(f"sum(b): {params.b.sum():.12f}  mean(k): {params.k.mean():.3e}")
    return rms_in


def time_split_backtest_lc(
    ages: np.ndarray,
    years: np.ndarray,
    m: np.ndarray,
    train_end: int,
) -> Dict[str, np.ndarray | float]:
    """
    Backtest Lee–Carter with an explicit time split.

    Fits LC on years <= train_end (training set), then produces a
    deterministic forecast of log m on years > train_end (test set)
    and computes the out-of-sample RMSE on log m.
    """
    if m.ndim != 2:
        raise ValueError("m must be a 2D array (A, T).")
    if years.ndim != 1 or years.shape[0] != m.shape[1]:
        raise ValueError("years must be 1D and match m.shape[1].")

    if train_end < years[0] or train_end >= years[-1]:
        raise ValueError(f"train_end must be in [{years[0]}, {years[-1] - 1}].")

    # masks train / test
    tr_mask = years <= train_end
    te_mask = years > train_end

    yrs_tr = years[tr_mask]
    yrs_te = years[te_mask]
    m_tr = m[:, tr_mask]
    m_te = m[:, te_mask]

    # fit LC on train
    params_tr = fit_lee_carter(m_tr)

    # deterministic RW+drift forecast on k_t
    from pymort.models import estimate_rw_params

    mu, _sigma = estimate_rw_params(params_tr.k)
    H = m_te.shape[1]
    k_det = params_tr.k[-1] + mu * np.arange(1, H + 1)

    ln_pred = params_tr.a[:, None] + np.outer(params_tr.b, k_det)
    ln_true = np.log(m_te)
    rmse_log = float(np.sqrt(np.mean((ln_true - ln_pred) ** 2)))

    return {
        "train_years": yrs_tr,
        "test_years": yrs_te,
        "rmse_log": rmse_log,
    }
