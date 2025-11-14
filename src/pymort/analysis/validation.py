from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Any

# --- Core LC diagnostics -----------------------------------------------------

def lc_explained_variance(m: np.ndarray) -> float:
    """
    Proportion of total variance in log-mortality explained by the
    first singular component used by Lee–Carter.

    Parameters
    ----------
    m : np.ndarray, shape (A, T)
        Death-rate matrix (strictly positive).

    Returns
    -------
    float
        Explained variance in [0, 1]. For ages 60+, values >= 0.8 are typical.
    """
    if m.ndim != 2:
        raise ValueError("m must be 2D (A, T).")
    if (m <= 0).any() or not np.isfinite(m).all():
        raise ValueError("m must be strictly positive and finite.")
    ln_m = np.log(m)
    a = ln_m.mean(axis=1, keepdims=True)
    Z = ln_m - a
    _, s, _ = np.linalg.svd(Z, full_matrices=False)
    return float((s[0] ** 2) / np.sum(s ** 2))


def reconstruction_rmse_log(m_true: np.ndarray, ln_m_hat: np.ndarray) -> float:
    """
    Root-mean-square error on the log scale between observed m and reconstructed ln m.

    Parameters
    ----------
    m_true : np.ndarray, shape (A, T)
        Observed death rates (positive).
    ln_m_hat : np.ndarray, shape (A, T)
        Reconstructed log death rates (e.g., a_x + b_x k_t).

    Returns
    -------
    float
        RMS error on log scale.
    """
    if m_true.shape != ln_m_hat.shape:
        raise ValueError("Shapes must match.")
    if (m_true <= 0).any():
        raise ValueError("m_true must be > 0.")
    ln_m_true = np.log(m_true)
    return float(np.sqrt(np.mean((ln_m_true - ln_m_hat) ** 2)))


# --- Simple train/test split for backtesting ---------------------------------

def split_train_test(
    ages: np.ndarray,
    years: np.ndarray,
    m: np.ndarray,
    *,
    train_end: int | None = None,
    holdout: int = 4,
    min_train: int = 20,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Time-based split (no shuffling). Either set `train_end` explicitly,
    or hold out the last `holdout` years.

    Returns
    -------
    (ages_tr, years_tr, m_tr), (ages_te, years_te, m_te)
    """
    years = np.asarray(years)
    if years.ndim != 1:
        raise ValueError("years must be 1D.")
    if m.shape != (len(ages), len(years)):
        raise ValueError("m shape must be (len(ages), len(years)).")

    if train_end is None:
        if len(years) <= holdout + min_train:
            raise ValueError("Not enough years for the requested holdout/min_train.")
        cut = -holdout
        tr_mask = np.arange(len(years)) < (len(years) + cut)
    else:
        tr_mask = years <= train_end
        if tr_mask.sum() < min_train or (~tr_mask).sum() == 0:
            raise ValueError("Invalid split: training too short or empty test set.")

    te_mask = ~tr_mask
    return (
        (ages, years[tr_mask], m[:, tr_mask]),
        (ages, years[te_mask], m[:, te_mask]),
    )


# --- Optional: quick LC backtest (uses the public LC API) --------------------

def backtest_lee_carter(
    ages: np.ndarray,
    years: np.ndarray,
    m: np.ndarray,
    *,
    train_end: int | None = None,
    holdout: int = 4,
    n_sims: int = 0,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Fit LC on train, forecast test years, and report log-RMSE.
    Uses deterministic RW mean for speed if n_sims == 0.

    Returns
    -------
    dict with keys: 'train_years', 'test_years', 'rmse_log'
    """
    from pymort.models import fit_lee_carter, estimate_rw_params, reconstruct_log_m, simulate_k_paths

    (ages_tr, yrs_tr, m_tr), (_ages_te, yrs_te, m_te) = split_train_test(
        ages, years, m, train_end=train_end, holdout=holdout
    )

    params = fit_lee_carter(m_tr)
    mu, sigma = estimate_rw_params(params.k)
    H = len(yrs_te)

    if n_sims and n_sims > 0:
        k_paths = simulate_k_paths(params.k[-1], H, mu, sigma, n_sims=n_sims, seed=seed)
        ln_m_paths = params.a[:, None][None, :, :] + params.b[:, None][None, :, :] * k_paths[:, None, :]
        ln_m_pred = ln_m_paths.mean(axis=0)  # expected log m under P
    else:
        k_det = params.k[-1] + mu * np.arange(1, H + 1)
        ln_m_pred = params.a[:, None] + np.outer(params.b, k_det)

    rmse = reconstruction_rmse_log(m_te, ln_m_pred)
    return {"train_years": yrs_tr, "test_years": yrs_te, "rmse_log": rmse}