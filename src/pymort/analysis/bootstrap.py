from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pymort.lifetables import m_to_q, validate_q
from pymort.models.apc_m3 import APCM3
from pymort.models.cbd_m5 import CBDM5, _logit
from pymort.models.cbd_m6 import CBDM6
from pymort.models.cbd_m7 import CBDM7
from pymort.models.lc_m1 import LCM1
from pymort.models.lc_m2 import LCM2

ResampleMode = Literal["cell", "year_block"]


@dataclass
class BootstrapResult:
    """Container for bootstrap outputs.

    Attributes:
    ----------
    params_list : list
        List of fitted params objects, one per bootstrap replication.
    mu_sigma : np.ndarray
        Array of shape (B, d) with drift/vol estimates.
        d=2 for LC/APC (mu, sigma),
        d=4 for CBD M5/M6 (mu1,sig1, mu2,sig2),
        d=6 for CBD M7 (mu1,sig1, mu2,sig2, mu3,sig3).
    seed : int | None
        RNG seed used.
    """

    params_list: list[Any]
    mu_sigma: np.ndarray
    seed: int | None = None


def _resample_residuals(
    resid: np.ndarray,
    mode: ResampleMode,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample residual matrix resid[A,T] into resid_star[A,T].

    - cell: iid resampling of all A*T residuals
    - year_block: resampling whole years (columns) with replacement
    """
    A, T = resid.shape
    if mode == "cell":
        flat = resid.ravel()
        idx = rng.integers(0, flat.size, size=flat.size)
        return flat[idx].reshape(A, T)

    if mode == "year_block":
        years_idx = rng.integers(0, T, size=T)
        return resid[:, years_idx]

    raise ValueError("mode must be 'cell' or 'year_block'.")


def bootstrap_logm_model(
    model_cls: type,
    m: np.ndarray,
    ages: np.ndarray,
    years: np.ndarray,
    B: int = 500,
    seed: int | None = None,
    resample: ResampleMode = "year_block",
    fit_kwargs: dict | None = None,
) -> BootstrapResult:
    """Residual bootstrap for models fitted on log m (LCM1/LCM2/APCM3).

    Parameters
    ----------
    model_cls : Type
        Class to bootstrap (LCM1, LCM2, APCM3).
        Must implement:
            - fit(m, ages?, years?)
            - predict_log_m() -> (A,T)
            - estimate_rw() returning (mu, sigma)
            - params attribute
    m : np.ndarray
        Mortality surface (A,T), central death rates.
    ages, years : np.ndarray
        Grids used in fit.
    B : int
        Number of bootstrap replications.
    seed : int | None
        RNG seed.
    resample : {"cell","year_block"}
        Residual resampling scheme.
    fit_kwargs : dict | None
        Extra kwargs for fit, if any.

    Returns:
    -------
    BootstrapResult
    """
    if B <= 0:
        raise ValueError("B must be strictly positive.")
    fit_kwargs = fit_kwargs or {}
    rng = np.random.default_rng(seed)

    # Fit on original data
    if model_cls is LCM1:
        model0 = model_cls().fit(m, **fit_kwargs)
    elif model_cls in (LCM2, APCM3):
        model0 = model_cls().fit(m, ages, years, **fit_kwargs)
    else:
        raise ValueError(
            f"bootstrap_logm_model does not support model_cls={model_cls.__name__}. "
            "Expected LCM1, LCM2 or APCM3."
        )
    if model0.params is None:
        raise RuntimeError(f"{model_cls.__name__}.fit() returned None params on original data")

    ln_hat0 = model0.predict_log_m()
    m = np.asarray(m, dtype=float)
    if not np.all(np.isfinite(m)) or (m <= 0).any():
        raise ValueError("bootstrap_logm_model: m must be strictly positive and finite.")
    ln_true = np.log(m)

    resid0 = ln_true - ln_hat0

    params_list = []
    mu_sigma = np.zeros((B, 2), dtype=float)

    for b in range(B):
        resid_star = _resample_residuals(resid0, resample, rng)

        ln_star = ln_hat0 + resid_star
        m_star = np.exp(ln_star)

        # re-fit
        if model_cls is LCM1:
            model_b = model_cls().fit(m_star, **fit_kwargs)
        elif model_cls in (LCM2, APCM3):
            model_b = model_cls().fit(m_star, ages, years, **fit_kwargs)
        else:
            raise ValueError(
                f"bootstrap_logm_model does not support model_cls={model_cls.__name__}."
            )
        if model_b.params is None:
            raise RuntimeError(
                f"{model_cls.__name__}.fit() returned None params in bootstrap replication {b}"
            )

        params_list.append(model_b.params)

        mu_b, sig_b = model_b.estimate_rw()
        mu_sigma[b, :] = (mu_b, sig_b)

    return BootstrapResult(params_list=params_list, mu_sigma=mu_sigma, seed=seed)


def bootstrap_logitq_model(
    model_cls: type,
    q: np.ndarray,
    ages: np.ndarray,
    years: np.ndarray,
    B: int = 500,
    seed: int | None = None,
    resample: ResampleMode = "year_block",
    fit_kwargs: dict | None = None,
) -> BootstrapResult:
    """Residual bootstrap for models fitted on logit(q) (CBDM5/M6/M7).

    model_cls must implement:
        - fit(q, ages, years?)
        - predict_logit_q() -> (A,T)
        - estimate_rw() returning d params
        - params attribute
    """
    if B <= 0:
        raise ValueError("B must be strictly positive.")
    fit_kwargs = fit_kwargs or {}
    rng = np.random.default_rng(seed)

    validate_q(q)

    if model_cls is CBDM5:
        model0 = model_cls().fit(q, ages, **fit_kwargs)
    elif model_cls in (CBDM6, CBDM7):
        model0 = model_cls().fit(q, ages, years, **fit_kwargs)
    else:
        raise ValueError(
            f"bootstrap_logitq_model does not support model_cls={model_cls.__name__}. "
            "Expected CBDM5, CBDM6 or CBDM7."
        )
    if model0.params is None:
        raise RuntimeError(f"{model_cls.__name__}.fit() returned None params on original data")

    logit_hat0 = model0.predict_logit_q()
    logit_true = _logit(q)

    resid0 = logit_true - logit_hat0

    params_list = []
    # dimension drift/vol depends on model
    # we infer by calling estimate_rw once
    rw0 = model0.estimate_rw()
    d = len(rw0)
    mu_sigma = np.zeros((B, d), dtype=float)

    # keep original rw0? not needed, we re-estimate per bootstrap

    for b in range(B):
        resid_star = _resample_residuals(resid0, resample, rng)

        logit_star = logit_hat0 + resid_star
        q_star = 1.0 / (1.0 + np.exp(-logit_star))
        q_star = np.clip(q_star, 1e-10, 1 - 1e-10)
        validate_q(q_star)

        if model_cls is CBDM5:
            model_b = model_cls().fit(q_star, ages, **fit_kwargs)
        elif model_cls in (CBDM6, CBDM7):
            model_b = model_cls().fit(q_star, ages, years, **fit_kwargs)
        else:
            raise ValueError(
                f"bootstrap_logitq_model does not support model_cls={model_cls.__name__}."
            )
        if model_b.params is None:
            raise RuntimeError(
                f"{model_cls.__name__}.fit() returned None params in bootstrap replication {b}"
            )

        params_list.append(model_b.params)
        mu_sigma[b, :] = np.array(model_b.estimate_rw(), dtype=float)

    return BootstrapResult(params_list=params_list, mu_sigma=mu_sigma, seed=seed)


def bootstrap_from_m(
    model_cls: type,
    m: np.ndarray,
    ages: np.ndarray,
    years: np.ndarray,
    B: int = 500,
    seed: int | None = None,
    resample: ResampleMode = "year_block",
) -> BootstrapResult:
    """Convenience wrapper.
    - If model is CBD family, converts to q and bootstraps on logit(q).
    - Else bootstraps on log m.
    """
    name = model_cls.__name__.lower()
    if "cbd" in name:
        q = m_to_q(m)
        return bootstrap_logitq_model(model_cls, q, ages, years, B=B, seed=seed, resample=resample)
    return bootstrap_logm_model(model_cls, m, ages, years, B=B, seed=seed, resample=resample)
