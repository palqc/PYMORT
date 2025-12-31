"""Hedging utilities for longevity risk.

Note:
    Docstrings follow Google style and type hints use NDArray for clarity.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from scipy.optimize import lsq_linear

from pymort._types import FloatArray
from pymort.pricing.utils import pv_matrix_from_cf_paths


@dataclass
class HedgeResult:
    """Result of a scenario-based variance-minimizing hedge.

    We choose weights w_j to minimize the dispersion of:
        Net_n = L_n + sum_j w_j * H_{n,j}.
    """

    weights: FloatArray  # (M,)
    instrument_names: list[str]  # length M
    liability_pv_paths: FloatArray  # (N,)
    hedge_pv_paths: FloatArray  # (N,)
    net_pv_paths: FloatArray  # (N,)
    summary: dict[str, object]


class _LinearModel(Protocol):
    coef_: FloatArray

    def fit(self, X: FloatArray, y: FloatArray) -> _LinearModel: ...


class _LinearModelFactory(Protocol):
    def __call__(self, **kwargs: object) -> _LinearModel: ...


_sklearn_linear_model = importlib.import_module("sklearn.linear_model")
Lasso = cast(_LinearModelFactory, _sklearn_linear_model.Lasso)
Ridge = cast(_LinearModelFactory, _sklearn_linear_model.Ridge)


def _default_instrument_names(m: int) -> list[str]:
    return [f"Instr{j}" for j in range(m)]


def compute_min_variance_hedge(
    liability_pv_paths: FloatArray,
    instruments_pv_paths: FloatArray,
    instrument_names: list[str] | None = None,
) -> HedgeResult:
    """Compute a minimum-variance hedge using OLS.

    Args:
        liability_pv_paths: Liability PV paths. Shape (N,).
        instruments_pv_paths: Instrument PV paths. Shape (N, M) or (M, N).
        instrument_names: Optional instrument names.

    Returns:
        HedgeResult with optimal weights and summary statistics.
    """
    L = np.asarray(liability_pv_paths, dtype=float).reshape(-1)
    if L.ndim != 1:
        raise ValueError("liability_pv_paths must be 1D or (N,1).")

    H = np.asarray(instruments_pv_paths, dtype=float)
    if H.ndim != 2:
        raise ValueError("instruments_pv_paths must be 2D (N,M) or (M,N).")

    n = L.shape[0]
    if H.shape[0] != n and H.shape[1] == n:
        H = H.T

    if H.shape[0] != n:
        raise ValueError(
            f"Inconsistent shapes: liability has N={n} scenarios, "
            f"but instruments_pv_paths has shape {H.shape}."
        )

    m = H.shape[1]
    if instrument_names is None:
        instrument_names = _default_instrument_names(m)
    elif len(instrument_names) != m:
        raise ValueError(f"len(instrument_names)={len(instrument_names)} != M={m}.")

    if n < m:
        raise ValueError(f"Not enough scenarios N={n} for M={m} instruments; need N >= M.")

    w, _, rank, _ = np.linalg.lstsq(H, -L, rcond=None)

    hedge_pv_paths = H @ w
    net_pv_paths = L + hedge_pv_paths

    mean_L = float(L.mean())
    std_L = float(L.std(ddof=0))
    mean_net = float(net_pv_paths.mean())
    std_net = float(net_pv_paths.std(ddof=0))

    corr_L_net = (
        float(np.corrcoef(L, net_pv_paths)[0, 1]) if (std_L > 0.0 and std_net > 0.0) else np.nan
    )

    var_L = std_L**2
    var_net = std_net**2
    var_reduction = 1.0 - (var_net / var_L) if var_L > 0.0 else np.nan

    summary: dict[str, object] = {
        "mean_liability": mean_L,
        "std_liability": std_L,
        "mean_net": mean_net,
        "std_net": std_net,
        "var_liability": var_L,
        "var_net": var_net,
        "var_reduction": var_reduction,
        "corr_L_net": corr_L_net,
        "rank_H": int(rank),
    }

    return HedgeResult(
        weights=np.asarray(w, dtype=float).reshape(-1),
        instrument_names=list(instrument_names),
        liability_pv_paths=L,
        hedge_pv_paths=hedge_pv_paths,
        net_pv_paths=net_pv_paths,
        summary=summary,
    )


def compute_multihorizon_hedge(
    liability_cf_paths: FloatArray,
    instruments_cf_paths: FloatArray,
    discount_factors: FloatArray | None = None,
    time_weights: FloatArray | None = None,
    instrument_names: list[str] | None = None,
    mode: str = "pv_by_horizon",
) -> HedgeResult:
    """Compute a multi-horizon hedge based on cashflows.

    The optimization solves:
        min_w || L_flat + H_flat w ||^2,
    where (scenario, time) pairs are flattened.

    Args:
        liability_cf_paths: Liability cashflows. Shape (N, T).
        instruments_cf_paths: Instrument cashflows. Shape (N, M, T) or (M, N, T).
        discount_factors: Discount factors for PV calculations, shape (T,) or (N, T).
        time_weights: Optional time weights, shape (T,).
        instrument_names: Optional instrument names.
        mode: "pv_by_horizon" or "pv_cashflows".

    Returns:
        HedgeResult with optimal weights and summary statistics.
    """
    L_cf = np.asarray(liability_cf_paths, dtype=float)
    if L_cf.ndim != 2:
        raise ValueError("liability_cf_paths must have shape (N, T).")
    n, t = L_cf.shape

    H_cf = np.asarray(instruments_cf_paths, dtype=float)
    if H_cf.ndim != 3:
        raise ValueError("instruments_cf_paths must have shape (N,M,T) or (M,N,T).")

    # Normalize to (N, M, T)
    if H_cf.shape[0] == n:
        n_h, m, t_h = H_cf.shape
        if t_h != t:
            raise ValueError(f"instruments_cf_paths has T={t_h} but liability has T={t}.")
    elif H_cf.shape[1] == n:
        m, n_h, t_h = H_cf.shape
        if n_h != n or t_h != t:
            raise ValueError(
                f"instruments_cf_paths has shape {H_cf.shape}, "
                f"but liability_cf_paths is (N={n}, T={t})."
            )
        H_cf = np.transpose(H_cf, (1, 0, 2))
    else:
        raise ValueError(
            "instruments_cf_paths must have shape (N,M,T) or (M,N,T) "
            "with the same N,T as liability_cf_paths."
        )

    # ---------- Build per-(n,t) sqrt-weights for WLS ----------
    W = np.ones((n, t), dtype=float)

    # Time weights ONLY (do not mix with discount factors)
    if time_weights is not None:
        w_t = np.asarray(time_weights, dtype=float).reshape(-1)
        if w_t.shape[0] != t:
            raise ValueError(f"time_weights must have length T={t}, got {w_t.shape[0]}.")
        if not np.all(np.isfinite(w_t)) or np.any(w_t < 0.0):
            raise ValueError("time_weights must be non-negative and finite.")
        W *= w_t[None, :]

    W_sqrt = np.sqrt(W)

    # --- discount factors are REQUIRED for pv_by_horizon ---
    mode = mode.lower()
    if mode == "pv_by_horizon" and discount_factors is None:
        raise ValueError("mode='pv_by_horizon' requires discount_factors (shape (T,) or (N,T)).")

    if mode not in {"pv_by_horizon", "pv_cashflows"}:
        raise ValueError("mode must be one of {'pv_by_horizon', 'pv_cashflows'}.")

    # Normalize discount factors to (N,T) for consistent operations
    df_pv: FloatArray | None = None
    if discount_factors is not None:
        df_arr = np.asarray(discount_factors, dtype=float)
        if df_arr.ndim == 1:
            if df_arr.shape[0] != t:
                raise ValueError(f"discount_factors must have length T={t}, got {df_arr.shape[0]}.")
            df_pv = np.repeat(df_arr[None, :], n, axis=0)  # (N,T)
        elif df_arr.ndim == 2:
            if df_arr.shape != (n, t):
                raise ValueError(
                    f"discount_factors must have shape (N,T)={(n, t)}, got {df_arr.shape}."
                )
            df_pv = df_arr
        else:
            raise ValueError("discount_factors must be (T,) or (N,T).")
        if not np.all(np.isfinite(df_pv)) or np.any(df_pv <= 0.0):
            raise ValueError("discount_factors must be positive and finite.")

    # ---------- Build regression targets/features ----------
    if mode == "pv_by_horizon":
        # "Proper" multihorizon: fit on PV-at-horizon matrices (N,T)
        # L_pv_mat[n,h] = PV at horizon h of remaining liability CFs
        df_pv_mat = cast(FloatArray, df_pv)
        L_pv_mat = pv_matrix_from_cf_paths(L_cf, df_pv_mat)  # (N,T)

        # H_pv_mat[n,j,h] = PV at horizon h of remaining CFs of instrument j
        H_pv_mat = np.empty((n, m, t), dtype=float)
        for j in range(m):
            H_pv_mat[:, j, :] = pv_matrix_from_cf_paths(H_cf[:, j, :], df_pv_mat)

        # WLS on (n,h) pairs
        L_flat = (L_pv_mat * W_sqrt).reshape(n * t)  # (N*T,)
        H_flat = (H_pv_mat.transpose(0, 2, 1) * W_sqrt[:, :, None]).reshape(n * t, m)

        w, _, rank, _ = np.linalg.lstsq(H_flat, -L_flat, rcond=None)

        # Summary PV at t=0 (horizon 0)
        liability_pv_paths = L_pv_mat[:, 0]
        hedge_pv_paths = H_pv_mat[:, :, 0] @ w  # (N,)
        net_pv_paths = liability_pv_paths + hedge_pv_paths

    else:
        # Legacy behaviour: fit on PV cashflows (CF * DF) directly
        if df_pv is None:
            L_cf_pv = L_cf
            H_cf_pv = H_cf
        else:
            L_cf_pv = L_cf * df_pv
            H_cf_pv = H_cf * df_pv[:, None, :]

        L_flat = (L_cf_pv * W_sqrt).reshape(n * t)
        H_flat = (H_cf_pv.transpose(0, 2, 1) * W_sqrt[:, :, None]).reshape(n * t, m)

        w, _, rank, _ = np.linalg.lstsq(H_flat, -L_flat, rcond=None)

        liability_pv_paths = L_cf_pv.sum(axis=1)
        hedge_cf_paths = np.einsum("nmt,m->nt", H_cf_pv, w)
        hedge_pv_paths = hedge_cf_paths.sum(axis=1)
        net_pv_paths = liability_pv_paths + hedge_pv_paths

    mean_L = float(liability_pv_paths.mean())
    std_L = float(liability_pv_paths.std(ddof=0))
    mean_net = float(net_pv_paths.mean())
    std_net = float(net_pv_paths.std(ddof=0))

    corr_L_net = (
        float(np.corrcoef(liability_pv_paths, net_pv_paths)[0, 1])
        if (std_L > 0.0 and std_net > 0.0)
        else np.nan
    )

    var_L = std_L**2
    var_net = std_net**2
    var_reduction = 1.0 - (var_net / var_L) if var_L > 0.0 else np.nan

    if instrument_names is None:
        instrument_names = _default_instrument_names(m)
    elif len(instrument_names) != m:
        raise ValueError(f"len(instrument_names)={len(instrument_names)} != M={m}.")

    summary: dict[str, object] = {
        "mean_liability": mean_L,
        "std_liability": std_L,
        "mean_net": mean_net,
        "std_net": std_net,
        "var_liability": var_L,
        "var_net": var_net,
        "var_reduction": var_reduction,
        "corr_L_net": corr_L_net,
        "rank_H": int(rank),
    }

    return HedgeResult(
        weights=np.asarray(w, dtype=float).reshape(-1),
        instrument_names=list(instrument_names),
        liability_pv_paths=liability_pv_paths,
        hedge_pv_paths=hedge_pv_paths,
        net_pv_paths=net_pv_paths,
        summary=summary,
    )


@dataclass
class GreekHedgeResult:
    """Result of a greek-matching hedge."""

    weights: FloatArray  # (M,)
    instrument_names: list[str]  # length M
    liability_greeks: FloatArray  # (K,)
    instruments_greeks: FloatArray  # (K, M)
    residuals: FloatArray  # (K,)
    method: str  # "ols", "ridge", "lasso"
    alpha: float


def compute_greek_matching_hedge(
    liability_greeks: Iterable[float],
    instruments_greeks: FloatArray,
    instrument_names: list[str] | None = None,
    *,
    method: str = "ols",
    alpha: float = 1.0,
) -> GreekHedgeResult:
    """Match multiple greeks using OLS, ridge, or lasso.

    Args:
        liability_greeks: Liability greek targets. Length K.
        instruments_greeks: Instrument greek matrix. Shape (K, M) or (M, K).
        instrument_names: Optional instrument names.
        method: Solver method ("ols", "ridge", "lasso").
        alpha: Regularization strength for ridge/lasso.

    Returns:
        GreekHedgeResult with optimal weights and residuals.
    """
    g = np.asarray(list(liability_greeks), dtype=float).reshape(-1)
    if g.size == 0 or not np.all(np.isfinite(g)):
        raise ValueError("liability_greeks must be non-empty and finite.")
    k = g.shape[0]

    G = np.asarray(instruments_greeks, dtype=float)
    if G.ndim != 2:
        raise ValueError("instruments_greeks must be 2D (K,M) or (M,K).")

    if G.shape[0] == k:
        _, m = G.shape
    elif G.shape[1] == k:
        m, _ = G.shape
        G = G.T
    else:
        raise ValueError(f"instruments_greeks has shape {G.shape}, incompatible with K={k}.")

    if instrument_names is None:
        instrument_names = _default_instrument_names(m)
    elif len(instrument_names) != m:
        raise ValueError(f"len(instrument_names)={len(instrument_names)} != M={m}.")

    method = method.lower()
    if method not in {"ols", "ridge", "lasso"}:
        raise ValueError("method must be one of {'ols', 'ridge', 'lasso'}.")

    if method == "ols":
        w, _, _, _ = np.linalg.lstsq(G, -g, rcond=None)
    elif method == "ridge":
        reg = Ridge(alpha=float(alpha), fit_intercept=False)
        reg.fit(G, -g)
        w = np.asarray(reg.coef_, dtype=float).reshape(-1)
    else:
        reg = Lasso(alpha=float(alpha), fit_intercept=False, max_iter=10_000)
        reg.fit(G, -g)
        w = np.asarray(reg.coef_, dtype=float).reshape(-1)

    residuals_vec = G @ w + g

    return GreekHedgeResult(
        weights=np.asarray(w, dtype=float).reshape(-1),
        instrument_names=list(instrument_names),
        liability_greeks=g,
        instruments_greeks=G,
        residuals=residuals_vec,
        method=method,
        alpha=float(alpha),
    )


def compute_duration_matching_hedge(
    liability_dPdr: float,
    instruments_dPdr: Iterable[float],
    instrument_names: list[str] | None = None,
    *,
    method: str = "ols",
    alpha: float = 1.0,
) -> GreekHedgeResult:
    """Match duration only using a greek-matching hedge.

    Args:
        liability_dPdr: Liability duration (dP/dr).
        instruments_dPdr: Instrument durations.
        instrument_names: Optional instrument names.
        method: Solver method ("ols", "ridge", "lasso").
        alpha: Regularization strength for ridge/lasso.

    Returns:
        GreekHedgeResult with optimal weights.
    """
    g = np.array([float(liability_dPdr)], dtype=float)
    d1 = np.asarray(list(instruments_dPdr), dtype=float).reshape(-1)
    if d1.size == 0:
        raise ValueError("instruments_dPdr must contain at least one instrument.")
    if not np.all(np.isfinite(d1)) or not np.all(np.isfinite(g)):
        raise ValueError("Greeks must be finite.")

    G = d1.reshape(1, -1)
    return compute_greek_matching_hedge(
        liability_greeks=g,
        instruments_greeks=G,
        instrument_names=instrument_names,
        method=method,
        alpha=alpha,
    )


def compute_duration_convexity_matching_hedge(
    liability_dPdr: float,
    liability_d2Pdr2: float,
    instruments_dPdr: Iterable[float],
    instruments_d2Pdr2: Iterable[float],
    instrument_names: list[str] | None = None,
    *,
    method: str = "ols",
    alpha: float = 1.0,
) -> GreekHedgeResult:
    """Match duration and convexity using a greek-matching hedge.

    Args:
        liability_dPdr: Liability duration (dP/dr).
        liability_d2Pdr2: Liability convexity (d2P/dr2).
        instruments_dPdr: Instrument durations.
        instruments_d2Pdr2: Instrument convexities.
        instrument_names: Optional instrument names.
        method: Solver method ("ols", "ridge", "lasso").
        alpha: Regularization strength for ridge/lasso.

    Returns:
        GreekHedgeResult with optimal weights.
    """
    d1 = np.asarray(list(instruments_dPdr), dtype=float).reshape(-1)
    d2 = np.asarray(list(instruments_d2Pdr2), dtype=float).reshape(-1)

    if d1.size == 0 or d2.size == 0:
        raise ValueError("Instrument greeks must contain at least one instrument.")
    if d1.shape != d2.shape:
        raise ValueError("instruments_dPdr and instruments_d2Pdr2 must have same length.")
    if not np.all(np.isfinite(d1)) or not np.all(np.isfinite(d2)):
        raise ValueError("Instrument greeks must be finite.")

    g = np.array([float(liability_dPdr), float(liability_d2Pdr2)], dtype=float)
    if not np.all(np.isfinite(g)):
        raise ValueError("Liability greeks must be finite.")

    G = np.vstack([d1, d2])
    return compute_greek_matching_hedge(
        liability_greeks=g,
        instruments_greeks=G,
        instrument_names=instrument_names,
        method=method,
        alpha=alpha,
    )


def compute_min_variance_hedge_constrained(
    liability_pv_paths: FloatArray,
    instruments_pv_paths: FloatArray,
    instrument_names: list[str] | None = None,
    *,
    lb: float = 0.0,
    ub: float = np.inf,
) -> HedgeResult:
    """Compute a minimum-variance hedge with bounds.

    Args:
        liability_pv_paths: Liability PV paths. Shape (N,).
        instruments_pv_paths: Instrument PV paths. Shape (N, M) or (M, N).
        instrument_names: Optional instrument names.
        lb: Lower bound on weights.
        ub: Upper bound on weights.

    Returns:
        HedgeResult with optimal bounded weights and summary statistics.
    """
    L = np.asarray(liability_pv_paths, dtype=float).reshape(-1)

    H = np.asarray(instruments_pv_paths, dtype=float)
    if H.ndim != 2:
        raise ValueError("instruments_pv_paths must be 2D (N,M) or (M,N).")

    n = L.shape[0]
    if H.shape[0] != n and H.shape[1] == n:
        H = H.T
    if H.shape[0] != n:
        raise ValueError(f"Inconsistent shapes: L has N={n}, H has shape {H.shape}.")

    m = H.shape[1]
    if instrument_names is None:
        instrument_names = _default_instrument_names(m)
    elif len(instrument_names) != m:
        raise ValueError(f"len(instrument_names)={len(instrument_names)} != M={m}.")

    res = lsq_linear(H, -L, bounds=(lb, ub))
    w = np.asarray(res.x, dtype=float).reshape(-1)

    hedge_pv_paths = H @ w
    net_pv_paths = L + hedge_pv_paths

    mean_L = float(L.mean())
    std_L = float(L.std(ddof=0))
    mean_net = float(net_pv_paths.mean())
    std_net = float(net_pv_paths.std(ddof=0))

    corr_L_net = (
        float(np.corrcoef(L, net_pv_paths)[0, 1]) if (std_L > 0.0 and std_net > 0.0) else np.nan
    )

    var_L = std_L**2
    var_net = std_net**2
    var_reduction = 1.0 - (var_net / var_L) if var_L > 0.0 else np.nan

    summary: dict[str, object] = {
        "mean_liability": mean_L,
        "std_liability": std_L,
        "mean_net": mean_net,
        "std_net": std_net,
        "var_liability": var_L,
        "var_net": var_net,
        "var_reduction": var_reduction,
        "corr_L_net": corr_L_net,
        "rank_H": int(np.linalg.matrix_rank(H)),
        "constrained": True,
        "bounds": (float(lb), float(ub)),
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "cost": float(res.cost),
    }

    return HedgeResult(
        weights=w,
        instrument_names=list(instrument_names),
        liability_pv_paths=L,
        hedge_pv_paths=hedge_pv_paths,
        net_pv_paths=net_pv_paths,
        summary=summary,
    )
