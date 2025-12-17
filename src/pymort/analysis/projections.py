"""
projections.py — Stochastic mortality projections with param + process uncertainty
===============================================================================

This module produces forward mortality scenarios by combining:

1) **Parameter uncertainty** via residual bootstrap:
   - user supplies a BootstrapResult with params_list and mu_sigma.

2) **Process uncertainty** via random-walk-with-drift dynamics on period factors:
   - for LC/APC models: k_t or kappa_t
   - for CBD models: kappa1_t, kappa2_t, and optionally kappa3_t

The engine is fully vectorized:
- For each bootstrap replicate b, we simulate n_process RW paths in one shot.
- Total scenarios N = B * n_process.

Outputs are suitable for pricing longevity-linked cashflows.

Expected bootstrap contract
---------------------------
bootstrap_result must expose:
- params_list : list of fitted params objects (length B)
- mu_sigma    : np.ndarray of shape (B, d)
    d = 2  for LCM1/LCM2/APCM3  -> (mu, sigma)
    d = 4  for CBDM5/CBDM6      -> (mu1, sig1, mu2, sig2)
    d = 6  for CBDM7           -> (mu1, sig1, mu2, sig2, mu3, sig3)

CRN (Common Random Numbers)
---------------------------
To reduce Monte-Carlo noise in finite differences (e.g. vega wrt sigma scaling),
you can pass pre-generated innovations eps to reuse the SAME shocks across runs.

- LC/APC: eps_rw shape (B, n_process, H)
- CBD:    eps1, eps2 shape (B, n_process, H)
- CBDM7:  eps3 also shape (B, n_process, H)

If eps are None, innovations are drawn internally as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Type

import numpy as np

from pymort.lifetables import m_to_q, validate_q


@dataclass
class ProjectionResult:
    years: np.ndarray  # (H_out,)
    q_paths: np.ndarray  # (N, A, H_out)
    m_paths: Optional[np.ndarray]  # (N, A, H_out) for log-m models, else None
    k_paths: Optional[
        np.ndarray
    ]  # (N, H_out) for LC/APC, or (N, d_factors, H_out) for CBD


def simulate_random_walk_paths(
    k_last: float,
    mu: float,
    sigma: float,
    horizon: int,
    n_sims: int,
    rng: np.random.Generator,
    include_last: bool = False,
) -> np.ndarray:
    """
    Vectorized RW+drift simulation:

        k_t = k_{t-1} + mu + sigma * eps_t
    """
    H = int(horizon)
    n_sims = int(n_sims)
    if H <= 0:
        raise ValueError("horizon must be > 0.")
    if n_sims <= 0:
        raise ValueError("n_sims must be > 0.")

    mu = float(mu)
    sigma = float(sigma)
    if not np.isfinite(mu) or not np.isfinite(sigma):
        raise ValueError("mu and sigma must be finite.")
    if sigma < 0:
        sigma = abs(sigma)

    eps = rng.normal(size=(n_sims, H))
    steps = mu + sigma * eps
    paths = k_last + np.cumsum(steps, axis=1)

    if include_last:
        paths = np.concatenate(
            [np.full((n_sims, 1), k_last, dtype=float), paths], axis=1
        )
    return paths


def simulate_random_walk_paths_with_eps(
    k_last: float,
    mu: float,
    sigma: float,
    eps: np.ndarray,
    include_last: bool = False,
) -> np.ndarray:
    """
    Same as simulate_random_walk_paths, but takes eps explicitly (CRN-friendly).
    """
    mu = float(mu)
    sigma = float(sigma)
    if sigma < 0:
        sigma = abs(sigma)

    eps = np.asarray(eps, dtype=float)
    if eps.ndim != 2:
        raise ValueError("eps must be 2D (n_sims, H).")
    if not np.isfinite(eps).all():
        raise ValueError("eps must be finite.")

    steps = mu + sigma * eps
    paths = k_last + np.cumsum(steps, axis=1)

    if include_last:
        paths = np.concatenate(
            [np.full((paths.shape[0], 1), k_last, dtype=float), paths], axis=1
        )
    return paths


def project_mortality_from_bootstrap(
    model_cls: Type,
    ages: np.ndarray,
    years: np.ndarray,
    m: np.ndarray,
    bootstrap_result,
    horizon: int = 50,
    n_process: int = 200,
    seed: Optional[int] = None,
    include_last: bool = False,
    drift_overrides: Optional[np.ndarray] = None,
    scale_sigma: float | np.ndarray = 1.0,
    sigma_overrides: Optional[np.ndarray] = None,
    # --- CRN innovations (optional) ---
    eps_rw: Optional[np.ndarray] = None,  # (B, n_process, H) for LC/APC
    eps1: Optional[np.ndarray] = None,  # (B, n_process, H) for CBD
    eps2: Optional[np.ndarray] = None,  # (B, n_process, H) for CBD
    eps3: Optional[np.ndarray] = None,  # (B, n_process, H) for CBDM7 only
) -> ProjectionResult:
    """
    Forecast mortality by combining bootstrap parameter uncertainty and RW process risk.
    """
    rng_master = np.random.default_rng(seed)

    params_list = bootstrap_result.params_list
    mu_sigma_mat = np.asarray(bootstrap_result.mu_sigma)

    B = len(params_list)
    if mu_sigma_mat.shape[0] != B:
        raise ValueError(
            "bootstrap_result.mu_sigma must have same length as params_list."
        )

    A = len(ages)
    H = int(horizon)
    n_process = int(n_process)
    if H <= 0 or n_process <= 0:
        raise ValueError("horizon and n_process must be > 0.")

    if include_last:
        years_future = np.arange(int(years[-1]), int(years[-1]) + H + 1)
        H_out = H + 1
    else:
        years_future = np.arange(int(years[-1]) + 1, int(years[-1]) + 1 + H)
        H_out = H

    is_cbd = "CBD" in model_cls.__name__.upper()

    N = B * n_process
    q_paths = np.zeros((N, A, H_out), dtype=float)
    m_paths = None if is_cbd else np.zeros((N, A, H_out), dtype=float)

    # k_paths: preallocate depending on model family
    if is_cbd:
        d_mu = mu_sigma_mat.shape[1]
        if d_mu == 4:
            d_factors = 2
        elif d_mu == 6:
            d_factors = 3
        else:
            raise ValueError(
                f"CBD projection: expected mu_sigma with 4 (M5/M6) or 6 (M7) columns, got {d_mu}."
            )
        k_paths = np.zeros((N, d_factors, H_out), dtype=float)
    else:
        k_paths = np.zeros((N, H_out), dtype=float)

    # drift_overrides validation
    if drift_overrides is not None:
        drift_overrides = np.asarray(drift_overrides, dtype=float).reshape(-1)
        expected_len = 1 if not is_cbd else d_factors
        if drift_overrides.shape[0] != expected_len:
            raise ValueError(
                f"drift_overrides must have length {expected_len} for this model, got {drift_overrides.shape[0]}."
            )
        if not np.all(np.isfinite(drift_overrides)):
            raise ValueError("drift_overrides must be finite.")

    # --- sigma handling (scale and/or overrides) ---
    if scale_sigma is None:
        scale_sigma = 1.0

    if is_cbd:
        scale_vec = np.asarray(scale_sigma, dtype=float).reshape(-1)
        if scale_vec.size == 1:
            scale_vec = np.full(d_factors, float(scale_vec[0]))
        if scale_vec.size != d_factors:
            raise ValueError(
                f"scale_sigma must have length 1 or {d_factors} for CBD, got {scale_vec.size}."
            )
    else:
        scale_vec = np.asarray(scale_sigma, dtype=float).reshape(-1)
        if scale_vec.size != 1:
            raise ValueError(
                f"scale_sigma must be scalar (or length 1) for LC/APC, got {scale_vec.size}."
            )
        scale_vec = np.array([float(scale_vec[0])])

    if not np.all(np.isfinite(scale_vec)) or np.any(scale_vec <= 0.0):
        raise ValueError("scale_sigma must be finite and > 0.")

    if sigma_overrides is not None:
        sigma_overrides = np.asarray(sigma_overrides, dtype=float).reshape(-1)
        expected = d_factors if is_cbd else 1
        if sigma_overrides.size != expected:
            raise ValueError(
                f"sigma_overrides must have length {expected}, got {sigma_overrides.size}."
            )
        if not np.all(np.isfinite(sigma_overrides)) or np.any(sigma_overrides <= 0.0):
            raise ValueError("sigma_overrides must be finite and > 0.")

    # --- CRN validation helpers ---
    def _check_eps(name: str, arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=float)
        if arr.shape != (B, n_process, H):
            raise ValueError(
                f"{name} must have shape (B, n_process, H)=({B},{n_process},{H}), got {arr.shape}."
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite.")
        return arr

    if is_cbd:
        eps1 = _check_eps("eps1", eps1)
        eps2 = _check_eps("eps2", eps2)
        eps3 = _check_eps("eps3", eps3)
        # If user uses CRN for CBD, eps1 and eps2 must be provided.
        if (eps1 is None) != (eps2 is None):
            raise ValueError("For CBD CRN, provide BOTH eps1 and eps2 (or neither).")
    else:
        eps_rw = _check_eps("eps_rw", eps_rw)

    out = 0
    for b in range(B):
        params = params_list[b]
        if params is None:
            raise RuntimeError(
                f"bootstrap_result.params_list[{b}] is None; did the bootstrap fit fail?"
            )

        mu_sigma = mu_sigma_mat[b].copy()

        # Apply drift overrides if requested
        if drift_overrides is not None:
            if is_cbd and len(mu_sigma) == 4:
                mu_sigma[0] = drift_overrides[0]
                mu_sigma[2] = drift_overrides[1]
            elif is_cbd and len(mu_sigma) == 6:
                mu_sigma[0] = drift_overrides[0]
                mu_sigma[2] = drift_overrides[1]
                mu_sigma[4] = drift_overrides[2]
            elif (not is_cbd) and len(mu_sigma) == 2:
                mu_sigma[0] = drift_overrides[0]
            else:
                raise RuntimeError(
                    "Unexpected combination of drift_overrides and bootstrap mu_sigma."
                )

        rng_b = np.random.default_rng(rng_master.integers(0, 2**32 - 1))

        # --- choose eps for this bootstrap replicate (CRN if provided) ---
        if is_cbd:
            if eps1 is None:
                eps1_b = rng_b.normal(size=(n_process, H))
                eps2_b = rng_b.normal(size=(n_process, H))
                eps3_b = (
                    rng_b.normal(size=(n_process, H)) if mu_sigma.size == 6 else None
                )
            else:
                eps1_b = eps1[b]
                eps2_b = eps2[b]
                eps3_b = eps3[b] if (mu_sigma.size == 6) else None
                if mu_sigma.size == 6 and eps3 is None:
                    raise ValueError("eps3 must be provided for CBDM7 when using CRN.")
        else:
            if eps_rw is None:
                eps_rw_b = rng_b.normal(size=(n_process, H))
            else:
                eps_rw_b = eps_rw[b]

        # CBD family: logit(q)
        if is_cbd:
            if len(mu_sigma) == 4:
                mu1, sig1, mu2, sig2 = mu_sigma

                if sigma_overrides is None:
                    sig1_eff = float(sig1) * float(scale_vec[0])
                    sig2_eff = float(sig2) * float(scale_vec[1])
                else:
                    sig1_eff = float(sigma_overrides[0])
                    sig2_eff = float(sigma_overrides[1])

                k1_block = simulate_random_walk_paths_with_eps(
                    params.kappa1[-1], mu1, sig1_eff, eps1_b, include_last=include_last
                )
                k2_block = simulate_random_walk_paths_with_eps(
                    params.kappa2[-1], mu2, sig2_eff, eps2_b, include_last=include_last
                )

                z = ages - params.x_bar
                logit_q_block = (
                    k1_block[:, None, :] + z[None, :, None] * k2_block[:, None, :]
                )

                if hasattr(params, "gamma_for_age_at_last_year"):
                    gamma_last = np.array(
                        [params.gamma_for_age_at_last_year(float(ax)) for ax in ages],
                        dtype=float,
                    )
                    logit_q_block += gamma_last[None, :, None]

                q_block = 1.0 / (1.0 + np.exp(-logit_q_block))

                q_paths[out : out + n_process] = q_block
                k_paths[out : out + n_process, 0, :] = k1_block
                k_paths[out : out + n_process, 1, :] = k2_block
                out += n_process
                continue

            if len(mu_sigma) == 6:
                mu1, sig1, mu2, sig2, mu3, sig3 = mu_sigma

                if sigma_overrides is None:
                    sig1_eff = float(sig1) * float(scale_vec[0])
                    sig2_eff = float(sig2) * float(scale_vec[1])
                    sig3_eff = float(sig3) * float(scale_vec[2])
                else:
                    sig1_eff = float(sigma_overrides[0])
                    sig2_eff = float(sigma_overrides[1])
                    sig3_eff = float(sigma_overrides[2])

                k1_block = simulate_random_walk_paths_with_eps(
                    params.kappa1[-1], mu1, sig1_eff, eps1_b, include_last=include_last
                )
                k2_block = simulate_random_walk_paths_with_eps(
                    params.kappa2[-1], mu2, sig2_eff, eps2_b, include_last=include_last
                )
                k3_block = simulate_random_walk_paths_with_eps(
                    params.kappa3[-1], mu3, sig3_eff, eps3_b, include_last=include_last
                )

                z = ages - params.x_bar
                z2c = z**2 - params.sigma2_x

                logit_q_block = (
                    k1_block[:, None, :]
                    + z[None, :, None] * k2_block[:, None, :]
                    + z2c[None, :, None] * k3_block[:, None, :]
                )

                if hasattr(params, "gamma_for_age_at_last_year"):
                    gamma_last = np.array(
                        [params.gamma_for_age_at_last_year(float(ax)) for ax in ages],
                        dtype=float,
                    )
                    logit_q_block += gamma_last[None, :, None]

                q_block = 1.0 / (1.0 + np.exp(-logit_q_block))

                q_paths[out : out + n_process] = q_block
                k_paths[out : out + n_process, 0, :] = k1_block
                k_paths[out : out + n_process, 1, :] = k2_block
                k_paths[out : out + n_process, 2, :] = k3_block
                out += n_process
                continue

            raise RuntimeError("Unexpected mu_sigma length for CBD model.")

        # LC / APC family: log(m)
        else:
            if len(mu_sigma) != 2:
                raise ValueError(
                    "LC/APC bootstrap mu_sigma must have length 2 (mu, sigma)."
                )
            mu, sigma = mu_sigma

            # LC-like
            if hasattr(params, "k") and hasattr(params, "a") and hasattr(params, "b"):
                k_last = params.k[-1]
                a = params.a
                b_age = params.b
                if a.shape[0] != A:
                    raise ValueError(
                        f"Projection: params have {a.shape[0]} ages but ages grid has {A}."
                    )

            # APC-like
            elif hasattr(params, "kappa") and hasattr(params, "beta_age"):
                k_last = params.kappa[-1]
                a = params.beta_age
                b_age = np.ones_like(a)
                if a.shape[0] != A:
                    raise ValueError(
                        f"Projection: params have {a.shape[0]} ages but ages grid has {A}."
                    )

            else:
                raise RuntimeError(
                    "Unknown parameter structure: expected LC params (a,b,k) or APC params (beta_age,kappa)."
                )

            if sigma_overrides is None:
                sigma_eff = float(sigma) * float(scale_vec[0])
            else:
                sigma_eff = float(sigma_overrides[0])

            k_block = simulate_random_walk_paths_with_eps(
                k_last, mu, sigma_eff, eps_rw_b, include_last=include_last
            )

            ln_m_block = a[None, :, None] + b_age[None, :, None] * k_block[:, None, :]

            if hasattr(params, "gamma_for_age_at_last_year"):
                gamma_last = np.array(
                    [params.gamma_for_age_at_last_year(float(ax)) for ax in ages],
                    dtype=float,
                )
                ln_m_block += gamma_last[None, :, None]

            m_block = np.exp(ln_m_block)
            q_block = m_to_q(m_block)

            q_paths[out : out + n_process] = q_block
            m_paths[out : out + n_process] = m_block
            k_paths[out : out + n_process] = k_block
            out += n_process

    validate_q(q_paths)

    return ProjectionResult(
        years=years_future,
        q_paths=q_paths,
        m_paths=m_paths,
        k_paths=k_paths,
    )
