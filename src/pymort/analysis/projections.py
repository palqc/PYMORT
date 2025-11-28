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

    Parameters
    ----------
    k_last : float
        Last observed value.
    mu, sigma : float
        Drift and volatility.
    horizon : int
        Number of future steps.
    n_sims : int
        Number of simulated paths.
    rng : np.random.Generator
        RNG for reproducibility.
    include_last : bool
        If True, prepend k_last as first column.

    Returns
    -------
    paths : np.ndarray
        Shape (n_sims, H) or (n_sims, H+1) if include_last=True.
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
) -> ProjectionResult:
    """
    Forecast mortality by combining bootstrap parameter uncertainty and RW process risk.

    Parameters
    ----------
    model_cls : Type
        One of (LCM1, LCM2, APCM3, CBDM5, CBDM6, CBDM7).
        Used only to detect family (CBD vs log-m).
    ages, years : np.ndarray
        Grids from the historical fit.
    m : np.ndarray
        Central death rates surface (A, T).
        Only used for conversion m->q for log-m models.
    bootstrap_result : BootstrapResult
        Output of bootstrap_from_m / bootstrap_logm_model / bootstrap_logitq_model.
    horizon : int
        Forecast horizon H (years ahead).
    n_process : int
        Number of RW innovations per bootstrap replicate.
        Total scenarios N = B * n_process.
    seed : int | None
        RNG seed.

    Returns
    -------
    ProjectionResult
        years_future: (H_out,)
        q_paths: (N, A, H_out)
        m_paths: (N, A, H_out) or None
        k_paths: (N, H_out) or (N, d_factors, H_out)
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
        # d_factors = 2 for M5/M6, 3 for M7 (inferred per bootstrap, must be constant!)
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
        # LC/APC have single period index
        k_paths = np.zeros((N, H_out), dtype=float)

    out = 0
    for b in range(B):
        params = params_list[b]
        if params is None:
            raise RuntimeError(
                f"bootstrap_result.params_list[{b}] is None; "
                "did the bootstrap fit fail?"
            )
        mu_sigma = mu_sigma_mat[b]
        rng_b = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        # CBD family: logit(q)
        if is_cbd:
            if len(mu_sigma) == 4:
                mu1, sig1, mu2, sig2 = mu_sigma
                k1_block = simulate_random_walk_paths(
                    params.kappa1[-1],
                    mu1,
                    sig1,
                    H,
                    n_process,
                    rng=rng_b,
                    include_last=include_last,
                )
                k2_block = simulate_random_walk_paths(
                    params.kappa2[-1],
                    mu2,
                    sig2,
                    H,
                    n_process,
                    rng=rng_b,
                    include_last=include_last,
                )

                z = ages - params.x_bar
                logit_q_block = (
                    k1_block[:, None, :] + z[None, :, None] * k2_block[:, None, :]
                )

                # cohort freeze if available
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
                k1_block = simulate_random_walk_paths(
                    params.kappa1[-1],
                    mu1,
                    sig1,
                    H,
                    n_process,
                    rng=rng_b,
                    include_last=include_last,
                )
                k2_block = simulate_random_walk_paths(
                    params.kappa2[-1],
                    mu2,
                    sig2,
                    H,
                    n_process,
                    rng=rng_b,
                    include_last=include_last,
                )
                k3_block = simulate_random_walk_paths(
                    params.kappa3[-1],
                    mu3,
                    sig3,
                    H,
                    n_process,
                    rng=rng_b,
                    include_last=include_last,
                )

                z = ages - params.x_bar
                z2c = z**2 - params.sigma2_x

                logit_q_block = (
                    k1_block[:, None, :]
                    + z[None, :, None] * k2_block[:, None, :]
                    + z2c[None, :, None] * k3_block[:, None, :]
                )

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

            k_block = simulate_random_walk_paths(
                k_last, mu, sigma, H, n_process, rng=rng_b, include_last=include_last
            )  # (n_process, H)

            ln_m_block = (
                a[None, :, None]  # (1, A, 1)
                + b_age[None, :, None] * k_block[:, None, :]  # (n_process, A, H)
            )

            # cohort freeze if available
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
