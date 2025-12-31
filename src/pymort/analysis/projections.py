"""Stochastic mortality projections with parameter and process uncertainty.

This module builds forward mortality scenarios by combining:
    1) Parameter uncertainty via residual bootstrap (BootstrapResult with
       `params_list` and `mu_sigma`).
    2) Process uncertainty via random-walk-with-drift factors:
       - LC/APC models: k_t or kappa_t
       - CBD models: kappa1_t, kappa2_t, and optionally kappa3_t

The engine is vectorized: for each bootstrap replicate b, we simulate
`n_process` paths in one shot. Total scenarios N = B * n_process.

Expected bootstrap contract:
    params_list: List of fitted parameter objects (length B).
    mu_sigma: Array of shape (B, d) with drift/vol estimates, where
        d=2 (LCM1/LCM2/APCM3), d=4 (CBDM5/CBDM6), d=6 (CBDM7).

CRN (common random numbers):
    Pass eps arrays to reuse the same shocks across runs and reduce
    Monte Carlo noise in finite differences.
    - LC/APC: eps_rw shape (B, n_process, H)
    - CBD: eps1/eps2 shape (B, n_process, H)
    - CBDM7: eps3 shape (B, n_process, H)

Note:
    Docstrings follow Google style for clarity and spec alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np

from pymort._types import AnyArray, FloatArray, IntArray
from pymort.analysis.bootstrap import BootstrapResult
from pymort.lifetables import m_to_q, validate_q


class _CBDParamsBase(Protocol):
    kappa1: FloatArray
    kappa2: FloatArray
    x_bar: float


class _CBDM7Params(_CBDParamsBase, Protocol):
    kappa3: FloatArray
    sigma2_x: float


class _HasGamma(Protocol):
    def gamma_for_age_at_last_year(self, age: float) -> float: ...


@dataclass
class ProjectionResult:
    """Container for projection outputs.

    Attributes:
        years (np.ndarray): Projection years, shape (H_out,).
        q_paths (np.ndarray): Death probabilities, shape (N, A, H_out).
        m_paths (np.ndarray | None): Central death rates for log-m models,
            shape (N, A, H_out), or None for logit-q models.
        k_paths (np.ndarray | None): Factor paths, shape (N, H_out) for LC/APC
            or (N, d_factors, H_out) for CBD, or None if not stored.
    """

    years: IntArray  # (H_out,)
    q_paths: FloatArray  # (N, A, H_out)
    m_paths: FloatArray | None  # (N, A, H_out) for log-m models, else None
    k_paths: FloatArray | None  # (N, H_out) for LC/APC, or (N, d_factors, H_out) for CBD


def simulate_random_walk_paths(
    k_last: float,
    mu: float,
    sigma: float,
    horizon: int,
    n_sims: int,
    rng: np.random.Generator,
    include_last: bool = False,
) -> FloatArray:
    """Simulate random-walk-with-drift paths (vectorized).

    The model is:
        k_t = k_{t-1} + mu + sigma * eps_t

    Args:
        k_last: Last observed factor value.
        mu: Drift of the random walk.
        sigma: Volatility of the random walk.
        horizon: Number of future steps (H).
        n_sims: Number of simulated paths (N).
        rng: NumPy random generator.
        include_last: If True, prepend k_last to each path.

    Returns:
        Array of shape (N, H) or (N, H+1) if include_last is True.

    Raises:
        ValueError: If horizon or n_sims are non-positive, or parameters are invalid.
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
        paths = np.concatenate([np.full((n_sims, 1), k_last, dtype=float), paths], axis=1)
    return paths


def simulate_random_walk_paths_with_eps(
    k_last: float,
    mu: float,
    sigma: float,
    eps: FloatArray,
    include_last: bool = False,
) -> FloatArray:
    """Simulate random-walk paths using pre-drawn eps (CRN-friendly).

    Args:
        k_last: Last observed factor value.
        mu: Drift of the random walk.
        sigma: Volatility of the random walk.
        eps: Pre-drawn innovations, shape (N, H).
        include_last: If True, prepend k_last to each path.

    Returns:
        Array of shape (N, H) or (N, H+1) if include_last is True.

    Raises:
        ValueError: If eps is not 2D or contains non-finite values.
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
        paths = np.concatenate([np.full((paths.shape[0], 1), k_last, dtype=float), paths], axis=1)
    return paths


def project_mortality_from_bootstrap(
    model_cls: type,
    ages: AnyArray,
    years: AnyArray,
    m: FloatArray,
    bootstrap_result: BootstrapResult,
    horizon: int = 50,
    n_process: int = 200,
    seed: int | None = None,
    include_last: bool = False,
    drift_overrides: FloatArray | None = None,
    scale_sigma: float | FloatArray = 1.0,
    sigma_overrides: FloatArray | None = None,
    # --- CRN innovations (optional) ---
    eps_rw: FloatArray | None = None,  # (B, n_process, H) for LC/APC
    eps1: FloatArray | None = None,  # (B, n_process, H) for CBD
    eps2: FloatArray | None = None,  # (B, n_process, H) for CBD
    eps3: FloatArray | None = None,  # (B, n_process, H) for CBDM7 only
) -> ProjectionResult:
    """Project mortality by combining bootstrap and process uncertainty.

    Args:
        model_cls: Model class used to generate projections.
        ages: Age grid, shape (A,).
        years: Year grid, shape (T,).
        m: Central death rates, shape (A, T).
        bootstrap_result: BootstrapResult with params_list and mu_sigma.
        horizon: Projection horizon in years.
        n_process: Number of process simulations per bootstrap replicate.
        seed: RNG seed.
        include_last: If True, include the last observed year in output.
        drift_overrides: Optional drift overrides (length 1 for LC/APC, or
            length d_factors for CBD).
        scale_sigma: Scale factor(s) for sigma (scalar or length d_factors).
        sigma_overrides: Optional sigma overrides (length 1 for LC/APC, or
            length d_factors for CBD).
        eps_rw: Pre-drawn innovations for LC/APC, shape (B, n_process, H).
        eps1: Pre-drawn innovations for CBD factor 1, shape (B, n_process, H).
        eps2: Pre-drawn innovations for CBD factor 2, shape (B, n_process, H).
        eps3: Pre-drawn innovations for CBD factor 3, shape (B, n_process, H).

    Returns:
        ProjectionResult with q_paths, optional m_paths, and factor paths.
    """
    rng_master = np.random.default_rng(seed)

    params_list = bootstrap_result.params_list
    mu_sigma_mat = np.asarray(bootstrap_result.mu_sigma)

    B = len(params_list)
    if mu_sigma_mat.shape[0] != B:
        raise ValueError("bootstrap_result.mu_sigma must have same length as params_list.")
    m = np.asarray(m, dtype=float)
    if m.shape != (len(ages), len(years)):
        raise ValueError(f"m must have shape (A,T)=({len(ages)},{len(years)}), got {m.shape}.")
    if not np.isfinite(m).all():
        raise ValueError("m must be finite.")

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
        k_paths = cast(FloatArray, np.zeros((N, d_factors, H_out), dtype=float))
    else:
        k_paths = cast(FloatArray, np.zeros((N, H_out), dtype=float))

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

    scale_vec: FloatArray
    if is_cbd:
        scale_vec = cast(FloatArray, np.asarray(scale_sigma, dtype=float).reshape(-1))
        if scale_vec.size == 1:
            scale_vec = cast(FloatArray, np.full(d_factors, float(scale_vec[0])))
        if scale_vec.size != d_factors:
            raise ValueError(
                f"scale_sigma must have length 1 or {d_factors} for CBD, got {scale_vec.size}."
            )
    else:
        scale_vec = cast(FloatArray, np.asarray(scale_sigma, dtype=float).reshape(-1))
        if scale_vec.size != 1:
            raise ValueError(
                f"scale_sigma must be scalar (or length 1) for LC/APC, got {scale_vec.size}."
            )
        scale_vec = cast(FloatArray, np.array([float(scale_vec[0])]))

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
    def _check_eps(name: str, arr: FloatArray | None) -> FloatArray | None:
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
                eps3_b = rng_b.normal(size=(n_process, H)) if mu_sigma.size == 6 else None
            else:
                if eps2 is None:
                    raise ValueError("For CBD CRN, provide BOTH eps1 and eps2 (or neither).")
                eps1_b = eps1[b]
                eps2_b = eps2[b]
                if mu_sigma.size == 6:
                    if eps3 is None:
                        raise ValueError("eps3 must be provided for CBDM7 when using CRN.")
                    eps3_b = eps3[b]
                else:
                    eps3_b = None
        elif eps_rw is None:
            eps_rw_b = rng_b.normal(size=(n_process, H))
        else:
            eps_rw_b = eps_rw[b]

        # CBD family: logit(q)
        if is_cbd:
            if len(mu_sigma) == 4:
                mu1, sig1, mu2, sig2 = mu_sigma
                params_cbd = cast(_CBDParamsBase, params)

                if sigma_overrides is None:
                    sig1_eff = float(sig1) * float(scale_vec[0])
                    sig2_eff = float(sig2) * float(scale_vec[1])
                else:
                    sig1_eff = float(sigma_overrides[0])
                    sig2_eff = float(sigma_overrides[1])

                k1_block = simulate_random_walk_paths_with_eps(
                    params_cbd.kappa1[-1], mu1, sig1_eff, eps1_b, include_last=include_last
                )
                k2_block = simulate_random_walk_paths_with_eps(
                    params_cbd.kappa2[-1], mu2, sig2_eff, eps2_b, include_last=include_last
                )

                z = ages - params_cbd.x_bar
                logit_q_block = k1_block[:, None, :] + z[None, :, None] * k2_block[:, None, :]

                if hasattr(params, "gamma_for_age_at_last_year"):
                    params_gamma = cast(_HasGamma, params)
                    gamma_last = np.array(
                        [params_gamma.gamma_for_age_at_last_year(float(ax)) for ax in ages],
                        dtype=float,
                    )
                    logit_q_block += gamma_last[None, :, None]

                q_block = cast(FloatArray, 1.0 / (1.0 + np.exp(-logit_q_block)))

                q_paths[out : out + n_process] = q_block
                k_paths[out : out + n_process, 0, :] = k1_block
                k_paths[out : out + n_process, 1, :] = k2_block
                out += n_process
                continue

            if len(mu_sigma) == 6:
                mu1, sig1, mu2, sig2, mu3, sig3 = mu_sigma
                params_cbd7 = cast(_CBDM7Params, params)

                if sigma_overrides is None:
                    sig1_eff = float(sig1) * float(scale_vec[0])
                    sig2_eff = float(sig2) * float(scale_vec[1])
                    sig3_eff = float(sig3) * float(scale_vec[2])
                else:
                    sig1_eff = float(sigma_overrides[0])
                    sig2_eff = float(sigma_overrides[1])
                    sig3_eff = float(sigma_overrides[2])

                k1_block = simulate_random_walk_paths_with_eps(
                    params_cbd7.kappa1[-1], mu1, sig1_eff, eps1_b, include_last=include_last
                )
                k2_block = simulate_random_walk_paths_with_eps(
                    params_cbd7.kappa2[-1], mu2, sig2_eff, eps2_b, include_last=include_last
                )
                if eps3_b is None:
                    raise RuntimeError("eps3_b must be set for CBDM7 projections.")
                k3_block = simulate_random_walk_paths_with_eps(
                    params_cbd7.kappa3[-1], mu3, sig3_eff, eps3_b, include_last=include_last
                )

                z = ages - params_cbd7.x_bar
                z2c = z**2 - params_cbd7.sigma2_x

                logit_q_block = (
                    k1_block[:, None, :]
                    + z[None, :, None] * k2_block[:, None, :]
                    + z2c[None, :, None] * k3_block[:, None, :]
                )

                if hasattr(params, "gamma_for_age_at_last_year"):
                    params_gamma = cast(_HasGamma, params)
                    gamma_last = np.array(
                        [params_gamma.gamma_for_age_at_last_year(float(ax)) for ax in ages],
                        dtype=float,
                    )
                    logit_q_block += gamma_last[None, :, None]

                q_block = cast(FloatArray, 1.0 / (1.0 + np.exp(-logit_q_block)))

                q_paths[out : out + n_process] = q_block
                k_paths[out : out + n_process, 0, :] = k1_block
                k_paths[out : out + n_process, 1, :] = k2_block
                k_paths[out : out + n_process, 2, :] = k3_block
                out += n_process
                continue

            raise RuntimeError("Unexpected mu_sigma length for CBD model.")

        # LC / APC family: log(m)
        if len(mu_sigma) != 2:
            raise ValueError("LC/APC bootstrap mu_sigma must have length 2 (mu, sigma).")
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
        if m_paths is None:
            raise RuntimeError("m_paths should be initialized for LC/APC projections.")
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
