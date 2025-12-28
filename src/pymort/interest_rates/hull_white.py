from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class InterestRateScenarioSet:
    """Container for simulated short-rate and discount factor paths.

    Attributes:
    ----------
    r_paths : np.ndarray
        Short-rate paths, shape (N, T).
    discount_factors : np.ndarray
        Discount factors along each path, shape (N, T).
    times : np.ndarray
        Time grid in years, shape (T,).
    metadata : dict
        Optional metadata (parameters, curve info, seed, etc.).
    """

    r_paths: np.ndarray
    discount_factors: np.ndarray
    times: np.ndarray
    metadata: dict[str, Any]

    def n_scenarios(self) -> int:
        return int(self.r_paths.shape[0])

    def horizon(self) -> int:
        return int(self.times.shape[0])


def _fwd_from_zero(times: np.ndarray, zero_rates: np.ndarray) -> np.ndarray:
    """Approximate instantaneous forward f(0,t) from zero curve z(t).
    z(t) assumed continuous-compounded: P(0,t)=exp(-z(t)*t).
    """
    times = np.asarray(times, dtype=float)
    z = np.asarray(zero_rates, dtype=float)
    if times.shape != z.shape:
        raise ValueError("times and zero_rates must have the same shape.")
    if np.any(times <= 0):
        raise ValueError("times must be strictly positive for forward rate derivation.")
    # f(0,t) = d/dt (t z(t))
    tz = times * z
    dt = np.diff(times, prepend=times[0] - (times[1] - times[0]))
    dt[dt == 0] = 1e-8
    deriv = np.gradient(tz, times)
    return deriv


def calibrate_theta_from_zero_curve(
    times: np.ndarray, zero_rates: np.ndarray, a: float, sigma: float
) -> np.ndarray:
    """Calibrate deterministic theta(t) to fit the initial zero curve under Hull–White.
    Formula: theta(t) = f(0,t) + (1/a) * df/dt + (sigma^2/(2a^2)) * (1 - e^{-2 a t})
    using finite differences for df/dt.
    """
    times = np.asarray(times, dtype=float)
    z = np.asarray(zero_rates, dtype=float)
    if times.ndim != 1 or z.ndim != 1:
        raise ValueError("times and zero_rates must be 1D.")
    f0 = _fwd_from_zero(times, z)
    df_dt = np.gradient(f0, times)
    a = float(a)
    sigma = float(sigma)
    if a <= 0.0 or sigma < 0.0:
        raise ValueError("a must be >0 and sigma >=0.")
    theta = f0 + df_dt / a + (sigma**2) / (2 * a**2) * (1.0 - np.exp(-2 * a * times))
    return theta


def simulate_hull_white_paths(
    *,
    a: float,
    theta: np.ndarray,
    sigma: float,
    r0: float,
    times: np.ndarray,
    n_scenarios: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate short-rate paths and discount factors under Hull–White (1-factor).
    Exact discretization (Gaussian).

    Returns:
    -------
    r_paths : (N, T)
    discount_factors : (N, T)
    """
    a = float(a)
    sigma = float(sigma)
    if a <= 0.0:
        raise ValueError("Mean-reversion a must be positive.")
    times = np.asarray(times, dtype=float)
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing.")
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if theta.shape[0] != times.shape[0]:
        raise ValueError("theta must have same length as times.")

    n = int(n_scenarios)
    T = times.shape[0]
    rng = np.random.default_rng(seed)

    r_paths = np.zeros((n, T), dtype=float)
    df_paths = np.zeros((n, T), dtype=float)
    r_paths[:, 0] = r0
    dt = np.diff(times, prepend=times[0])

    for t in range(1, T):
        dt_t = dt[t]
        if dt_t <= 0:
            dt_t = 1e-8
        exp_term = np.exp(-a * dt_t)
        mean = r_paths[:, t - 1] * exp_term + theta[t - 1] * (1.0 - exp_term)
        var = sigma**2 * (1.0 - np.exp(-2.0 * a * dt_t)) / (2.0 * a)
        eps = rng.normal(size=n)
        r_paths[:, t] = mean + np.sqrt(var) * eps

    # discount factors: cumulative integral of r
    # simple trapezoidal rule
    dt_all = np.diff(times, prepend=0.0)
    # average rate between steps for integral
    r_mid = 0.5 * (r_paths[:, :-1] + r_paths[:, 1:])
    integrals = np.cumsum(r_mid * dt_all[1:], axis=1)
    # align to (N,T)
    df_paths[:, 0] = np.exp(-r_paths[:, 0] * times[0])
    df_paths[:, 1:] = np.exp(-integrals)

    return r_paths, df_paths


def build_interest_rate_scenarios(
    *,
    times: np.ndarray,
    zero_rates: np.ndarray,
    a: float,
    sigma: float,
    n_scenarios: int,
    r0: float | None = None,
    seed: int | None = None,
) -> InterestRateScenarioSet:
    """Convenience wrapper: calibrate theta to zero curve, simulate HW paths,
    and return an InterestRateScenarioSet.
    """
    times = np.asarray(times, dtype=float)
    zero_rates = np.asarray(zero_rates, dtype=float)
    if r0 is None:
        r0 = float(zero_rates[0])

    theta = calibrate_theta_from_zero_curve(times, zero_rates, a=a, sigma=sigma)
    r_paths, df_paths = simulate_hull_white_paths(
        a=a,
        theta=theta,
        sigma=sigma,
        r0=r0,
        times=times,
        n_scenarios=n_scenarios,
        seed=seed,
    )

    metadata: dict[str, Any] = {
        "model": "Hull-White 1F",
        "a": float(a),
        "sigma": float(sigma),
        "r0": float(r0),
        "seed": seed,
    }
    return InterestRateScenarioSet(
        r_paths=r_paths,
        discount_factors=df_paths,
        times=times,
        metadata=metadata,
    )


def save_ir_scenarios_npz(ir_set: InterestRateScenarioSet, path: str) -> None:
    np.savez_compressed(
        path,
        r_paths=ir_set.r_paths,
        discount_factors=ir_set.discount_factors,
        times=ir_set.times,
        metadata=np.array(ir_set.metadata, dtype=object),
    )


def load_ir_scenarios_npz(path: str) -> InterestRateScenarioSet:
    data = np.load(path, allow_pickle=True)
    metadata = data["metadata"].item() if "metadata" in data else {}
    return InterestRateScenarioSet(
        r_paths=np.asarray(data["r_paths"]),
        discount_factors=np.asarray(data["discount_factors"]),
        times=np.asarray(data["times"]),
        metadata=metadata,
    )
