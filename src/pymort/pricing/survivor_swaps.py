"""Survivor swap pricing helpers.

Note:
    Docstrings follow Google style and type hints use NDArray for clarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np

from pymort._types import FloatArray, IntArray
from pymort.analysis import MortalityScenarioSet
from pymort.pricing.utils import (
    build_discount_factors,
    cohort_survival_full_horizon_from_q,
    find_nearest_age_index,
    pv_from_cf_paths,
)


class SurvivorSwapPricingResult(TypedDict, total=False):
    """Typed payload for survivor swap pricing."""

    price: float
    pv_paths: FloatArray
    age_index: int
    discount_factors: FloatArray
    discount_factors_input: FloatArray
    strike: float
    expected_cashflows: FloatArray
    metadata: dict[str, object]
    cf_paths: FloatArray
    times: IntArray


@dataclass
class SurvivorSwapSpec:
    """Specification for a simple survivor swap on a single cohort.

    The net cashflow (from the payer's perspective) at payment date t is:
    - payer == "fixed": notional * (S_x(t) - K)
    - payer == "floating": notional * (K - S_x(t))

    Attributes:
        age: Cohort age at valuation.
        maturity_years: Swap maturity in years.
        notional: Swap notional (per unit of survival index).
        strike: Fixed survival level K. If None, an ATM strike is used.
        payer: "fixed" or "floating".
        payment_times: Optional payment schedule as year offsets (1..T).
    """

    age: float
    maturity_years: int
    notional: float = 1.0
    strike: float | None = None
    payer: str = "fixed"  # "fixed" or "floating"
    payment_times: IntArray | None = None


def price_survivor_swap(
    scen_set: MortalityScenarioSet,
    spec: SurvivorSwapSpec,
    *,
    short_rate: float | None = None,
    discount_factors: FloatArray | None = None,
    return_cf_paths: bool = False,
) -> SurvivorSwapPricingResult:
    """Price a survivor swap using mortality scenarios.

    Net cashflow at each payment date t = 1, ..., T:

        if payer == "fixed":
            CF_t = notional * (S_x(t) - K)
        else:  # payer == "floating"
            CF_t = notional * (K - S_x(t))

    Present value is the discounted expectation over scenarios.

    Args:
        scen_set: Scenario set with S_paths of shape (N, A, H).
        spec: Survivor swap specification.
        short_rate: Flat continuous rate used when no discount factors are provided.
        discount_factors: Discount factors of shape (H,) or (N, H).
        return_cf_paths: Whether to include cashflow paths in the output.

    Returns:
        Pricing payload with price, PV paths, and metadata.
    """
    # Basic checks on scenario set
    S_paths = np.asarray(scen_set.S_paths, dtype=float)
    if S_paths.ndim != 3:
        raise ValueError(f"Expected S_paths with shape (N, A, H), got {S_paths.shape}.")

    N, _A, H_full = S_paths.shape

    # Maturity (max horizon we may need)
    T = int(spec.maturity_years)
    if T <= 0:
        raise ValueError("spec.maturity_years must be > 0.")
    if H_full < T:
        raise ValueError(f"spec.maturity_years={T} exceeds available projection horizon {H_full}.")

    # Payment schedule (indices in 1..T, aligned with yearly grid)
    if spec.payment_times is None:
        pay_times = np.arange(1, T + 1, dtype=int)  # 1..T
    else:
        pay_times = np.asarray(spec.payment_times, dtype=int).reshape(-1)
        if pay_times.size == 0:
            raise ValueError("spec.payment_times must be non-empty if provided.")
        if np.any(pay_times <= 0):
            raise ValueError("spec.payment_times must contain strictly positive integers.")
        if np.any(pay_times > T):
            raise ValueError(
                f"spec.payment_times must be <= maturity_years={T}. Got max={int(pay_times.max())}."
            )
        # unique + sorted to avoid double-counting
        pay_times = np.unique(pay_times)

    # Convert payment times (1..T) to zero-based indices for array slicing
    pay_idx = pay_times - 1  # 0..T-1

    # Age index
    age_idx = find_nearest_age_index(scen_set.ages, spec.age)

    # Survival for this cohort at payment dates: shape (N, P)
    S_age = S_paths[:, age_idx, pay_idx]  # (N, P)

    if not np.isfinite(S_age).all():
        q_paths = np.asarray(scen_set.q_paths, dtype=float)
        ages_grid = np.asarray(scen_set.ages, dtype=float)

        S_full = cohort_survival_full_horizon_from_q(
            q_paths=q_paths,
            ages=ages_grid,
            age0=float(spec.age),
            horizon=int(T),
            age_fit_min=80,
            age_fit_max=min(95, int(ages_grid.max())),
        )  # (N, T)

        S_age = S_full[:, pay_idx]

        if not np.isfinite(S_age).all():
            raise ValueError("Some S_age values are not finite even after fallback.")

    P = S_age.shape[1]

    # Discount factors for payment dates: shape (P,)
    df_full = build_discount_factors(
        scen_set=scen_set,
        short_rate=short_rate,
        discount_factors=discount_factors,
        H=T,
    )  # (T,)
    if df_full.ndim == 1:
        df_full_eff = df_full[None, :]  # (1,T)
    elif df_full.ndim == 2:
        if df_full.shape[0] not in (1, N):
            raise ValueError(
                f"discount_factors must have first dim 1 or N={N}; got {df_full.shape}."
            )
        df_full_eff = df_full if df_full.shape[0] == N else np.repeat(df_full, N, axis=0)
    else:
        raise ValueError("discount_factors must be 1D or 2D.")

    df = df_full_eff[:, pay_idx]  # (N,P) or (1,P)

    # Fixed strike: user-provided or ATM (zero-value) strike
    if spec.strike is None:
        # ATM strike so that PV(floating) = PV(fixed)
        # PV_floating = notional * sum_t D_t * E[S_x(t)]
        # PV_fixed    = notional * K * sum_t D_t
        # => K = sum_t D_t * E[S_x(t)] / sum_t D_t
        S_mean_vec = S_age.mean(axis=0)  # (P,)
        num = float(np.sum(df.mean(axis=0) * S_mean_vec))
        den = float(np.sum(df.mean(axis=0)))
        if den <= 0.0 or not np.isfinite(den):
            raise RuntimeError("Invalid discount factors when computing ATM strike.")
        K = num / den
    else:
        K = float(spec.strike)

    # Floating / fixed legs per scenario & time
    # Floating_t = notional * S_x(t)
    # Fixed_t    = notional * K
    if spec.payer not in ("fixed", "floating"):
        raise ValueError("spec.payer must be 'fixed' or 'floating'.")

    # Pays fixed, receives floating: CF_t = N(S - K)
    cf = (
        spec.notional * (S_age - K) if spec.payer == "fixed" else spec.notional * (K - S_age)
    )  # (N, P)

    # Build full-grid cashflow paths on annual grid (N,T), with zeros off-schedule
    cf_paths_full = np.zeros((N, T), dtype=float)  # (N,T)
    cf_paths_full[:, pay_idx] = cf  # insert scheduled CFs

    # Discount factors on full grid: (1,T) or (N,T)
    if df_full_eff.shape[1] != T:
        raise RuntimeError("Internal error: df_full_eff horizon mismatch.")
    pv_paths = pv_from_cf_paths(cf_paths_full, df_full_eff)  # (N,)
    price = float(pv_paths.mean())

    times = np.arange(1, T + 1, dtype=int)
    expected_cashflows = cf_paths_full.mean(axis=0)  # (T,)

    # Expected legs (useful diagnostics)
    df_pay_mean = df_full_eff[:, pay_idx].mean(axis=0)  # (P,)
    float_leg_expected = float(np.sum(spec.notional * S_age.mean(axis=0) * df_pay_mean))
    fixed_leg_expected = float(np.sum(spec.notional * K * df_pay_mean))

    metadata: dict[str, object] = {
        "N_scenarios": int(N),
        "age": float(spec.age),
        "age_index": int(age_idx),
        "maturity_years": T,
        "n_payments": int(P),
        "payment_times": pay_times.astype(int).tolist(),
        "notional": float(spec.notional),
        "strike": float(K),
        "payer": spec.payer,
        "float_leg_pv_expected": float(float_leg_expected),
        "fixed_leg_pv_expected": float(fixed_leg_expected),
    }

    payload: SurvivorSwapPricingResult = {
        "price": price,
        "pv_paths": pv_paths,
        "age_index": age_idx,
        "discount_factors": df_full_eff,  # (1,T) or (N,T)
        "discount_factors_input": df_full,
        "strike": K,
        "expected_cashflows": expected_cashflows,
        "metadata": metadata,
    }

    if return_cf_paths:
        payload["cf_paths"] = cf_paths_full  # (N,T)
        payload["times"] = times  # (T,)

    return payload
