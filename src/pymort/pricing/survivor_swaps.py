from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymort.analysis import MortalityScenarioSet
from pymort.pricing.utils import (
    build_discount_factors,
    cohort_survival_full_horizon_from_q,
    find_nearest_age_index,
    pv_from_cf_paths,
)


@dataclass
class SurvivorSwapSpec:
    """Specification of a simple survivor swap on a single cohort.

    At each payment date t = 1, ..., T (years from start):

        Floating leg:  notional * S_{x}(t)
        Fixed leg:     notional * K

    Net cashflow from the perspective of the payer:

        - if payer == "fixed":
            CF_t = notional * (S_x(t) - K)
        - if payer == "floating":
            CF_t = notional * (K - S_x(t))

    where S_x(t) is read from scen_set.S_paths at the chosen age.

    Attributes:
    ----------
    age : float
        Age of the cohort at valuation date (e.g. 65). The closest age
        in `scen_set.ages` is used internally.
    maturity_years : int
        Number of years from the first projection year to the swap maturity.
        Payments are assumed annual and aligned with the mortality grid.
    notional : float
        Notional of the swap (per unit of survival index).
    strike : float | None
        Fixed survival level K. If None, K is chosen so that the swap is
        at-the-money (zero value) at t=0 under the scenario measure:
            K = sum_t D_t * E[S_x(t)] / sum_t D_t
    payer : {"fixed", "floating"}
        Which leg the holder pays:
            - "fixed"    : pays fixed, receives floating
            - "floating" : pays floating, receives fixed
    Optional payment schedule in years from valuation date.

    - If None: default annual payments at t = 1, 2, ..., maturity_years.
    - If provided: must be a 1D array of strictly positive integers (aligned with
      the mortality grid yearly steps), each <= maturity_years.

    Examples:
    --------
    Annual: None  -> [1,2,...,T]
    Custom: np.array([1,3,5,10])  (pays only at these maturities)
    """

    age: float
    maturity_years: int
    notional: float = 1.0
    strike: float | None = None
    payer: str = "fixed"  # "fixed" or "floating"
    payment_times: np.ndarray | None = None


def price_survivor_swap(
    scen_set: MortalityScenarioSet,
    spec: SurvivorSwapSpec,
    *,
    short_rate: float | None = None,
    discount_factors: np.ndarray | None = None,
    return_cf_paths: bool = False,
) -> dict[str, Any]:
    """Price a survivor swap using mortality scenarios.

    Net cashflow at each payment date t = 1, ..., T:

        if payer == "fixed":
            CF_t = notional * (S_x(t) - K)
        else:  # payer == "floating"
            CF_t = notional * (K - S_x(t))

    Present value is the discounted expectation over scenarios.
    """
    # Basic checks on scenario set
    S_paths = np.asarray(scen_set.S_paths, dtype=float)
    if S_paths.ndim != 3:
        raise ValueError(f"Expected S_paths with shape (N, A, H), got {S_paths.shape}.")

    N, A, H_full = S_paths.shape

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

    if spec.payer == "fixed":
        # Pays fixed, receives floating: CF_t = N(S - K)
        cf = spec.notional * (S_age - K)  # (N, P)
    else:
        # Pays floating, receives fixed: CF_t = N(K - S)
        cf = spec.notional * (K - S_age)  # (N, P)

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

    metadata: dict[str, Any] = {
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

    payload: dict[str, Any] = {
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
