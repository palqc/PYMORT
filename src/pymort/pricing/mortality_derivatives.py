"""Mortality derivative pricing helpers (q-forwards and s-forwards).

Note:
    Docstrings follow Google style and type hints use NDArray for clarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from pymort.analysis import MortalityScenarioSet
from pymort.pricing.utils import (
    build_discount_factors,
    cohort_survival_full_horizon_from_q,
    find_nearest_age_index,
    pv_from_cf_paths,
)

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


class ForwardPricingResult(TypedDict, total=False):
    """Typed payload for forward pricing."""

    price: float
    pv_paths: FloatArray
    age_index: int
    measurement_index: int
    settlement_index: int
    strike: float
    discount_factors: FloatArray
    expected_cashflows: FloatArray
    metadata: dict[str, object]
    cf_paths: FloatArray
    times: IntArray


@dataclass
class QForwardSpec:
    """Specification for a q-forward.

    The contract measures q at Tm but settles at Ts (Ts >= Tm). Payoff:
        payoff_Ts = notional * (q_realised(Tm) - strike).
    """

    age: float
    maturity_years: int  # measurement Tm (years from start, 1..H)
    notional: float = 1.0
    strike: float | None = None  # if None, ATM (mean of q_{x,Tm})
    settlement_years: int | None = None  # if None -> Ts = Tm


@dataclass
class SForwardSpec:
    """Specification for an s-forward (survival forward).

    The contract measures S at Tm but settles at Ts (Ts >= Tm). Payoff:
        payoff_Ts = notional * (S_realised(Tm) - strike).
    """

    age: float
    maturity_years: int  # measurement Tm
    notional: float = 1.0
    strike: float | None = None  # if None, ATM (mean of S_{x,Tm})
    settlement_years: int | None = None  # if None -> Ts = Tm


def price_q_forward(
    scen_set: MortalityScenarioSet,
    spec: QForwardSpec,
    *,
    short_rate: float | None = None,
    discount_factors: FloatArray | None = None,
    return_cf_paths: bool = False,
) -> ForwardPricingResult:
    """Price a q-forward using mortality scenarios.

    Measurement at Tm:
        q_Tm = q_{x, Tm}.

    Payoff settled at Ts (Ts >= Tm):
        payoff_Ts = notional * (q_Tm - K).

    PV uses discount factor D(Ts).

    Args:
        scen_set: Scenario set with q_paths of shape (N, A, H).
        spec: Q-forward specification.
        short_rate: Flat continuous rate used when no discount factors are provided.
        discount_factors: Discount factors of shape (H,) or (N, H).
        return_cf_paths: Whether to include cashflow paths in the output.

    Returns:
        Pricing payload with price, PV paths, and metadata.
    """
    q_paths = np.asarray(scen_set.q_paths, dtype=float)
    if q_paths.ndim != 3:
        raise ValueError(f"Expected q_paths with shape (N, A, H), got {q_paths.shape}.")

    N, _A, H_full = q_paths.shape

    # Measurement date (Tm): maturity_years = 1 -> index 0, etc.
    Tm = int(spec.maturity_years)
    if Tm <= 0:
        raise ValueError("spec.maturity_years must be > 0.")
    if Tm > H_full:
        raise ValueError(f"spec.maturity_years={Tm} exceeds available projection horizon {H_full}.")
    tm_idx = Tm - 1

    # Settlement date (Ts): default Ts = Tm
    Ts = int(spec.settlement_years) if spec.settlement_years is not None else Tm
    if Ts < Tm:
        raise ValueError("spec.settlement_years must be >= spec.maturity_years.")
    if Ts > H_full:
        raise ValueError(
            f"spec.settlement_years={Ts} exceeds available projection horizon {H_full}."
        )
    ts_idx = Ts - 1

    # Age index
    age_idx = find_nearest_age_index(scen_set.ages, spec.age)

    # q measured at Tm: (N,)
    q_Tm = q_paths[:, age_idx, tm_idx]
    if not np.isfinite(q_Tm).all():
        raise ValueError("Some q_Tm values are not finite.")

    # Strike: user-provided or ATM on measurement date
    K = float(q_Tm.mean()) if spec.strike is None else float(spec.strike)

    payoff_paths = spec.notional * (q_Tm - K)  # (N,)

    # Build full-grid CF paths (N,Ts): all zeros, payoff at settlement (last column)
    cf_paths = np.zeros((N, Ts), dtype=float)
    cf_paths[:, ts_idx] = payoff_paths

    # Use full discount factors up to Ts
    df_full = build_discount_factors(
        scen_set=scen_set,
        short_rate=short_rate,
        discount_factors=discount_factors,
        H=Ts,
    )

    pv_paths = pv_from_cf_paths(cf_paths, df_full)  # (N,)
    price = float(pv_paths.mean())

    times = np.arange(1, Ts + 1, dtype=int)
    expected_cashflows = cf_paths.mean(axis=0)  # (Ts,)

    # For metadata convenience
    df_arr = np.asarray(df_full, dtype=float)
    df_settle = float(df_arr[ts_idx]) if df_arr.ndim == 1 else float(df_arr[:, ts_idx].mean())

    metadata: dict[str, object] = {
        "N_scenarios": int(N),
        "age": float(spec.age),
        "age_index": int(age_idx),
        "measurement_years": int(Tm),
        "settlement_years": int(Ts),
        "measurement_index": int(tm_idx),
        "settlement_index": int(ts_idx),
        "notional": float(spec.notional),
        "strike": float(K),
        "discount_factor_settlement": float(df_settle),
    }

    payload: ForwardPricingResult = {
        "price": price,
        "pv_paths": pv_paths,
        "age_index": age_idx,
        "measurement_index": tm_idx,
        "settlement_index": ts_idx,
        "strike": K,
        "discount_factors": df_full,
        "expected_cashflows": expected_cashflows,
        "metadata": metadata,
    }

    if return_cf_paths:
        payload["cf_paths"] = cf_paths
        payload["times"] = times

    return payload


def price_s_forward(
    scen_set: MortalityScenarioSet,
    spec: SForwardSpec,
    *,
    short_rate: float | None = None,
    discount_factors: FloatArray | None = None,
    return_cf_paths: bool = False,
) -> ForwardPricingResult:
    """Price an s-forward (survival forward) using mortality scenarios.

    Measurement at Tm:
        S_Tm = S_{x, Tm}.

    Payoff settled at Ts (Ts >= Tm):
        payoff_Ts = notional * (S_Tm - K).

    PV uses discount factor D(Ts).

    Args:
        scen_set: Scenario set with S_paths of shape (N, A, H).
        spec: S-forward specification.
        short_rate: Flat continuous rate used when no discount factors are provided.
        discount_factors: Discount factors of shape (H,) or (N, H).
        return_cf_paths: Whether to include cashflow paths in the output.

    Returns:
        Pricing payload with price, PV paths, and metadata.
    """
    S_paths = np.asarray(scen_set.S_paths, dtype=float)
    if S_paths.ndim != 3:
        raise ValueError(f"Expected S_paths with shape (N, A, H), got {S_paths.shape}.")

    N, _A, H_full = S_paths.shape

    q_paths = np.asarray(scen_set.q_paths, dtype=float)

    # Measurement date (Tm)
    Tm = int(spec.maturity_years)
    if Tm <= 0:
        raise ValueError("spec.maturity_years must be > 0.")
    if Tm > H_full:
        raise ValueError(f"spec.maturity_years={Tm} exceeds available projection horizon {H_full}.")
    tm_idx = Tm - 1

    # Settlement date (Ts): default Ts = Tm
    Ts = int(spec.settlement_years) if spec.settlement_years is not None else Tm
    if Ts < Tm:
        raise ValueError("spec.settlement_years must be >= spec.maturity_years.")
    if Ts > H_full:
        raise ValueError(
            f"spec.settlement_years={Ts} exceeds available projection horizon {H_full}."
        )
    ts_idx = Ts - 1

    # Age index
    age_idx = find_nearest_age_index(scen_set.ages, spec.age)

    S_Tm = S_paths[:, age_idx, tm_idx]
    if not np.isfinite(S_Tm).all():
        # Fallback: rebuild diagonal cohort survival up to Tm using q_paths + Gompertz tail
        S_curve = cohort_survival_full_horizon_from_q(
            q_paths=q_paths,
            ages=np.asarray(scen_set.ages, dtype=float),
            age0=float(spec.age),
            horizon=Tm,
            age_fit_min=80,
            age_fit_max=95,
        )

        S_Tm = S_curve[:, -1]

        if not np.isfinite(S_Tm).all():
            raise ValueError("Some S_Tm values are not finite (even after Gompertz fallback).")

    # Strike: user-provided or ATM on measurement date
    K = float(S_Tm.mean()) if spec.strike is None else float(spec.strike)

    df_full = build_discount_factors(
        scen_set=scen_set,
        short_rate=short_rate,
        discount_factors=discount_factors,
        H=Ts,
    )

    payoff_paths = spec.notional * (S_Tm - K)  # (N,)

    cf_paths = np.zeros((N, Ts), dtype=float)
    cf_paths[:, ts_idx] = payoff_paths

    pv_paths = pv_from_cf_paths(cf_paths, df_full)  # (N,)
    price = float(pv_paths.mean())

    times = np.arange(1, Ts + 1, dtype=int)
    expected_cashflows = cf_paths.mean(axis=0)

    df_arr = np.asarray(df_full, dtype=float)
    df_settle = float(df_arr[ts_idx]) if df_arr.ndim == 1 else float(df_arr[:, ts_idx].mean())

    metadata: dict[str, object] = {
        "N_scenarios": int(N),
        "age": float(spec.age),
        "age_index": int(age_idx),
        "measurement_years": int(Tm),
        "settlement_years": int(Ts),
        "measurement_index": int(tm_idx),
        "settlement_index": int(ts_idx),
        "notional": float(spec.notional),
        "strike": float(K),
        "discount_factor_settlement": float(df_settle),
    }

    payload: ForwardPricingResult = {
        "price": price,
        "pv_paths": pv_paths,
        "age_index": age_idx,
        "measurement_index": tm_idx,
        "settlement_index": ts_idx,
        "strike": K,
        "discount_factors": df_full,
        "expected_cashflows": expected_cashflows,
        "metadata": metadata,
    }

    if return_cf_paths:
        payload["cf_paths"] = cf_paths
        payload["times"] = times

    return payload
