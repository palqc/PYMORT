from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from pymort.analysis import MortalityScenarioSet
from pymort.pricing.utils import build_discount_factors, find_nearest_age_index


@dataclass
class QForwardSpec:
    """
    q-forward with (optional) separate measurement vs settlement.

    We measure q at Tm, but settle/pay at Ts (Ts >= Tm).
    Payoff at settlement:
        payoff_Ts = notional * (q_realised(Tm) - strike)
    """

    age: float
    maturity_years: int  # measurement Tm (years from start, 1..H)
    notional: float = 1.0
    strike: Optional[float] = None  # if None, ATM (mean of q_{x,Tm})
    settlement_years: Optional[int] = None  # if None -> Ts = Tm


@dataclass
class SForwardSpec:
    """
    s-forward with (optional) separate measurement vs settlement.

    Measure S at Tm, settle at Ts (Ts >= Tm).
    Payoff at settlement:
        payoff_Ts = notional * (S_realised(Tm) - strike)
    """

    age: float
    maturity_years: int  # measurement Tm
    notional: float = 1.0
    strike: Optional[float] = None  # if None, ATM (mean of S_{x,Tm})
    settlement_years: Optional[int] = None  # if None -> Ts = Tm


def price_q_forward(
    scen_set: MortalityScenarioSet,
    spec: QForwardSpec,
    *,
    short_rate: Optional[float] = None,
    discount_factors: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Price a q-forward using mortality scenarios, with optional separate
    measurement vs settlement.

    Measurement at Tm:
        q_Tm = q_{x, Tm}

    Payoff settled at Ts (Ts >= Tm):
        payoff_Ts = notional * (q_Tm - K)

    PV uses discount factor D(Ts).
    """
    q_paths = np.asarray(scen_set.q_paths, dtype=float)
    if q_paths.ndim != 3:
        raise ValueError(f"Expected q_paths with shape (N, A, H), got {q_paths.shape}.")

    N, A, H_full = q_paths.shape

    # Measurement date (Tm): maturity_years = 1 -> index 0, etc.
    Tm = int(spec.maturity_years)
    if Tm <= 0:
        raise ValueError("spec.maturity_years must be > 0.")
    if Tm > H_full:
        raise ValueError(
            f"spec.maturity_years={Tm} exceeds available projection horizon {H_full}."
        )
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

    # Discount factor at settlement date Ts
    df_vec = build_discount_factors(
        scen_set=scen_set,
        short_rate=short_rate,
        discount_factors=discount_factors,
        H=Ts,
    )
    if df_vec.ndim == 1:
        D_Ts = float(df_vec[ts_idx])
    elif df_vec.ndim == 2:
        if df_vec.shape[0] not in (1, N):
            raise ValueError(
                f"discount_factors must have first dim 1 or N={N}; got {df_vec.shape}."
            )
        df_eff = df_vec if df_vec.shape[0] == N else np.repeat(df_vec, N, axis=0)
        D_Ts = df_eff[:, ts_idx]
    else:
        raise ValueError("discount_factors must be 1D or 2D.")

    payoff_paths = spec.notional * (q_Tm - K)  # measured at Tm
    if np.ndim(D_Ts) == 0:
        pv_paths = payoff_paths * D_Ts
    else:
        pv_paths = payoff_paths * D_Ts
    price = float(pv_paths.mean())

    metadata: Dict[str, Any] = {
        "N_scenarios": int(N),
        "age": float(spec.age),
        "age_index": int(age_idx),
        "measurement_years": int(Tm),
        "settlement_years": int(Ts),
        "measurement_index": int(tm_idx),
        "settlement_index": int(ts_idx),
        "notional": float(spec.notional),
        "strike": float(K),
        "discount_factor_settlement": float(D_Ts),
    }

    return {
        "price": price,
        "pv_paths": pv_paths,
        "age_index": age_idx,
        "measurement_index": tm_idx,
        "settlement_index": ts_idx,
        "strike": K,
        "discount_factor_settlement": D_Ts,
        "metadata": metadata,
    }


def price_s_forward(
    scen_set: MortalityScenarioSet,
    spec: SForwardSpec,
    *,
    short_rate: Optional[float] = None,
    discount_factors: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Price an s-forward (survival forward) using mortality scenarios, with optional
    separate measurement vs settlement.

    Measurement at Tm:
        S_Tm = S_{x, Tm}

    Payoff settled at Ts (Ts >= Tm):
        payoff_Ts = notional * (S_Tm - K)

    PV uses discount factor D(Ts).
    """
    S_paths = np.asarray(scen_set.S_paths, dtype=float)
    if S_paths.ndim != 3:
        raise ValueError(f"Expected S_paths with shape (N, A, H), got {S_paths.shape}.")

    N, A, H_full = S_paths.shape

    # Measurement date (Tm)
    Tm = int(spec.maturity_years)
    if Tm <= 0:
        raise ValueError("spec.maturity_years must be > 0.")
    if Tm > H_full:
        raise ValueError(
            f"spec.maturity_years={Tm} exceeds available projection horizon {H_full}."
        )
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

    # Survival measured at Tm: (N,)
    S_Tm = S_paths[:, age_idx, tm_idx]
    if not np.isfinite(S_Tm).all():
        raise ValueError("Some S_Tm values are not finite.")

    # Strike: user-provided or ATM on measurement date
    K = float(S_Tm.mean()) if spec.strike is None else float(spec.strike)

    # Discount factor at settlement date Ts
    df_vec = build_discount_factors(
        scen_set=scen_set,
        short_rate=short_rate,
        discount_factors=discount_factors,
        H=Ts,
    )
    if df_vec.ndim == 1:
        D_Ts = float(df_vec[ts_idx])
    elif df_vec.ndim == 2:
        if df_vec.shape[0] not in (1, N):
            raise ValueError(
                f"discount_factors must have first dim 1 or N={N}; got {df_vec.shape}."
            )
        df_eff = df_vec if df_vec.shape[0] == N else np.repeat(df_vec, N, axis=0)
        D_Ts = df_eff[:, ts_idx]
    else:
        raise ValueError("discount_factors must be 1D or 2D.")

    payoff_paths = spec.notional * (S_Tm - K)  # measured at Tm
    pv_paths = payoff_paths * D_Ts
    price = float(pv_paths.mean())

    metadata: Dict[str, Any] = {
        "N_scenarios": int(N),
        "age": float(spec.age),
        "age_index": int(age_idx),
        "measurement_years": int(Tm),
        "settlement_years": int(Ts),
        "measurement_index": int(tm_idx),
        "settlement_index": int(ts_idx),
        "notional": float(spec.notional),
        "strike": float(K),
        "discount_factor_settlement": float(D_Ts),
    }

    return {
        "price": price,
        "pv_paths": pv_paths,
        "age_index": age_idx,
        "measurement_index": tm_idx,
        "settlement_index": ts_idx,
        "strike": K,
        "discount_factor_settlement": D_Ts,
        "metadata": metadata,
    }
