from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from pymort.analysis import MortalityScenarioSet
from pymort.pricing.utils import build_discount_factors, find_nearest_age_index


@dataclass
class LongevityBondSpec:
    """
    Specification of a simple cohort-based longevity bond.

    This corresponds to structures similar to the EIB/BNP longevity bond:
    - coupons proportional to the survival of a given cohort (age x at issue),
    - principal repayment also scaled by survival at maturity.

    Attributes
    ----------
    issue_age : float
        Age of the cohort at the valuation/issue date, e.g. 65.
        The closest age in `scenario.ages` is used internally.
    notional : float
        Nominal notional (face value) of the bond.
    include_principal : bool
        If True, a final payment at maturity equal to
            notional * S_x(T)
        is included in addition to the last coupon.
    maturity_years : int | None
        Number of years from the first projection year to maturity.
        If None, the whole projection horizon in `scen_set.years` is used.
    """

    issue_age: float
    notional: float = 1.0
    include_principal: bool = True
    maturity_years: Optional[int] = None


def price_simple_longevity_bond(
    scen_set: MortalityScenarioSet,
    spec: LongevityBondSpec,
    *,
    short_rate: Optional[float] = None,
    discount_factors: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Price a simple cohort-based longevity bond from mortality scenarios.

    Structure:
        For a given cohort age x:

            Coupon_t  = notional * S_x(t)
            Principal = notional * S_x(T)  (if spec.include_principal is True)

        where S_x(t) is read from scen_set.S_paths for the chosen age.

    Parameters
    ----------
    scen_set : MortalityScenarioSet
        Stochastic mortality scenarios (output of the PYMORT pipeline).
    spec : LongevityBondSpec
        Product specification (cohort age, notional, maturity, etc.).
    short_rate : float | None
        If provided and no discount_factors are given, a flat continuous
        short rate used to build discount factors exp(-r * t) with t=1..H in years.
    discount_factors : np.ndarray | None
        Optional explicit discount factors D_t of shape (H,). If provided,
        these override scen_set.discount_factors and short_rate.

    Returns
    -------
    dict
        A dictionary with keys:
            - "price": float, Monte-Carlo estimate of present value
            - "pv_paths": np.ndarray, shape (N,) present value per scenario
            - "age_index": int, index of the cohort in scen_set.ages
            - "discount_factors": np.ndarray, shape (H,)
            - "metadata": dict with extra info (spec, N, horizon, etc.)
    """
    # Basic shape checks
    q_paths = np.asarray(scen_set.q_paths, dtype=float)
    S_paths = np.asarray(scen_set.S_paths, dtype=float)

    if q_paths.shape != S_paths.shape:
        raise ValueError(
            f"q_paths and S_paths must have the same shape; "
            f"got {q_paths.shape} vs {S_paths.shape}."
        )

    N, A, H_full = S_paths.shape

    # Determine maturity horizon in time steps
    if spec.maturity_years is None:
        H = H_full
    else:
        H = int(spec.maturity_years)
        if H <= 0:
            raise ValueError("spec.maturity_years must be > 0.")
        if H > H_full:
            raise ValueError(
                f"spec.maturity_years={H} exceeds available "
                f"projection horizon {H_full}."
            )

    # Choose cohort age index
    age_idx = find_nearest_age_index(scen_set.ages, spec.issue_age)

    # Slice survival for this age and horizon: (N, H)
    S_age = S_paths[:, age_idx, :H]

    # Build discount factors D_t, shape (H,)
    df = build_discount_factors(
        scen_set=scen_set,
        short_rate=short_rate,
        discount_factors=discount_factors,
        H=H,
    )

    if np.any(df <= 0.0) or not np.all(np.isfinite(df)):
        raise ValueError("discount_factors must be positive and finite.")
    if df.ndim == 1:
        df_eff = df[None, :]  # (1,H) broadcast
    elif df.ndim == 2:
        if df.shape[0] not in (1, N):
            raise ValueError(
                f"discount_factors must have first dim 1 or N={N}; got {df.shape}."
            )
        df_eff = df if df.shape[0] == N else np.repeat(df, N, axis=0)
    else:
        raise ValueError("discount_factors must be 1D or 2D.")

    # Coupons: C_t = notional * S_x(t)
    coupons = spec.notional * S_age  # (N, H)

    # Principal at maturity: N_T = notional * S_x(T)
    if spec.include_principal:
        # Ajoute le nominal à la dernière colonne des coupons
        coupons[:, -1] += spec.notional * S_age[:, -1]

    # Present value per scenario
    pv_paths = (coupons * df_eff).sum(axis=1)  # (N,)
    expected_cashflows = coupons.mean(axis=0)  # (H,)
    price = float(pv_paths.mean())

    metadata: Dict[str, Any] = {
        "N_scenarios": int(N),
        "horizon_used": int(H),
        "issue_age": float(spec.issue_age),
        "age_index": int(age_idx),
        "notional": float(spec.notional),
        "include_principal": bool(spec.include_principal),
        "maturity_years": (
            None if spec.maturity_years is None else int(spec.maturity_years)
        ),
    }

    return {
        "price": price,
        "pv_paths": pv_paths,
        "age_index": age_idx,
        "discount_factors": df,
        "expected_cashflows": expected_cashflows,
        "metadata": metadata,
    }
