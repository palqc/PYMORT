"""Longevity bond pricing helpers.

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


class LongevityBondPricingResult(TypedDict, total=False):
    """Typed payload for longevity bond pricing."""

    price: float
    pv_paths: FloatArray
    age_index: int
    discount_factors: FloatArray
    expected_cashflows: FloatArray
    metadata: dict[str, object]
    cf_paths: FloatArray
    times: IntArray


@dataclass
class LongevityBondSpec:
    """Specification for a simple cohort-based longevity bond.

    Coupons are proportional to cohort survival, and the principal can also be
    linked to survival at maturity.

    Attributes:
        issue_age: Age of the cohort at valuation.
        notional: Face value of the bond.
        include_principal: Whether to include survival-linked principal at maturity.
        maturity_years: Pricing horizon in years; defaults to full scenario horizon.
    """

    issue_age: float
    notional: float = 1.0
    include_principal: bool = True
    maturity_years: int | None = None


def price_simple_longevity_bond(
    scen_set: MortalityScenarioSet,
    spec: LongevityBondSpec,
    *,
    short_rate: float | None = None,
    discount_factors: FloatArray | None = None,
    return_cf_paths: bool = False,
) -> LongevityBondPricingResult:
    """Price a simple cohort-based longevity bond from mortality scenarios.

    Structure:
        For a given cohort age x:

            Coupon_t  = notional * S_x(t)
            Principal = notional * S_x(T)  (if spec.include_principal is True)

        where S_x(t) is read from scen_set.S_paths for the chosen age.

    Args:
        scen_set: Scenario set with q_paths and S_paths of shape (N, A, H).
        spec: Product specification.
        short_rate: Flat continuous rate used when no discount factors are provided.
        discount_factors: Discount factors of shape (H,) or (N, H).
        return_cf_paths: Whether to include cashflow paths in the output.

    Returns:
        Pricing payload with price, PV paths, discount factors, and metadata.
    """
    # Basic shape checks
    q_paths = np.asarray(scen_set.q_paths, dtype=float)
    S_paths = np.asarray(scen_set.S_paths, dtype=float)

    if q_paths.shape != S_paths.shape:
        raise ValueError(
            f"q_paths and S_paths must have the same shape; got {q_paths.shape} vs {S_paths.shape}."
        )

    N, _A, H_full = S_paths.shape

    # Determine maturity horizon in time steps
    if spec.maturity_years is None:
        H = H_full
    else:
        H = int(spec.maturity_years)
        if H <= 0:
            raise ValueError("spec.maturity_years must be > 0.")
        if H_full < H:
            raise ValueError(
                f"spec.maturity_years={H} exceeds available projection horizon {H_full}."
            )

    # Choose cohort age index
    age_idx = find_nearest_age_index(scen_set.ages, spec.issue_age)

    S_age = S_paths[:, age_idx, :H]

    if not np.isfinite(S_age).all():
        ages_grid = np.asarray(scen_set.ages, dtype=float)

        S_age = cohort_survival_full_horizon_from_q(
            q_paths=q_paths,
            ages=ages_grid,
            age0=float(spec.issue_age),
            horizon=int(H),
            age_fit_min=80,
            age_fit_max=min(95, int(ages_grid.max())),
        )

        if not np.isfinite(S_age).all():
            raise ValueError("Some S_age values are not finite even after fallback.")

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
            raise ValueError(f"discount_factors must have first dim 1 or N={N}; got {df.shape}.")
        df_eff = df if df.shape[0] == N else np.repeat(df, N, axis=0)
    else:
        raise ValueError("discount_factors must be 1D or 2D.")

    # Coupons: C_t = notional * S_x(t)
    coupons = spec.notional * S_age  # (N, H)

    # Principal at maturity: N_T = notional * S_x(T)
    if spec.include_principal:
        # Add principal to the final coupon payment.
        coupons[:, -1] += spec.notional * S_age[:, -1]

    # Present value per scenario (consistent PV <-> CF)
    pv_paths = pv_from_cf_paths(coupons, df_eff)  # (N,)
    expected_cashflows = coupons.mean(axis=0)  # (H,)
    price = float(pv_paths.mean())

    times = np.asarray(scen_set.years[:H], dtype=int)

    metadata: dict[str, object] = {
        "N_scenarios": int(N),
        "horizon_used": int(H),
        "issue_age": float(spec.issue_age),
        "age_index": int(age_idx),
        "notional": float(spec.notional),
        "include_principal": bool(spec.include_principal),
        "maturity_years": (None if spec.maturity_years is None else int(spec.maturity_years)),
    }

    payload: LongevityBondPricingResult = {
        "price": price,
        "pv_paths": pv_paths,
        "age_index": age_idx,
        "discount_factors": df_eff,
        "expected_cashflows": expected_cashflows,
        "metadata": metadata,
    }

    if return_cf_paths:
        payload["cf_paths"] = coupons  # (N,H)
        payload["times"] = times  # (H,)

    return payload
