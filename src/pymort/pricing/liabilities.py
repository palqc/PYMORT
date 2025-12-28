"""Liability pricing helpers for cohort life annuities.

Note:
    Docstrings follow Google style and type hints use NDArray for clarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from pymort.analysis import MortalityScenarioSet
from pymort.lifetables import validate_survival_monotonic
from pymort.pricing.utils import (
    build_discount_factors,
    cohort_survival_full_horizon_from_q,
    find_nearest_age_index,
    pv_from_cf_paths,
)

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


class CohortLifeAnnuityResult(TypedDict, total=False):
    """Typed payload for cohort life annuity pricing."""

    price: float
    pv_paths: FloatArray
    age_index: int
    discount_factors: FloatArray
    expected_cashflows: FloatArray
    metadata: dict[str, object]
    cf_paths: FloatArray
    times: IntArray


@dataclass
class CohortLifeAnnuitySpec:
    """Specification for a cohort-based life annuity.

    Annual payments are proportional to cohort survival S_x(t), starting after
    an optional deferral period and possibly including a terminal benefit.

    Attributes:
        issue_age: Age of the cohort at valuation. The nearest age in the
            scenario grid is used.
        payment_per_survivor: Annual payment per surviving member.
        maturity_years: Number of projection years to price. If None, uses
            the full scenario horizon.
        defer_years: Number of years with zero payments at the start.
        exposure_at_issue: Cohort size (scaling factor).
        include_terminal: Whether to include a terminal survival benefit.
        terminal_notional: Notional for the terminal benefit.
    """

    issue_age: float
    payment_per_survivor: float = 1.0
    maturity_years: int | None = None

    defer_years: int = 0
    exposure_at_issue: float = 1.0
    include_terminal: bool = False
    terminal_notional: float = 0.0


def price_cohort_life_annuity(
    scen_set: MortalityScenarioSet,
    spec: CohortLifeAnnuitySpec,
    *,
    short_rate: float | None = None,
    discount_factors: FloatArray | None = None,
    return_cf_paths: bool = False,
) -> CohortLifeAnnuityResult:
    """Price a cohort-based life annuity from mortality scenarios.

    Expected cashflow at year t (on the projection grid):

        CF_t = exposure_at_issue * payment_per_survivor * S_x(t)

    for t >= defer_years, and CF_t = 0 for t < defer_years.

    Optionally, a terminal benefit at maturity T is added:

        Terminal_T = exposure_at_issue * terminal_notional * S_x(T)
        (if spec.include_terminal is True).

    Present value per scenario:
        PV_n = sum_{t=0..H-1} CF_{n,t} * D_t,
    where D_t is the discount factor at time t.

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

    # Deferral sanity checks
    defer = int(spec.defer_years)
    if defer < 0:
        raise ValueError("spec.defer_years must be >= 0.")
    if defer >= H:
        raise ValueError(
            f"spec.defer_years={defer} must be strictly less than "
            f"the effective horizon H={H} (otherwise no payments occur)."
        )

    # Choose cohort age index
    age_idx = find_nearest_age_index(scen_set.ages, spec.issue_age)

    # Slice survival for this age and horizon: (N, H)
    S_age = S_paths[:, age_idx, :H]

    # If censored (NaN/inf) beyond age_max slice, rebuild cohort survival using q_paths + Gompertz tail
    if not np.isfinite(S_age).all():
        S_age = cohort_survival_full_horizon_from_q(
            q_paths=q_paths,
            ages=np.asarray(scen_set.ages, dtype=float),
            age0=float(spec.issue_age),
            horizon=int(H),
            age_fit_min=80,
            age_fit_max=95,
        )

    # Quick sanity: mean survival should be non-increasing over time
    S_mean = S_age.mean(axis=0, keepdims=True)  # shape (1, H)
    validate_survival_monotonic(S_mean)

    # Build discount factors D_t, shape (H,)
    df = build_discount_factors(
        scen_set=scen_set,
        short_rate=short_rate,
        discount_factors=discount_factors,
        H=H,
    )
    if df.ndim == 1:
        df_eff = df[None, :]  # (1,H)
    elif df.ndim == 2:
        if df.shape[0] not in (1, N):
            raise ValueError(f"discount_factors must have first dim 1 or N={N}; got {df.shape}.")
        df_eff = df if df.shape[0] == N else np.repeat(df, N, axis=0)
    else:
        raise ValueError("discount_factors must be 1D or 2D.")

    # Base payments: CF_t = payment_per_survivor * S_x(t)
    cashflows = spec.payment_per_survivor * S_age  # (N, H)

    # Apply deferral: no payments before defer_years
    if defer > 0:
        cashflows[:, :defer] = 0.0

    # Terminal benefit at maturity (if requested)
    if spec.include_terminal:
        cashflows[:, -1] += spec.terminal_notional * S_age[:, -1]

    # Scale by exposure_at_issue (size of the cohort / portfolio)
    if spec.exposure_at_issue != 1.0:
        cashflows *= float(spec.exposure_at_issue)

    # Present value per scenario
    # Present value per scenario (consistent PV <-> CF)
    pv_paths = pv_from_cf_paths(cashflows, df_eff)  # (N,)
    price = float(pv_paths.mean())

    # Expected cashflow profile E[CF_t] across scenarios
    expected_cashflows = cashflows.mean(axis=0)  # (H,)
    times = np.arange(1, H + 1, dtype=int)

    metadata: dict[str, object] = {
        "N_scenarios": int(N),
        "horizon_used": int(H),
        "issue_age": float(spec.issue_age),
        "age_index": int(age_idx),
        "payment_per_survivor": float(spec.payment_per_survivor),
        "maturity_years": (None if spec.maturity_years is None else int(spec.maturity_years)),
        "defer_years": defer,
        "exposure_at_issue": float(spec.exposure_at_issue),
        "include_terminal": bool(spec.include_terminal),
        "terminal_notional": float(spec.terminal_notional),
    }

    payload: CohortLifeAnnuityResult = {
        "price": price,
        "pv_paths": pv_paths,
        "age_index": age_idx,
        "discount_factors": df_eff,
        "expected_cashflows": expected_cashflows,
        "metadata": metadata,
    }

    if return_cf_paths:
        payload["cf_paths"] = cashflows
        payload["times"] = times

    return payload
