from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from pymort.analysis import MortalityScenarioSet
from pymort.lifetables import validate_survival_monotonic
from pymort.pricing.utils import (
    build_discount_factors,
    cohort_survival_full_horizon_from_q,
    find_nearest_age_index,
    pv_from_cf_paths,
)


@dataclass
class CohortLifeAnnuitySpec:
    """
    Specification of a cohort-based life annuity.

    Structure (discrete yearly times):

        For a given cohort age x at issue:

            Payment_t = exposure_at_issue
                        * payment_per_survivor
                        * S_x(t)

        for t = defer_years, ..., T (or up to maturity_years), where:
            - S_x(t) is the cohort survival probability at time t,
              read from scen_set.S_paths for the chosen age.
            - Payments are annual, on the same grid as scen_set.years.

    This is the liability side of a pension fund paying
    `payment_per_survivor` per surviving member each year, scaled
    by `exposure_at_issue`.

    Attributes
    ----------
    issue_age : float
        Age of the cohort at the valuation / issue date, e.g. 65.
        The closest age in `scen_set.ages` is used internally.

    payment_per_survivor : float, default 1.0
        Annual payment amount per surviving member of the cohort.
        If the cohort has been normalised to 1 at issue, this is the
        annual payment per unit of initial exposure.

    maturity_years : int | None, default None
        Number of years from the first projection year to maturity.
        If None, the whole projection horizon in `scen_set.years` is used.

    defer_years : int, default 0
        Number of initial years during which no annuity payments are made
        (deferred annuity). For example, defer_years=5 means that payments
        start at year index 5 of the projection grid.

    exposure_at_issue : float, default 1.0
        Size of the cohort at issue (or scaling factor). All cashflows are
        multiplied by this exposure. For example, exposure_at_issue=1e6
        means "1 million lives" if the model is per-life.

    include_terminal : bool, default False
        If True, a terminal benefit is paid at maturity equal to:
            exposure_at_issue * terminal_notional * S_x(T).

    terminal_notional : float, default 0.0
        Notional used for the terminal benefit if include_terminal=True.
        Often used to mimic a lump-sum payment at maturity contingent on
        survival of the cohort.
    """

    issue_age: float
    payment_per_survivor: float = 1.0
    maturity_years: Optional[int] = None

    defer_years: int = 0
    exposure_at_issue: float = 1.0
    include_terminal: bool = False
    terminal_notional: float = 0.0


def price_cohort_life_annuity(
    scen_set: MortalityScenarioSet,
    spec: CohortLifeAnnuitySpec,
    *,
    short_rate: Optional[float] = None,
    discount_factors: Optional[np.ndarray] = None,
    return_cf_paths: bool = False,
) -> Dict[str, Any]:
    """
    Price a cohort-based life annuity from mortality scenarios.

    Expected cashflow at year t (on the projection grid):

        CF_t = exposure_at_issue * payment_per_survivor * S_x(t)

    for t >= defer_years, and CF_t = 0 for t < defer_years.

    Optionally, a terminal benefit at maturity T is added:

        Terminal_T = exposure_at_issue * terminal_notional * S_x(T)
        (if spec.include_terminal is True).

    Present value in each scenario n:

        PV_n = sum_{t=0..H-1} CF_{n,t} * D_t

    with D_t the discount factor at time index t.

    Parameters
    ----------
    scen_set : MortalityScenarioSet
        Stochastic mortality scenarios (output of the PYMORT pipeline).
    spec : CohortLifeAnnuitySpec
        Product specification (cohort age, payment, maturity, deferral, etc.).
    short_rate : float | None
        If provided and no discount_factors are given, a flat continuous
        short rate used to build discount factors exp(-r * (year - year0)).
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
            - "expected_cashflows": np.ndarray, shape (H,), E[CF_t] per year
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
            raise ValueError(
                f"discount_factors must have first dim 1 or N={N}; got {df.shape}."
            )
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

    metadata: Dict[str, Any] = {
        "N_scenarios": int(N),
        "horizon_used": int(H),
        "issue_age": float(spec.issue_age),
        "age_index": int(age_idx),
        "payment_per_survivor": float(spec.payment_per_survivor),
        "maturity_years": (
            None if spec.maturity_years is None else int(spec.maturity_years)
        ),
        "defer_years": defer,
        "exposure_at_issue": float(spec.exposure_at_issue),
        "include_terminal": bool(spec.include_terminal),
        "terminal_notional": float(spec.terminal_notional),
    }

    payload: Dict[str, Any] = {
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
