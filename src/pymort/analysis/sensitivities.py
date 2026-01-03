"""Sensitivity analysis utilities for mortality-linked products.

This module provides rate sensitivities, mortality deltas, and volatility
vegas for pricing diagnostics.

Note:
    Docstrings follow Google style for clarity and spec alignment.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import cast

import numpy as np

from pymort._types import FloatArray
from pymort.analysis import MortalityScenarioSet
from pymort.analysis.scenario_analysis import clone_scen_set_with
from pymort.lifetables import (
    cohort_survival_from_q_paths,
    validate_q,
    validate_survival_monotonic,
)
from pymort.pricing.liabilities import CohortLifeAnnuitySpec, price_cohort_life_annuity
from pymort.pricing.longevity_bonds import (
    LongevityBondSpec,
    price_simple_longevity_bond,
)
from pymort.pricing.mortality_derivatives import (
    QForwardSpec,
    SForwardSpec,
    price_q_forward,
    price_s_forward,
)
from pymort.pricing.survivor_swaps import SurvivorSwapSpec, price_survivor_swap


@dataclass
class RateSensitivity:
    """Sensitivity of a price to the short rate.

    Attributes:
        base_rate (float): Base short rate (continuously compounded).
        bump (float): Bump size around base_rate.
        price_base (float): Price at base_rate.
        price_up (float): Price at base_rate + bump.
        price_down (float): Price at base_rate - bump.
        dP_dr (float): Numerical derivative dP/dr (central difference).
        duration (float): Macaulay-like duration approximation,
            duration ≈ - (1 / P) * dP/dr.
        dv01 (float): Approximate DV01 for a 1 bp rate change,
            dv01 ≈ dP/dr * 1e-4.
    """

    base_rate: float
    bump: float
    price_base: float
    price_up: float
    price_down: float
    dP_dr: float
    duration: float
    dv01: float


@dataclass
class MortalityDeltaByAge:
    """Collection of mortality deltas by age.

    Attributes:
        base_price (float): Price under the original scenario set.
        rel_bump (float): Relative bump applied to q, e.g. 0.01 for +1%.
        ages (np.ndarray): Ages for which deltas were computed.
        deltas (np.ndarray): Array of shape (A_sel,) with dP/d(1+eps) for
            each selected age, interpreted as sensitivity to proportional
            mortality increases at that age.
    """

    base_price: float
    rel_bump: float
    ages: FloatArray
    deltas: FloatArray


def rate_sensitivity(
    price_func: Callable[..., float],
    scen_set: MortalityScenarioSet,
    *,
    base_short_rate: float,
    bump: float = 1e-4,
    **price_kwargs: object,
) -> RateSensitivity:
    """Compute numerical sensitivity of a price to the short rate.

    The pricing function is expected to follow:
        price_func(scen_set=scen_set, short_rate=r, **kwargs) -> float

    Args:
        price_func: Callable returning a scalar price (e.g., bond or forward pricer).
        scen_set: Scenario set used for base/up/down evaluations.
        base_short_rate: Base short rate (continuously compounded).
        bump: Additive bump around base_short_rate for central difference.
        **price_kwargs: Extra keyword arguments forwarded to price_func.

    Returns:
        RateSensitivity with base/up/down prices, dP/dr, duration, DV01.
    """
    r0 = float(base_short_rate)
    h = float(bump)

    # Base, up, down prices
    p0 = float(price_func(scen_set=scen_set, short_rate=r0, **price_kwargs))
    p_up = float(price_func(scen_set=scen_set, short_rate=r0 + h, **price_kwargs))
    p_dn = float(price_func(scen_set=scen_set, short_rate=r0 - h, **price_kwargs))

    # Central difference derivative
    dP_dr = (p_up - p_dn) / (2.0 * h)

    EPS = 1e-8
    duration = -dP_dr / p0 if np.isfinite(p0) and abs(p0) > EPS else np.nan
    dv01 = dP_dr * 1e-4  # change in price for 1 bp bump

    return RateSensitivity(
        base_rate=r0,
        bump=h,
        price_base=p0,
        price_up=p_up,
        price_down=p_dn,
        dP_dr=float(dP_dr),
        duration=float(duration),
        dv01=float(dv01),
    )


def mortality_delta_by_age(
    price_func: Callable[[MortalityScenarioSet], float],
    scen_set: MortalityScenarioSet,
    *,
    ages: Iterable[float] | None = None,
    rel_bump: float = 0.01,
) -> MortalityDeltaByAge:
    """Compute numerical mortality delta by age via proportional q bumps.

    For each selected age x:
        1) multiply q_{x,t} by (1 + rel_bump),
        2) recompute survival S_{x,t},
        3) rebuild a bumped scenario set,
        4) recompute the price.

    The delta for age x is:
        Delta_x ≈ (P_bumped(x) - P_base) / rel_bump

    Args:
        price_func: Function that maps a scenario set to a scalar price.
        scen_set: Base scenario set (P or Q measure).
        ages: Ages at which to compute deltas. If None, use all ages.
        rel_bump: Relative bump applied to q (e.g., 0.01 = +1%).

    Returns:
        MortalityDeltaByAge with base price and deltas by age.

    Notes:
        This brute-force finite-difference approach is expensive: if you have
        A ages, you re-price A times. Recomputing S keeps survival curves
        consistent and monotone for the bumped age.
    """
    q_base = np.asarray(scen_set.q_paths, dtype=float)
    S_base = np.asarray(scen_set.S_paths, dtype=float)

    if q_base.shape != S_base.shape:
        raise ValueError(
            f"q_paths and S_paths must have the same shape; got {q_base.shape} vs {S_base.shape}."
        )
    eps = float(rel_bump)
    if eps == 0.0:
        raise ValueError("rel_bump must be non-zero.")

    ages_all = np.asarray(scen_set.ages, dtype=float)
    ages_sel = ages_all if ages is None else np.asarray(list(ages), dtype=float)

    # indices of ages to bump
    idx_map: dict[float, int] = {}
    for x in ages_sel:
        # closest age in grid
        i = int(np.argmin(np.abs(ages_all - float(x))))
        idx_map[float(x)] = i

    # Base price
    base_price = float(price_func(scen_set))
    sign = 1.0 if base_price >= 0.0 else -1.0

    deltas = np.zeros(len(ages_sel), dtype=float)

    for k, x in enumerate(ages_sel):
        i_age = idx_map[float(x)]

        # Copy q
        q_bump = q_base.copy()

        # Bump q for age index i_age
        q_bump[:, i_age, :] *= 1.0 + eps

        # Validate q (ensure still in (0,1])
        validate_q(q_bump)

        # Recompute FULL cohort survival paths (diagonals age+t)
        S_bump = cohort_survival_from_q_paths(q_bump)
        validate_survival_monotonic(S_bump)

        # Rebuild bumped scenario set
        scen_bump = clone_scen_set_with(scen_set, q_paths=q_bump, S_paths=S_bump)

        # Price under bumped mortality for that age
        p_bump = float(price_func(scen_bump))

        deltas[k] = ((sign * p_bump) - (sign * base_price)) / eps

    return MortalityDeltaByAge(
        base_price=base_price,
        rel_bump=eps,
        ages=ages_sel,
        deltas=deltas,
    )


# ============================================================================
# 3) Convexity de taux (optionnelle mais "top niveau")
# ============================================================================


@dataclass
class RateConvexity:
    """Rate convexity estimated by finite differences.

    We use:
        d2P/dr2 ≈ (P(r+h) - 2P(r) + P(r-h)) / h^2
    and normalize by price to express convexity per unit price.
    """

    base_rate: float
    bump: float
    price_base: float
    price_up: float
    price_down: float
    convexity: float


def rate_convexity(
    price_func: Callable[..., float],
    scen_set: MortalityScenarioSet,
    *,
    base_short_rate: float,
    bump: float = 1e-4,
    **price_kwargs: object,
) -> RateConvexity:
    """Approximate rate convexity with finite differences.

    Args:
        price_func: Pricing callable of the form
            price_func(scen_set=scen_set, short_rate=r, **kwargs) -> float.
        scen_set: Scenario set used for the r, r±h evaluations.
        base_short_rate: Base short rate (continuously compounded).
        bump: Increment h around base_short_rate.
        **price_kwargs: Extra kwargs forwarded to price_func.

    Returns:
        RateConvexity with base/up/down prices and normalized convexity.
    """
    r0 = float(base_short_rate)
    h = float(bump)

    p0 = float(price_func(scen_set=scen_set, short_rate=r0, **price_kwargs))
    p_up = float(price_func(scen_set=scen_set, short_rate=r0 + h, **price_kwargs))
    p_dn = float(price_func(scen_set=scen_set, short_rate=r0 - h, **price_kwargs))

    if p0 != 0.0:
        d2P_dr2 = (p_up - 2.0 * p0 + p_dn) / (h**2)
        convexity = d2P_dr2 / p0
    else:
        convexity = float("nan")

    return RateConvexity(
        base_rate=r0,
        bump=h,
        price_base=p0,
        price_up=p_up,
        price_down=p_dn,
        convexity=float(convexity),
    )


# ============================================================================
# 4) Mortality Vega via scaling de la volatilité des facteurs
# ============================================================================


@dataclass
class MortalityVega:
    """Sensitivity of price to scaling mortality-factor volatility.

    The bump is interpreted as:
        sigma -> (1 ± rel_bump) * sigma

    Attributes:
        base_price (float): Price at scale_sigma = 1.0.
        rel_bump (float): Relative bump applied to sigma scale.
        price_up (float): Price at scale_sigma = 1 + rel_bump.
        price_down (float): Price at scale_sigma = max(1 - rel_bump, eps).
        vega (float): Numerical derivative dP/d(scale_sigma) at 1.0,
            vega ≈ (P_up - P_down) / (2 * rel_bump).
    """

    base_price: float
    rel_bump: float
    price_up: float
    price_down: float
    vega: float


def mortality_vega_via_sigma_scale(
    build_scenarios_func: Callable[[float], MortalityScenarioSet],
    price_func: Callable[[MortalityScenarioSet], float],
    *,
    rel_bump: float = 0.05,
) -> MortalityVega:
    """Compute a mortality "vega" by scaling factor volatility.

    We evaluate:
        P0   = price_func(build_scenarios_func(1.0))
        P_up = price_func(build_scenarios_func(1 + rel_bump))
        P_dn = price_func(build_scenarios_func(max(1 - rel_bump, eps)))
    and compute:
        Vega ≈ (P_up - P_dn) / (2 * rel_bump)

    Args:
        build_scenarios_func: Function that rebuilds a scenario set given
            a sigma scale factor.
        price_func: Pricing function that maps a scenario set to a scalar price.
        rel_bump: Relative bump applied to sigma scale (e.g., 0.05 = ±5%).

    Returns:
        MortalityVega with P0, P_up, P_dn, and the corresponding vega.
    """
    eps = float(rel_bump)
    if eps <= 0.0:
        raise ValueError("rel_bump doit être > 0.")

    # Scales
    scale_0 = 1.0
    scale_up = 1.0 + eps
    scale_dn = max(1.0 - eps, 1e-6)

    scen_0 = build_scenarios_func(scale_0)
    scen_up = build_scenarios_func(scale_up)
    scen_dn = build_scenarios_func(scale_dn)

    p0 = float(price_func(scen_0))
    p_up = float(price_func(scen_up))
    p_dn = float(price_func(scen_dn))

    vega = (p_up - p_dn) / (2.0 * eps)

    return MortalityVega(
        base_price=p0,
        rel_bump=eps,
        price_up=p_up,
        price_down=p_dn,
        vega=float(vega),
    )


# ============================================================================
# 5) Wrappers helpers: price one product / all products
# ============================================================================

ProductSpec = (
    LongevityBondSpec | SForwardSpec | QForwardSpec | SurvivorSwapSpec | CohortLifeAnnuitySpec
)


def make_single_product_pricer(
    *,
    kind: str,
    spec: ProductSpec,
    short_rate: float = 0.02,
) -> Callable[[MortalityScenarioSet], float]:
    """Return a pricing function for one instrument.

    Supported kinds:
        - "longevity_bond" (alias: "bond")
        - "s_forward"
        - "q_forward"
        - "survivor_swap"
        - "life_annuity" (alias: "annuity")

    Args:
        kind: Instrument kind string.
        spec: Instrument specification dataclass.
        short_rate: Flat short rate for discounting.

    Returns:
        Callable that maps a MortalityScenarioSet to a scalar price.
    """
    k = str(kind).strip().lower()

    # ---- aliases (backward-compatible / friendly) ----
    alias_map = {
        "bond": "longevity_bond",
        "longevitybond": "longevity_bond",
        "longevity-bond": "longevity_bond",
        "annuity": "life_annuity",
        "swap": "survivor_swap",
        "lifeannuity": "life_annuity",
        "life-annuity": "life_annuity",
    }
    k = alias_map.get(k, k)

    if k == "longevity_bond":
        spec_lb = cast(LongevityBondSpec, spec)
        return lambda scen: float(
            price_simple_longevity_bond(scen_set=scen, spec=spec_lb, short_rate=short_rate)["price"]
        )

    if k == "s_forward":
        spec_sf = cast(SForwardSpec, spec)
        return lambda scen: float(
            price_s_forward(scen_set=scen, spec=spec_sf, short_rate=short_rate)["price"]
        )

    if k == "q_forward":
        spec_qf = cast(QForwardSpec, spec)
        return lambda scen: float(
            price_q_forward(scen_set=scen, spec=spec_qf, short_rate=short_rate)["price"]
        )

    if k == "survivor_swap":
        spec_ss = cast(SurvivorSwapSpec, spec)
        return lambda scen: float(
            price_survivor_swap(scen_set=scen, spec=spec_ss, short_rate=short_rate)["price"]
        )

    if k == "life_annuity":
        spec_ann = cast(CohortLifeAnnuitySpec, spec)
        return lambda scen: float(
            price_cohort_life_annuity(
                scen_set=scen,
                spec=spec_ann,
                short_rate=short_rate,
                discount_factors=None,
            )["price"]
        )

    raise ValueError(
        f"Unknown kind='{kind}'. Expected one of "
        "['longevity_bond','bond','s_forward','q_forward','survivor_swap','life_annuity','annuity']."
    )


def price_all_products(
    scen: MortalityScenarioSet,
    *,
    specs: Mapping[str, ProductSpec],
    short_rate: float = 0.02,
) -> dict[str, float]:
    """Price all instruments in `specs` on the same scenario set.

    Args:
        scen: Scenario set used for pricing.
        specs: Mapping from kind to instrument spec.
        short_rate: Flat short rate for discounting.

    Returns:
        Dict mapping kind to price.
    """
    out: dict[str, float] = {}
    for kind, spec in specs.items():
        pricer = make_single_product_pricer(kind=kind, spec=spec, short_rate=short_rate)
        out[str(kind)] = float(pricer(scen))
    return out


def mortality_vega_all_products(
    build_scenarios_func: Callable[[float], MortalityScenarioSet],
    *,
    specs: Mapping[str, ProductSpec],
    short_rate: float = 0.02,
    rel_bump: float = 0.05,
) -> dict[str, float]:
    """Compute sigma-scale Vega for all instruments in `specs`.

    Vega_kind ≈ (P_kind(1+eps) - P_kind(1-eps)) / (2 * eps)

    Args:
        build_scenarios_func: Function that rebuilds scenarios given sigma scale.
        specs: Mapping from kind to instrument spec.
        short_rate: Flat short rate for discounting.
        rel_bump: Relative bump applied to sigma scale.

    Returns:
        Dict mapping kind to vega estimate.
    """
    eps = float(rel_bump)
    if eps <= 0.0:
        raise ValueError("rel_bump must be > 0.")

    scen_0 = build_scenarios_func(1.0)
    scen_up = build_scenarios_func(1.0 + eps)
    scen_dn = build_scenarios_func(max(1.0 - eps, 1e-6))

    P0 = price_all_products(scen_0, specs=specs, short_rate=short_rate)
    Pup = price_all_products(scen_up, specs=specs, short_rate=short_rate)
    Pdn = price_all_products(scen_dn, specs=specs, short_rate=short_rate)

    vega: dict[str, float] = {}
    for k in P0:
        vega[k] = (Pup[k] - Pdn[k]) / (2.0 * eps)

    return vega


def rate_sensitivity_all_products(
    scen_set: MortalityScenarioSet,
    *,
    specs: Mapping[str, ProductSpec],
    base_short_rate: float,
    bump: float = 1e-4,
) -> dict[str, RateSensitivity]:
    """Compute rate sensitivity for each instrument in specs.

    The same scenario set is reused; only the short rate is bumped.

    Args:
        scen_set: Scenario set used for pricing.
        specs: Mapping from kind to instrument spec.
        base_short_rate: Base short rate.
        bump: Rate bump size for finite differences.

    Returns:
        Dict mapping kind to RateSensitivity.
    """
    out: dict[str, RateSensitivity] = {}

    for kind, spec in specs.items():
        spec_used = freeze_atm_strike(
            scen_set, kind=str(kind), spec=spec, short_rate=float(base_short_rate)
        )

        # IMPORTANT: here we need a pricer that accepts short_rate as arg
        # So we wrap manually instead of using pricer(scenset).
        def price_func(
            *,
            scen_set: MortalityScenarioSet,
            short_rate: float,
            _kind: str = kind,
            _spec: ProductSpec = spec_used,
        ) -> float:
            return float(
                make_single_product_pricer(
                    kind=str(_kind), spec=_spec, short_rate=float(short_rate)
                )(scen_set)
            )

        out[str(kind)] = rate_sensitivity(
            price_func,
            scen_set,
            base_short_rate=float(base_short_rate),
            bump=float(bump),
        )

    return out


def rate_convexity_all_products(
    scen_set: MortalityScenarioSet,
    *,
    specs: Mapping[str, ProductSpec],
    base_short_rate: float,
    bump: float = 1e-4,
) -> dict[str, RateConvexity]:
    """Compute rate convexity for each instrument in specs.

    Args:
        scen_set: Scenario set used for pricing.
        specs: Mapping from kind to instrument spec.
        base_short_rate: Base short rate.
        bump: Rate bump size for finite differences.

    Returns:
        Dict mapping kind to RateConvexity.
    """
    out: dict[str, RateConvexity] = {}

    for kind, spec in specs.items():
        spec_used = freeze_atm_strike(
            scen_set, kind=str(kind), spec=spec, short_rate=float(base_short_rate)
        )

        def price_func(
            *,
            scen_set: MortalityScenarioSet,
            short_rate: float,
            _kind: str = kind,
            _spec: ProductSpec = spec_used,
        ) -> float:
            return float(
                make_single_product_pricer(
                    kind=str(_kind), spec=_spec, short_rate=float(short_rate)
                )(scen_set)
            )

        out[str(kind)] = rate_convexity(
            price_func,
            scen_set,
            base_short_rate=float(base_short_rate),
            bump=float(bump),
        )

    return out


def freeze_atm_strike(
    scen_set: MortalityScenarioSet,
    *,
    kind: str,
    spec: ProductSpec,
    short_rate: float,
) -> ProductSpec:
    """Return a copy of spec with strike fixed at the ATM/par strike under scen_set."""
    k = str(kind).lower()

    if k == "q_forward":
        res_qf = price_q_forward(
            scen_set=scen_set, spec=cast(QForwardSpec, spec), short_rate=short_rate
        )
        return replace(cast(QForwardSpec, spec), strike=float(res_qf["strike"]))

    if k == "s_forward":
        res_sf = price_s_forward(
            scen_set=scen_set, spec=cast(SForwardSpec, spec), short_rate=short_rate
        )
        return replace(cast(SForwardSpec, spec), strike=float(res_sf["strike"]))

    if k == "survivor_swap":
        res_ss = price_survivor_swap(
            scen_set=scen_set, spec=cast(SurvivorSwapSpec, spec), short_rate=short_rate
        )
        return replace(cast(SurvivorSwapSpec, spec), strike=float(res_ss["strike"]))

    return spec


def mortality_delta_by_age_all_products(
    scen_set: MortalityScenarioSet,
    *,
    specs: Mapping[str, ProductSpec],
    short_rate: float = 0.02,
    ages: Iterable[float] | None = None,
    rel_bump: float = 0.01,
) -> dict[str, MortalityDeltaByAge]:
    """Compute mortality delta-by-age for each instrument on one scen_set.

    Args:
        scen_set: Scenario set used for pricing.
        specs: Mapping from kind to instrument spec.
        short_rate: Flat short rate for discounting.
        ages: Ages to include in delta-by-age. If None, use all ages.
        rel_bump: Relative bump applied to q.

    Returns:
        Dict mapping kind to MortalityDeltaByAge.
    """
    out: dict[str, MortalityDeltaByAge] = {}

    for kind, spec in specs.items():
        spec_used = freeze_atm_strike(
            scen_set, kind=str(kind), spec=spec, short_rate=float(short_rate)
        )
        pricer = make_single_product_pricer(
            kind=str(kind), spec=spec_used, short_rate=float(short_rate)
        )
        out[str(kind)] = mortality_delta_by_age(
            pricer, scen_set, ages=ages, rel_bump=float(rel_bump)
        )

    return out


# ============================================================================
# 6) One-shot: compute ALL sensitivities for ALL products
# ============================================================================


@dataclass
class AllSensitivities:
    """Bundle of sensitivity results for multiple instruments.

    Attributes:
        prices_base (dict[str, float]): Base prices at scale_sigma=1.0.
        vega_sigma_scale (dict[str, float]): Vega wrt sigma scaling.
        delta_by_age (dict[str, MortalityDeltaByAge]): Mortality delta-by-age.
        rate_sensitivity (dict[str, RateSensitivity]): Rate sensitivity per instrument.
        rate_convexity (dict[str, RateConvexity]): Rate convexity per instrument.
        meta (dict[str, object]): Convenience metadata (bumps, rates, etc.).
    """

    prices_base: dict[str, float]
    vega_sigma_scale: dict[str, float]
    delta_by_age: dict[str, MortalityDeltaByAge]
    rate_sensitivity: dict[str, RateSensitivity]
    rate_convexity: dict[str, RateConvexity]
    meta: dict[str, object]


def compute_all_sensitivities(
    build_scenarios_func: Callable[[float], MortalityScenarioSet],
    *,
    specs: Mapping[str, ProductSpec],
    base_short_rate: float = 0.02,
    short_rate_for_pricing: float | None = None,
    # bumps
    sigma_rel_bump: float = 0.05,
    q_rel_bump: float = 0.01,
    rate_bump: float = 1e-4,
    # delta-by-age selection
    ages_for_delta: Iterable[float] | None = None,
) -> AllSensitivities:
    """Compute prices and sensitivities for all instruments in `specs`.

    The outputs include sigma-scale vega, delta-by-age, rate sensitivity,
    and rate convexity.

    Args:
        build_scenarios_func: Function that rebuilds scenarios given sigma scale.
        specs: Mapping from kind to instrument spec.
        base_short_rate: Base short rate for sensitivity calculations.
        short_rate_for_pricing: Optional pricing rate override.
        sigma_rel_bump: Relative bump for sigma scaling.
        q_rel_bump: Relative bump for q in delta-by-age.
        rate_bump: Rate bump for sensitivity/convexity.
        ages_for_delta: Optional ages to include for delta-by-age.

    Returns:
        AllSensitivities bundle for the provided instruments.

    Notes:
        Vega is computed by rebuilding scenarios at scale_sigma = 1±eps. Other
        sensitivities are computed on the base scenario set (scale_sigma=1.0).
    """
    r0 = float(base_short_rate)
    r_pr = r0 if short_rate_for_pricing is None else float(short_rate_for_pricing)

    # --- base scenario (for prices/delta/rate/convexity) ---
    scen0 = build_scenarios_func(1.0)

    # Base prices at r_pr (single pass)
    prices_base = price_all_products(scen0, specs=specs, short_rate=r_pr)

    # --- Vega (sigma-scale) ---
    vega_sigma_scale = mortality_vega_all_products(
        build_scenarios_func,
        specs=specs,
        short_rate=r_pr,
        rel_bump=float(sigma_rel_bump),
    )

    # --- Delta by age (q bumps on scen0) ---
    delta_by_age = mortality_delta_by_age_all_products(
        scen0,
        specs=specs,
        short_rate=r_pr,
        ages=ages_for_delta,
        rel_bump=float(q_rel_bump),
    )

    # --- Rate sensitivity & convexity (rate bumps on scen0) ---
    rate_sens = rate_sensitivity_all_products(
        scen0,
        specs=specs,
        base_short_rate=r0,
        bump=float(rate_bump),
    )

    rate_conv = rate_convexity_all_products(
        scen0,
        specs=specs,
        base_short_rate=r0,
        bump=float(rate_bump),
    )

    meta: dict[str, object] = {
        "base_short_rate": r0,
        "short_rate_used_for_pricing": r_pr,
        "sigma_rel_bump": float(sigma_rel_bump),
        "q_rel_bump": float(q_rel_bump),
        "rate_bump": float(rate_bump),
        "delta_ages": None if ages_for_delta is None else list(ages_for_delta),
    }

    return AllSensitivities(
        prices_base=prices_base,
        vega_sigma_scale=vega_sigma_scale,
        delta_by_age=delta_by_age,
        rate_sensitivity=rate_sens,
        rate_convexity=rate_conv,
        meta=meta,
    )
