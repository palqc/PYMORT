from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import numpy as np

from pymort.analysis import MortalityScenarioSet
from pymort.analysis.scenario_analysis import clone_scen_set_with
from pymort.lifetables import survival_from_q, validate_q, validate_survival_monotonic
from pymort.pricing.liabilities import price_cohort_life_annuity
from pymort.pricing.longevity_bonds import price_simple_longevity_bond
from pymort.pricing.mortality_derivatives import price_q_forward, price_s_forward
from pymort.pricing.survivor_swaps import price_survivor_swap


@dataclass
class RateSensitivity:
    """
    Sensitivity of a price to the short rate.

    Attributes
    ----------
    base_rate : float
        Short rate used for the base price (continuously compounded).
    bump : float
        Bump size used around base_rate.
    price_base : float
        Price at base_rate.
    price_up : float
        Price at base_rate + bump.
    price_down : float
        Price at base_rate - bump.
    dP_dr : float
        Numerical derivative dP/dr (central difference).
    duration : float
        Macaulay-like duration approximation:
            duration ≈ - (1 / P) * dP/dr
    dv01 : float
        Approximate DV01 for a 1 bp change in rate:
            dv01 ≈ dP/dr * 1e-4
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
    """
    Collection of mortality deltas by age.

    Attributes
    ----------
    base_price : float
        Price under the original scenario set.
    rel_bump : float
        Relative bump applied to q for each age, e.g. 0.01 for +1%.
    ages : np.ndarray
        Ages for which delt_as were computed.
    deltas : np.ndarray
        Array of shape (A_sel,) with dP/d(1+eps) for each selected age.
        Interpret loosely as 'sensitivity of price to a proportional
        increase in mortality rates at that age'.
    """

    base_price: float
    rel_bump: float
    ages: np.ndarray
    deltas: np.ndarray


def rate_sensitivity(
    price_func: Callable[..., float],
    scen_set: MortalityScenarioSet,
    *,
    base_short_rate: float,
    bump: float = 1e-4,
    **price_kwargs,
) -> RateSensitivity:
    """
    Compute numerical sensitivity of a price to the short rate.

    The pricing function is expected to have a signature like:

        price_func(scen_set=scen_set, short_rate=r, **kwargs) -> float

    Parameters
    ----------
    price_func : callable
        Function that returns a scalar price. Typically one of:
            - price_simple_longevity_bond(...)
            - price_q_forward(...), etc.,
        wrapped to extract the "price" entry from the returned dict.
    scen_set : MortalityScenarioSet
        Scenario set used for all three evaluations (base, up, down).
    base_short_rate : float
        Short rate (continuously compounded) used as base point.
    bump : float, default 1e-4
        Additive bump around base_short_rate for central difference.
    price_kwargs :
        Additional keyword arguments forwarded to price_func.

    Returns
    -------
    RateSensitivity
        Dataclass with base/up/down prices, dP/dr, duration, DV01.
    """
    r0 = float(base_short_rate)
    h = float(bump)

    # Base, up, down prices
    p0 = float(price_func(scen_set=scen_set, short_rate=r0, **price_kwargs))
    p_up = float(price_func(scen_set=scen_set, short_rate=r0 + h, **price_kwargs))
    p_dn = float(price_func(scen_set=scen_set, short_rate=r0 - h, **price_kwargs))

    # Central difference derivative
    dP_dr = (p_up - p_dn) / (2.0 * h)

    duration = -dP_dr / p0 if p0 != 0.0 else np.nan
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
    ages: Optional[Iterable[float]] = None,
    rel_bump: float = 0.01,
) -> MortalityDeltaByAge:
    """
    Compute numerical "mortality delta" by age via proportional bumps in q.

    For each selected age x, we:
        1. multiply q_{x,t} by (1 + rel_bump),
        2. recompute the corresponding survival path S_{x,t},
        3. rebuild a bumped scenario set,
        4. recompute the price.

    The delta for age x is then:

        Delta_x ≈ (P_bumped(x) - P_base) / rel_bump

    Parameters
    ----------
    price_func : callable
        Function that takes a MortalityScenarioSet and returns a scalar price,
        e.g. a small wrapper around price_simple_longevity_bond(...) that
        extracts the "price" entry:
            lambda scen: price_simple_longevity_bond(scen, spec, short_rate=0.02)["price"]
    scen_set : MortalityScenarioSet
        Base scenario set (under P or Q, depending on your pipeline).
    ages : iterable of float, optional
        Ages at which to compute deltas. If None, all ages in scen_set.ages
        are used.
    rel_bump : float, default 0.01
        Relative bump applied multiplicatively to q, e.g. 0.01 -> +1%.

    Returns
    -------
    MortalityDeltaByAge
        Dataclass with base price, bump size, ages and deltas per age.

    Notes
    -----
    - This is a brute-force finite-difference approach. It is expensive:
      if you have A ages, you will re-price A times.
    - Bumping q and recomputing S ensures we keep survival curves
      internally consistent and monotone in time for the bumped age.
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

    N, A, H = q_base.shape

    ages_all = np.asarray(scen_set.ages, dtype=float)
    if ages is None:
        ages_sel = ages_all
    else:
        ages_sel = np.asarray(list(ages), dtype=float)

    # indices of ages to bump
    idx_map: Dict[float, int] = {}
    for x in ages_sel:
        # closest age in grid
        i = int(np.argmin(np.abs(ages_all - float(x))))
        idx_map[float(x)] = i

    # Base price
    base_price = float(price_func(scen_set))

    deltas = np.zeros(len(ages_sel), dtype=float)

    for k, x in enumerate(ages_sel):
        i_age = idx_map[float(x)]

        # Copy q and S
        q_bump = q_base.copy()
        S_bump = S_base.copy()

        # Bump q for age index i_age
        q_bump[:, i_age, :] *= 1.0 + eps

        # Validate q (ensure still in (0,1])
        validate_q(q_bump)

        # Recompute S for this age based on bumped q
        # For each scenario, survival_from_q works along the last axis
        q_age = q_bump[:, i_age, :]  # (N, H)
        S_age = survival_from_q(q_age)  # (N, H)
        validate_survival_monotonic(S_age)

        S_bump[:, i_age, :] = S_age

        # Rebuild bumped scenario set
        scen_bump = clone_scen_set_with(scen_set, q_paths=q_bump, S_paths=S_bump)

        # Price under bumped mortality for that age
        p_bump = float(price_func(scen_bump))

        deltas[k] = (p_bump - base_price) / eps

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
    """
    Convexity approximée par différences finies sur le short rate.

    On utilise les mêmes évaluations que pour la duration, mais on
    exploite la formule de dérivée seconde numérique :

        d2P/dr2 ≈ (P(r+h) - 2P(r) + P(r-h)) / h^2

    et on normalise par le prix pour obtenir une convexity "par unité de prix".
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
    **price_kwargs,
) -> RateConvexity:
    """
    Approxime la convexity de taux par différences finies.

    Paramètres
    ----------
    price_func : callable
        Fonction de pricing de type
            price_func(scen_set=scen_set, short_rate=r, **kwargs) -> float
        (souvent un petit wrapper qui retourne res["price"]).
    scen_set : MortalityScenarioSet
        Ensemble de scénarios utilisé pour les trois évaluations (r, r±h).
    base_short_rate : float
        Taux court (continu) de base.
    bump : float, défaut 1e-4
        Incrément h autour de base_short_rate.
    price_kwargs :
        kwargs supplémentaires passés à price_func (spec, etc.).

    Retourne
    --------
    RateConvexity
        Avec prix base/up/down et convexity normalisée (≈ d2P/dr2 / P).
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
    """
    Sensibilité du prix à un scaling de la volatilité des facteurs de mortalité.

    On interprète le bump comme :
        sigma -> (1 ± rel_bump) * sigma

    Attributes
    ----------
    base_price : float
        Prix pour scale_sigma = 1.0.
    rel_bump : float
        Bump relatif appliqué au scale de sigma (ex: 0.05 -> ±5%).
    price_up : float
        Prix pour scale_sigma = 1 + rel_bump.
    price_down : float
        Prix pour scale_sigma = max(1 - rel_bump, eps).
    vega : float
        Approximation numérique de dP/d(scale_sigma) au point 1.0 :
            vega ≈ (P_up - P_down) / (2 * rel_bump)
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
    """
    Calcule une "Vega de mortalité" en rescalant la volatilité des facteurs.

    Idée
    ----
    - build_scenarios_func(scale_sigma) doit :
        * reconstruire un MortalityScenarioSet en multipliant les σ_kappa
          (ou σ_k) par scale_sigma dans le moteur de projections.
    - price_func(scen_set) doit renvoyer un prix scalaire (float).

    On évalue :
        P0   = price_func(build_scenarios_func(1.0))
        P_up = price_func(build_scenarios_func(1 + rel_bump))
        P_dn = price_func(build_scenarios_func(max(1 - rel_bump, eps)))

    Puis :
        Vega ≈ (P_up - P_dn) / (2 * rel_bump)

    Paramètres
    ----------
    build_scenarios_func : Callable[[float], MortalityScenarioSet]
        Fonction qui prend un scale_sigma et renvoie un scénario de mortalité.
        Exemple typique (à placer côté risk_neutral / projections) :

            def build_Q(scale_sigma: float) -> MortalityScenarioSet:
                return build_scenarios_under_lambda(
                    lambda_esscher=lambda_star,
                    ages=ages,
                    years=years,
                    m=m,
                    model_name="LCM2",
                    B_bootstrap=50,
                    n_process=200,
                    horizon=H,
                    seed=123,
                    include_last=False,
                    sigma_scale=scale_sigma,   # à gérer dans ton moteur
                )

    price_func : Callable[[MortalityScenarioSet], float]
        Fonction qui retourne un prix scalaire à partir d'un scénario.
        Exemple :

            price_func = lambda scen: price_simple_longevity_bond(
                scen_set=scen,
                spec=bond_spec,
                short_rate=0.02,
            )["price"]

    rel_bump : float, défaut 0.05
        Bump relatif (±5% par défaut) sur le scale des volatilities.

    Retourne
    --------
    MortalityVega
        Dataclass avec P0, P_up, P_dn et la vega correspondante.
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

ProductSpec = Any  # specs are dataclasses (LongevityBondSpec, SForwardSpec, etc.)


def make_single_product_pricer(
    *,
    kind: str,
    spec: ProductSpec,
    short_rate: float = 0.02,
) -> Callable[[MortalityScenarioSet], float]:
    """
    Returns a function scen_set -> price for ONE instrument.

    kind in:
      - "longevity_bond"
      - "s_forward"
      - "q_forward"
      - "survivor_swap"
      - "life_annuity"
    """
    k = str(kind).lower()

    if k == "longevity_bond":
        return lambda scen: float(
            price_simple_longevity_bond(
                scen_set=scen, spec=spec, short_rate=short_rate
            )["price"]
        )

    if k == "s_forward":
        return lambda scen: float(
            price_s_forward(scen_set=scen, spec=spec, short_rate=short_rate)["price"]
        )

    if k == "q_forward":
        return lambda scen: float(
            price_q_forward(scen_set=scen, spec=spec, short_rate=short_rate)["price"]
        )

    if k == "survivor_swap":
        return lambda scen: float(
            price_survivor_swap(scen_set=scen, spec=spec, short_rate=short_rate)[
                "price"
            ]
        )

    if k == "life_annuity":
        return lambda scen: float(
            price_cohort_life_annuity(
                scen_set=scen,
                spec=spec,
                short_rate=short_rate,
                discount_factors=None,
            )["price"]
        )

    raise ValueError(
        f"Unknown kind='{kind}'. Expected one of "
        "['longevity_bond','s_forward','q_forward','survivor_swap','life_annuity']."
    )


def price_all_products(
    scen: MortalityScenarioSet,
    *,
    specs: Mapping[str, ProductSpec],
    short_rate: float = 0.02,
) -> dict[str, float]:
    """
    Price all instruments present in `specs` on the same scenario set.
    Returns a dict {kind: price}.
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
    """
    Compute sigma-scale Vega for ALL instruments in `specs` at once.

    Vega_kind ≈ (P_kind(1+eps) - P_kind(1-eps)) / (2 eps)
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
    for k in P0.keys():
        vega[k] = (Pup[k] - Pdn[k]) / (2.0 * eps)

    return vega


def rate_sensitivity_all_products(
    scen_set: MortalityScenarioSet,
    *,
    specs: Mapping[str, ProductSpec],
    base_short_rate: float,
    bump: float = 1e-4,
) -> dict[str, RateSensitivity]:
    """
    Compute rate sensitivity (dP/dr, duration, DV01) for each instrument in specs.
    Reuses the SAME scen_set; only bumps the short rate inside pricers.
    """
    out: dict[str, RateSensitivity] = {}

    for kind, spec in specs.items():
        # IMPORTANT: here we need a pricer that accepts short_rate as arg
        # So we wrap manually instead of using pricer(scenset).
        def price_func(
            *, scen_set: MortalityScenarioSet, short_rate: float, _kind=kind, _spec=spec
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
    """
    Compute rate convexity for each instrument in specs.
    """
    out: dict[str, RateConvexity] = {}

    for kind, spec in specs.items():

        def price_func(
            *, scen_set: MortalityScenarioSet, short_rate: float, _kind=kind, _spec=spec
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


def mortality_delta_by_age_all_products(
    scen_set: MortalityScenarioSet,
    *,
    specs: Mapping[str, ProductSpec],
    short_rate: float = 0.02,
    ages: Optional[Iterable[float]] = None,
    rel_bump: float = 0.01,
) -> dict[str, MortalityDeltaByAge]:
    """
    Compute mortality delta-by-age for each instrument in specs, on the SAME scen_set.
    """
    out: dict[str, MortalityDeltaByAge] = {}

    for kind, spec in specs.items():
        pricer = make_single_product_pricer(
            kind=str(kind), spec=spec, short_rate=float(short_rate)
        )
        out[str(kind)] = mortality_delta_by_age(
            pricer,
            scen_set,
            ages=ages,
            rel_bump=float(rel_bump),
        )

    return out


# ============================================================================
# 6) One-shot: compute ALL sensitivities for ALL products
# ============================================================================


@dataclass
class AllSensitivities:
    """
    Bundle of results for multiple instruments.

    Attributes
    ----------
    prices_base : dict[str, float]
        Prices at base scenario (scale_sigma=1.0) and base_short_rate.
    vega_sigma_scale : dict[str, float]
        Vega wrt sigma scaling: dP/d(scale_sigma) around 1.0 (central diff).
    delta_by_age : dict[str, MortalityDeltaByAge]
        Mortality delta-by-age per instrument (bump q at each age).
    rate_sensitivity : dict[str, RateSensitivity]
        Rate sensitivity (dP/dr, duration, DV01) per instrument.
    rate_convexity : dict[str, RateConvexity]
        Rate convexity per instrument.
    meta : dict[str, object]
        Convenience metadata (bumps, rates, etc.).
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
    short_rate_for_pricing: Optional[float] = None,
    # bumps
    sigma_rel_bump: float = 0.05,
    q_rel_bump: float = 0.01,
    rate_bump: float = 1e-4,
    # delta-by-age selection
    ages_for_delta: Optional[Iterable[float]] = None,
) -> AllSensitivities:
    """
    Compute prices + (sigma-scale vega) + (delta-by-age) + (rate sens) + (rate convexity)
    for all instruments provided in `specs`.

    Notes
    -----
    - Vega is computed by rebuilding scenarios at scale_sigma = 1±eps.
    - Delta-by-age, rate sensitivity, convexity are computed on ONE base scenario set
      built at scale_sigma=1.0 (so they are conditional on that scenario set).
    - Rates: pricing uses `base_short_rate` unless `short_rate_for_pricing` is set.
      (Most of the time you leave it None.)
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

    meta = {
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
