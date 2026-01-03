"""Scenario shocks and stress-testing helpers.

This module applies deterministic mortality shocks to scenario sets and
builds bundles of stressed scenarios for analysis.

Note:
    Docstrings follow Google style for clarity and spec alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymort._types import FloatArray
from pymort.analysis import MortalityScenarioSet
from pymort.lifetables import (
    m_to_q,
    q_to_m,
    survival_from_q,
    validate_q,
    validate_survival_monotonic,
)

# ============================================================================
# 1) Outil interne : cloner un MortalityScenarioSet avec q / S modifiés
# ============================================================================


def clone_scen_set_with(
    scen_set: MortalityScenarioSet,
    *,
    q_paths: FloatArray | None = None,
    S_paths: FloatArray | None = None,
    discount_factors: FloatArray | None = None,
    metadata: dict[str, object] | None = None,
) -> MortalityScenarioSet:
    """Clone a MortalityScenarioSet with optional replacements.

    Args:
        scen_set: Base scenario set to clone.
        q_paths: Optional replacement q_paths, shape (N, A, T).
        S_paths: Optional replacement S_paths, shape (N, A, T).
        discount_factors: Optional replacement discount factors.
        metadata: Optional replacement metadata.

    Returns:
        New MortalityScenarioSet with requested fields replaced.
    """
    field_names = list(MortalityScenarioSet.__dataclass_fields__.keys())
    kwargs: dict[str, object] = {}

    for name in field_names:
        if name == "q_paths" and q_paths is not None:
            kwargs[name] = q_paths
        elif name == "S_paths" and S_paths is not None:
            kwargs[name] = S_paths
        elif name == "discount_factors" and discount_factors is not None:
            kwargs[name] = discount_factors
        elif name == "metadata" and metadata is not None:
            kwargs[name] = metadata
        else:
            kwargs[name] = getattr(scen_set, name)

    return MortalityScenarioSet(**kwargs)  # type: ignore[arg-type]


# ============================================================================
# 2) Fonction générique : appliquer un "mortality shock"
# ============================================================================


def apply_mortality_shock(
    scen_set: MortalityScenarioSet,
    *,
    shock_type: str = "long_life",
    magnitude: float = 0.10,
    pandemic_year: int | None = None,
    pandemic_duration: int = 1,
    plateau_start_year: int | None = None,
    accel_start_year: int | None = None,
) -> MortalityScenarioSet:
    """Apply a mortality shock to a scenario set.

    Supported shock types:
        - "long_life": Uniform mortality improvement,
          q' = (1 - magnitude) * q.
        - "short_life": Uniform mortality deterioration,
          q' = (1 + magnitude) * q.
        - "pandemic": Temporary spike over a forward window,
          q'_{t in window} = (1 + magnitude) * q_t,
          where window = [pandemic_year, pandemic_year + pandemic_duration - 1].
        - "plateau": Freeze improvements from a start year,
          q'_{t >= plateau_start} = q_{t_plateau_start}.
        - "accel_improvement": Accelerate improvements from a start year,
          q'_{t0+h} = q_{t0+h} * (1 - magnitude)^h.

    Args:
        scen_set: Base scenario set (P or Q measure).
        shock_type: One of {"long_life", "short_life", "pandemic", "plateau",
            "accel_improvement"}.
        magnitude: Shock intensity (interpretation depends on shock_type).
        pandemic_year: First year of the pandemic shock window.
        pandemic_duration: Number of years in the pandemic window.
        plateau_start_year: Year from which to freeze mortality rates.
        accel_start_year: Year from which to accelerate improvements.

    Returns:
        New MortalityScenarioSet with modified q_paths and S_paths.
    """
    q_base = np.asarray(scen_set.q_paths, dtype=float)
    S_base = np.asarray(scen_set.S_paths, dtype=float)

    if q_base.shape != S_base.shape:
        raise ValueError(...)

    _N, _A, H = q_base.shape
    years = np.asarray(scen_set.years, dtype=int)
    if years.shape[0] != H:
        raise ValueError("scen_set.years length must match q_paths horizon.")

    # Prefer existing m_paths if available, else derive m from q (stable)
    if getattr(scen_set, "m_paths", None) is not None:
        m_base = np.asarray(scen_set.m_paths, dtype=float)
        if m_base.shape != q_base.shape:
            raise ValueError(
                f"m_paths must match q_paths shape; got {m_base.shape} vs {q_base.shape}."
            )
    else:
        m_base = q_to_m(q_base)

    m_new = np.asarray(m_base, dtype=float).copy()
    eps = float(magnitude)
    shock_type = shock_type.lower()

    eps = float(magnitude)

    shock_type = shock_type.lower()

    if magnitude < 0:
        raise ValueError("magnitude must be >= 0.")
    if shock_type in {"long_life", "short_life", "accel_improvement"} and magnitude >= 1:
        raise ValueError("magnitude must be < 1 for this shock_type.")

    if shock_type == "long_life":
        # Baisse uniforme des mortalités
        m_new *= 1.0 - eps

    elif shock_type == "short_life":
        # Hausse uniforme des mortalités
        m_new *= 1.0 + eps

    elif shock_type == "pandemic":
        if pandemic_year is None:
            raise ValueError("pandemic_year must be provided for shock_type='pandemic'.")
        if pandemic_duration <= 0:
            raise ValueError("pandemic_duration must be > 0.")

        # Fenêtre en années
        start = pandemic_year
        end = pandemic_year + pandemic_duration - 1
        mask = (years >= start) & (years <= end)

        if not np.any(mask):
            # Rien dans la fenêtre -> pas de choc
            return scen_set

        # On spike q sur ces années
        m_new[:, :, mask] *= 1.0 + eps

    elif shock_type == "plateau":
        if plateau_start_year is None:
            raise ValueError("plateau_start_year must be provided for shock_type='plateau'.")

        # index à partir duquel on fige
        idx_start = np.searchsorted(years, plateau_start_year)
        if idx_start >= H:
            # plateau en dehors de l'horizon -> pas de changement
            return scen_set

        # On fige toutes les années t >= idx_start au niveau de idx_start
        m_new[:, :, idx_start:] = m_new[:, :, idx_start][:, :, None]

    elif shock_type == "accel_improvement":
        # Accélération des améliorations = baisse plus rapide des q dans le temps.
        t0_idx = 0 if accel_start_year is None else int(np.searchsorted(years, accel_start_year))
        if t0_idx >= H - 1:
            # Rien à accélérer
            return scen_set

        # Pour chaque année h après t0, multiplier par (1 - eps)^h
        offsets = np.arange(H - t0_idx, dtype=float)  # 0,1,...,H-t0-1
        factors = (1.0 - eps) ** offsets  # shape (H - t0,)
        # Broadcast sur (N, A, H-t0)
        m_new[:, :, t0_idx:] *= factors[None, None, :]

    else:
        raise ValueError(f"Unknown shock_type='{shock_type}'.")

    # Keep m non-negative (safety)
    m_new = np.maximum(m_new, 0.0)

    # Convert back to q, clamp for strict (0,1) constraints
    q_new = m_to_q(m_new)
    q_new = np.clip(q_new, 1e-12, 1.0 - 1e-12)
    validate_q(q_new)

    S_new = survival_from_q(q_new)
    validate_survival_monotonic(S_new)

    return clone_scen_set_with(scen_set, q_paths=q_new, S_paths=S_new)


# ============================================================================
# 3) Générateur de scénarios "stressés" (bundle Base / Optimistic / Pessimistic / Stress)
# ============================================================================


@dataclass
class ScenarioBundle:
    """Container for a coherent set of stressed scenarios.

    Attributes:
        base (MortalityScenarioSet): Base scenario set (often Q-measure).
        optimistic (MortalityScenarioSet): "Long life" scenarios (lower mortality).
        pessimistic (MortalityScenarioSet): "Short life" scenarios (higher mortality).
        pandemic_stress (MortalityScenarioSet | None): Pandemic shock scenarios.
        plateau (MortalityScenarioSet | None): Improvement plateau scenarios.
        accel_improvement (MortalityScenarioSet | None): Accelerated improvement scenarios.
    """

    base: MortalityScenarioSet
    optimistic: MortalityScenarioSet
    pessimistic: MortalityScenarioSet
    pandemic_stress: MortalityScenarioSet | None = None
    plateau: MortalityScenarioSet | None = None
    accel_improvement: MortalityScenarioSet | None = None


def generate_stressed_bundle(
    base_scen_set: MortalityScenarioSet,
    *,
    long_life_bump: float = 0.10,
    short_life_bump: float = 0.10,
    pandemic_year: int | None = None,
    pandemic_severity: float = 1.00,
    pandemic_duration: int = 1,
    plateau_start_year: int | None = None,
    accel_improvement_rate: float = 0.01,
    accel_start_year: int | None = None,
) -> ScenarioBundle:
    """Generate a bundle of stressed scenarios from a base set.

    Stress logic:
        - base: the input scenarios.
        - optimistic / long life: q' = (1 - long_life_bump) * q.
        - pessimistic / short life: q' = (1 + short_life_bump) * q.
        - pandemic_stress: mortality spike over a window around pandemic_year.
        - plateau: freeze improvements from plateau_start_year onward.
        - accel_improvement: accelerate improvements from accel_start_year,
          q_{t+h} *= (1 - accel_improvement_rate)^h.

    Args:
        base_scen_set: Base scenario set (from P or Q engine).
        long_life_bump: Long-life shock magnitude (lower q).
        short_life_bump: Short-life shock magnitude (higher q).
        pandemic_year: Start year for the pandemic shock window.
        pandemic_severity: Additional multiplier during the pandemic window:
            q' = (1 + pandemic_severity) * q.
        pandemic_duration: Pandemic window length in years.
        plateau_start_year: Year from which improvements stop.
        accel_improvement_rate: Annual acceleration of improvements.
        accel_start_year: Year from which acceleration applies.

    Returns:
        ScenarioBundle with base, optimistic, pessimistic, and optional stresses.
    """
    base = base_scen_set

    optimistic = apply_mortality_shock(
        base,
        shock_type="long_life",
        magnitude=long_life_bump,
    )

    pessimistic = apply_mortality_shock(
        base,
        shock_type="short_life",
        magnitude=short_life_bump,
    )

    pandemic_scen = None
    if pandemic_year is not None:
        pandemic_scen = apply_mortality_shock(
            base,
            shock_type="pandemic",
            magnitude=pandemic_severity,
            pandemic_year=pandemic_year,
            pandemic_duration=pandemic_duration,
        )

    plateau_scen = None
    if plateau_start_year is not None:
        plateau_scen = apply_mortality_shock(
            base,
            shock_type="plateau",
            magnitude=0.0,  # magnitude n'est pas utilisé pour plateau
            plateau_start_year=plateau_start_year,
        )

    accel_scen = None
    if accel_improvement_rate is not None and accel_improvement_rate != 0.0:
        accel_scen = apply_mortality_shock(
            base,
            shock_type="accel_improvement",
            magnitude=accel_improvement_rate,
            accel_start_year=accel_start_year,
        )

    return ScenarioBundle(
        base=base,
        optimistic=optimistic,
        pessimistic=pessimistic,
        pandemic_stress=pandemic_scen,
        plateau=plateau_scen,
        accel_improvement=accel_scen,
    )


# ============================================================================
# 4) Stress test calibré : +Δ années d'espérance de vie (life expectancy)
# ============================================================================


def _life_expectancy_from_q(
    q_1d: FloatArray,
    *,
    include_half_year: bool = True,
) -> float:
    """Approximate remaining life expectancy on an annual grid.

    Args:
        q_1d: One-year death probabilities by year, shape (H,).
        include_half_year: If True, apply a half-year continuity correction.

    Returns:
        Remaining life expectancy in years.

    Notes:
        With half-year correction:
            E[T] ≈ 0.5 + sum_{t=1}^{H-1} S[t]
        Without correction:
            E[T] ≈ sum_{t=0}^{H-1} S[t]
    """
    q_1d = np.asarray(q_1d, dtype=float).reshape(-1)
    validate_q(q_1d[None, None, :])

    S = survival_from_q(q_1d[None, None, :])[0, 0, :]
    validate_survival_monotonic(S[None, :])

    if include_half_year:
        # start at t=1 because S[0] is end-of-year-0 survival
        return float(0.5 + S[1:].sum())
    return float(S.sum())


def apply_life_expectancy_shift(
    scen_set: MortalityScenarioSet,
    *,
    age: float,
    delta_years: float = 2.0,
    year_start: int | None = None,
    bracket: tuple[float, float] = (0.0, 0.5),
    tol: float = 1e-4,
    max_iter: int = 60,
) -> MortalityScenarioSet:
    """Apply a longevity shift to increase remaining life expectancy.

    We solve for alpha so that, over the future window (>= year_start):
        q' = q * (1 - alpha)
    and the change in remaining life expectancy satisfies:
        e'(age) - e(age) ≈ delta_years.

    Args:
        scen_set: Base scenario set.
        age: Target age (closest grid age is used).
        delta_years: Desired increase in remaining life expectancy.
        year_start: First projection year to apply the shift.
        bracket: Initial bracket for alpha root-finding.
        tol: Root-finding tolerance.
        max_iter: Maximum number of iterations.

    Returns:
        Scenario set with adjusted q_paths and S_paths.
    """
    if delta_years <= 0:
        raise ValueError("delta_years must be > 0.")
    years = np.asarray(scen_set.years, dtype=int)
    q_base = np.asarray(scen_set.q_paths, dtype=float)

    # trouver l'index âge
    ages_grid = np.asarray(scen_set.ages, dtype=float)
    age_idx = int(np.argmin(np.abs(ages_grid - float(age))))

    # fenêtre à shifter (par défaut toute la projection)
    if year_start is None:
        t0 = 0
    else:
        t0 = int(np.searchsorted(years, int(year_start)))
        t0 = max(0, min(t0, q_base.shape[2] - 1))

    # courbe moyenne de q pour cet âge (sur scénarios)
    q_mean = q_base[:, age_idx, :].mean(axis=0)  # (H,)
    e0 = _life_expectancy_from_q(q_mean[t0:], include_half_year=True)

    target = e0 + float(delta_years)

    def gap(alpha: float) -> float:
        a = float(alpha)
        q_shift = q_mean.copy()
        q_shift[t0:] = q_shift[t0:] * (1.0 - a)
        # clamp soft (validate_q fera le vrai check)
        q_shift = np.clip(q_shift, 0.0, 1.0)
        e1 = _life_expectancy_from_q(q_shift[t0:], include_half_year=True)
        return e1 - target  # root when = 0

    a, b = float(bracket[0]), float(bracket[1])
    fa, fb = float(gap(a)), float(gap(b))

    # Auto-expand upper bound if not bracketed
    if fa * fb > 0:
        # we can only improve by lowering q, so expand b upward toward 1
        b_try = b
        for _ in range(12):
            b_try = min(0.999999, 0.5 * (b_try + 1.0))  # move toward 1
            fb_try = float(gap(b_try))
            if fa * fb_try <= 0:
                b, fb = b_try, fb_try
                break
        else:
            raise ValueError(
                "Life-expectancy root not bracketed. Widen `bracket` (e.g. (0, 0.99)) "
                f"or choose a smaller delta_years. gap(a)={fa}, gap(b)={fb}."
            )

    left, right = float(a), float(b)
    f_left, _f_right = float(fa), float(fb)

    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = float(gap(mid))
        if abs(f_mid) < tol or 0.5 * (right - left) < tol:
            alpha_star = mid
            break
        if f_left * f_mid <= 0:
            right, _f_right = mid, f_mid
        else:
            left, f_left = mid, f_mid
    else:
        alpha_star = 0.5 * (left + right)

    # appliquer alpha* à tous les scénarios sur (âge_idx, t>=t0)
    q_new = q_base.copy()
    q_new[:, age_idx, t0:] *= 1.0 - alpha_star
    validate_q(q_new)
    S_new = survival_from_q(q_new)
    validate_survival_monotonic(S_new)

    return clone_scen_set_with(scen_set, q_paths=q_new, S_paths=S_new)


# ============================================================================
# 5) Cohort trends shock (année de naissance = year - age)
# ============================================================================


def apply_cohort_trend_shock(
    scen_set: MortalityScenarioSet,
    *,
    cohort_start: int,
    cohort_end: int,
    magnitude: float = 0.05,
    direction: str = "favorable",
    ramp: bool = True,
) -> MortalityScenarioSet:
    """Apply a cohort-trend shock to selected birth cohorts.

    Cohort definition:
        cohort_year = calendar_year - age

    Shock:
        - direction="favorable": improvement, q' = q * (1 - w * magnitude)
        - direction="adverse": deterioration, q' = q * (1 + w * magnitude)
        where w in [0, 1] optionally ramps between cohort_start and cohort_end.

    Args:
        scen_set: Base scenario set.
        cohort_start: First cohort year affected.
        cohort_end: Last cohort year affected.
        magnitude: Maximum intensity (e.g., 0.05 = +/-5%).
        direction: "favorable" or "adverse".
        ramp: If True, ramp weights linearly between start and end.

    Returns:
        Scenario set with adjusted q_paths and S_paths.
    """
    if cohort_end < cohort_start:
        raise ValueError("cohort_end must be >= cohort_start.")
    if magnitude < 0:
        raise ValueError("magnitude must be >= 0.")
    direction = direction.lower()
    if direction not in {"favorable", "adverse"}:
        raise ValueError("direction must be 'favorable' or 'adverse'.")

    q_base = np.asarray(scen_set.q_paths, dtype=float)
    S_base = np.asarray(scen_set.S_paths, dtype=float)
    if q_base.shape != S_base.shape:
        raise ValueError(
            f"q_paths and S_paths must have same shape; got {q_base.shape} vs {S_base.shape}."
        )

    ages_int = np.round(scen_set.ages).astype(int)  # (A,)
    years = np.asarray(scen_set.years, dtype=int)  # (H,)
    _N, _A, H = q_base.shape
    if years.shape[0] != H:
        raise ValueError("scen_set.years length must match q_paths horizon.")

    # cohort_year[a,t] = year[t] - age[a]
    cohort_year = years[None, :] - ages_int[:, None]  # (A, H) int
    # masque cohortes ciblées
    mask = (cohort_year >= cohort_start) & (cohort_year <= cohort_end)  # (A, H)

    if not np.any(mask):
        # Rien à faire
        return scen_set

    # poids w (A,H)
    if ramp and cohort_end > cohort_start:
        w = (cohort_year - float(cohort_start)) / float(cohort_end - cohort_start)
        w = np.clip(w, 0.0, 1.0)
        w = w * mask.astype(float)
    else:
        w = mask.astype(float)

    q_new = q_base.copy()

    factor = (
        1.0 - float(magnitude) * w if direction == "favorable" else 1.0 + float(magnitude) * w
    )  # (A,H)

    # broadcast (N,A,H) * (A,H)
    q_new *= factor[None, :, :]

    validate_q(q_new)
    S_new = survival_from_q(q_new)
    validate_survival_monotonic(S_new)

    return clone_scen_set_with(scen_set, q_paths=q_new, S_paths=S_new)


@dataclass(frozen=True)
class ShockSpec:
    name: str
    shock_type: str
    params: dict[str, Any]


def apply_shock_spec(
    scen_set: MortalityScenarioSet,
    spec: ShockSpec,
) -> MortalityScenarioSet:
    st = spec.shock_type.lower()

    if st == "cohort":
        return apply_cohort_trend_shock(scen_set, **spec.params)

    if st == "life_expectancy":
        return apply_life_expectancy_shift(scen_set, **spec.params)

    # sinon, shocks “q-based” gérés par apply_mortality_shock
    return apply_mortality_shock(scen_set, shock_type=st, **spec.params)


def generate_stressed_scenarios(
    base_scen_set: MortalityScenarioSet,
    *,
    shock_list: list[ShockSpec] | None = None,
) -> dict[str, MortalityScenarioSet]:
    """Return a dict of stressed scenarios by name.

    Args:
        base_scen_set: Base scenario set.
        shock_list: List of ShockSpec definitions. If None, only "base" is returned.

    Returns:
        Mapping {scenario_name: MortalityScenarioSet}.
    """
    out: dict[str, MortalityScenarioSet] = {"base": base_scen_set}
    if not shock_list:
        return out

    for spec in shock_list:
        out[spec.name] = apply_shock_spec(base_scen_set, spec)

    return out


def apply_shock_chain(
    scen_set: MortalityScenarioSet,
    chain: list[ShockSpec],
) -> MortalityScenarioSet:
    out = scen_set
    for spec in chain:
        out = apply_shock_spec(out, spec)
    return out
