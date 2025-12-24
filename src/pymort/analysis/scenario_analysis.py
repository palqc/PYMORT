from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pymort.analysis import MortalityScenarioSet
from pymort.lifetables import survival_from_q, validate_q, validate_survival_monotonic

# ============================================================================
# 1) Outil interne : cloner un MortalityScenarioSet avec q / S modifiés
# ============================================================================


def clone_scen_set_with(
    scen_set: MortalityScenarioSet,
    *,
    q_paths: Optional[np.ndarray] = None,
    S_paths: Optional[np.ndarray] = None,
) -> MortalityScenarioSet:
    """
    Clone un MortalityScenarioSet, en remplaçant éventuellement q_paths / S_paths.

    On reste robuste à l'évolution de la dataclass en introspectant ses champs.
    """
    field_names = list(MortalityScenarioSet.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    kwargs: Dict[str, object] = {}

    for name in field_names:
        if name == "q_paths" and q_paths is not None:
            kwargs[name] = q_paths
        elif name == "S_paths" and S_paths is not None:
            kwargs[name] = S_paths
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
    pandemic_year: Optional[int] = None,
    pandemic_duration: int = 1,
    plateau_start_year: Optional[int] = None,
    accel_start_year: Optional[int] = None,
) -> MortalityScenarioSet:
    """
    Applique un choc de mortalité sur un ensemble de scénarios.

    Shock types supportés
    ---------------------
    - "long_life" :
        Baisse globale des mortalités :
            q' = (1 - magnitude) * q
        ex : magnitude=0.10 -> -10% sur tous les q (survie meilleure).

    - "short_life" :
        Hausse globale des mortalités :
            q' = (1 + magnitude) * q

    - "pandemic" :
        Spike temporaire de mortalité sur une fenêtre **forward** de plusieurs années :
            q'_{t in window} = (1 + magnitude) * q_t

        La fenêtre est définie comme :
            [pandemic_year, pandemic_year + pandemic_duration - 1]

        Paramètres :
            pandemic_year : int (obligatoire)
                Première année affectée par le choc pandémique.
            pandemic_duration : int, default 1
                Nombre d'années consécutives impactées par le choc.

    - "plateau" :
        Plateau des améliorations : à partir de plateau_start_year, on
        fige les taux de mortalité dans le futur :
            q'_{t >= plateau_start} = q_{t_plateau_start}

    - "accel_improvement" :
        Accélération des améliorations de longévité (scénario "long life ++").
        À partir de accel_start_year (ou la 1re année si None) :
            pour un offset h années après, on applique :
                q'_{t0+h} = q_{t0+h} * (1 - magnitude)^h
        ex : magnitude=0.01 -> +1% d'amélioration *supplémentaire* par an.

    Paramètres
    ----------
    scen_set : MortalityScenarioSet
        Scénarios de base (sous P ou Q, peu importe).
    shock_type : str
        "long_life", "short_life", "pandemic", "plateau", "accel_improvement".
    magnitude : float
        Intensité du choc, interprétation dépend du type (voir ci-dessus).
    pandemic_year : int, optionnel
        Première année affectée par le choc pandémique.
    pandemic_duration : int, défaut 1
        Nombre d'années consécutives impactées par le choc,
        à partir de pandemic_year :
            [pandemic_year, pandemic_year + pandemic_duration - 1].
    plateau_start_year : int, optionnel
        À partir de cette année, on fige les q dans le futur ("plateau").
    accel_start_year : int, optionnel
        Année à partir de laquelle on accélère les améliorations.

    Retourne
    --------
    MortalityScenarioSet
        Nouveau set de scénarios avec q_paths et S_paths modifiés.
    """
    q_base = np.asarray(scen_set.q_paths, dtype=float)
    S_base = np.asarray(scen_set.S_paths, dtype=float)

    if q_base.shape != S_base.shape:
        raise ValueError(
            f"q_paths and S_paths must have the same shape; got {q_base.shape} vs {S_base.shape}."
        )

    _N, _A, H = q_base.shape
    years = np.asarray(scen_set.years, dtype=int)
    if years.shape[0] != H:
        raise ValueError("scen_set.years length must match q_paths horizon.")

    q_new = q_base.copy()

    eps = float(magnitude)

    shock_type = shock_type.lower()

    if magnitude < 0:
        raise ValueError("magnitude must be >= 0.")
    if (
        shock_type in {"long_life", "short_life", "accel_improvement"}
        and magnitude >= 1
    ):
        raise ValueError("magnitude must be < 1 for this shock_type.")

    if shock_type == "long_life":
        # Baisse uniforme des mortalités
        q_new *= 1.0 - eps

    elif shock_type == "short_life":
        # Hausse uniforme des mortalités
        q_new *= 1.0 + eps

    elif shock_type == "pandemic":
        if pandemic_year is None:
            raise ValueError(
                "pandemic_year must be provided for shock_type='pandemic'."
            )
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
        q_new[:, :, mask] *= 1.0 + eps

    elif shock_type == "plateau":
        if plateau_start_year is None:
            raise ValueError(
                "plateau_start_year must be provided for shock_type='plateau'."
            )

        # index à partir duquel on fige
        idx_start = np.searchsorted(years, plateau_start_year)
        if idx_start >= H:
            # plateau en dehors de l'horizon -> pas de changement
            return scen_set

        # On fige toutes les années t >= idx_start au niveau de idx_start
        q_new[:, :, idx_start:] = q_new[:, :, idx_start][:, :, None]

    elif shock_type == "accel_improvement":
        # Accélération des améliorations = baisse plus rapide des q dans le temps.
        if accel_start_year is None:
            t0_idx = 0
        else:
            t0_idx = int(np.searchsorted(years, accel_start_year))
        if t0_idx >= H - 1:
            # Rien à accélérer
            return scen_set

        # Pour chaque année h après t0, multiplier par (1 - eps)^h
        offsets = np.arange(H - t0_idx, dtype=float)  # 0,1,...,H-t0-1
        factors = (1.0 - eps) ** offsets  # shape (H - t0,)
        # Broadcast sur (N, A, H-t0)
        q_new[:, :, t0_idx:] *= factors[None, None, :]

    else:
        raise ValueError(f"Unknown shock_type='{shock_type}'.")

    # Validation des q
    validate_q(q_new)

    # Recalcul complet des S à partir des q bumpés
    # (plus simple/robuste que de bricoler seulement certaines tranches)
    S_new = survival_from_q(q_new)
    validate_survival_monotonic(S_new)

    return clone_scen_set_with(scen_set, q_paths=q_new, S_paths=S_new)


# ============================================================================
# 3) Générateur de scénarios "stressés" (bundle Base / Optimistic / Pessimistic / Stress)
# ============================================================================


@dataclass
class ScenarioBundle:
    """
    Petit conteneur pour un ensemble cohérent de scénarios.

    Attributes
    ----------
    base : MortalityScenarioSet
        Scénarios de base (souvent Q-measure calibrés).
    optimistic : MortalityScenarioSet
        Scénarios "long life" (améliorations plus rapides).
    pessimistic : MortalityScenarioSet
        Scénarios "short life" (mortalité plus élevée).
    pandemic_stress : Optional[MortalityScenarioSet]
        Scénarios avec choc pandémique.
    plateau : Optional[MortalityScenarioSet]
        Scénarios avec plateau des améliorations.
    accel_improvement : Optional[MortalityScenarioSet]
        Scénarios avec accélération d'amélioration (long life ++).
    """

    base: MortalityScenarioSet
    optimistic: MortalityScenarioSet
    pessimistic: MortalityScenarioSet
    pandemic_stress: Optional[MortalityScenarioSet] = None
    plateau: Optional[MortalityScenarioSet] = None
    accel_improvement: Optional[MortalityScenarioSet] = None


def generate_stressed_bundle(
    base_scen_set: MortalityScenarioSet,
    *,
    long_life_bump: float = 0.10,
    short_life_bump: float = 0.10,
    pandemic_year: Optional[int] = None,
    pandemic_severity: float = 1.00,
    pandemic_duration: int = 1,
    plateau_start_year: Optional[int] = None,
    accel_improvement_rate: float = 0.01,
    accel_start_year: Optional[int] = None,
) -> ScenarioBundle:
    """
    Génère un bundle de scénarios stressés à partir d'un scénario de base.

    Idée
    ----
    - base : les scénarios fournis (souvent sous Q).
    - optimistic / "long life" :
        q' = (1 - long_life_bump) * q  (par défaut -10% sur tous les q).
    - pessimistic / "short life" :
        q' = (1 + short_life_bump) * q.
    - pandemic_stress :
        spike de mortalité sur une fenêtre autour de `pandemic_year`.
    - plateau :
        à partir de `plateau_start_year`, les q sont figés (plus de progrès).
    - accel_improvement :
        à partir de `accel_start_year` (ou début), on accélère les
        améliorations :
            q_{t+h} *= (1 - accel_improvement_rate)^h

    Paramètres
    ----------
    base_scen_set : MortalityScenarioSet
        Scénarios de base (issus de ton moteur P ou Q).
    long_life_bump : float, défaut 0.10
        Intensité du choc "long life" (baisse des q).
    short_life_bump : float, défaut 0.10
        Intensité du choc "short life" (hausse des q).
    pandemic_year : int, optionnel
        Année centrale du choc pandémique.
    pandemic_severity : float, défaut 1.0
        Multiplicateur additionnel sur q pendant la pandémie :
            q' = (1 + pandemic_severity) * q.
    pandemic_duration : int, défaut 1
        Durée (en années) du choc pandémique.
    plateau_start_year : int, optionnel
        À partir de cette année, les améliorations s'arrêtent.
    accel_improvement_rate : float, défaut 0.01
        Accélération annuelle des améliorations (ex: 0.01 -> +1%/an).
    accel_start_year : int, optionnel
        Année à partir de laquelle l'accélération s'applique.

    Retourne
    --------
    ScenarioBundle
        Contient base, optimistic, pessimistic, et, si spécifiés,
        les scénarios pandemic_stress / plateau / accel_improvement.
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
    q_1d: np.ndarray,
    *,
    include_half_year: bool = True,
) -> float:
    """
    Approximate discrete remaining life expectancy on an annual grid.

    - q_1d[t] : 1-year death probability during year t
    - S[t]    : survival prob at END of year t

    If include_half_year=True, apply a standard continuity correction:
        E[T] ≈ 0.5 + sum_{t=1}^{H-1} S[t]
    Else:
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
    year_start: Optional[int] = None,
    bracket: Tuple[float, float] = (0.0, 0.5),
    tol: float = 1e-4,
    max_iter: int = 60,
) -> MortalityScenarioSet:
    """
    Applique un choc de longévité calibré pour obtenir +delta_years d'espérance
    de vie résiduelle à partir de la courbe moyenne (sur scénarios) à l'âge 'age'.

    On cherche α tel que, sur la fenêtre future (>= year_start):
        q' = q * (1 - α)
    et e'(age) - e(age) ≈ delta_years.
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
    f_left, f_right = float(fa), float(fb)

    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = float(gap(mid))
        if abs(f_mid) < tol or 0.5 * (right - left) < tol:
            alpha_star = mid
            break
        if f_left * f_mid <= 0:
            right, f_right = mid, f_mid
        else:
            left, f_left = mid, f_mid
    else:
        alpha_star = 0.5 * (left + right)

    # appliquer α* à tous les scénarios sur (âge_idx, t>=t0)
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
    """
    Applique un choc "cohort trends" : certaines générations (cohortes)
    ont une mortalité systématiquement plus basse/haute que prévu.

    Définition cohorte
    ------------------
    cohort_year = calendar_year - age

    Shock
    -----
    - direction="favorable" : amélioration => q' = q * (1 - w * magnitude)
    - direction="adverse"   : dégradation => q' = q * (1 + w * magnitude)

    où w ∈ [0,1] est un poids optionnel (ramp) qui fait un tilt progressif
    entre cohort_start et cohort_end.

    Paramètres
    ----------
    cohort_start, cohort_end : int
        Intervalle de cohortes (années de naissance) impactées.
    magnitude : float
        Intensité max (ex 0.05 = +/-5% sur q) au centre/plateau du choc.
    direction : {"favorable","adverse"}
    ramp : bool
        Si True, poids linéaire w entre start/end. Si False, choc uniforme.

    Returns
    -------
    MortalityScenarioSet
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
    N, A, H = q_base.shape
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

    if direction == "favorable":
        # baisse de q => longévité meilleure
        factor = 1.0 - float(magnitude) * w  # (A,H)
    else:
        # hausse de q => mortalité pire
        factor = 1.0 + float(magnitude) * w

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
    params: Dict[str, Any]


def apply_shock_spec(
    scen_set: MortalityScenarioSet,
    spec: ShockSpec,
) -> MortalityScenarioSet:
    st = spec.shock_type.lower()

    if st == "cohort":
        return apply_cohort_trend_shock(scen_set, **spec.params)  # type: ignore[arg-type]

    if st == "life_expectancy":
        return apply_life_expectancy_shift(scen_set, **spec.params)  # type: ignore[arg-type]

    # sinon, shocks “q-based” gérés par apply_mortality_shock
    return apply_mortality_shock(scen_set, shock_type=st, **spec.params)  # type: ignore[arg-type]


def generate_stressed_scenarios(
    base_scen_set: MortalityScenarioSet,
    *,
    shock_list: Optional[list[ShockSpec]] = None,
) -> Dict[str, MortalityScenarioSet]:
    """
    Retourne un dict {scenario_name: MortalityScenarioSet}.
    - Si shock_list est None, retourne juste {"base": base}
    - Sinon, applique chaque shock indépendamment à base.
    """
    out: Dict[str, MortalityScenarioSet] = {"base": base_scen_set}
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
