from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pymort.analysis.projections import ProjectionResult
from pymort.lifetables import (
    cohort_survival_from_q_paths,
    validate_q,
    validate_survival_monotonic,
)


@dataclass
class MortalityScenarioSet:
    """
    Container for stochastic mortality scenarios, ready for pricing.

    This is the standard interface between the mortality modeling
    layer (fit + projections) and the pricing layer (longevity bonds,
    survivor swaps, q-forwards, etc.).

    Attributes
    ----------
    years : np.ndarray
        Future calendar years of shape (H_out,).
        Typically starts at last_observed_year (+ optional t=0 anchor).
    ages : np.ndarray
        Ages grid of shape (A,).

    q_paths : np.ndarray
        Death probabilities q_{x,t} under the chosen measure (P or Q),
        with shape (N, A, H_out), where
        - N : number of scenarios (bootstrap * process risk),
        - A : number of ages,
        - H_out : number of future time points.

    S_paths : np.ndarray
        Survival probabilities S_{x,t} corresponding to q_paths,
        shape (N, A, H_out). Usually S_paths is obtained from q_paths
        via ``survival_paths_from_q_paths``.

    m_paths : Optional[np.ndarray]
        Optional central death rates m_{x,t} (N, A, H_out).
        Only populated for log-m models (e.g., LC/APC). For CBD
        models (logit q), this may be None.

    discount_factors : Optional[np.ndarray]
        Optional discount factors D_t of shape (H_out,). If None,
        pricing routines may assume deterministic or external
        discount curves.

    metadata : dict
        Optional dictionary for extra context, e.g.:
        - "measure": "P" or "Q"
        - "model": "LCM2", "CBDM7", ...
        - "source": "bootstrap_from_m + project_mortality_from_bootstrap"
    """

    years: np.ndarray
    ages: np.ndarray

    q_paths: np.ndarray
    S_paths: np.ndarray

    m_paths: Optional[np.ndarray] = None
    discount_factors: Optional[np.ndarray] = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def n_scenarios(self) -> int:
        return int(self.q_paths.shape[0])

    def n_ages(self) -> int:
        return int(self.ages.shape[0])

    def horizon(self) -> int:
        return int(self.years.shape[0])


def validate_scenario_set(scen_set: MortalityScenarioSet) -> None:
    """
    Lightweight validation of a MortalityScenarioSet.
    Raises ValueError on any detected inconsistency.
    """
    years = np.asarray(scen_set.years)
    ages = np.asarray(scen_set.ages)
    q = np.asarray(scen_set.q_paths, dtype=float)
    S = np.asarray(scen_set.S_paths, dtype=float)

    if q.ndim != 3:
        raise ValueError(f"q_paths must be 3D (N,A,T); got shape {q.shape}.")
    if S.ndim != 3:
        raise ValueError(f"S_paths must be 3D (N,A,T); got shape {S.shape}.")

    N, A, T = q.shape
    if S.shape != (N, A, T):
        raise ValueError(
            f"S_paths shape {S.shape} incompatible with q_paths {q.shape}."
        )
    if ages.shape != (A,):
        raise ValueError(f"ages must have shape ({A},); got {ages.shape}.")
    if years.shape != (T,):
        raise ValueError(f"years must have shape ({T},); got {years.shape}.")

    validate_q(q)
    try:
        validate_survival_monotonic(S)
    except AssertionError as exc:
        raise ValueError(str(exc)) from exc
    if np.any(S < 0) or np.any(S > 1) or not np.isfinite(S).all():
        raise ValueError("S_paths must lie in [0,1] and be finite.")
    df = scen_set.discount_factors
    if df is not None:
        df_arr = np.asarray(df, dtype=float)
        if df_arr.ndim == 1:
            if df_arr.shape[0] < T:
                raise ValueError(
                    f"discount_factors length must be >= T={T}; got {df_arr.shape[0]}."
                )
        elif df_arr.ndim == 2:
            if df_arr.shape[1] < T:
                raise ValueError(
                    f"discount_factors second dim must be >= T={T}; got {df_arr.shape}."
                )
            if df_arr.shape[0] not in (1, N):
                raise ValueError(
                    f"discount_factors first dim must be 1 or N={N}; got {df_arr.shape[0]}."
                )
        else:
            raise ValueError("discount_factors must be 1D or 2D.")
        if not (np.isfinite(df_arr).all() and np.all(df_arr > 0)):
            raise ValueError("discount_factors must be positive and finite.")


def save_scenario_set_npz(scen_set: MortalityScenarioSet, path: Path | str) -> None:
    """
    Persist a MortalityScenarioSet to disk as compressed NPZ.
    """
    target = Path(path)
    payload: dict[str, Any] = {
        "q_paths": scen_set.q_paths,
        "S_paths": scen_set.S_paths,
        "ages": scen_set.ages,
        "years": scen_set.years,
        "metadata": json.dumps(scen_set.metadata),
    }
    if scen_set.m_paths is not None:
        payload["m_paths"] = scen_set.m_paths
    if scen_set.discount_factors is not None:
        payload["discount_factors"] = scen_set.discount_factors
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **payload)


def load_scenario_set_npz(path: Path | str) -> MortalityScenarioSet:
    """
    Load a MortalityScenarioSet saved by ``save_scenario_set_npz``.
    """
    data = np.load(Path(path), allow_pickle=True)
    metadata_raw = data.get("metadata", "{}")
    try:
        metadata = json.loads(str(metadata_raw))
    except Exception:
        metadata = {}
    scen_set = MortalityScenarioSet(
        years=np.asarray(data["years"]),
        ages=np.asarray(data["ages"]),
        q_paths=np.asarray(data["q_paths"]),
        S_paths=np.asarray(data["S_paths"]),
        m_paths=data["m_paths"] if "m_paths" in data else None,
        discount_factors=(
            data["discount_factors"] if "discount_factors" in data else None
        ),
        metadata=metadata,
    )
    return scen_set


def build_scenario_set_from_projection(
    proj: ProjectionResult,
    ages: np.ndarray,
    discount_factors: Optional[np.ndarray] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> MortalityScenarioSet:
    """
    Build a MortalityScenarioSet from a ProjectionResult.

    Parameters
    ----------
    proj : ProjectionResult
        Output of project_mortality_from_bootstrap, containing
        years, q_paths, m_paths, k_paths.
    ages : np.ndarray
        Ages grid of shape (A,).
    discount_factors : np.ndarray | None
        Optional discount factors D_t for the same timeline as proj.years.
        If None, pricing routines can use an external yield curve or
        assume deterministic rates.
    metadata : dict | None
        Optional metadata (e.g. {"model": "LCM2", "measure": "P"}).

    Returns
    -------
    MortalityScenarioSet
        Ready-to-use mortality scenario container for pricing.
    """
    q_paths = np.asarray(proj.q_paths, dtype=float)
    validate_q(q_paths)
    ages = np.asarray(ages, dtype=float)
    if q_paths.ndim != 3:
        raise ValueError(
            f"proj.q_paths must have shape (N, A, H), got {q_paths.shape}."
        )

    N, A, H = q_paths.shape
    if A != ages.shape[0]:
        raise ValueError(
            f"Age dimension mismatch: q_paths has A={A}, ages has {ages.shape[0]}."
        )
    if H != proj.years.shape[0]:
        raise ValueError(
            f"Time dimension mismatch: q_paths has H={H}, proj.years has "
            f"{proj.years.shape[0]}."
        )
    S_paths = cohort_survival_from_q_paths(q_paths)
    validate_survival_monotonic(S_paths)

    if discount_factors is not None:
        df = np.asarray(discount_factors, dtype=float)
        if df.ndim == 1:
            if df.shape[0] != proj.years.shape[0]:
                raise ValueError(
                    "discount_factors must have same length as proj.years: "
                    f"{df.shape[0]} vs {proj.years.shape[0]}"
                )
        elif df.ndim == 2:
            if df.shape[1] != proj.years.shape[0]:
                raise ValueError(
                    "discount_factors must have shape (N,H) with H=len(proj.years); "
                    f"got {df.shape}"
                )
        else:
            raise ValueError("discount_factors must be 1D or 2D.")
        if not (np.all(df > 0) and np.isfinite(df).all()):
            raise ValueError("discount_factors must be positive and finite.")
        discount_factors = df

    if metadata is None:
        metadata = {}
    else:
        metadata = dict(metadata)

    return MortalityScenarioSet(
        years=proj.years,
        ages=ages,
        q_paths=q_paths,
        S_paths=S_paths,
        m_paths=proj.m_paths,
        discount_factors=discount_factors,
        metadata=metadata,
    )
