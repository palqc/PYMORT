"""Scenario containers and serialization helpers.

This module defines the standard mortality scenario container used by pricing
and provides helpers to validate and serialize scenarios.

Note:
    Docstrings follow Google style for clarity and spec alignment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pymort._types import AnyArray, FloatArray
from pymort.analysis.projections import ProjectionResult
from pymort.lifetables import (
    cohort_survival_from_q_paths,
    validate_q,
    validate_survival_monotonic,
)


@dataclass
class MortalityScenarioSet:
    """Container for stochastic mortality scenarios, ready for pricing.

    This is the standard interface between the modeling layer (fit + projections)
    and the pricing layer (longevity bonds, swaps, forwards, etc.).

    Attributes:
        years (np.ndarray): Future calendar years, shape (H_out,). Typically
            starts at the last observed year (+ optional t=0 anchor).
        ages (np.ndarray): Age grid, shape (A,).
        q_paths (np.ndarray): Death probabilities q_{x,t} under the chosen
            measure (P or Q), shape (N, A, H_out).
        S_paths (np.ndarray): Survival probabilities S_{x,t}, shape (N, A, H_out),
            usually computed from q_paths.
        m_paths (np.ndarray | None): Central death rates m_{x,t}, shape
            (N, A, H_out) for log-m models; None for logit-q models.
        discount_factors (np.ndarray | None): Discount factors D_t, shape (H_out,)
            or scenario-specific factors (N, H_out). If None, pricing routines
            may assume deterministic or external discount curves.
        metadata (dict[str, Any]): Extra context such as measure and model name.
    """

    years: AnyArray
    ages: AnyArray

    q_paths: FloatArray
    S_paths: FloatArray

    m_paths: FloatArray | None = None
    discount_factors: FloatArray | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def n_scenarios(self) -> int:
        """Return the number of scenarios stored in the set."""
        return int(self.q_paths.shape[0])

    def n_ages(self) -> int:
        """Return the number of ages in the scenario grid."""
        return int(self.ages.shape[0])

    def horizon(self) -> int:
        """Return the projection horizon length."""
        return int(self.years.shape[0])


def validate_scenario_set(scen_set: MortalityScenarioSet) -> None:
    """Validate a MortalityScenarioSet for basic consistency.

    Args:
        scen_set: Scenario set to validate.

    Raises:
        ValueError: If any shape, finiteness, or monotonicity checks fail.
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
        raise ValueError(f"S_paths shape {S.shape} incompatible with q_paths {q.shape}.")
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
    """Persist a MortalityScenarioSet to disk as compressed NPZ.

    Args:
        scen_set: Scenario set to serialize.
        path: Output path for the NPZ file.
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
    """Load a MortalityScenarioSet saved by `save_scenario_set_npz`.

    Args:
        path: NPZ file path.

    Returns:
        Deserialized MortalityScenarioSet.
    """
    data = np.load(Path(path), allow_pickle=True)
    metadata_raw = data.get("metadata", "{}")
    try:
        metadata = json.loads(str(metadata_raw))
    except Exception:
        metadata = {}
    return MortalityScenarioSet(
        years=np.asarray(data["years"]),
        ages=np.asarray(data["ages"]),
        q_paths=np.asarray(data["q_paths"]),
        S_paths=np.asarray(data["S_paths"]),
        m_paths=data.get("m_paths", None),
        discount_factors=(data.get("discount_factors", None)),
        metadata=metadata,
    )


def build_scenario_set_from_projection(
    proj: ProjectionResult,
    ages: AnyArray,
    discount_factors: FloatArray | None = None,
    metadata: dict[str, Any] | None = None,
) -> MortalityScenarioSet:
    """Build a MortalityScenarioSet from a ProjectionResult.

    Args:
        proj: Output of `project_mortality_from_bootstrap` with years and paths.
        ages: Age grid, shape (A,).
        discount_factors: Optional discount factors aligned with proj.years.
        metadata: Optional metadata (e.g., {"model": "LCM2", "measure": "P"}).

    Returns:
        MortalityScenarioSet ready for pricing.
    """
    q_paths = np.asarray(proj.q_paths, dtype=float)
    validate_q(q_paths)
    ages = np.asarray(ages, dtype=float)
    if q_paths.ndim != 3:
        raise ValueError(f"proj.q_paths must have shape (N, A, H), got {q_paths.shape}.")

    _N, A, H = q_paths.shape
    if ages.shape[0] != A:
        raise ValueError(f"Age dimension mismatch: q_paths has A={A}, ages has {ages.shape[0]}.")
    if proj.years.shape[0] != H:
        raise ValueError(
            f"Time dimension mismatch: q_paths has H={H}, proj.years has {proj.years.shape[0]}."
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
                    f"discount_factors must have shape (N,H) with H=len(proj.years); got {df.shape}"
                )
        else:
            raise ValueError("discount_factors must be 1D or 2D.")
        if not (np.all(df > 0) and np.isfinite(df).all()):
            raise ValueError("discount_factors must be positive and finite.")
        discount_factors = df

    metadata = {} if metadata is None else dict(metadata)

    return MortalityScenarioSet(
        years=proj.years,
        ages=ages,
        q_paths=q_paths,
        S_paths=S_paths,
        m_paths=proj.m_paths,
        discount_factors=discount_factors,
        metadata=metadata,
    )
