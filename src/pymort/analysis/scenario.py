from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from pymort.analysis.projections import ProjectionResult
from pymort.lifetables import survival_paths_from_q_paths


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
    S_paths = survival_paths_from_q_paths(q_paths)

    if discount_factors is not None:
        discount_factors = np.asarray(discount_factors, dtype=float)
        if discount_factors.shape[0] != proj.years.shape[0]:
            raise ValueError(
                "discount_factors must have same length as proj.years: "
                f"{discount_factors.shape[0]} vs {proj.years.shape[0]}"
            )

    if metadata is None:
        metadata = {}

    return MortalityScenarioSet(
        years=proj.years,
        ages=ages,
        q_paths=q_paths,
        S_paths=S_paths,
        m_paths=proj.m_paths,
        discount_factors=discount_factors,
        metadata=metadata,
    )
