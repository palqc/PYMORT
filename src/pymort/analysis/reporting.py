from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from pymort.analysis import MortalityScenarioSet

# =============================================================================
# 1) Core : RiskReport dataclass
# =============================================================================


@dataclass
class RiskReport:
    """
    Résumé structuré du risque pour un portefeuille (liability seul ou hedgé).

    Attributes
    ----------
    name : str
        Nom du portefeuille / produit (ex: "Cohort annuity", "Annuity + hedge").
    n_scenarios : int
        Nombre de scénarios Monte Carlo.
    mean_pv : float
        Espérance du PV.
    std_pv : float
        Écart-type du PV.
    pv_min : float
        Minimum observé.
    pv_max : float
        Maximum observé.
    var_level : float
        Niveau de VaR (ex: 0.99).
    var : float
        Value-at-Risk (perte quantile) à var_level.
    cvar : float
        Conditional VaR (Expected Shortfall) à var_level.
    quantiles : Dict[float, float]
        Quelques quantiles intermédiaires (par ex. 1%, 5%, 50%, 95%, 99%).
    hedge_var_reduction : Optional[float]
        Réduction de variance par rapport à une référence (si fournie).
    extra : Dict[str, float]
        Espace libre pour rajouter d'autres métriques.
    """

    name: str
    n_scenarios: int
    mean_pv: float
    std_pv: float
    pv_min: float
    pv_max: float
    var_level: float
    var: float
    cvar: float
    quantiles: Dict[float, float]
    hedge_var_reduction: Optional[float] = None
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """
        Conversion pratique en dict (pour logs, JSON, tableaux, etc.).
        """
        return asdict(self)


# =============================================================================
# 2) Helpers stats : VaR / CVaR
# =============================================================================


def _compute_var_cvar(
    pv_paths: np.ndarray,
    alpha: float = 0.99,
) -> Tuple[float, float]:
    """
    Compute empirical Value-at-Risk and Conditional VaR (Expected Shortfall).

    Convention :
      - pv_paths : chemin de PV (N,)
      - on s'intéresse au *côté défavorable* du passif.
        Typiquement, si PV>0 est une perte pour l'assureur, VaR_α est
        le quantile α des PV.
    """
    x = np.asarray(pv_paths, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("pv_paths must contain at least one scenario.")

    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")

    # VaR = quantile alpha
    var = float(np.quantile(x, alpha))

    # CVaR = moyenne des PV au-delà de VaR (queue droite)
    tail = x[x >= var]
    if tail.size == 0:
        cvar = var
    else:
        cvar = float(tail.mean())

    return var, cvar


# =============================================================================
# 3) Rapport de risque : un portefeuille, avec éventuellement un hedge
# =============================================================================


def generate_risk_report(
    pv_paths: np.ndarray,
    *,
    name: str = "Portfolio",
    var_level: float = 0.99,
    ref_pv_paths: Optional[np.ndarray] = None,
) -> RiskReport:
    """
    Génère un RiskReport pour un ensemble de PV de scénario.

    Paramètres
    ----------
    pv_paths : np.ndarray
        Shape (N,) – PV du portefeuille (liability seul ou net hedgé).
    name : str
        Nom du portefeuille.
    var_level : float, default 0.99
        Niveau de quantile pour la VaR / CVaR.
    ref_pv_paths : np.ndarray, optional
        PV d'une référence (ex: liability seul). Si fourni, on calcule
        la réduction de variance :
            1 - Var(portefeuille) / Var(ref)

    Returns
    -------
    RiskReport
        Rapport structuré avec stats + VaR + CVaR.
    """
    x = np.asarray(pv_paths, dtype=float).reshape(-1)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("pv_paths must be a non-empty 1D array.")

    n = x.size
    mean_pv = float(x.mean())
    std_pv = float(x.std(ddof=0))
    pv_min = float(x.min())
    pv_max = float(x.max())

    var, cvar = _compute_var_cvar(x, alpha=var_level)

    # Quelques quantiles utiles
    qs = [0.01, 0.05, 0.50, 0.95, var_level]
    quantiles = {q: float(np.quantile(x, q)) for q in qs}

    # Réduction de variance vs référence
    hedge_var_reduction: Optional[float] = None
    if ref_pv_paths is not None:
        ref = np.asarray(ref_pv_paths, dtype=float).reshape(-1)
        if ref.size != n:
            raise ValueError(
                "ref_pv_paths must have same number of scenarios as pv_paths."
            )
        var_ref = float(ref.var(ddof=0))
        var_port = float(x.var(ddof=0))
        hedge_var_reduction = 1.0 - var_port / var_ref if var_ref > 0.0 else None

    return RiskReport(
        name=name,
        n_scenarios=n,
        mean_pv=mean_pv,
        std_pv=std_pv,
        pv_min=pv_min,
        pv_max=pv_max,
        var_level=var_level,
        var=var,
        cvar=cvar,
        quantiles=quantiles,
        hedge_var_reduction=hedge_var_reduction,
        extra={},
    )


# =============================================================================
# 4) Graphiques : survival fan, distribution de prix, hedge performance
# =============================================================================


def plot_survival_fan(
    scen_set: MortalityScenarioSet,
    age: float,
    *,
    ax=None,
    quantiles: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
):
    """
    Trace un "survival fan" pour un âge donné :
        - courbe de médiane
        - bandes inter-quantiles.

    Nécessite matplotlib, mais on gère l'import de façon souple.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is required for plot_survival_fan but is not installed."
        ) from e

    required = {0.05, 0.25, 0.50, 0.75, 0.95}
    if set(quantiles) != required:
        raise ValueError("quantiles must be exactly (0.05, 0.25, 0.50, 0.75, 0.95).")

    ages = np.asarray(scen_set.ages, dtype=float)
    years = np.asarray(scen_set.years, dtype=float)
    S_paths = np.asarray(scen_set.S_paths, dtype=float)  # (N, A, H)

    # Trouver l'indice d'âge le plus proche
    idx_age = int(np.argmin(np.abs(ages - float(age))))
    S_age = S_paths[:, idx_age, :]  # (N, H)

    # Stats par année
    bands = {q: np.quantile(S_age, q, axis=0) for q in quantiles}

    if ax is None:
        fig, ax = plt.subplots()

    # Médiane
    ax.plot(years, bands[0.50], label="Median survival", linewidth=2)

    # Bandes inf/sup
    ax.fill_between(
        years,
        bands[0.25],
        bands[0.75],
        alpha=0.3,
        label="50% band (25–75%)",
    )
    ax.fill_between(
        years,
        bands[0.05],
        bands[0.95],
        alpha=0.2,
        label="90% band (5–95%)",
    )

    ax.set_title(f"Survival fan – age {ages[idx_age]:.0f}")
    ax.set_xlabel("Calendar year")
    ax.set_ylabel("Survival probability S(x,t)")
    ax.legend()
    ax.grid(True)

    return ax


def plot_price_distribution(
    pv_paths: np.ndarray,
    *,
    ax=None,
    bins: int = 50,
    density: bool = True,
    label: str = "PV distribution",
):
    """
    Histogramme de la distribution des PV de scénario.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is required for plot_price_distribution but is not installed."
        ) from e

    x = np.asarray(pv_paths, dtype=float).reshape(-1)

    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(x, bins=bins, density=density, alpha=0.7)
    ax.set_title(label)
    ax.set_xlabel("Present value")
    ax.set_ylabel("Density" if density else "Count")
    ax.grid(True)

    return ax


def plot_hedge_performance(
    liability_pv_paths: np.ndarray,
    net_pv_paths: np.ndarray,
    *,
    ax=None,
    label_liability: str = "Liability PV",
    label_net: str = "Net (Liability + Hedge) PV",
):
    """
    Scatter plot comparant PV du liability vs PV net hedgé
    pour visualiser la réduction de dispersion (variance).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is required for plot_hedge_performance but is not installed."
        ) from e

    L = np.asarray(liability_pv_paths, dtype=float).reshape(-1)
    N = np.asarray(net_pv_paths, dtype=float).reshape(-1)
    if L.size != N.size:
        raise ValueError("liability_pv_paths and net_pv_paths must have same length.")

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(L, N, alpha=0.4, s=10)
    ax.set_title("Hedge performance – scenario scatter")
    ax.set_xlabel(label_liability)
    ax.set_ylabel(label_net)
    ax.axline((0, 0), slope=1.0, color="black", linewidth=1, linestyle="--")
    ax.grid(True)

    return ax
