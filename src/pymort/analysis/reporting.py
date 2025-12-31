"""Risk reporting utilities for scenario PV paths.

This module builds structured risk summaries (VaR/CVaR, moments, and
comparisons) from simulated present-value paths.

Note:
    Docstrings follow Google style for clarity and spec alignment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from pymort._types import FloatArray
from pymort.analysis import MortalityScenarioSet

PVKind = Literal["cost", "value", "pnl"]

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# =============================================================================
# 1) Core dataclass
# =============================================================================


@dataclass
class RiskReport:
    """Structured risk summary for scenario PVs.

    Convention:
        Risk is computed on a *loss* variable L:
          - pv_kind="cost":  L =  PV   (bigger PV = worse)
          - pv_kind="value": L = -PV   (bigger PV = better, so loss is -PV)
          - pv_kind="pnl":   L = -PnL  (profit positive -> loss is negative)

    Reported VaR/CVaR are computed on L (loss units).
    """

    name: str
    pv_kind: PVKind

    n_scenarios: int

    # PV stats
    mean_pv: float
    std_pv: float
    pv_min: float
    pv_max: float

    # Loss stats (derived from PV convention)
    mean_loss: float
    std_loss: float

    var_level: float
    var: float  # VaR on losses
    cvar: float  # CVaR/ES on losses

    quantiles_pv: dict[float, float]
    quantiles_loss: dict[float, float]

    # Relative-to-reference analytics (optional)
    hedge_var_reduction: float | None = None
    hedge_var_reduction_loss: float | None = None
    hedge_var_reduction_var: float | None = None
    hedge_var_reduction_cvar: float | None = None
    corr_with_ref: float | None = None
    beta_vs_ref: float | None = None
    tracking_error: float | None = None

    # Extra metrics
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# =============================================================================
# 2) Helpers
# =============================================================================


def _as_1d(x: FloatArray, *, name: str) -> FloatArray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one scenario.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    return arr


def _pv_to_loss(pv: FloatArray, pv_kind: PVKind) -> FloatArray:
    if pv_kind == "cost":
        return pv
    if pv_kind in ("value", "pnl"):
        return -pv
    raise ValueError(f"Unknown pv_kind: {pv_kind!r}")


def _compute_var_cvar(loss_paths: FloatArray, alpha: float) -> tuple[float, float]:
    if not (0.0 < alpha < 1.0):
        raise ValueError("var_level (alpha) must be in (0,1).")

    L = loss_paths
    var = float(np.quantile(L, alpha))
    tail = L[var <= L]
    cvar = float(tail.mean()) if tail.size > 0 else var
    return var, cvar


def _skew_kurtosis(x: FloatArray) -> tuple[float, float]:
    # population moments (ddof=0)
    mu = float(x.mean())
    s = float(x.std(ddof=0))
    if s <= 0.0:
        return 0.0, 0.0
    z = (x - mu) / s
    skew = float(np.mean(z**3))
    kurt_excess = float(np.mean(z**4) - 3.0)
    return skew, kurt_excess


# =============================================================================
# 3) Main report generator
# =============================================================================


def generate_risk_report(
    pv_paths: FloatArray,
    *,
    name: str = "Portfolio",
    var_level: float = 0.99,
    ref_pv_paths: FloatArray | None = None,
    pv_kind: PVKind = "cost",
    quantile_grid: Sequence[float] = (0.01, 0.05, 0.50, 0.95, 0.99),
    loss_threshold: float | None = None,
) -> RiskReport:
    """Generate a RiskReport from scenario PV paths.

    Args:
        pv_paths: Scenario PVs, shape (N,).
        name: Report label.
        var_level: Quantile level for VaR/CVaR (e.g., 0.99).
        ref_pv_paths: Optional reference PVs (same N) for reduction metrics.
        pv_kind: "cost", "value", or "pnl" to define the loss convention.
        quantile_grid: PV and loss quantiles to store.
        loss_threshold: Optional loss threshold for exceedance probability.

    Returns:
        RiskReport with summary statistics and risk measures.
    """
    pv = _as_1d(pv_paths, name="pv_paths")
    N = pv.size

    loss = _pv_to_loss(pv, pv_kind)

    mean_pv = float(pv.mean())
    std_pv = float(pv.std(ddof=0))
    pv_min = float(pv.min())
    pv_max = float(pv.max())

    mean_loss = float(loss.mean())
    std_loss = float(loss.std(ddof=0))

    # Quantiles (clean + sorted + unique + in (0,1))
    qs = sorted({float(q) for q in quantile_grid} | {float(var_level)})
    for q in qs:
        if not (0.0 < q < 1.0):
            raise ValueError("quantile_grid values must be in (0,1).")

    quantiles_pv = {q: float(np.quantile(pv, q)) for q in qs}
    quantiles_loss = {q: float(np.quantile(loss, q)) for q in qs}

    var, cvar = _compute_var_cvar(loss, alpha=float(var_level))

    # Extra moments (on loss is usually what risk teams want)
    skew_loss, kurt_loss = _skew_kurtosis(loss)

    extra: dict[str, float] = {
        "skew_loss": float(skew_loss),
        "kurtosis_excess_loss": float(kurt_loss),
        "downside_deviation_loss": float(np.sqrt(np.mean(np.minimum(loss - mean_loss, 0.0) ** 2))),
    }

    if loss_threshold is not None:
        thr = float(loss_threshold)
        extra["prob_loss_exceeds_threshold"] = float(np.mean(loss >= thr))

    # Reference comparisons
    hedge_var_reduction = None
    hedge_var_reduction_loss = None
    hedge_var_reduction_var = None
    hedge_var_reduction_cvar = None
    corr_with_ref = None
    beta_vs_ref = None
    tracking_error = None

    if ref_pv_paths is not None:
        ref_pv = _as_1d(ref_pv_paths, name="ref_pv_paths")
        if ref_pv.size != N:
            raise ValueError("ref_pv_paths must have same number of scenarios as pv_paths.")
        ref_loss = _pv_to_loss(ref_pv, pv_kind)

        var_ref_pv = float(ref_pv.var(ddof=0))
        var_port_pv = float(pv.var(ddof=0))
        hedge_var_reduction = (1.0 - var_port_pv / var_ref_pv) if var_ref_pv > 0.0 else None

        var_ref_loss = float(ref_loss.var(ddof=0))
        var_port_loss = float(loss.var(ddof=0))
        hedge_var_reduction_loss = (
            (1.0 - var_port_loss / var_ref_loss) if var_ref_loss > 0.0 else None
        )

        ref_var, ref_cvar = _compute_var_cvar(ref_loss, alpha=float(var_level))
        hedge_var_reduction_var = (1.0 - var / ref_var) if ref_var > 0.0 else None
        hedge_var_reduction_cvar = (1.0 - cvar / ref_cvar) if ref_cvar > 0.0 else None

        # Corr/beta on losses (risk lens)
        s_ref = float(ref_loss.std(ddof=0))
        s_port = float(loss.std(ddof=0))
        if s_ref > 0.0 and s_port > 0.0:
            corr_with_ref = float(np.corrcoef(ref_loss, loss)[0, 1])
            cov = float(np.mean((ref_loss - ref_loss.mean()) * (loss - loss.mean())))
            beta_vs_ref = float(cov / (s_ref**2)) if s_ref > 0.0 else None

        # Tracking error: std(loss - ref_loss)
        tracking_error = float((loss - ref_loss).std(ddof=0))

    return RiskReport(
        name=str(name),
        pv_kind=pv_kind,
        n_scenarios=int(N),
        mean_pv=mean_pv,
        std_pv=std_pv,
        pv_min=pv_min,
        pv_max=pv_max,
        mean_loss=mean_loss,
        std_loss=std_loss,
        var_level=float(var_level),
        var=float(var),
        cvar=float(cvar),
        quantiles_pv=quantiles_pv,
        quantiles_loss=quantiles_loss,
        hedge_var_reduction=hedge_var_reduction,
        hedge_var_reduction_loss=hedge_var_reduction_loss,
        hedge_var_reduction_var=hedge_var_reduction_var,
        hedge_var_reduction_cvar=hedge_var_reduction_cvar,
        corr_with_ref=corr_with_ref,
        beta_vs_ref=beta_vs_ref,
        tracking_error=tracking_error,
        extra=extra,
    )


# =============================================================================
# 4) Plots (robust)
# =============================================================================


def plot_survival_fan(
    scen_set: MortalityScenarioSet,
    age: float,
    *,
    ax: Axes | None = None,
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
) -> Axes:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is required for plot_survival_fan but is not installed."
        ) from e

    qs = tuple(float(q) for q in quantiles)
    if any((q <= 0.0 or q >= 1.0) for q in qs):
        raise ValueError("quantiles must be in (0,1).")
    if 0.50 not in qs:
        raise ValueError("quantiles must include 0.50 (median).")
    if len(set(qs)) != len(qs):
        raise ValueError("quantiles must be unique.")

    ages = np.asarray(scen_set.ages, dtype=float)
    years = np.asarray(scen_set.years, dtype=float)
    S_paths = np.asarray(scen_set.S_paths, dtype=float)  # (N, A, H)

    idx_age = int(np.argmin(np.abs(ages - float(age))))
    S_age = S_paths[:, idx_age, :]  # (N, H)

    bands = {q: np.quantile(S_age, q, axis=0) for q in qs}

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(years, bands[0.50], label="Median survival", linewidth=2)

    # Optional fills if the common bands exist
    if 0.25 in bands and 0.75 in bands:
        ax.fill_between(years, bands[0.25], bands[0.75], alpha=0.3, label="50% band (25–75%)")
    if 0.05 in bands and 0.95 in bands:
        ax.fill_between(years, bands[0.05], bands[0.95], alpha=0.2, label="90% band (5–95%)")

    ax.set_title(f"Survival fan – age {ages[idx_age]:.0f}")
    ax.set_xlabel("Calendar year")
    ax.set_ylabel("Survival probability S(x,t)")
    ax.legend()
    ax.grid(True)
    return ax


def plot_price_distribution(
    pv_paths: FloatArray,
    *,
    ax: Axes | None = None,
    bins: int = 50,
    density: bool = True,
    label: str = "PV distribution",
) -> Axes:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is required for plot_price_distribution but is not installed."
        ) from e

    x = _as_1d(pv_paths, name="pv_paths")

    if ax is None:
        _, ax = plt.subplots()

    ax.hist(x, bins=int(bins), density=bool(density), alpha=0.7)
    ax.set_title(label)
    ax.set_xlabel("Present value")
    ax.set_ylabel("Density" if density else "Count")
    ax.grid(True)
    return ax


def plot_hedge_performance(
    liability_pv_paths: FloatArray,
    net_pv_paths: FloatArray,
    *,
    ax: Axes | None = None,
    label_liability: str = "Liability PV",
    label_net: str = "Net (Liability + Hedge) PV",
) -> Axes:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is required for plot_hedge_performance but is not installed."
        ) from e

    L = _as_1d(liability_pv_paths, name="liability_pv_paths")
    N = _as_1d(net_pv_paths, name="net_pv_paths")
    if L.size != N.size:
        raise ValueError("liability_pv_paths and net_pv_paths must have same length.")

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(L, N, alpha=0.4, s=10)
    ax.set_title("Hedge performance – scenario scatter")
    ax.set_xlabel(label_liability)
    ax.set_ylabel(label_net)

    # 45-degree reference line
    lo = float(min(L.min(), N.min()))
    hi = float(max(L.max(), N.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.grid(True)
    return ax
