# streamlit_app/pages/7_Hedging.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from pymort.pipeline import hedging_pipeline

st.set_page_config(page_title="Hedging", page_icon="🛡️", layout="wide")
st.title("🛡️ Hedging")
st.caption("Build hedge weights from scenario PV/CF paths computed on the Pricing page.")


# -----------------------------
# Guards: require pricing outputs
# -----------------------------
prices = st.session_state.get("prices")
specs = st.session_state.get("pricing_specs")
pv_paths = st.session_state.get("pv_paths")
cf_paths = st.session_state.get("cf_paths")

if prices is None or specs is None:
    st.info("Run **Pricing** first (select products → Price).")
    st.stop()

if pv_paths is None or not isinstance(pv_paths, dict) or len(pv_paths) == 0:
    st.warning("No pv_paths found. In Pricing, make sure you store per-scenario pv_paths.")
    st.stop()

# cf_paths is optional (only needed for multihorizon)
if cf_paths is not None and not isinstance(cf_paths, dict):
    st.warning("cf_paths exists but is not a dict; ignoring it.")
    cf_paths = None


# -----------------------------
# Helpers
# -----------------------------
def _as_1d(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.size == 0:
        raise ValueError("Empty array.")
    if not np.all(np.isfinite(a)):
        raise ValueError("Array contains non-finite values.")
    return a


def _as_2d_cols(mat: np.ndarray, n: int) -> np.ndarray:
    H = np.asarray(mat, dtype=float)
    if H.ndim != 2:
        raise ValueError("Expected 2D array.")
    # normalize to (N, M)
    if H.shape[0] == n:
        return H
    if H.shape[1] == n:
        return H.T
    raise ValueError(f"Inconsistent shapes: expected N={n}, got {H.shape}.")


def _as_cf_mat(cf: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(cf, dtype=float)
    if a.ndim != 2:
        raise ValueError("cf_paths must be 2D (N,T).")
    if a.shape[0] != n:
        raise ValueError(f"cf_paths first dim must be N={n}, got {a.shape[0]}.")
    if not np.all(np.isfinite(a)):
        raise ValueError("cf_paths contains non-finite values.")
    return a


def _stack_pv(selected_names: list[str]) -> np.ndarray:
    cols = [_as_1d(pv_paths[name]) for name in selected_names]
    return np.column_stack(cols)  # (N, M)


def _stack_cf(selected_names: list[str], n: int, t: int) -> np.ndarray:
    # returns (N, M, T)
    mats = []
    for name in selected_names:
        if cf_paths is None or name not in cf_paths:
            raise ValueError(f"Missing cf_paths for instrument '{name}'.")
        cf = _as_cf_mat(cf_paths[name], n)
        if cf.shape[1] != t:
            # strict: require common T (your Pricing page should pad to common H)
            raise ValueError(
                f"CF horizon mismatch for '{name}': got T={cf.shape[1]}, expected T={t}."
            )
        mats.append(cf[:, None, :])  # (N,1,T)
    return np.concatenate(mats, axis=1)  # (N,M,T)


def _try_get_discount_factors(n: int, t: int) -> np.ndarray | None:
    # 1) Prefer discount factors explicitly used at pricing time
    df = st.session_state.get("pricing_discount_factors")
    if df is not None:
        df = np.asarray(df, dtype=float)
    else:
        scen = st.session_state.get("scen_Q") or st.session_state.get("scen_P")
        if scen is None:
            return None
        df = getattr(scen, "discount_factors", None)
        if df is None:
            return None
        df = np.asarray(df, dtype=float)

    # 2) Shape handling
    if df.ndim == 1:
        if df.shape[0] < t:
            return None
        df = df[:t]
        if np.any(df <= 0):
            return None
        return np.repeat(df[None, :], n, axis=0)

    if df.ndim == 2:
        if df.shape[1] < t:
            return None
        df = df[:, :t]
        if df.shape[0] == 1:
            df = np.repeat(df, n, axis=0)
        if df.shape[0] != n:
            return None
        if np.any(df <= 0):
            return None
        return df

    return None


# -----------------------------
# Sidebar controls
# -----------------------------
all_names = list(specs.keys())

with st.sidebar:
    st.header("Hedging settings")

    st.write("Pricing measure:", st.session_state.get("pricing_measure", "?"))
    st.write("Short rate:", st.session_state.get("pricing_short_rate", "?"))

    st.divider()
    st.subheader("1) Select liability")
    default_liab = "life_annuity" if "life_annuity" in all_names else all_names[0]
    liability_name = st.selectbox(
        "Liability instrument", options=all_names, index=all_names.index(default_liab)
    )

    st.subheader("2) Select hedging instruments")
    hedge_candidates = [x for x in all_names if x != liability_name]
    default_hedges = [
        x
        for x in hedge_candidates
        if x in ("longevity_bond", "survivor_swap", "q_forward", "s_forward")
    ]
    selected_hedges = st.multiselect(
        "Hedge set",
        options=hedge_candidates,
        default=default_hedges[:4] if default_hedges else hedge_candidates[:3],
    )

    st.divider()
    st.subheader("3) Method")
    method = st.selectbox(
        "Hedge method",
        options=[
            "min_variance (PV)",
            "min_variance_constrained (PV)",
            "multihorizon (CF)",
        ],
        index=0,
    )

    constraints = {}

    if method == "min_variance_constrained (PV)":
        lb = st.number_input("Lower bound (lb)", value=-10.0, step=0.5)
        ub = st.number_input("Upper bound (ub)", value=10.0, step=0.5)
        constraints = {"lb": float(lb), "ub": float(ub)}

    if method == "multihorizon (CF)":
        mode = st.selectbox("Multihorizon mode", options=["pv_by_horizon", "pv_cashflows"], index=0)
        constraints["mode"] = mode

        use_time_weights = st.checkbox("Use time weights", value=False)
        if use_time_weights:
            st.caption("Example: heavier weight on short maturities.")
            t_power = st.number_input("weight(t) = 1 / t^p  (p)", value=1.0, step=0.1)
            constraints["time_weights_power"] = float(t_power)

    st.divider()
    run = st.button("🚀 Compute hedge", type="primary")


# -----------------------------
# Main
# -----------------------------
st.subheader("Inputs")

if not selected_hedges:
    st.warning("Select at least one hedging instrument.")
    st.stop()

# Build arrays
L = _as_1d(pv_paths[liability_name])
N = int(L.shape[0])

H_pv = _stack_pv(selected_hedges)  # (N,M)
M = int(H_pv.shape[1])

c1, c2, c3 = st.columns(3)
c1.metric("N scenarios", f"{N}")
c2.metric("Hedges (M)", f"{M}")
c3.metric("Liability", liability_name)

st.write("Hedges:", ", ".join(selected_hedges))


# -----------------------------
# Run hedge
# -----------------------------
if run:
    try:
        if method.startswith("min_variance"):
            hedge_method = (
                "min_variance" if "constrained" not in method else "min_variance_constrained"
            )
            res = hedging_pipeline(
                liability_pv_paths=L,
                hedge_pv_paths=H_pv,
                method=hedge_method,
                constraints=constraints or None,
            )

        else:
            # multihorizon requires CF paths
            if cf_paths is None:
                raise ValueError(
                    "cf_paths not available. In Pricing, store cf_paths for each instrument."
                )

            L_cf = _as_cf_mat(cf_paths[liability_name], N)
            T = int(L_cf.shape[1])

            H_cf = _stack_cf(selected_hedges, n=N, t=T)  # (N,M,T)

            # Build discount_factors if needed
            df = _try_get_discount_factors(n=N, t=T)
            if constraints.get("mode") == "pv_by_horizon" and df is None:
                raise ValueError(
                    "multihorizon mode='pv_by_horizon' requires discount_factors on the scenario set. "
                    "Either build/join rate scenarios or use mode='pv_cashflows'."
                )
            if df is not None:
                constraints["discount_factors"] = df

            # Optional time weights
            if "time_weights_power" in constraints:
                p = float(constraints["time_weights_power"])
                tw = 1.0 / (np.arange(1, T + 1, dtype=float) ** p)
                constraints["time_weights"] = tw

            res = hedging_pipeline(
                liability_pv_paths=L,  # still used for summary consistency
                hedge_pv_paths=H_pv,  # still used for summary consistency
                liability_cf_paths=L_cf,
                hedge_cf_paths=H_cf,
                method="multihorizon",
                constraints=constraints or None,
                discount_factors=df,
            )

        # Store in session for downstream reporting
        st.session_state["hedge_result"] = res

    except Exception as e:
        st.error(f"Hedging failed: {e}")
        st.stop()


# -----------------------------
# Display results
# -----------------------------
res = st.session_state.get("hedge_result")
if res is None:
    st.info("Set inputs and click **Compute hedge**.")
    st.stop()

st.success("Hedge computed ✅")

# Weights table
weights = np.asarray(res.weights, dtype=float).reshape(-1)
df_w = pd.DataFrame({"instrument": list(selected_hedges), "weight": weights}).sort_values(
    "instrument"
)

st.subheader("Hedge weights")
st.dataframe(df_w, use_container_width=True)

# Summary
st.subheader("Summary (before vs after)")
summary = res.summary if hasattr(res, "summary") else {}
if summary:
    s = summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean liability PV", f"{s.get('mean_liability', float('nan')):,.4f}")
    c2.metric("Std liability PV", f"{s.get('std_liability', float('nan')):,.4f}")
    c3.metric("Std net PV", f"{s.get('std_net', float('nan')):,.4f}")
    c4.metric("Var reduction", f"{100.0 * s.get('var_reduction', float('nan')):,.2f}%")
    st.caption(
        f"Corr(L, Net): {s.get('corr_L_net', float('nan')):,.4f} | rank(H): {s.get('rank_H', 'n/a')}"
    )

# Distributions
L_plot = _as_1d(res.liability_pv_paths)
net_plot = _as_1d(res.net_pv_paths)

st.subheader("Distributions")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Liability PV paths**")
    hist_L = np.histogram(L_plot, bins=40)
    st.bar_chart(pd.Series(hist_L[0]))

with col2:
    st.markdown("**Net PV paths (after hedge)**")
    hist_N = np.histogram(net_plot, bins=40)
    st.bar_chart(pd.Series(hist_N[0]))

# Quick quantiles table
st.subheader("Risk stats (PV)")


def qstats(x: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=0)),
        "p01": float(np.quantile(x, 0.01)),
        "p05": float(np.quantile(x, 0.05)),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
    }


stats_df = pd.DataFrame(
    [
        {"series": "Liability", **qstats(L_plot)},
        {"series": "Net (hedged)", **qstats(net_plot)},
    ]
)
st.dataframe(stats_df, use_container_width=True)

with st.expander("Session debug (hedging)"):
    st.write("Available session keys:", list(st.session_state.keys()))
    st.write("pv_paths instruments:", list(pv_paths.keys()))
    st.write("cf_paths instruments:", None if cf_paths is None else list(cf_paths.keys()))
    st.write("Selected liability:", liability_name)
    st.write("Selected hedges:", selected_hedges)
    st.write("Method:", method)
    st.write("Constraints:", constraints)
