from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from assets.logo import LOGO_PATH, add_logo_top_right

from pymort.pipeline import sensitivities_pipeline

add_logo_top_right()
st.set_page_config(page_title="Sensitivities", page_icon=LOGO_PATH, layout="wide")
st.title("Sensitivities")
st.caption(
    "Compute base prices + optional rate DV01/duration, convexity, mortality delta-by-age, and sigma-scale vega."
)

# -----------------------------
# Guards: require scenarios + specs
# -----------------------------
specs = st.session_state.get("pricing_specs")
scen_Q = st.session_state.get("scen_Q")
scen_P = st.session_state.get("scen_P")

if not specs or not isinstance(specs, dict):
    st.info("Run **Pricing** first so `pricing_specs` exists.")
    st.stop()

if scen_Q is None and scen_P is None:
    st.warning("No scenario set found (`scen_Q` / `scen_P`). Run **Pricing** first.")
    st.stop()


# -----------------------------
# Guards: require scenarios + specs
# -----------------------------


def clean_small(x, eps=1e-8):
    return 0 if abs(x) < eps else x


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Sensitivities settings")

    base_measure = st.radio(
        "Base measure",
        options=["Q (risk-neutral)", "P (physical)"],
        index=0 if scen_Q is not None else 1,
    )
    scen = scen_Q if base_measure.startswith("Q") else scen_P
    if scen is None:
        st.error("Selected base measure is not available. Build it in Pricing first.")
        st.stop()

    default_r = float(st.session_state.get("pricing_short_rate", 0.02))
    short_rate = st.number_input(
        "Short rate used for pricing/sensitivities",
        value=default_r,
        step=0.0025,
        format="%.3f",
    )

    st.divider()
    st.subheader("Select instruments")
    all_kinds = list(specs.keys())
    default_sel = [
        k
        for k in all_kinds
        if k
        in ("life_annuity", "longevity_bond", "q_forward", "s_forward", "survivor_swap")
    ]
    selected_kinds = st.multiselect(
        "Compute on",
        options=all_kinds,
        default=default_sel if default_sel else all_kinds,
    )

    st.divider()
    st.subheader("What to compute")
    sensitivities_list = []

    compute_rate = st.checkbox("Rate sensitivity (duration, DV01)", value=True)
    rate_bump = None
    if compute_rate:
        rate_bump = st.number_input(
            "Rate bump (h)", value=1e-4, step=1e-5, format="%.5f"
        )
        sensitivities_list.append("rate_sensitivity")

    compute_convexity = st.checkbox("Rate convexity", value=False)
    if compute_convexity:
        sensitivities_list.append("rate_convexity")

    compute_delta = st.checkbox("Mortality delta-by-age (q bumps)", value=False)
    q_rel_bump = None
    ages_for_delta = None
    if compute_delta:
        sensitivities_list.append("delta_by_age")
        q_rel_bump = st.number_input(
            "q relative bump (eps)", value=0.01, step=0.005, format="%.3f"
        )

        ages_grid = np.asarray(getattr(scen, "ages", []), dtype=float)
        if ages_grid.size > 0:
            a_min, a_max = float(np.min(ages_grid)), float(np.max(ages_grid))
        else:
            a_min, a_max = 40.0, 100.0

        st.caption("Tip: delta-by-age is expensive if you compute many ages.")
        mode = st.selectbox(
            "Age selection",
            options=["Coarse grid", "Range + step", "All ages"],
            index=0,
        )

        if mode == "All ages":
            ages_for_delta = None  # None => use all in engine
        elif mode == "Coarse grid":
            # pick ~15 ages evenly spaced
            n_pts = st.slider("How many ages", 5, 40, 15, 1)
            ages_for_delta = np.linspace(a_min, a_max, int(n_pts)).tolist()
        else:
            lo = st.number_input("Age min", value=float(a_min), step=1.0)
            hi = st.number_input("Age max", value=float(a_max), step=1.0)
            step = st.number_input("Age step", value=5.0, step=1.0)
            if step <= 0:
                step = 5.0
            ages_for_delta = np.arange(
                float(lo), float(hi) + 1e-9, float(step)
            ).tolist()

    compute_vega = st.checkbox(
        "Mortality vega via sigma scaling (requires calibration)", value=False
    )
    sigma_rel_bump = None
    vega_available = False
    cache = None
    lam = None
    if compute_vega:
        sensitivities_list.append("vega_sigma_scale")
        sigma_rel_bump = st.number_input(
            "Sigma scale bump (eps)", value=0.05, step=0.01, format="%.2f"
        )

        # Try to auto-grab calibration info from scenario metadata
        md = getattr(scen, "metadata", {}) or {}
        cache = md.get("calibration_cache")
        lam = md.get("lambda_star")

        # also try common session keys (if you store them elsewhere)
        if cache is None:
            cache = st.session_state.get("calibration_cache")
        if lam is None:
            lam = st.session_state.get("lambda_star")

        vega_available = (cache is not None) and (lam is not None)
        if not vega_available:
            st.warning(
                "Vega requires `calibration_cache` and `lambda_star` (lambda_esscher). "
                "Rebuild Q scenarios via calibration in Pricing so scen_Q.metadata contains them."
            )

    st.divider()
    run = st.button(
        "Compute sensitivities",
        type="primary",
        disabled=(len(sensitivities_list) == 0),
        help=(
            "Select at least one sensitivity to run the analysis."
            if len(sensitivities_list) == 0
            else None
        ),
    )

# -----------------------------
# Main
# -----------------------------
st.subheader("Base info")
c1, c2, c3 = st.columns(3)
c1.metric("Measure", "Q" if base_measure.startswith("Q") else "P")
c2.metric("Short rate", f"{float(short_rate):.2%}")
c3.metric("# instruments", f"{len(selected_kinds)}")

if not selected_kinds:
    st.warning("Select at least one instrument.")
    st.stop()

# -----------------------------
# Run pipeline
# -----------------------------
if run:
    try:
        sel_specs = {k: specs[k] for k in selected_kinds}

        bumps = {}
        if compute_rate or compute_convexity:
            bumps["rate_bump"] = float(rate_bump) if rate_bump is not None else 1e-4
        if compute_delta:
            bumps["q_rel_bump"] = float(q_rel_bump) if q_rel_bump is not None else 0.01
            bumps["ages_for_delta"] = ages_for_delta
        if compute_vega:
            bumps["sigma_rel_bump"] = (
                float(sigma_rel_bump) if sigma_rel_bump is not None else 0.05
            )
            if vega_available:
                bumps["calibration_cache"] = cache
                bumps["lambda_esscher"] = lam
            else:
                # force off to avoid hard error
                compute_vega = False

        res = sensitivities_pipeline(
            scen,
            specs=sel_specs,
            short_rate=float(short_rate),
            compute_rate=bool(compute_rate),
            compute_convexity=bool(compute_convexity),
            compute_delta_by_age=bool(compute_delta),
            compute_vega=bool(compute_vega),
            bumps=bumps if bumps else None,
        )

        st.session_state["sensitivities_result"] = {
            "base_measure": "Q" if base_measure.startswith("Q") else "P",
            "short_rate": float(short_rate),
            "selected_instruments": list(selected_kinds),
            "compute_rate": bool(compute_rate),
            "compute_convexity": bool(compute_convexity),
            "compute_delta": bool(compute_delta),
            "compute_vega": bool(compute_vega),
            "bumps": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in (bumps or {}).items()
            },
            "result": res,
        }

    except Exception as e:
        st.error(f"Sensitivities failed: {e}")
        st.stop()

saved = st.session_state.get("sensitivities_result")
if saved is None:
    st.info("Set options and click **Compute sensitivities**.")
    st.stop()

res = saved["result"]

# -----------------------------
# Display: base prices
# -----------------------------
st.divider()
st.subheader("Base prices")
st.markdown("")
col_left, col_mid, col_right = st.columns([1, 3, 1])

with col_mid:
    prices_base = res.get("prices_base", {}) or {}
    df_prices = pd.DataFrame(
        [
            {"instrument": k, "price": clean_small(float(v))}
            for k, v in prices_base.items()
        ]
    ).sort_values("instrument")
    st.dataframe(df_prices, use_container_width=True)

# -----------------------------
# Display: rate sensitivity
# -----------------------------
if "rate_sensitivity" in res:
    st.divider()
    st.subheader("Rate sensitivity (DV01 / duration)")
    st.markdown("")
    col_left, col_mid, col_right = st.columns([0.5, 6, 0.5])

    with col_mid:
        rs = res["rate_sensitivity"] or {}
        rows = []
        for inst, obj in rs.items():
            rows.append(
                {
                    "instrument": inst,
                    "price": clean_small(float(getattr(obj, "price_base", np.nan))),
                    "duration": clean_small(float(getattr(obj, "duration", np.nan))),
                    "dv01": clean_small(float(getattr(obj, "dv01", np.nan))),
                    "bump": clean_small(float(getattr(obj, "bump", np.nan))),
                }
            )
        df_rs = pd.DataFrame(rows).sort_values("instrument")
        st.dataframe(df_rs, use_container_width=True)

    st.divider()
    st.subheader("Rate sensitivity charts")
    st.markdown("")
    col1, col2, col3, col4, col5 = st.columns([0.3, 5, 0.7, 5, 0.6])

    DV01_EPS = 1e-6

    df_dv01_plot = df_rs[
        np.isfinite(df_rs["dv01"]) & (np.abs(df_rs["dv01"]) > DV01_EPS)
    ]

    if df_rs.shape[0] > 0:
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(
                df_dv01_plot["instrument"].values,
                df_dv01_plot["dv01"].values,
            )
            ax.axhline(0.0, linewidth=1.0)
            ax.set_title("DV01 by instrument", pad=15)
            ax.set_xlabel("Instrument")
            ax.set_ylabel("DV01")
            ax.grid(True, alpha=0.2)
            ax.tick_params(axis="x", rotation=20)
            st.pyplot(fig, use_container_width=True)

        with col4:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(df_rs["instrument"].values, df_rs["duration"].values)
            ax.axhline(0.0, linewidth=1.0)
            ax.set_title("Duration by instrument", pad=15)
            ax.set_xlabel("Instrument")
            ax.set_ylabel("Duration")
            ax.grid(True, alpha=0.2)
            ax.tick_params(axis="x", rotation=20)
            st.pyplot(fig, use_container_width=True)

# -----------------------------
# Display: rate convexity
# -----------------------------
if "rate_convexity" in res:
    st.divider()
    st.subheader("Rate convexity")
    st.markdown("")
    col_left, col_mid, col_right = st.columns([1, 5, 1])

    with col_mid:
        rc = res["rate_convexity"] or {}
        CONV_EPS = 1e-6
        DV01_EPS = 1e-6

        rows = []
        for inst, obj in rc.items():
            price = float(getattr(obj, "price_base", np.nan))
            dv01 = (
                float(getattr(obj, "dv01", np.nan)) if hasattr(obj, "dv01") else np.nan
            )
            convexity = float(getattr(obj, "convexity", np.nan))

            if abs(price) < CONV_EPS or abs(dv01) < DV01_EPS:
                convexity = np.nan  # <-- clé

            rows.append(
                {
                    "instrument": inst,
                    "price": clean_small(price),
                    "convexity": convexity,
                    "bump": getattr(obj, "bump", np.nan),
                }
            )
        df_rc = pd.DataFrame(rows).sort_values("instrument")
        st.dataframe(df_rc)

# -----------------------------
# Display: sigma-scale vega
# -----------------------------
if "vega_sigma_scale" in res:
    st.divider()
    st.subheader("Mortality vega (sigma-scale)")
    st.markdown("")
    col_left, col_mid, col_right = st.columns([1, 3, 1])

    with col_mid:
        vega = res["vega_sigma_scale"] or {}
        df_v = pd.DataFrame(
            [{"instrument": k, "vega": float(v)} for k, v in vega.items()]
        ).sort_values("instrument")
        st.dataframe(df_v)
    st.markdown("")

    if df_v.shape[0] > 0:
        col_left, col_mid, col_right = st.columns([0.7, 5, 1])

        with col_mid:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(df_v["instrument"].values, df_v["vega"].values)
            ax.axhline(0.0, linewidth=1.0)
            ax.set_title("Vega (sigma-scale) by instrument", pad=15)
            ax.set_xlabel("Instrument")
            ax.set_ylabel("Vega")
            ax.grid(True, alpha=0.2)
            ax.tick_params(axis="x", rotation=20)
            st.pyplot(fig, use_container_width=True)

# -----------------------------
# Display: delta-by-age
# -----------------------------
if "delta_by_age" in res:
    st.divider()
    st.subheader("Mortality delta-by-age")
    dba = res["delta_by_age"] or {}

    col1, col2 = st.columns([1, 3])
    with col1:
        inst_list = sorted(list(dba.keys()))
        pick = st.selectbox("Instrument", options=inst_list, index=0)

    obj = dba[pick]
    eps = float(getattr(obj, "rel_bump", 0.01))

    st.markdown("")
    st.caption(
        f"Convention: dP/dε ≈ (P_bumped - P_base) / ε. "
        f"Impact PV for the selected bump ({eps:.3%}) = (dP/dε) × ε."
    )

    col_left, col_mid, col_right = st.columns([1, 2, 1])
    with col_mid:
        ages = np.asarray(getattr(obj, "ages", []), dtype=float)
        dP_d_eps = np.asarray(getattr(obj, "deltas", []), dtype=float)

        df_d = pd.DataFrame(
            {
                "age": ages,
                "dP_d_eps": dP_d_eps,
                "ΔPV": dP_d_eps * eps,
            }
        )

        st.dataframe(
            df_d,
            use_container_width=True,
            column_config={
                "age": st.column_config.NumberColumn(format="%.1f"),
                "dP_d_eps": st.column_config.NumberColumn("dP / dε", format="%.4f"),
                "ΔPV": st.column_config.NumberColumn(
                    f"ΔPV (for ε={eps:.3%})", format="%.4f"
                ),
            },
        )
        st.markdown("")

    if df_d.shape[0] > 0:
        col_left, col_mid, col_right = st.columns([0.7, 5, 1])
        with col_mid:
            fig, ax = plt.subplots(figsize=(9, 4))
            y = df_d["ΔPV"].values
            ax.plot(df_d["age"].values, y, linewidth=2)
            ax.set_ylabel(f"ΔPV (for ε={eps:.3%})")
            ax.set_title(f"Mortality impact-by-age — {pick} (ε={eps:.3%})", pad=15)
            ax.set_xlabel("Age")
            ax.grid(True, alpha=0.2)
            st.pyplot(fig, use_container_width=True)

    st.divider()
    st.subheader("Top 10 ages by |ΔPV|")
    st.markdown("")
    col_left, col_mid, col_right = st.columns([1, 3, 1])
    with col_mid:
        df_top = df_d.copy()
        df_top["abs"] = np.abs(df_top["ΔPV"])
        df_top = df_top.sort_values("abs", ascending=False).head(10)
        st.dataframe(df_top.drop(columns=["abs"]), use_container_width=True)
        st.markdown("")

st.markdown("")
st.success("Next: go to **Report Export** page.")
