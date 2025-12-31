from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from pymort.analysis.scenario_analysis import ShockSpec
from pymort.analysis.sensitivities import price_all_products
from pymort.pipeline import stress_testing_pipeline

st.set_page_config(page_title="Scenario Analysis", page_icon="🧪", layout="wide")
st.title("🧪 Scenario Analysis (Stress Tests)")
st.caption(
    "Apply mortality stress scenarios to the base scenario set and re-price all instruments."
)


# -----------------------------
# Guards: require base scenarios + specs
# -----------------------------
specs = st.session_state.get("pricing_specs")
scen_Q = st.session_state.get("scen_Q")
scen_P = st.session_state.get("scen_P")

if not specs or not isinstance(specs, dict):
    st.info("Run **Pricing** first (select products → Price) so `pricing_specs` exists.")
    st.stop()

if scen_Q is None and scen_P is None:
    st.warning("No scenario set found in session (`scen_Q` / `scen_P`). Run **Pricing** first.")
    st.stop()


# -----------------------------
# Sidebar: base set + shocks
# -----------------------------
with st.sidebar:
    st.header("Scenario Analysis settings")

    base_measure = st.radio(
        "Base measure",
        options=["Q (risk-neutral)", "P (physical)"],
        index=0 if scen_Q is not None else 1,
    )
    base_scen = scen_Q if base_measure.startswith("Q") else scen_P
    if base_scen is None:
        st.error("Selected base measure is not available in session. Build it in Pricing first.")
        st.stop()

    # Rate used for repricing
    default_r = float(st.session_state.get("pricing_short_rate", 0.02))
    short_rate = st.number_input(
        "Short rate used for repricing", value=default_r, step=0.0025, format="%.6f"
    )

    st.divider()
    st.subheader("Select instruments")
    all_kinds = list(specs.keys())
    default_sel = [
        k
        for k in all_kinds
        if k
        in (
            "life_annuity",
            "longevity_bond",
            "q_forward",
            "s_forward",
            "survivor_swap",
        )
    ]
    selected_kinds = st.multiselect(
        "Reprice these",
        options=all_kinds,
        default=default_sel if default_sel else all_kinds,
    )

    st.divider()
    st.subheader("Stress scenarios")

    shock_list: list[ShockSpec] = []

    # 1) Long life (uniform -q)
    use_long = st.checkbox("Long life (uniform improvement)", value=False)
    if use_long:
        long_bump = st.slider("Long life bump (q *= 1 - bump)", 0.0, 0.50, 0.10, 0.01)
        if use_long:
            shock_list.append(
                ShockSpec(
                    name=f"long_life_{int(long_bump * 100)}pct",
                    shock_type="long_life",
                    params={"magnitude": float(long_bump)},
                )
            )

    # 2) Short life (uniform +q)
    use_short = st.checkbox("Short life (uniform deterioration)", value=False)
    if use_short:
        short_bump = st.slider("Short life bump (q *= 1 + bump)", 0.0, 0.50, 0.10, 0.01)
        if use_short:
            shock_list.append(
                ShockSpec(
                    name=f"short_life_{int(short_bump * 100)}pct",
                    shock_type="short_life",
                    params={"magnitude": float(short_bump)},
                )
            )

    # 3) Pandemic
    use_pand = st.checkbox("Pandemic shock (spike over window)", value=False)
    if use_pand:
        pandemic_year = st.number_input(
            "Pandemic start year",
            value=int(np.asarray(base_scen.years, dtype=int)[0]),
            step=1,
        )
        pandemic_duration = st.number_input(
            "Pandemic duration (years)", value=1, step=1, min_value=1
        )
        pandemic_severity = st.slider("Pandemic severity (q *= 1 + severity)", 0.0, 5.0, 1.0, 0.1)
    if use_pand:
        shock_list.append(
            ShockSpec(
                name=f"pandemic_{pandemic_year}_{int(pandemic_severity * 100)}bp_{pandemic_duration}y",
                shock_type="pandemic",
                params={
                    "magnitude": float(pandemic_severity),
                    "pandemic_year": int(pandemic_year),
                    "pandemic_duration": int(pandemic_duration),
                },
            )
        )

    # 4) Plateau
    use_plateau = st.checkbox("Plateau (freeze improvements from year)", value=False)
    if use_plateau:
        plateau_start_year = st.number_input(
            "Plateau start year",
            value=int(np.asarray(base_scen.years, dtype=int)[0]),
            step=1,
        )
    if use_plateau:
        shock_list.append(
            ShockSpec(
                name=f"plateau_from_{plateau_start_year}",
                shock_type="plateau",
                params={"plateau_start_year": int(plateau_start_year)},
            )
        )

    # 5) Acceleration of improvement
    use_accel = st.checkbox("Accel improvement (extra improvement over time)", value=False)
    if use_accel:
        accel_rate = st.slider("Accel rate (annual extra improvement)", 0.0, 0.05, 0.01, 0.001)
        accel_start_year = st.number_input(
            "Accel start year (optional)",
            value=int(np.asarray(base_scen.years, dtype=int)[0]),
            step=1,
        )
    if use_accel and accel_rate > 0:
        shock_list.append(
            ShockSpec(
                name=f"accel_{accel_rate:.3f}_from_{accel_start_year}",
                shock_type="accel_improvement",
                params={
                    "magnitude": float(accel_rate),
                    "accel_start_year": int(accel_start_year),
                },
            )
        )

    # 6) Life expectancy shift (+Δ years for a given age)
    use_le = st.checkbox("Life expectancy shift (+Δ years at one age)", value=False)
    if use_le:
        ages_grid = np.asarray(base_scen.ages, dtype=float)
        default_age = float(ages_grid[len(ages_grid) // 2]) if ages_grid.size else 65.0
        le_age = st.number_input("Target age", value=float(default_age), step=1.0)
        le_delta = st.slider("Δ life expectancy (years)", 0.5, 10.0, 2.0, 0.5)
        le_year_start = st.number_input(
            "Apply from year (optional)",
            value=int(np.asarray(base_scen.years, dtype=int)[0]),
            step=1,
        )
    if use_le:
        shock_list.append(
            ShockSpec(
                name=f"LE_plus_{le_delta:.1f}y_at_{int(le_age)}",
                shock_type="life_expectancy",
                params={
                    "age": float(le_age),
                    "delta_years": float(le_delta),
                    "year_start": int(le_year_start),
                },
            )
        )

    # 7) Cohort trend shock
    use_cohort = st.checkbox("Cohort trend shock (birth-year band)", value=False)
    if use_cohort:
        years_grid = np.asarray(base_scen.years, dtype=int)
        cohort_start = st.number_input("Cohort start (birth year)", value=1960, step=1)
        cohort_end = st.number_input("Cohort end (birth year)", value=1970, step=1)
        cohort_mag = st.slider("Cohort magnitude", 0.0, 0.30, 0.05, 0.01)
        cohort_dir = st.selectbox("Direction", options=["favorable", "adverse"], index=0)
        cohort_ramp = st.checkbox("Ramp (tilt across band)", value=True)
    if use_cohort:
        shock_list.append(
            ShockSpec(
                name=f"cohort_{cohort_dir}_{cohort_start}_{cohort_end}",
                shock_type="cohort",
                params={
                    "cohort_start": int(cohort_start),
                    "cohort_end": int(cohort_end),
                    "magnitude": float(cohort_mag),
                    "direction": str(cohort_dir),
                    "ramp": bool(cohort_ramp),
                },
            )
        )

    run = st.button("🚀 Run scenario analysis", type="primary")


# -----------------------------
# Main: run & display
# -----------------------------
st.subheader("Base info")
c1, c2, c3 = st.columns(3)
c1.metric("Base measure", "Q" if base_measure.startswith("Q") else "P")
c2.metric("Short rate", f"{float(short_rate):.4%}")
c3.metric("Instruments", f"{len(selected_kinds)}")

if not selected_kinds:
    st.warning("Select at least one instrument.")
    st.stop()

if run:
    try:
        # restrict specs to selected instruments
        sel_specs = {k: specs[k] for k in selected_kinds}

        scen_dict = stress_testing_pipeline(base_scen, shock_specs=shock_list)

        # Price each scenario
        rows = []
        base_prices = price_all_products(
            scen_dict["base"], specs=sel_specs, short_rate=float(short_rate)
        )

        for scen_name, scen in scen_dict.items():
            prices = price_all_products(scen, specs=sel_specs, short_rate=float(short_rate))
            for inst, p in prices.items():
                EPS = 1e-8

                p0 = float(base_prices.get(inst, np.nan))
                dp = float(p - p0) if np.isfinite(p0) else np.nan

                if np.isfinite(p0) and abs(p0) > EPS:
                    pct = dp / p0
                else:
                    pct = np.nan
                rows.append(
                    {
                        "scenario": scen_name,
                        "instrument": inst,
                        "price": float(p),
                        "base_price": p0,
                        "delta": dp,
                        "pct_delta": pct,
                    }
                )

        df = pd.DataFrame(rows).sort_values(["instrument", "scenario"])

        st.session_state["scenario_analysis_result"] = {
            "base_measure": "Q" if base_measure.startswith("Q") else "P",
            "short_rate": float(short_rate),
            "selected_instruments": list(selected_kinds),
            "shocks": [s.__dict__ for s in shock_list],
            "prices_table": df,
        }

    except Exception as e:
        st.error(f"Scenario analysis failed: {e}")
        st.stop()


res = st.session_state.get("scenario_analysis_result")
if res is None:
    st.info("Set your shocks and click **Run scenario analysis**.")
    st.stop()

df = res["prices_table"].copy()

st.success("Scenario analysis computed ✅")

st.subheader("Results table")
st.caption(
    "% change shown only when |base_price| > 1e-8; "
    "near-zero prices are dominated by numerical noise."
)
# nicer formatting
df_show = df.copy()
df_show["pct_delta"] = 100.0 * df_show["pct_delta"]
st.dataframe(
    df_show,
    use_container_width=True,
    column_config={
        "price": st.column_config.NumberColumn(format="%.6f"),
        "base_price": st.column_config.NumberColumn(format="%.6f"),
        "delta": st.column_config.NumberColumn(format="%.6f"),
        "pct_delta": st.column_config.NumberColumn("Δ vs base (%)", format="%.3f"),
    },
)

st.divider()
st.subheader("Quick view: % change vs base")
inst_list = sorted(df["instrument"].unique().tolist())
inst_pick = st.selectbox("Instrument", options=inst_list, index=0)

df_inst = df[df["instrument"] == inst_pick].copy()
df_inst = df_inst.sort_values("scenario")
ser = pd.Series(100.0 * df_inst["pct_delta"].values, index=df_inst["scenario"].values)
st.bar_chart(ser)

st.caption(
    "Tip: use this page to identify which instruments are most exposed to longevity shocks, then hedge with your Hedging page."
)

with st.expander("Session debug (scenario analysis)"):
    st.write("Session keys:", list(st.session_state.keys()))
    st.write("Base scenario chosen:", res.get("base_measure"))
    st.write("Short rate:", res.get("short_rate"))
    st.write("Selected instruments:", res.get("selected_instruments"))
    st.write("Shocks:", res.get("shocks"))
