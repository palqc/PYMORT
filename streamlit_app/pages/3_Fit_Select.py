from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from assets.logo import LOGO_PATH, add_logo_top_right

from pymort.analysis.fitting import select_and_fit_best_model_for_pricing

add_logo_top_right()
st.set_page_config(
    page_title="Fit & Model Selection", page_icon=LOGO_PATH, layout="wide"
)
st.title("Fit & Model Selection")
st.caption(
    "Select best mortality model (or force one) and fit on the chosen training window."
)

# ---- Get data (prefer sliced if available)
ages_slice = st.session_state.get("ages_slice")
years_slice = st.session_state.get("years_slice")
m_slice = st.session_state.get("m_slice")

ages = ages_slice if ages_slice is not None else st.session_state.get("ages")
years = years_slice if years_slice is not None else st.session_state.get("years")
m = m_slice if m_slice is not None else st.session_state.get("m")

if ages is None or years is None or m is None:
    st.error(
        "No mortality surface found. Go to **Data Upload** (and optionally Data Slicing) first."
    )
    st.stop()

ages = np.asarray(ages, dtype=float)
years = np.asarray(years, dtype=int)
m = np.asarray(m, dtype=float)

# ---- Controls
with st.sidebar:
    st.header("Selection / fit options")

    mode = st.radio(
        "Mode", ["Auto (select best + fit)", "Manual (choose model)"], index=0
    )

    model_names_all = ["LCM1", "LCM2", "APCM3", "CBDM5", "CBDM6", "CBDM7"]
    if mode.startswith("Auto"):
        model_names = st.multiselect(
            "Candidate models", model_names_all, default=model_names_all
        )
    else:
        model_choice = st.selectbox("Model to fit", model_names_all, index=1)
        model_names = [model_choice]

    metric = st.selectbox("Selection metric", ["logit_q", "log_m"], index=0)

    # train_end: last in-sample year
    y_min, y_max = int(years.min()), int(years.max())
    default_train_end = y_max - 15
    train_end = st.slider(
        "train_end (last in-sample year)",
        min_value=y_min,
        max_value=y_max - 1,
        value=default_train_end,
    )

    st.divider()
    st.subheader("CPsplines (optional)")
    use_cps = st.toggle("Use CPsplines smoothing", value=True)

    cpsplines_kwargs = None
    if use_cps:
        # Keep minimal & safe defaults; you can expose more later
        k = st.number_input(
            "k (basis dimension)", min_value=5, max_value=20, value=12, step=5
        )
        deg = st.number_input(
            "deg (B-spline degree)", min_value=1, max_value=5, value=3, step=1
        )
        ord_ = st.number_input(
            "ord (penalty order)", min_value=1, max_value=4, value=2, step=1
        )
        cpsplines_kwargs = {"k": int(k), "deg": int(deg), "ord": int(ord_)}

    st.divider()
    run = st.button("Run selection / fit", type="primary")

# ---- Run
if not run:
    st.info("Choose options on the left, then click **Run selection / fit**.")
    st.stop()

if not model_names:
    st.error("Select at least one model.")
    st.stop()

with st.spinner("Fitting / selecting model..."):
    selected_df, fitted_best = select_and_fit_best_model_for_pricing(
        ages=ages,
        years=years,
        m=m,
        train_end=int(train_end),
        model_names=tuple(model_names),  # type: ignore[arg-type]
        metric=metric,  # "log_m" or "logit_q"
        cpsplines_kwargs=cpsplines_kwargs,
    )

# ---- Persist
st.session_state["fit_selection_table"] = selected_df
st.session_state["fitted_model"] = fitted_best
st.session_state["fit_cfg"] = {
    "mode": mode,
    "model_names": list(model_names),
    "metric": metric,
    "train_end": int(train_end),
    "cpsplines_kwargs": cpsplines_kwargs,
    "using_slice": bool(st.session_state.get("m_slice") is not None),
}

# Reset downstream objects
for k in [
    "proj_P",
    "scen_P",
    "scen_Q",
    "calibration_summary",
    "calibration_cache",
    "prices",
    "hedge_result",
    "risk_report",
]:
    st.session_state[k] = None

# ---- Display results
st.subheader("Selection table")
st.markdown("")
st.markdown("")
col, col1, col2 = st.columns([1, 6, 1])
with col1:
    if isinstance(selected_df, pd.DataFrame) and not selected_df.empty:
        st.dataframe(selected_df, use_container_width=True)
    else:
        st.write(selected_df)

st.markdown("")
st.markdown("")
st.write("**Selected model:**", getattr(fitted_best, "name", None))


st.markdown("")
st.markdown("")
st.success("Next: go to **Projection P** page.")
