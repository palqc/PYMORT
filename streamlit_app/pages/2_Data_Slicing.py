from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from assets.logo import add_logo_top_right

add_logo_top_right()
st.set_page_config(page_title="Data & Slicing", page_icon="✂️", layout="wide")
st.title("✂️ Data Slicing (fit / projection subset)")
st.caption(
    "Define the subset of ages/years used for model fitting & projections. "
    "This does NOT modify the raw uploaded data."
)

# -----------------------------
# Guard: need data loaded
# -----------------------------
if "m" not in st.session_state or st.session_state.get("m") is None:
    st.info("Load data first (page: Data Upload).")
    st.stop()

ages = np.asarray(st.session_state["ages"], dtype=float)
years = np.asarray(st.session_state["years"], dtype=int)
m = np.asarray(st.session_state["m"], dtype=float)

slice_cfg = st.session_state.get("slice_cfg") or {}

default_age_min = int(slice_cfg.get("age_min", int(np.min(ages))))
default_age_max = int(slice_cfg.get("age_max", min(int(np.max(ages)), 95)))
default_year_min = int(slice_cfg.get("year_min", int(np.min(years))))
default_year_max = int(slice_cfg.get("year_max", int(np.max(years))))

c1, c2 = st.columns(2, gap="large")
with c1:
    fit_age_min, fit_age_max = st.slider(
        "Fit age range",
        min_value=int(np.min(ages)),
        max_value=int(np.max(ages)),
        value=(default_age_min, default_age_max),
        step=1,
    )
with c2:
    fit_year_min, fit_year_max = st.slider(
        "Fit year range",
        min_value=int(np.min(years)),
        max_value=int(np.max(years)),
        value=(default_year_min, default_year_max),
        step=1,
    )

apply_slice = st.button("Apply slicing", type="primary", use_container_width=False)


def _apply_slice() -> None:
    a_mask = (ages >= fit_age_min) & (ages <= fit_age_max)
    y_mask = (years >= fit_year_min) & (years <= fit_year_max)

    if not np.any(a_mask) or not np.any(y_mask):
        raise ValueError("Empty slice (no ages or no years). Adjust ranges.")

    ages_s = ages[a_mask]
    years_s = years[y_mask]
    m_s = m[np.ix_(a_mask, y_mask)]

    st.session_state["slice_cfg"] = {
        "age_min": int(fit_age_min),
        "age_max": int(fit_age_max),
        "year_min": int(fit_year_min),
        "year_max": int(fit_year_max),
    }
    st.session_state["ages_slice"] = ages_s
    st.session_state["years_slice"] = years_s
    st.session_state["m_slice"] = m_s

    # Reset downstream objects that depend on slice
    for k in [
        "fitted_model",
        "proj_P",
        "scen_P",
        "scen_Q",
        "calibration_summary",
        "calibration_cache",
        "prices",
        "pv_paths",
        "cf_paths",
        "hedge_result",
        "risk_report",
    ]:
        st.session_state[k] = None


if apply_slice:
    try:
        _apply_slice()
        st.success("Slice applied ✅ (saved as ages_slice / years_slice / m_slice)")
    except Exception as e:
        st.error(str(e))

# -----------------------------
# Display current slice status
# -----------------------------
ages_s = st.session_state.get("ages_slice")
years_s = st.session_state.get("years_slice")
m_s = st.session_state.get("m_slice")

st.divider()
st.subheader("Current slice")

if ages_s is None or years_s is None or m_s is None:
    st.info("No slice applied yet. Click **Apply slicing**.")
else:
    ages_s = np.asarray(ages_s, dtype=float)
    years_s = np.asarray(years_s, dtype=int)
    m_s = np.asarray(m_s, dtype=float)

    s1, s2, s3 = st.columns(3)
    s1.metric("Slice ages", f"{int(ages_s.min())} → {int(ages_s.max())}", f"{len(ages_s)} pts")
    s2.metric(
        "Slice years",
        f"{int(years_s.min())} → {int(years_s.max())}",
        f"{len(years_s)} pts",
    )
    s3.metric("Slice m shape", f"{m_s.shape[0]} × {m_s.shape[1]}", "age × year")

    st.subheader("Slice preview")
    preview_s = pd.DataFrame(
        m_s[: min(10, m_s.shape[0]), : min(10, m_s.shape[1])],
        index=ages_s[: min(10, len(ages_s))],
        columns=years_s[: min(10, len(years_s))],
    )
    st.dataframe(preview_s, use_container_width=True)

    with st.expander("Slice diagnostics"):
        st.write("NaNs:", int(np.isnan(m_s).sum()))
        st.write("Min m:", float(np.min(m_s)))
        st.write("Max m:", float(np.max(m_s)))
        st.json(st.session_state.get("slice_cfg", {}))

st.success("Next step ✅ Go to **Fit & Model Selection**.")
