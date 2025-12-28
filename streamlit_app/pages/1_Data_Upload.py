from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from pymort.lifetables import load_m_from_excel_any

st.set_page_config(page_title="Data Upload", page_icon="📥", layout="wide")
st.title("📥 Data Upload (HMD Excel)")
st.caption("Drag & drop an HMD-style period death-rate table (Mx_1x1-like) in .xlsx.")

# ---- UI controls
with st.sidebar:
    st.header("Loader options")
    sex = st.selectbox("Sex column", ["Total", "Female", "Male"], index=0)
    year_min = st.number_input("Year min (optional)", value=1970, min_value=0, step=1)
    year_max = st.number_input("Year max (optional)", value=2023, min_value=0, step=1)

    st.divider()
    m_floor = st.number_input("m_floor", value=1e-12, format="%.2e")
    drop_years_str = st.text_input("Drop years (comma-separated)", value="")

uploaded = st.file_uploader(
    "Upload Excel (.xlsx)",
    type=["xlsx"],
    accept_multiple_files=False,
)

# ---- Parse optional inputs
year_min_opt = int(year_min) if year_min and int(year_min) > 0 else None
year_max_opt = int(year_max) if year_max and int(year_max) > 0 else None

drop_years = None
if drop_years_str.strip():
    try:
        drop_years = [int(x.strip()) for x in drop_years_str.split(",") if x.strip()]
    except Exception:
        st.error("Could not parse 'Drop years'. Use e.g. 1918, 2020")
        st.stop()

# ---- Load
if uploaded is None:
    st.info("Upload an .xlsx file to begin.")
    st.stop()

# Defensive check (Streamlit already filters by type, but keep it explicit)
if not uploaded.name.lower().endswith(".xlsx"):
    st.error("Please upload an .xlsx file (Excel).")
    st.stop()

try:
    # NEW: no temp file needed; we pass the UploadedFile directly
    res = load_m_from_excel_any(
        uploaded,
        sex=sex,  # type: ignore[arg-type]
        age_min=60,
        age_max=110,
        year_min=year_min_opt,
        year_max=year_max_opt,
        m_floor=float(m_floor),
        drop_years=drop_years,
    )
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

ages, years, m = res["m"]
ages = np.asarray(ages, dtype=float)
years = np.asarray(years, dtype=int)
m = np.asarray(m, dtype=float)

# ---- Store in session state (canonical keys used by app.py)
st.session_state["ages"] = ages
st.session_state["years"] = years
st.session_state["m"] = m
st.session_state["raw_df"] = None  # optional: could store long df later
st.session_state["data_meta"] = {
    "source_file": uploaded.name,
    "sex": sex,
    "age_min": int(60),
    "age_max": int(110),
    "year_min": year_min_opt,
    "year_max": year_max_opt,
    "drop_years": drop_years,
}

# Reset downstream objects (so user doesn't keep stale fitted/scenarios)
for k in [
    "slice_cfg",
    "ages_slice",
    "years_slice",
    "m_slice",
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
    if k == "slice_cfg":
        st.session_state[k] = {}
    else:
        st.session_state[k] = None

# ---- Display summary
col1, col2, col3 = st.columns(3)
col1.metric("Ages", f"{int(np.min(ages))} → {int(np.max(ages))}", f"{len(ages)} points")
col2.metric(
    "Years", f"{int(np.min(years))} → {int(np.max(years))}", f"{len(years)} points"
)
col3.metric("m shape", f"{m.shape[0]} × {m.shape[1]}", "age × year")

st.subheader("Preview")
preview = pd.DataFrame(
    m[: min(10, m.shape[0]), : min(10, m.shape[1])],
    index=ages[: min(10, len(ages))],
    columns=years[: min(10, len(years))],
)
st.dataframe(preview, use_container_width=True)

with st.expander("Diagnostics"):
    st.write("NaNs:", int(np.isnan(m).sum()))
    st.write("Min m:", float(np.min(m)))
    st.write("Max m:", float(np.max(m)))
    st.json(st.session_state["data_meta"])

st.success("Data loaded ✅ Go to the next page: Data & Slicing.")
