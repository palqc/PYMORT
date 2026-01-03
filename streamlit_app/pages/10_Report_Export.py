from __future__ import annotations

import datetime as dt
import json
from html import escape

import numpy as np
import pandas as pd
import streamlit as st
from assets.logo import LOGO_PATH, add_logo_top_right

add_logo_top_right()
st.set_page_config(page_title="Report / Export", page_icon=LOGO_PATH, layout="wide")
st.title("Report / Export")
st.caption("Generate a lightweight HTML report of the current session.")


# -----------------------------
# Helpers
# -----------------------------
def _now_iso() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_json(obj) -> str:
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return json.dumps(str(obj), indent=2)


def _df_to_html(df: pd.DataFrame, *, max_rows=200) -> str:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "<p><em>No data.</em></p>"
    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)
        note = f"<p><em>Showing first {max_rows} rows.</em></p>"
    else:
        note = ""
    return note + df2.to_html(index=False, border=0, classes="tbl")


def _scen_summary(scen, label: str) -> str:
    if scen is None:
        return f"<p><em>{label}: not available.</em></p>"
    try:
        q = np.asarray(scen.q_paths)
        N, A, H = q.shape
        ages = np.asarray(scen.ages, dtype=float)
        years = np.asarray(scen.years, dtype=int)
        md = getattr(scen, "metadata", {}) or {}
        return f"""
        <ul>
          <li><b>{escape(label)}</b></li>
          <li>N scenarios: <b>{N}</b></li>
          <li>Ages: <b>{int(np.min(ages))} → {int(np.max(ages))}</b> ({A} pts)</li>
          <li>Horizon: <b>{H}</b> years</li>
          <li>Years: <b>{int(np.min(years))} → {int(np.max(years))}</b></li>
          <li>Metadata.measure: <b>{escape(str(md.get("measure", "n/a")))}</b></li>
        </ul>
        """
    except Exception as e:
        return f"<p><em>{label}: could not summarize ({escape(str(e))}).</em></p>"


def _specs_summary(specs: dict) -> pd.DataFrame:
    rows = []
    if not isinstance(specs, dict):
        return pd.DataFrame()
    for k, v in specs.items():
        rows.append(
            {"instrument": k, "spec_type": type(v).__name__, "spec_repr": repr(v)[:300]}
        )
    return pd.DataFrame(rows)


def _extract_hedge_tables(hedge_result):
    if hedge_result is None:
        return None, None
    # weights
    try:
        w = np.asarray(hedge_result.weights, dtype=float).reshape(-1)
    except Exception:
        w = None
    # summary dict
    try:
        s = hedge_result.summary if hasattr(hedge_result, "summary") else {}
        if s is None:
            s = {}
    except Exception:
        s = {}
    return w, s


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Export settings")
    report_title = st.text_input("Report title", value="PYMORT — Session Report")
    include_debug = st.checkbox("Include session debug keys", value=False)
    max_rows = st.slider("Max rows per table", 50, 500, 200, 50)
    st.divider()
    build = st.button("Build HTML report", type="primary")
    st.session_state["risk_report"] = build


# -----------------------------
# Collect session state objects
# -----------------------------
slice_cfg = st.session_state.get("slice_cfg")
fit_cfg = st.session_state.get("fit_cfg")
fit_selection_table = st.session_state.get("fit_selection_table")
fitted_model = st.session_state.get("fitted_model")

scen_P = st.session_state.get("scen_P")
projP_cfg = st.session_state.get("projP_cfg")
calibration_cache = st.session_state.get("calibration_cache")

scen_Q = st.session_state.get("scen_Q")
calibration_summary = st.session_state.get("calibration_summary")
risk_neutral_mode = st.session_state.get("risk_neutral_mode")

pricing_measure = st.session_state.get("pricing_measure")
pricing_short_rate = st.session_state.get("pricing_short_rate")
pricing_specs = st.session_state.get("pricing_specs")
prices = st.session_state.get("prices")

scenario_analysis_result = st.session_state.get("scenario_analysis_result")
sensitivities_result = st.session_state.get("sensitivities_result")
hedge_result = st.session_state.get("hedge_result")


# -----------------------------
# Build HTML
# -----------------------------
if not build:
    st.info("Click **Build HTML report** to generate a downloadable HTML summary.")
    st.stop()

# Tables
df_specs = _specs_summary(pricing_specs)
df_prices = (
    pd.DataFrame(
        [{"instrument": k, "price": float(v)} for k, v in (prices or {}).items()]
    )
    if prices
    else pd.DataFrame()
)

# Scenario analysis table
df_scen = None
scen_shocks = None
if isinstance(scenario_analysis_result, dict):
    df_scen = scenario_analysis_result.get("prices_table")
    scen_shocks = scenario_analysis_result.get("shocks")

# Sensitivities tables
df_sens_base = None
df_sens_rate = None
df_sens_conv = None
df_sens_vega = None
df_sens_dba = None

if isinstance(sensitivities_result, dict):
    r = sensitivities_result.get("result", {}) or {}
    base = r.get("prices_base", {}) or {}
    df_sens_base = pd.DataFrame(
        [{"instrument": k, "price": float(v)} for k, v in base.items()]
    )

    if "rate_sensitivity" in r and r["rate_sensitivity"]:
        rows = []
        for inst, obj in r["rate_sensitivity"].items():
            rows.append(
                {
                    "instrument": inst,
                    "price": float(getattr(obj, "price_base", np.nan)),
                    "dP_dr": float(getattr(obj, "dP_dr", np.nan)),
                    "duration": float(getattr(obj, "duration", np.nan)),
                    "dv01": float(getattr(obj, "dv01", np.nan)),
                    "bump": float(getattr(obj, "bump", np.nan)),
                }
            )
        df_sens_rate = pd.DataFrame(rows)

    if "rate_convexity" in r and r["rate_convexity"]:
        rows = []
        for inst, obj in r["rate_convexity"].items():
            rows.append(
                {
                    "instrument": inst,
                    "price": float(getattr(obj, "price_base", np.nan)),
                    "convexity": float(getattr(obj, "convexity", np.nan)),
                    "bump": float(getattr(obj, "bump", np.nan)),
                }
            )
        df_sens_conv = pd.DataFrame(rows)

    if "vega_sigma_scale" in r and r["vega_sigma_scale"]:
        df_sens_vega = pd.DataFrame(
            [
                {"instrument": k, "vega": float(v)}
                for k, v in r["vega_sigma_scale"].items()
            ]
        )

    if "delta_by_age" in r and r["delta_by_age"]:
        # export only first instrument to keep HTML small
        first_key = sorted(list(r["delta_by_age"].keys()))[0]
        obj = r["delta_by_age"][first_key]
        ages = np.asarray(getattr(obj, "ages", []), dtype=float)
        deltas = np.asarray(getattr(obj, "deltas", []), dtype=float)
        df_sens_dba = pd.DataFrame(
            {"instrument": first_key, "age": ages, "delta": deltas}
        )

# Hedging
w, hedge_summary = _extract_hedge_tables(hedge_result)
df_hedge_summary = pd.DataFrame([hedge_summary]) if hedge_summary else pd.DataFrame()

# If we can’t map weights to instrument names safely, keep numeric list only
df_hedge_w = None
if w is not None:
    # best-effort: if pricing_specs exists and hedge page stored selected hedges? (not stored currently)
    df_hedge_w = pd.DataFrame({"weight": w})

model_name = getattr(fitted_model, "name", None) if fitted_model is not None else None

html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{escape(report_title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 0; }}
    .meta {{ color: #666; margin-top: 6px; }}
    h2 {{ margin-top: 28px; border-bottom: 1px solid #eee; padding-bottom: 6px; }}
    .tbl {{ border-collapse: collapse; width: 100%; }}
    .tbl th, .tbl td {{ border-bottom: 1px solid #eee; padding: 8px; text-align: left; vertical-align: top; }}
    code, pre {{ background: #f6f6f6; padding: 2px 4px; border-radius: 4px; }}
    pre {{ padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>{escape(report_title)}</h1>
  <div class="meta">Generated: {_now_iso()}</div>

  <h2>1) Data slicing</h2>
  <pre>{escape(_safe_json(slice_cfg or {}))}</pre>

  <h2>2) Fit & model selection</h2>
  <p><b>Selected model:</b> {escape(str(model_name))}</p>
  <pre>{escape(_safe_json(fit_cfg or {}))}</pre>
  <h3>Selection table</h3>
  {_df_to_html(fit_selection_table if isinstance(fit_selection_table, pd.DataFrame) else pd.DataFrame(), max_rows=max_rows)}

  <h2>3) Scenario sets</h2>
  <h3>P-measure</h3>
  <pre>{escape(_safe_json(projP_cfg or {}))}</pre>
  {_scen_summary(scen_P, "scen_P")}
  <h3>Q-measure</h3>
  <p><b>risk_neutral_mode:</b> {escape(str(risk_neutral_mode))}</p>
  <pre>{escape(_safe_json(calibration_summary or {}))}</pre>
  {_scen_summary(scen_Q, "scen_Q")}

  <h2>4) Pricing</h2>
  <p><b>Measure:</b> {escape(str(pricing_measure))} &nbsp; | &nbsp; <b>Short rate:</b> {escape(str(pricing_short_rate))}</p>
  <h3>Specs</h3>
  {_df_to_html(df_specs, max_rows=max_rows)}
  <h3>Prices</h3>
  {_df_to_html(df_prices, max_rows=max_rows)}

  <h2>5) Scenario analysis</h2>
  <p><b>Available:</b> {escape(str(isinstance(scenario_analysis_result, dict)))}</p>
  <h3>Shocks</h3>
  <pre>{escape(_safe_json(scen_shocks or []))}</pre>
  <h3>Prices table</h3>
  {_df_to_html(df_scen if isinstance(df_scen, pd.DataFrame) else pd.DataFrame(), max_rows=max_rows)}

  <h2>6) Sensitivities</h2>
  <p><b>Available:</b> {escape(str(isinstance(sensitivities_result, dict)))}</p>
  <h3>Base prices</h3>
  {_df_to_html(df_sens_base if isinstance(df_sens_base, pd.DataFrame) else pd.DataFrame(), max_rows=max_rows)}
  <h3>Rate sensitivity</h3>
  {_df_to_html(df_sens_rate if isinstance(df_sens_rate, pd.DataFrame) else pd.DataFrame(), max_rows=max_rows)}
  <h3>Convexity</h3>
  {_df_to_html(df_sens_conv if isinstance(df_sens_conv, pd.DataFrame) else pd.DataFrame(), max_rows=max_rows)}
  <h3>Vega (sigma-scale)</h3>
  {_df_to_html(df_sens_vega if isinstance(df_sens_vega, pd.DataFrame) else pd.DataFrame(), max_rows=max_rows)}
  <h3>Delta-by-age (first instrument only)</h3>
  {_df_to_html(df_sens_dba if isinstance(df_sens_dba, pd.DataFrame) else pd.DataFrame(), max_rows=max_rows)}

  <h2>7) Hedging</h2>
  <p><b>Available:</b> {escape(str(hedge_result is not None))}</p>
  <h3>Summary</h3>
  {_df_to_html(df_hedge_summary, max_rows=max_rows)}
  <h3>Weights (order depends on hedge selection)</h3>
  {_df_to_html(df_hedge_w if isinstance(df_hedge_w, pd.DataFrame) else pd.DataFrame(), max_rows=max_rows)}

"""

if include_debug:
    keys = list(st.session_state.keys())
    html += f"""
  <h2>Debug</h2>
  <p>Session keys:</p>
  <pre>{escape(_safe_json(keys))}</pre>
"""

html += """
</body>
</html>
"""

# -----------------------------
# Streamlit display + download
# -----------------------------
st.markdown("")
st.success("HTML report built successfully!")

st.download_button(
    "⬇️ Download HTML report",
    data=html.encode("utf-8"),
    file_name="pymort_session_report.html",
    mime="text/html",
)
