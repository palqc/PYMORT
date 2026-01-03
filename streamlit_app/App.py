from __future__ import annotations

from pathlib import Path

import streamlit as st
from assets.logo import add_logo_top_right

LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

# -----------------------------
# Session state helpers
# -----------------------------
STATE_KEYS = [
    # data
    "raw_df",
    "ages",
    "years",
    "m",
    "data_meta",
    # slicing
    "slice_cfg",
    "ages_slice",
    "years_slice",
    "m_slice",
    # fitted / scenarios
    "fitted_model",
    "proj_P",
    "scen_P",
    "scen_Q",
    "calibration_summary",
    "calibration_cache",
    # pricing / hedging outputs
    "prices",
    "pv_paths",
    "cf_paths",
    "hedge_result",
    "risk_report",
]


def init_session_state() -> None:
    """Create default keys if missing (do not overwrite existing values)."""
    defaults = {
        "raw_df": None,
        "ages": None,
        "years": None,
        "m": None,
        "data_meta": {},
        "slice_cfg": None,
        "ages_slice": None,
        "years_slice": None,
        "m_slice": None,
        "fitted_model": None,
        "proj_P": None,
        "scen_P": None,
        "scen_Q": None,
        "calibration_summary": None,
        "calibration_cache": None,
        "prices": None,
        "pv_paths": None,
        "cf_paths": None,
        "hedge_result": None,
        "risk_report": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def reset_session_state(keep: tuple[str, ...] = ()) -> None:
    """Remove known keys from session_state."""
    for k in STATE_KEYS:
        if k in keep:
            continue
        if k in st.session_state:
            del st.session_state[k]
    init_session_state()


def _is_set(key: str) -> bool:
    return st.session_state.get(key, None) is not None


def render_sidebar_state() -> None:
    """Sidebar: quick status indicators + reset."""
    st.sidebar.header("Project state")

    st.sidebar.write(("✅" if _is_set("m") else "❌") + " Data loaded")
    st.sidebar.write(("✅" if _is_set("m_slice") else "❌") + " Data sliced")
    st.sidebar.write(
        ("✅" if _is_set("fitted_model") else "❌") + " Model fitted/selected"
    )
    st.sidebar.write(("✅" if _is_set("scen_P") else "❌") + " Scenarios P built")
    st.sidebar.write(("✅" if _is_set("scen_Q") else "❌") + " Scenarios Q built")
    st.sidebar.write(("✅" if _is_set("prices") else "❌") + " Pricing done")
    st.sidebar.write(("✅" if _is_set("hedge_result") else "❌") + " Hedge computed")
    st.sidebar.write(
        ("✅" if _is_set("scenario_analysis_result") else "❌")
        + " Scenario analysis computed"
    )
    st.sidebar.write(
        ("✅" if _is_set("sensitivities_result") else "❌")
        + " Sensitivity analysis computed"
    )
    st.sidebar.write(("✅" if _is_set("risk_report") else "❌") + " Report generated")

    st.sidebar.divider()

    if st.sidebar.button("Reset session", type="primary"):
        reset_session_state()
        st.rerun()


# -----------------------------
# App
# -----------------------------
def main() -> None:
    add_logo_top_right()
    st.set_page_config(
        page_title="PYMORT — Longevity Risk Lab",
        page_icon=LOGO_PATH,
        layout="wide",
    )

    init_session_state()
    render_sidebar_state()

    st.title("PYMORT — Longevity Risk Lab")
    st.caption(
        "Workflow: HMD table → slice → fit/select → P scenarios → Q scenarios (λ) → pricing → hedging → scenario analysis → sensitivities → reporting."
    )

    st.markdown(
        """
### Start here
Use the left sidebar to navigate pages:
- **Data Upload**: drag & drop your HMD life table / mortality surface
- **Fit & Model Selection**
- **Projection (P)**
- **Risk-neutral (Q)**
- **Pricing / Hedging / Reporting**

Tip: the sidebar shows what’s already computed (✅/❌).
"""
    )

    # Quick peek at what's loaded
    with st.expander("Debug: current session keys", expanded=False):
        st.json(
            {
                "data_loaded": _is_set("m"),
                "slice_ready": _is_set("m_slice"),
                "fitted_model": _is_set("fitted_model"),
                "scen_P": _is_set("scen_P"),
                "scen_Q": _is_set("scen_Q"),
                "prices": _is_set("prices"),
                "hedge_result": _is_set("hedge_result"),
            }
        )


if __name__ == "__main__":
    main()
