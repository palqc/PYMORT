from __future__ import annotations

from pathlib import Path

import streamlit as st
from assets.logo import add_logo_top_right

LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

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
    # analysis outputs (fix)
    "scenario_analysis_result",
    "sensitivities_result",
]


def init_session_state() -> None:
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
        # analysis outputs (fix)
        "scenario_analysis_result": None,
        "sensitivities_result": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def reset_session_state(keep: tuple[str, ...] = ()) -> None:
    for k in STATE_KEYS:
        if k in keep:
            continue
        if k in st.session_state:
            del st.session_state[k]
    init_session_state()


def _is_set(key: str) -> bool:
    return st.session_state.get(key, None) is not None


def render_sidebar_state() -> None:
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


def render_session_summary() -> None:
    fitted = st.session_state.get("fitted_model")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data loaded", "Yes" if _is_set("m") else "No")
    c2.metric("Sliced", "Yes" if _is_set("m_slice") else "No")
    c3.metric("Model", type(fitted).__name__ if fitted is not None else "—")
    c4.metric("Pricing", "Done" if _is_set("prices") else "—")


def main() -> None:
    # ✅ set_page_config doit venir en premier
    st.set_page_config(
        page_title="PYMORT — Longevity Risk Lab",
        page_icon=str(LOGO_PATH),
        layout="wide",
    )
    add_logo_top_right()

    init_session_state()
    render_sidebar_state()

    st.title("PYMORT — Longevity Risk Lab")
    st.caption(
        "Workflow: HMD table → slice → fit/select → P scenarios → Q scenarios (λ) → pricing → hedging → scenario analysis → sensitivities → reporting."
    )

    render_session_summary()

    st.divider()

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Quickstart")
        st.markdown(
            """
1) **Data Upload** — import life table / mortality surface  
2) **Data Slicing** — select ages/years window  
3) **Fit Select** — choose model (LC / CBD / APC …) + diagnostics  
4) **Projection & Pricing** — generate P/Q scenarios → price → hedge → report
"""
        )
        st.subheader("Data requirements")
        st.markdown(
            """
- **Rates must be consistent** (e.g., central death rates `m_x,t` or equivalent)
- **No negative values**, handle missing values (NaN) before fit if possible
- Prefer **rectangular age×year grid**
"""
        )

        with st.expander("Troubleshooting", expanded=False):
            st.markdown(
                """
- If a page stays ❌ in the sidebar: check you didn’t reset the session.
- If fits fail: try narrower slicing (shorter years range) or remove sparse ages.
- For reproducibility: keep the same slice window when comparing models.
"""
            )

    with right:
        st.subheader("Pages")
        st.markdown(
            """
- **Data Upload**
- **Data Slicing**  
- **Fit Select**  
- **Projection (P)**  
- **Risk Neutral (Q)**  
- **Pricing**  
- **Hedging**  
- **Scenario Analysis**  
- **Sensitivities**  
- **Report Export**
"""
        )
    st.markdown("")
    st.info("Tip: the sidebar shows what’s already computed ( ✅ / ❌ ).")


if __name__ == "__main__":
    main()
