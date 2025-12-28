# streamlit_app/pages/6_Pricing.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from pymort.analysis.sensitivities import make_single_product_pricer
from pymort.pipeline import (
    HullWhiteConfig,
    apply_hull_white_discounting,
    pricing_pipeline,
)
from pymort.pricing.liabilities import CohortLifeAnnuitySpec, price_cohort_life_annuity
from pymort.pricing.longevity_bonds import (
    LongevityBondSpec,
    price_simple_longevity_bond,
)
from pymort.pricing.mortality_derivatives import (
    QForwardSpec,
    SForwardSpec,
    price_q_forward,
    price_s_forward,
)
from pymort.pricing.survivor_swaps import SurvivorSwapSpec, price_survivor_swap

st.set_page_config(page_title="Pricing", page_icon="💰", layout="wide")
st.title("💰 Pricing")
st.caption("Price instruments under P or Q scenarios and store results for hedging/risk reporting.")


# -----------------------------
# Guards
# -----------------------------
scen_P = st.session_state.get("scen_P")
scen_Q = st.session_state.get("scen_Q")

if scen_P is None and scen_Q is None:
    st.info("Build scenarios first (Projection P and/or Risk-neutral Q).")
    st.stop()


# -----------------------------
# Sidebar: choose measure + common params
# -----------------------------
with st.sidebar:
    st.header("Pricing settings")

    # Measure choice
    available = []
    if scen_P is not None:
        available.append("P")
    if scen_Q is not None:
        available.append("Q")

    measure = st.radio(
        "Measure to price under", available, index=(0 if "Q" not in available else 1)
    )
    scen = scen_Q if measure == "Q" else scen_P

    st.divider()

    short_rate = st.number_input(
        "Short rate r (cont., flat)",
        value=float(st.session_state.get("short_rate_pricing", 0.02)),
        step=0.005,
        format="%.4f",
        help="Used for discounting in product pricers (if they use flat-rate discounting).",
    )
    st.session_state["short_rate_pricing"] = float(short_rate)

    st.divider()
    st.subheader("Interest-rate model")

    use_hw = st.checkbox("Use Hull–White discounting", value=False)

    hw_cfg = HullWhiteConfig(enabled=False)

    if use_hw:
        hw_a = st.number_input("HW mean reversion a", value=0.10, step=0.01, format="%.3f")
        hw_sigma = st.number_input("HW vol sigma", value=0.01, step=0.001, format="%.4f")
        hw_seed = st.number_input("HW seed", value=0, step=1)

        hw_cfg = HullWhiteConfig(
            enabled=True,
            a=float(hw_a),
            sigma=float(hw_sigma),
            seed=int(hw_seed),
            zero_rates=None,
        )

    st.divider()
    st.subheader("Products")
    use_bond = st.checkbox("Longevity bond", value=True)
    use_swap = st.checkbox("Survivor swap", value=False)
    use_qfwd = st.checkbox("Q-forward", value=False)
    use_sfwd = st.checkbox("S-forward", value=False)
    use_ann = st.checkbox("Cohort life annuity (liability)", value=False)

    st.divider()
    run_price = st.button("🚀 Price", type="primary")


# -----------------------------
# Main: build specs
# -----------------------------
ages = np.asarray(scen.ages, dtype=float)
years = np.asarray(scen.years, dtype=int)
age_min, age_max = int(ages.min()), int(ages.max())
age_default = int(round(float(np.median(ages))))
H = len(years)
max_maturity = max(1, min(60, H))

st.subheader(f"Scenario set selected: {measure}-measure")
c1, c2, c3, c4 = st.columns(4)
c1.metric("N scenarios", f"{int(scen.q_paths.shape[0])}")
c2.metric("Ages", f"{age_min} → {age_max}")
c3.metric("Horizon", f"{H} years")
c4.metric("Model", st.session_state.get("fitted_model").name)

st.divider()
st.subheader("Instrument specs")

specs: dict[str, object] = {}

colA, colB = st.columns(2)

with colA:
    if use_bond:
        st.markdown("### Longevity bond")
        bond_age = st.slider("Issue age", age_min, age_max, age_default, 1, key="pr_bond_age")
        bond_T = st.slider(
            "Maturity (years)",
            1,
            max_maturity,
            min(20, max_maturity),
            1,
            key="pr_bond_T",
        )
        bond_notional = st.number_input("Notional", value=100.0, step=10.0, key="pr_bond_notional")
        bond_include_principal = st.checkbox(
            "Include principal", value=True, key="pr_bond_include_principal"
        )

        specs["longevity_bond"] = LongevityBondSpec(
            issue_age=float(bond_age),
            notional=float(bond_notional),
            include_principal=bool(bond_include_principal),
            maturity_years=int(bond_T),
        )

    if use_qfwd:
        st.markdown("### Q-forward")
        qf_age = st.slider("Age", age_min, age_max, age_default, 1, key="pr_qf_age")
        qf_T = st.slider(
            "Maturity (years)",
            1,
            max_maturity,
            min(20, max_maturity),
            1,
            key="pr_qf_T",
        )
        qf_notional = st.number_input("Notional", value=100.0, step=10.0, key="pr_qf_notional")

        specs["q_forward"] = QForwardSpec(
            age=float(qf_age),
            maturity_years=int(qf_T),
            notional=float(qf_notional),
            strike=None,  # ATM if None (as in your other pages)
        )

    # --- Cohort life annuity (liability) ---
    if use_ann:
        st.markdown("### Cohort life annuity (liability)")
        ann_age = st.slider("Cohort age", age_min, age_max, age_default, 1, key="pr_ann_age")
        ann_T = st.slider(
            "Payment horizon (years)",
            1,
            max_maturity,
            min(30, max_maturity),
            1,
            key="pr_ann_T",
        )
        ann_payment = st.number_input(
            "Annual payment per survivor",
            value=1.0,
            step=0.1,
            format="%.4f",
            key="pr_ann_payment",
            help="This is payment_per_survivor in CohortLifeAnnuitySpec.",
        )
        ann_defer = st.number_input(
            "Deferral (years)",
            min_value=0,
            max_value=max(0, int(ann_T) - 1),
            value=0,
            step=1,
            key="pr_ann_defer",
        )
        ann_exposure = st.number_input(
            "Exposure at issue",
            value=1.0,
            step=1.0,
            format="%.4f",
            key="pr_ann_exposure",
            help="Scaling factor (e.g., number of lives).",
        )
        ann_include_terminal = st.checkbox(
            "Include terminal benefit",
            value=False,
            key="pr_ann_include_terminal",
        )
        ann_terminal_notional = st.number_input(
            "Terminal notional (paid at maturity × survival)",
            value=0.0,
            step=1.0,
            format="%.4f",
            key="pr_ann_terminal_notional",
            disabled=not ann_include_terminal,
        )

        specs["life_annuity"] = CohortLifeAnnuitySpec(
            issue_age=float(ann_age),
            payment_per_survivor=float(ann_payment),
            maturity_years=int(ann_T),
            defer_years=int(ann_defer),
            exposure_at_issue=float(ann_exposure),
            include_terminal=bool(ann_include_terminal),
            terminal_notional=float(ann_terminal_notional),
        )

with colB:
    if use_swap:
        st.markdown("### Survivor swap")
        swap_age = st.slider("Age", age_min, age_max, age_default, 1, key="pr_swap_age")
        swap_T = st.slider(
            "Maturity (years)",
            1,
            max_maturity,
            min(20, max_maturity),
            1,
            key="pr_swap_T",
        )
        swap_notional = st.number_input("Notional", value=100.0, step=10.0, key="pr_swap_notional")
        payer = st.selectbox("Payer", ["fixed", "floating"], index=0, key="pr_swap_payer")

        specs["survivor_swap"] = SurvivorSwapSpec(
            age=float(swap_age),
            maturity_years=int(swap_T),
            notional=float(swap_notional),
            strike=None,
            payer=str(payer),
        )

    if use_sfwd:
        st.markdown("### S-forward")
        sf_age = st.slider("Age", age_min, age_max, age_default, 1, key="pr_sf_age")
        sf_T = st.slider(
            "Maturity (years)",
            1,
            max_maturity,
            min(20, max_maturity),
            1,
            key="pr_sf_T",
        )
        sf_notional = st.number_input("Notional", value=100.0, step=10.0, key="pr_sf_notional")

        specs["s_forward"] = SForwardSpec(
            age=float(sf_age),
            maturity_years=int(sf_T),
            notional=float(sf_notional),
            strike=None,
        )

if not specs:
    st.warning("Select at least one product.")
    st.stop()


# -----------------------------
# Pricing runner
# -----------------------------
def _try_price_single(kind: str, spec: object, scen_set, r: float) -> dict:
    """Optional richer output:
    - If pricer returns float -> {"price": float}
    - If dict -> keep keys (price, pv_paths, cf_paths, ...)
    """
    pricer = make_single_product_pricer(kind=kind, spec=spec, short_rate=float(r))
    out = pricer(scen_set)

    if isinstance(out, dict):
        res = dict(out)
        if "price" not in res and "pv" in res:
            res["price"] = res["pv"]
        return res

    return {"price": float(out)}


if scen is None:
    st.error("No scenario set selected/available. Build P or Q scenarios first.")
    st.stop()

if run_price:
    with st.spinner("Pricing..."):
        try:
            # 1) MAIN: pipeline prices (clean + consistent with your orchestration layer)
            prices = pricing_pipeline(
                scen_Q=scen,
                specs=specs,
                short_rate=float(short_rate),
                hull_white=hw_cfg,
            )

            # 2) Build scenario used for path outputs (with HW DF if enabled)
            scen_for_rich = (
                apply_hull_white_discounting(scen, hw=hw_cfg, short_rate=short_rate)
                if use_hw
                else scen
            )

            H_full = len(scen_for_rich.years)
            N = int(scen_for_rich.q_paths.shape[0])

            def _pad_cf(cf: np.ndarray, H: int) -> np.ndarray:
                cf = np.asarray(cf, dtype=float)
                if cf.ndim != 2:
                    raise ValueError("cf_paths must be 2D (N,T).")
                if cf.shape[1] == H:
                    return cf
                if cf.shape[1] > H:
                    return cf[:, :H]
                out = np.zeros((cf.shape[0], H), dtype=float)
                out[:, : cf.shape[1]] = cf
                return out

            results_rich: dict[str, dict] = {}
            pv_paths = {}
            cf_paths = {}

            for name, spec in specs.items():
                if name == "longevity_bond":
                    rich = price_simple_longevity_bond(
                        scen_for_rich,
                        spec,
                        short_rate=float(short_rate),
                        return_cf_paths=True,
                    )
                elif name == "survivor_swap":
                    rich = price_survivor_swap(
                        scen_for_rich,
                        spec,
                        short_rate=float(short_rate),
                        return_cf_paths=True,
                    )
                elif name == "q_forward":
                    rich = price_q_forward(
                        scen_for_rich,
                        spec,
                        short_rate=float(short_rate),
                        return_cf_paths=True,
                    )
                elif name == "s_forward":
                    rich = price_s_forward(
                        scen_for_rich,
                        spec,
                        short_rate=float(short_rate),
                        return_cf_paths=True,
                    )
                elif name == "life_annuity":
                    rich = price_cohort_life_annuity(
                        scen_for_rich,
                        spec,
                        short_rate=float(short_rate),
                        return_cf_paths=True,
                    )
                else:
                    raise ValueError(f"Unsupported kind for paths: {name}")

                results_rich[name] = rich
                pv_paths[name] = np.asarray(rich["pv_paths"], dtype=float).reshape(N)
                cf_paths[name] = _pad_cf(np.asarray(rich["cf_paths"], dtype=float), H_full)

            # Store for next pages
            st.session_state["pv_paths"] = pv_paths
            st.session_state["cf_paths"] = cf_paths
            st.session_state["pricing_results_rich"] = results_rich
            st.session_state["pricing_discount_factors"] = (
                None
                if scen_for_rich.discount_factors is None
                else np.asarray(scen_for_rich.discount_factors)
            )

            # 3) Store for next pages
            st.session_state["prices"] = {k: float(v) for k, v in prices.items()}
            st.session_state["pricing_measure"] = measure
            st.session_state["pricing_short_rate"] = float(short_rate)
            st.session_state["pricing_specs"] = specs

            st.session_state["pv_paths"] = pv_paths if pv_paths else None
            st.session_state["cf_paths"] = cf_paths if cf_paths else None
            st.session_state["pricing_results_rich"] = results_rich  # debug if needed
            # After scen_for_rich is defined (scen or HW-discounted scen)
            H = len(scen_for_rich.years)
            t = np.arange(1, H + 1, dtype=float)

            df = scen_for_rich.discount_factors
            if df is None:
                df = np.exp(-float(short_rate) * t)  # (H,)

            st.session_state["pricing_discount_factors"] = df

            # reset downstream
            for k in ["hedge_result", "risk_report"]:
                st.session_state[k] = None

            st.success("Pricing done ✅")

        except Exception as e:
            st.error(f"Pricing failed: {e}")
            st.stop()


# -----------------------------
# Display results if available
# -----------------------------
st.divider()
st.subheader("Results")

prices = st.session_state.get("prices")
if not prices:
    st.info("Set specs and click **Price**.")
    st.stop()

df = pd.DataFrame([{"instrument": k, "price": float(v)} for k, v in prices.items()]).sort_values(
    "instrument"
)

st.dataframe(df, use_container_width=True)

with st.expander("Session debug (pricing)"):
    st.write("pricing_measure:", st.session_state.get("pricing_measure"))
    st.write("pricing_short_rate:", st.session_state.get("pricing_short_rate"))
    st.write("pv_paths available:", st.session_state.get("pv_paths") is not None)
    st.write("cf_paths available:", st.session_state.get("cf_paths") is not None)

st.success("Next: go to **Hedging** or **Risk report** pages (if you have them).")
