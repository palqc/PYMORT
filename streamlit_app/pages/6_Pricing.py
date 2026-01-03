from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from assets.logo import LOGO_PATH, add_logo_top_right

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

add_logo_top_right()
st.set_page_config(page_title="Pricing", page_icon=LOGO_PATH, layout="wide")
st.title("Pricing")
st.caption(
    "Price instruments under P or Q scenarios and store results for hedging/risk reporting."
)


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
    st.subheader("Interest-rate model")

    use_hw = st.checkbox("Use Hull–White discounting", value=False)

    short_rate = st.number_input(
        "Short rate r (cont., flat)",
        value=float(st.session_state.get("short_rate_pricing", 0.02)),
        step=0.005,
        format="%.3f",
        help="Fallback flat rate. Used when discount_factors are not provided and/or as anchor for HW helper.",
    )
    st.session_state["short_rate_pricing"] = float(short_rate)

    hw_cfg = HullWhiteConfig(enabled=False)
    if use_hw:
        hw_a = st.number_input(
            "HW mean reversion a", value=0.10, step=0.01, format="%.2f"
        )
        hw_sigma = st.number_input(
            "HW vol sigma", value=0.01, step=0.001, format="%.3f"
        )
        hw_seed = st.number_input("HW seed", value=0, step=1)

        hw_cfg = HullWhiteConfig(
            enabled=True,
            a=float(hw_a),
            sigma=float(hw_sigma),
            seed=int(hw_seed),
            zero_rates=None,
        )
    st.session_state["pricing_ir_model"] = "Hull-White" if use_hw else "Flat"
    st.session_state["pricing_hw_params"] = (
        {"a": float(hw_cfg.a), "sigma": float(hw_cfg.sigma), "seed": int(hw_cfg.seed)}
        if use_hw
        else None
    )

    st.divider()
    st.subheader("Products")
    use_bond = st.checkbox("Longevity bond", value=True)
    use_swap = st.checkbox("Survivor swap", value=True)
    use_qfwd = st.checkbox("Q-forward", value=True)
    use_sfwd = st.checkbox("S-forward", value=True)
    use_ann = st.checkbox("Cohort life annuity (liability)", value=True)

    st.divider()
    run_price = st.button("Price", type="primary")


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
st.markdown("")

specs: dict[str, object] = {}

col1, col2, col3, col4, col5 = st.columns([0.01, 3, 1, 3, 0.5])

with col2:
    if use_bond:
        st.markdown("**Longevity bond**")
        bond_age = st.slider(
            "Issue age", age_min, age_max, age_default, 1, key="pr_bond_age"
        )
        bond_T = st.slider(
            "Maturity (years)",
            1,
            max_maturity,
            min(20, max_maturity),
            1,
            key="pr_bond_T",
        )
        bond_notional = st.number_input(
            "Notional", value=100.0, step=10.0, key="pr_bond_notional"
        )
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
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("**Q-forward**")
        qf_age = st.slider("Age", age_min, age_max, age_default, 1, key="pr_qf_age")
        qf_T = st.slider(
            "Maturity (years)",
            1,
            max_maturity,
            min(20, max_maturity),
            1,
            key="pr_qf_T",
        )
        qf_notional = st.number_input(
            "Notional", value=100.0, step=10.0, key="pr_qf_notional"
        )

        specs["q_forward"] = QForwardSpec(
            age=float(qf_age),
            maturity_years=int(qf_T),
            notional=float(qf_notional),
            strike=None,  # ATM if None (as in your other pages)
        )

    if use_sfwd:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("**S-forward**")
        sf_age = st.slider("Age", age_min, age_max, age_default, 1, key="pr_sf_age")
        sf_T = st.slider(
            "Maturity (years)",
            1,
            max_maturity,
            min(20, max_maturity),
            1,
            key="pr_sf_T",
        )
        sf_notional = st.number_input(
            "Notional", value=100.0, step=10.0, key="pr_sf_notional"
        )

        specs["s_forward"] = SForwardSpec(
            age=float(sf_age),
            maturity_years=int(sf_T),
            notional=float(sf_notional),
            strike=None,
        )

with col4:
    if use_swap:
        st.markdown("**Survivor swap**")
        swap_age = st.slider("Age", age_min, age_max, age_default, 1, key="pr_swap_age")
        swap_T = st.slider(
            "Maturity (years)",
            1,
            max_maturity,
            min(20, max_maturity),
            1,
            key="pr_swap_T",
        )
        swap_notional = st.number_input(
            "Notional", value=100.0, step=10.0, key="pr_swap_notional"
        )
        payer = st.selectbox(
            "Payer", ["fixed", "floating"], index=0, key="pr_swap_payer"
        )

        specs["survivor_swap"] = SurvivorSwapSpec(
            age=float(swap_age),
            maturity_years=int(swap_T),
            notional=float(swap_notional),
            strike=None,
            payer=str(payer),
        )

    # --- Cohort life annuity (liability) ---
    if use_ann:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("**Cohort life annuity (liability)**")
        ann_age = st.slider(
            "Cohort age", age_min, age_max, age_default, 1, key="pr_ann_age"
        )
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
            format="%.2f",
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
            format="%.2f",
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

            # Compare flat DF vs HW DF effect (only if HW enabled)
            if use_hw:
                st.session_state["pricing_hw_enabled"] = True
            else:
                st.session_state["pricing_hw_enabled"] = False

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
                cf_paths[name] = _pad_cf(
                    np.asarray(rich["cf_paths"], dtype=float), H_full
                )

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

        except Exception as e:
            st.error(f"Pricing failed: {e}")
            st.stop()


def clean_small(x, eps=1e-8):
    return 0 if abs(x) < eps else x


# -----------------------------
# Display results if available
# -----------------------------
prices_ss = st.session_state.get("prices")
if not prices_ss:
    st.info("Set specs and click **Price**.")
    st.stop()
st.divider()
st.subheader("Results")
st.markdown("")
col_left, col_mid, col_right = st.columns([1, 3, 1])
with col_mid:
    prices = st.session_state.get("prices")
    df = pd.DataFrame(
        [{"instrument": k, "price": clean_small(float(v))} for k, v in prices.items()]
    ).sort_values("instrument")

    st.dataframe(df, use_container_width=True)

st.divider()
st.subheader("PV paths diagnostics")
st.markdown("")

pv_paths = st.session_state.get("pv_paths")
if pv_paths is None:
    st.info("No pv_paths stored yet. Click **Price** to generate scenario PV paths.")
else:
    # ---- collect PV series
    series = {k: np.asarray(v, dtype=float).reshape(-1) for k, v in pv_paths.items()}
    names = list(series.keys())

    # ---- quick table of quantiles (to detect outliers)
    def _qtab(x):
        return {
            "min": float(np.min(x)),
            "p01": float(np.quantile(x, 0.01)),
            "p05": float(np.quantile(x, 0.05)),
            "p50": float(np.quantile(x, 0.50)),
            "p95": float(np.quantile(x, 0.95)),
            "p99": float(np.quantile(x, 0.99)),
            "max": float(np.max(x)),
            "std": float(np.std(x)),
        }

    df_q = pd.DataFrame(
        [{"instrument": k, **_qtab(v)} for k, v in series.items()]
    ).sort_values("instrument")
    st.dataframe(df_q, use_container_width=True)
    st.markdown("")

    # ---- controls
    use_logx = st.checkbox(
        "Log-scale x (signed)",
        value=True,
        help="Uses signed log transform for visualization only.",
    )

    # ---- build common x-limits based on pooled quantiles
    pooled = np.concatenate(list(series.values()))

    # ---- plotting
    col_left, col_mid, col_right = st.columns([1, 5, 1])

    with col_mid:
        fig, ax = plt.subplots(figsize=(10, 4))

        def _signed_log(z):
            z = np.asarray(z, dtype=float)
            return np.sign(z) * np.log1p(np.abs(z))

        for k, x in series.items():
            x_plot = x.copy()

            if use_logx:
                x_plot = _signed_log(x_plot)

            # histogram density
            ax.hist(x_plot, bins=60, density=True, alpha=0.25, label=k)

        ax.set_title("PV distributions (density)", pad=15)
        ax.set_ylabel("Density")
        ax.set_xlabel("PV" + (" (signed log1p)" if use_logx else ""))

        # nicer ticks: show real PV bounds even if log view
        if use_logx:
            ax.set_xlabel("PV (signed log1p)")

        ax.grid(True, alpha=0.2)
        ax.legend()
        st.pyplot(fig, use_container_width=True)

st.divider()
st.subheader("Cashflow term structure (scenario fan)")

cf_paths = st.session_state.get("cf_paths")
df_disc = st.session_state.get("pricing_discount_factors")


if cf_paths is None:
    st.info(
        "cf_paths not available. Enable return_cf_paths in pricers (already done in your code)."
    )
else:
    col1, col2 = st.columns([1, 3])

    with col1:
        insts = sorted(list(cf_paths.keys()))
        inst_cf = st.selectbox(
            "Instrument for CF term structure", options=insts, index=0, key="pr_cf_inst"
        )

    cf = np.asarray(cf_paths[inst_cf], dtype=float)  # (N, H)
    N, H = cf.shape

    years = np.asarray(scen.years, dtype=int)
    H2 = min(len(years), H)
    years = years[:H2]
    cf = cf[:, :H2]

    # Optional discounted CF (if DF available)
    use_disc = st.checkbox("Show discounted cashflows", value=False, key="pr_cf_disc")
    if use_disc:
        df = np.asarray(df_disc, dtype=float)
        if df.ndim == 1:
            df = df[:H2][None, :]
        else:
            df = df[:, :H2]
            if df.shape[0] == 1:
                df = np.repeat(df, N, axis=0)
        cf_plot = cf * df
        ylab = "Discounted CF"
    else:
        cf_plot = cf
        ylab = "Cashflow"

    qs = [0.05, 0.25, 0.50, 0.75, 0.95]
    qmat = np.quantile(cf_plot, qs, axis=0)  # (5, H2)
    col_left, col_mid, col_right = st.columns([1, 5, 1])

    with col_mid:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(years, qmat[2], linewidth=2, label="median")
        ax.fill_between(years, qmat[1], qmat[3], alpha=0.2, label="25–75%")
        ax.fill_between(years, qmat[0], qmat[4], alpha=0.12, label="5–95%")
        ax.set_title(f"{inst_cf} cashflow fan", pad=15)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.2)
        ax.legend()
        st.pyplot(fig, use_container_width=True)

st.divider()
st.subheader("Interest-rate model impact (HW vs flat)")

if st.session_state.get("pricing_hw_enabled", False):
    # We reprice quickly with flat DF on the same scen (no HW) for delta view.
    # Uses your rich pricers already.
    try:
        scen_flat = scen  # same scenario set but discounting is flat inside pricers
        pv_flat = {}
        pv_hw = st.session_state.get("pv_paths") or {}

        for name, spec in specs.items():
            if name == "longevity_bond":
                out = price_simple_longevity_bond(
                    scen_flat, spec, short_rate=float(short_rate), return_cf_paths=False
                )
            elif name == "survivor_swap":
                out = price_survivor_swap(
                    scen_flat, spec, short_rate=float(short_rate), return_cf_paths=False
                )
            elif name == "q_forward":
                out = price_q_forward(
                    scen_flat, spec, short_rate=float(short_rate), return_cf_paths=False
                )
            elif name == "s_forward":
                out = price_s_forward(
                    scen_flat, spec, short_rate=float(short_rate), return_cf_paths=False
                )
            elif name == "life_annuity":
                out = price_cohort_life_annuity(
                    scen_flat, spec, short_rate=float(short_rate), return_cf_paths=False
                )
            else:
                continue
            pv_flat[name] = np.asarray(out["pv_paths"], dtype=float).reshape(-1)
        col1, col2 = st.columns([1, 3])

        with col1:
            insts = sorted(list(set(pv_flat.keys()) & set(pv_hw.keys())))
            pick = st.selectbox(
                "Instrument", options=insts, index=0, key="pr_hw_delta_inst"
            )
        d = np.asarray(pv_hw[pick], dtype=float) - np.asarray(
            pv_flat[pick], dtype=float
        )
        col_left, col_mid, col_right = st.columns([1, 5, 1])

        with col_mid:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(d, bins=60, alpha=0.6)
            ax.axvline(0.0, linewidth=1.0)
            ax.set_title(f"ΔPV = PV(HW) - PV(flat) for {pick}", pad=15)
            ax.set_xlabel("ΔPV")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.2)
            st.pyplot(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not compute HW vs flat delta view: {e}")
else:
    st.caption("Enable Hull–White discounting to see HW vs flat impact.")

st.markdown("")
st.success("Next: go to **Hedging** page.")
