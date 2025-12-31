# streamlit_app/pages/5_Risk_Neutral_Q.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from assets.logo import add_logo_top_right

from pymort.pipeline import build_risk_neutral_pipeline
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
from pymort.pricing.risk_neutral import build_scenarios_under_lambda_fast
from pymort.pricing.survivor_swaps import SurvivorSwapSpec, price_survivor_swap


# -----------------------------
# Helpers
# -----------------------------
def _lambda_dim_from_cache(cache) -> int:
    name = str(getattr(cache, "model_name", "")).upper()
    if name == "CBDM7":
        return 3
    return 1


def _price_instrument(scen_set, key: str, spec: object, short_rate: float) -> float:
    if key == "longevity_bond":
        return float(price_simple_longevity_bond(scen_set, spec, short_rate=short_rate)["price"])
    if key == "survivor_swap":
        return float(price_survivor_swap(scen_set, spec, short_rate=short_rate)["price"])
    if key == "q_forward":
        return float(price_q_forward(scen_set, spec, short_rate=short_rate)["price"])
    if key == "s_forward":
        return float(price_s_forward(scen_set, spec, short_rate=short_rate)["price"])
    if key == "life_annuity":
        return float(price_cohort_life_annuity(scen_set, spec, short_rate=short_rate)["price"])
    raise ValueError(f"Unknown instrument key '{key}'")


def _build_price_table(scen_set, instruments: dict[str, object], short_rate: float) -> pd.DataFrame:
    rows = []
    for k, spec in instruments.items():
        p = _price_instrument(scen_set, k, spec, short_rate=float(short_rate))
        rows.append({"instrument": k, "model_price": float(p)})
    return pd.DataFrame(rows)


def _as_lambda_vec(x: list[float] | float, k: int) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(x, dtype=float)).reshape(-1)
    if arr.size == 1 and k > 1:
        arr = np.full(k, float(arr[0]), dtype=float)
    if arr.size != k:
        raise ValueError(f"lambda must have dim {k} (got {arr.size})")
    return arr


def _request_gen_synth() -> None:
    st.session_state["do_gen_synth"] = True


def _request_run_B() -> None:
    st.session_state["do_run_B"] = True


def _request_run_A() -> None:
    st.session_state["do_run_A"] = True


def format_lambda_for_display(lam):
    if lam is None:
        return "n/a"

    arr = np.atleast_1d(lam)

    if arr.size == 1:
        return f"{arr[0]:.2f}"

    return "[" + ", ".join(f"{x:.2f}" for x in arr) + "]"


# -----------------------------
# Page
# -----------------------------
add_logo_top_right()
st.set_page_config(page_title="Risk-neutral Q", page_icon="🌙", layout="wide")
st.title("🌙 Risk-neutral scenarios (Q-measure)")
st.caption(
    "Mode A: choose λ manually (no calibration). "
    "Mode B: create synthetic market prices to test calibration and recover λ."
)

# -----------------------------
# Session state init
# -----------------------------
st.session_state.setdefault("do_gen_synth", False)
st.session_state.setdefault("do_run_A", False)
st.session_state.setdefault("do_run_B", False)
st.session_state.setdefault("mkt_autofill", {})  # optional debug/cache

# -----------------------------
# Guards
# -----------------------------
scen_P = st.session_state.get("scen_P")
cache = st.session_state.get("calibration_cache")

if scen_P is None:
    st.info("Build **P scenarios** first (page: Projection P).")
    st.stop()

if cache is None:
    st.error(
        "Missing **CalibrationCache** in session_state. "
        "Go to **Projection P** and store it as `calibration_cache`."
    )
    st.stop()

# Defaults from P-scenarios
ages = np.asarray(scen_P.ages, dtype=float)
years = np.asarray(scen_P.years, dtype=int)
age_min, age_max = int(ages.min()), int(ages.max())
age_default = int(ages[len(ages) // 2])
H_P = len(years)
max_maturity = max(1, min(40, H_P))
k_lam = _lambda_dim_from_cache(cache)

# -----------------------------
# Sidebar: global settings
# -----------------------------
with st.sidebar:
    st.header("Q builder mode")
    mode = st.radio(
        "Choose mode",
        [
            "Mode A — Fixed λ (no calibration)",
            "Mode B — Synthetic calibration (λ_true → market prices → λ*)",
        ],
        index=0,
    )

    st.divider()
    st.header("Common settings")
    short_rate = st.number_input(
        "Short rate r (cont., flat)",
        value=0.02,
        step=0.005,
        format="%.4f",
        help="Used for discounting in pricing/calibration.",
    )

    scale_sigma = st.number_input(
        "Sigma scale (vol multiplier)",
        value=1.0,
        step=0.05,
        format="%.3f",
        help="Multiplies process volatility during Q build (and calibration in Mode B).",
    )

    st.divider()
    st.subheader("Instruments")
    use_bond = st.checkbox("Longevity bond", value=True)
    use_swap = st.checkbox("Survivor swap", value=False)
    use_qfwd = st.checkbox("Q-forward", value=False)
    use_sfwd = st.checkbox("S-forward", value=False)
    use_ann = st.checkbox("Cohort life annuity (liability)", value=False)

# -----------------------------
# Main: instrument specs
# -----------------------------
st.subheader("Instrument specs")
st.write(
    "Pick instruments + specs. "
    "Mode A builds Q directly from λ. Mode B uses market prices (synthetic or manual) to calibrate λ."
)

instruments: dict[str, object] = {}

col_specs, col_prices = st.columns(2)

with col_specs:
    st.markdown("### Specs")

    if use_bond:
        st.markdown("**Longevity bond**")
        bond_age = st.slider("Bond issue age", age_min, age_max, age_default, 1, key="bond_age")
        bond_T = st.slider(
            "Bond maturity (years)",
            1,
            max_maturity,
            min(20, max_maturity),
            1,
            key="bond_T",
        )
        bond_notional = st.number_input(
            "Bond notional", value=100.0, step=10.0, key="bond_notional"
        )
        bond_include_principal = st.checkbox(
            "Bond include principal", value=True, key="bond_include_principal"
        )

        instruments["longevity_bond"] = LongevityBondSpec(
            issue_age=float(bond_age),
            notional=float(bond_notional),
            include_principal=bool(bond_include_principal),
            maturity_years=int(bond_T),
        )

    if use_swap:
        st.markdown("**Survivor swap**")
        swap_age = st.slider("Swap age", age_min, age_max, age_default, 1, key="swap_age")
        swap_T = st.slider(
            "Swap maturity (years)",
            1,
            max_maturity,
            min(20, max_maturity),
            1,
            key="swap_T",
        )
        swap_notional = st.number_input(
            "Swap notional", value=100.0, step=10.0, key="swap_notional"
        )
        payer = st.selectbox("Swap payer", ["fixed", "floating"], index=0, key="swap_payer")

        instruments["survivor_swap"] = SurvivorSwapSpec(
            age=float(swap_age),
            maturity_years=int(swap_T),
            notional=float(swap_notional),
            strike=None,
            payer=str(payer),
        )

    if use_qfwd:
        st.markdown("**Q-forward**")
        qf_age = st.slider("QF age", age_min, age_max, age_default, 1, key="qf_age")
        qf_T = st.slider(
            "QF maturity (years)",
            1,
            max_maturity,
            min(20, max_maturity),
            1,
            key="qf_T",
        )
        qf_notional = st.number_input("QF notional", value=100.0, step=10.0, key="qf_notional")

        instruments["q_forward"] = QForwardSpec(
            age=float(qf_age),
            maturity_years=int(qf_T),
            notional=float(qf_notional),
            strike=None,
        )

    if use_sfwd:
        st.markdown("**S-forward**")
        sf_age = st.slider("SF age", age_min, age_max, age_default, 1, key="sf_age")
        sf_T = st.slider(
            "SF maturity (years)",
            1,
            max_maturity,
            min(20, max_maturity),
            1,
            key="sf_T",
        )
        sf_notional = st.number_input("SF notional", value=100.0, step=10.0, key="sf_notional")

        instruments["s_forward"] = SForwardSpec(
            age=float(sf_age),
            maturity_years=int(sf_T),
            notional=float(sf_notional),
            strike=None,
        )
    if use_ann:
        st.markdown("**Cohort life annuity (liability)**")
        ann_age = st.slider("Annuity issue age", age_min, age_max, age_default, 1, key="ann_age")
        ann_T = st.slider(
            "Annuity horizon (years)",
            1,
            max_maturity,
            min(30, max_maturity),
            1,
            key="ann_T",
        )
        ann_payment = st.number_input(
            "Payment per survivor",
            value=1.0,
            step=0.1,
            format="%.4f",
            key="ann_payment",
        )
        ann_defer = st.number_input(
            "Deferral (years)",
            min_value=0,
            max_value=max(0, int(ann_T) - 1),
            value=0,
            step=1,
            key="ann_defer",
        )
        ann_exposure = st.number_input(
            "Exposure at issue",
            value=1.0,
            step=1.0,
            format="%.4f",
            key="ann_exposure",
        )

        instruments["life_annuity"] = CohortLifeAnnuitySpec(
            issue_age=float(ann_age),
            payment_per_survivor=float(ann_payment),
            maturity_years=int(ann_T),
            defer_years=int(ann_defer),
            exposure_at_issue=float(ann_exposure),
            include_terminal=False,
            terminal_notional=0.0,
        )

if not instruments:
    st.warning("Select at least one instrument.")
    st.stop()
calibration_keys = [k for k in instruments if k != "life_annuity"]
instruments_calib = {k: instruments[k] for k in calibration_keys}

# -----------------------------
# Sidebar: mode-specific params + run buttons
# (we need lam_true_vec/noise/seed available BEFORE generating synth prices)
# -----------------------------
with st.sidebar:
    st.divider()

    if mode.startswith("Mode A"):
        st.subheader("Mode A — choose λ")
        if k_lam == 1:
            lam_A = st.number_input("λ (fixed)", value=0.0, step=0.1, format="%.3f", key="lam_A_1d")
            lam_A_vec = [float(lam_A)]
        else:
            st.caption("CBDM7 → λ is 3D (one per RW factor).")
            lam1 = st.number_input("λ1", value=0.0, step=0.1, format="%.3f", key="lam_A_1")
            lam2 = st.number_input("λ2", value=0.0, step=0.1, format="%.3f", key="lam_A_2")
            lam3 = st.number_input("λ3", value=0.0, step=0.1, format="%.3f", key="lam_A_3")
            lam_A_vec = [float(lam1), float(lam2), float(lam3)]

        st.button(
            "🚀 Build Q (Fixed λ)",
            type="primary",
            on_click=_request_run_A,
        )

        # placeholders for downstream references
        lam_true_vec = None
        noise_std = None
        seed_syn = None
        lambda0 = None
        lam_lb = None
        lam_ub = None

    else:
        st.subheader("Mode B — synthetic calibration")

        if k_lam == 1:
            lam_true = st.number_input(
                "λ_true", value=0.3, step=0.1, format="%.3f", key="lam_true_1d"
            )
            lam_true_vec = [float(lam_true)]
        else:
            st.caption("CBDM7 → λ_true is 3D.")
            lt1 = st.number_input("λ_true1", value=0.3, step=0.1, format="%.3f", key="lam_true_1")
            lt2 = st.number_input("λ_true2", value=0.0, step=0.1, format="%.3f", key="lam_true_2")
            lt3 = st.number_input("λ_true3", value=0.0, step=0.1, format="%.3f", key="lam_true_3")
            lam_true_vec = [float(lt1), float(lt2), float(lt3)]

        noise_std = st.number_input(
            "Synthetic noise std (absolute)",
            value=0.0,
            step=1.0,
            format="%.6f",
            help="Adds N(0, noise_std) to each synthetic market price.",
        )
        seed_syn = st.number_input("Synthetic seed", value=0, step=1)

        st.divider()
        lambda0 = st.number_input("Initial λ (solver)", value=0.0, step=0.1, format="%.3f")
        lam_lb = st.number_input("λ lower bound", value=-5.0, step=0.5, format="%.2f")
        lam_ub = st.number_input("λ upper bound", value=5.0, step=0.5, format="%.2f")

        st.button(
            "✨ Generate synthetic market prices from λ_true",
            on_click=_request_gen_synth,
        )
        st.button(
            "🚀 Calibrate λ and build Q",
            type="primary",
            on_click=_request_run_B,
        )

# ------------------------------------------------------------------
# MODE B: generate synthetic market prices
# ------------------------------------------------------------------
if mode.startswith("Mode B") and st.session_state.get("do_gen_synth", False):
    st.session_state["do_gen_synth"] = False  # consume flag

    try:
        lam_true_arr = _as_lambda_vec(lam_true_vec, k_lam)

        with st.spinner("Building Q(λ_true) and generating synthetic market prices..."):
            scen_Q_true = build_scenarios_under_lambda_fast(
                cache=cache,
                lambda_esscher=lam_true_arr,
                scale_sigma=float(scale_sigma),
            )

            df_prices = _build_price_table(
                scen_Q_true, instruments_calib, short_rate=float(short_rate)
            )

            rng = np.random.default_rng(int(seed_syn))
            noisy = df_prices["model_price"].values + rng.normal(
                0.0, float(noise_std), size=len(df_prices)
            )

        # IMPORTANT: write directly into widget keys BEFORE widgets are created
        for inst, mp in zip(df_prices["instrument"].tolist(), noisy.tolist()):
            st.session_state[f"mkt_{inst}"] = float(mp)
            st.session_state["mkt_autofill"][inst] = float(mp)

        st.session_state["synthetic_market_debug"] = {
            "lambda_true": lam_true_arr.tolist(),
            "noise_std": float(noise_std),
            "seed": int(seed_syn),
            "base_model_prices": df_prices.to_dict(orient="records"),
        }

        st.success("Synthetic market prices generated ✅")
        st.rerun()

    except Exception as e:
        st.error(f"Synthetic market generation failed: {e}")

# -----------------------------
# Mode-specific UI (Market prices)
# -----------------------------
market_prices: dict[str, float] = {}

with col_prices:
    st.markdown("### Market prices / Calibration inputs")

    if mode.startswith("Mode A"):
        st.info("Mode A: no market prices needed. We'll build Q directly from your chosen λ.")
    else:
        st.write(
            "Mode B: edit market prices manually, or click the synthetic generator in the sidebar."
        )

        for k in calibration_keys:
            widget_key = f"mkt_{k}"

            if widget_key in st.session_state:
                market_prices[k] = st.number_input(
                    f"Market price for '{k}'",
                    step=1.0,
                    format="%.6f",
                    key=widget_key,
                )
            else:
                market_prices[k] = st.number_input(
                    f"Market price for '{k}'",
                    value=0.0,
                    step=1.0,
                    format="%.6f",
                    key=widget_key,
                )

# -----------------------------
# Run Mode A
# -----------------------------
if mode.startswith("Mode A") and st.session_state.get("do_run_A", False):
    st.session_state["do_run_A"] = False

    try:
        lamA = _as_lambda_vec(lam_A_vec, k_lam)

        with st.spinner("Building Q scenarios under fixed λ..."):
            scen_Q = build_scenarios_under_lambda_fast(
                cache=cache,
                lambda_esscher=lamA,
                scale_sigma=float(scale_sigma),
            )

        summary = {
            "mode": "A_fixed_lambda",
            "lambda_used": lamA.tolist(),
            "scale_sigma": float(scale_sigma),
            "short_rate": float(short_rate),
            "success": True,
            "rmse_pricing_error": np.nan,
            "objective_value": np.nan,
            "residuals": [],
        }

        st.session_state["scen_Q"] = scen_Q
        st.session_state["calibration_summary"] = summary
        st.session_state["calibration_cache"] = cache
        st.session_state["risk_neutral_mode"] = "A"

        for kk in ["prices", "pv_paths", "cf_paths", "hedge_result", "risk_report"]:
            st.session_state[kk] = None

        st.success("Q scenarios built (Mode A) ✅")

    except Exception as e:
        st.error(f"Mode A failed: {e}")

# -----------------------------
# Run Mode B (calibrate)
# -----------------------------
if mode.startswith("Mode B") and st.session_state.get("do_run_B", False):
    st.session_state["do_run_B"] = False

    # read market prices from widgets
    market_prices = {k: float(st.session_state.get(f"mkt_{k}", 0.0)) for k in calibration_keys}

    try:
        calibration_kwargs = {
            "cache": cache,
            "lambda0": float(lambda0),
            "bounds": (float(lam_lb), float(lam_ub)),
            "horizon": int(H_P),
            "seed": st.session_state.get("projP_cfg", {}).get("seed", None),
            "scale_sigma": float(scale_sigma),
        }

        with st.spinner("Calibrating λ and building Q scenarios..."):
            scen_Q, calib_summary, cache_out = build_risk_neutral_pipeline(
                scen_P=scen_P,
                instruments=instruments_calib,
                market_prices=market_prices,
                short_rate=float(short_rate),
                calibration_kwargs=calibration_kwargs,
            )

        calib_summary = dict(calib_summary)
        calib_summary["mode"] = "B_synthetic_or_manual_calibration"
        if "synthetic_market_debug" in st.session_state:
            calib_summary["synthetic_market_debug"] = st.session_state["synthetic_market_debug"]

        st.session_state["scen_Q"] = scen_Q
        st.session_state["calibration_summary"] = calib_summary
        st.session_state["calibration_cache"] = cache_out
        st.session_state["risk_neutral_mode"] = "B"

        for kk in ["prices", "pv_paths", "cf_paths", "hedge_result", "risk_report"]:
            st.session_state[kk] = None

        st.success("Q scenarios built (Mode B) ✅")

    except Exception as e:
        st.error(f"Mode B calibration failed: {e}")

# -----------------------------
# Display results
# -----------------------------
st.divider()
st.subheader("Results")

scen_Q = st.session_state.get("scen_Q")
summary = st.session_state.get("calibration_summary")

if scen_Q is None or summary is None:
    st.info("Build Q using Mode A or Mode B.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)

lam_star = summary.get("lambda_star", None)
lam_used = summary.get("lambda_used", None)

if lam_star is None and lam_used is not None:
    lam_disp = lam_used
elif lam_star is not None:
    lam_disp = lam_star
else:
    lam_disp = None

lam = np.atleast_1d(lam_disp).astype(float)

if lam.size == 1:
    c4.metric("λ", f"{lam[0]:.3f}")
else:
    lc1, lc2, lc3 = st.columns(lam.size)
    for i, col in enumerate([lc1, lc2, lc3][: lam.size]):
        col.metric(f"λ{i + 1}", f"{lam[i]:.3f}")

c2.metric(
    "RMSE pricing error",
    (
        f"{float(summary.get('rmse_pricing_error', np.nan)):.4f}"
        if np.isfinite(float(summary.get("rmse_pricing_error", np.nan)))
        else "n/a"
    ),
)
c3.metric(
    "Objective",
    (
        f"{float(summary.get('objective_value', np.nan)):.4f}"
        if np.isfinite(float(summary.get("objective_value", np.nan)))
        else "n/a"
    ),
)
c1.metric("Success", str(bool(summary.get("success", True))))

residuals = summary.get("residuals", [])
if residuals:
    st.markdown("### Residuals (model vs market)")
    st.dataframe(pd.DataFrame(residuals), use_container_width=True)

st.markdown("### Model prices under current Q scenarios")
try:
    df_model = _build_price_table(scen_Q, instruments, short_rate=float(short_rate))
    st.dataframe(df_model, use_container_width=True)
except Exception as e:
    st.warning(f"Could not compute model prices table: {e}")

with st.expander("Full summary JSON"):
    st.json(summary)

st.subheader("Scenario set Q summary")
qQ = np.asarray(scen_Q.q_paths, dtype=float)
N, A, Hq = qQ.shape
d1, d2, d3, d4 = st.columns(4)
d1.metric("N scenarios", f"{N}")
d2.metric("Ages", f"{int(np.min(scen_Q.ages))} → {int(np.max(scen_Q.ages))}")
d3.metric("Horizon", f"{Hq} years")
d4.metric("Measure", str(scen_Q.metadata.get("measure", "Q")))

st.success("Next: go to **Pricing** once you're happy with Q scenarios.")
