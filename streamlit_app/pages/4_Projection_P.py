from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from pymort.pipeline import project_from_fitted_model

from assets.logo import add_logo_top_right

add_logo_top_right()
st.set_page_config(page_title="Projection P", page_icon="📈", layout="wide")
st.title("📈 Projections (P-measure)")
st.caption("Bootstrap + process risk mortality projections under the physical measure P.")

# -----------------------------
# Guard: need data loaded
# -----------------------------
if "m" not in st.session_state or st.session_state.get("m") is None:
    st.info("Load data first (page: Data Upload).")
    st.stop()

# ---- Get data (prefer sliced if available)
ages_slice = st.session_state.get("ages_slice")
years_slice = st.session_state.get("years_slice")
m_slice = st.session_state.get("m_slice")

ages = ages_slice if ages_slice is not None else st.session_state.get("ages")
years = years_slice if years_slice is not None else st.session_state.get("years")
m = m_slice if m_slice is not None else st.session_state.get("m")

ages = np.asarray(ages, dtype=float)
years = np.asarray(years, dtype=int)
m = np.asarray(m, dtype=float)

A, T = m.shape
year_min, year_max = int(years.min()), int(years.max())

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Projection settings")

    horizon = st.slider(
        "Projection horizon (years)",
        min_value=5,
        max_value=100,
        value=50,
        step=1,
    )

    st.divider()
    st.subheader("Bootstrap / process risk")
    # We let pipeline infer B_bootstrap & n_process from n_scenarios,
    # but expose overrides via bootstrap_kwargs.
    override_B = st.checkbox("Override bootstrap B", value=False)
    B_bootstrap = (
        st.number_input("B_bootstrap", min_value=1, max_value=300, value=70, step=20)
        if override_B
        else None
    )

    override_np = st.checkbox("Override n_process", value=False)
    n_process = (
        st.number_input("n_process", min_value=1, max_value=1000, value=300, step=100)
        if override_np
        else None
    )

    resample = st.selectbox("resample", ["year_block", "cell"], index=0)
    include_last = st.checkbox("include_last", value=True)
    seed = st.number_input("seed (optional)", min_value=0, value=0, step=1)

    st.divider()
    run = st.button("🚀 Build scenarios P", type="primary")

# -----------------------------
# Run projection
# -----------------------------
if run:
    slice_cfg = st.session_state.get("slice_cfg") or {}
    fit_age_max = slice_cfg.get("age_max")
    if fit_age_max is None:
        st.error("No slicing found. Go to Data Upload and click 'Apply slicing'.")
        st.stop()
    fit_age_max = int(fit_age_max)

    fitted = st.session_state.get("fitted_model")

    with st.spinner("Building P-measure scenarios..."):
        try:
            if fitted is None:
                st.error("No fitted model found. Go to **Fit & Model Selection** first.")
                st.stop()
            B_eff = int(B_bootstrap) if B_bootstrap is not None else 50
            nproc_eff = int(n_process) if n_process is not None else 200
            proj, scen_P, cache = project_from_fitted_model(
                fitted=fitted,
                B_bootstrap=B_eff,
                horizon=int(horizon),
                n_process=nproc_eff,
                seed=int(seed) if int(seed) > 0 else None,
                include_last=bool(include_last),
                resample=resample,
                ages_raw=st.session_state.get("ages"),
                years_raw=st.session_state.get("years"),
                m_raw=st.session_state.get("m"),
                plot_age_start=fit_age_max,
                plot_age_max=200,
            )
            scen_P.metadata["used_fitted_model_from_session"] = True
            scen_P.metadata["fitted_model_name"] = getattr(fitted, "name", None)

        except Exception as e:
            st.error(f"Projection failed: {e}")
            st.stop()

    # Store
    st.session_state["scen_P"] = scen_P
    st.session_state["proj_P"] = proj
    st.session_state["calibration_cache"] = cache
    for k in [
        "scen_Q",
        "calibration_summary",
        "prices",
        "pv_paths",
        "cf_paths",
        "hedge_result",
        "risk_report",
    ]:
        st.session_state[k] = None

    n_scenarios = B_eff * nproc_eff

    st.success("P scenarios built ✅")
    st.session_state["projP_cfg"] = {
        "requested_horizon": int(horizon),
        "n_scenarios": int(n_scenarios),
        "fitted_model_name": getattr(fitted, "name", None),
    }

# -----------------------------
# Display if available
# -----------------------------
scen_P = st.session_state.get("scen_P")
if scen_P is None:
    st.info("Configure parameters and click **Build scenarios P**.")
    st.stop()

q_paths = np.asarray(scen_P.q_paths, dtype=float)  # (N, A, H)
S_paths = np.asarray(scen_P.S_paths, dtype=float)  # (N, A, H)
years_proj = np.asarray(scen_P.years, dtype=int)

N, A_s, H = q_paths.shape

st.subheader("Scenario set summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("N scenarios", f"{N}")
c2.metric("Ages", f"{int(scen_P.ages.min())} → {int(scen_P.ages.max())}", f"{A_s} ages")
c3.metric("Horizon", f"{H} years")
c4.metric("Years", f"{int(years_proj.min())} → {int(years_proj.max())}")

# Choose an age to view
age_list = np.asarray(scen_P.ages, dtype=float)
default_age = int(round(np.median(age_list)))
age_view = st.slider(
    "Age to view",
    min_value=float(age_list.min()),
    max_value=float(age_list.max()),
    value=float(default_age),
    step=1.0,
)
age_idx = int(np.argmin(np.abs(age_list - age_view)))

# Quantiles for fan plots
qs = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])


def _fanplot(ax, x, y_paths, title, ylabel, *, xlim=None, ylim=None):
    # y_paths: (N, H) with possible NaNs in the tail
    qmat = np.nanquantile(y_paths, qs, axis=0)  # (len(qs), H)

    ax.plot(x, qmat[4], linewidth=2)  # median
    ax.fill_between(x, qmat[3], qmat[5], alpha=0.2)  # 25-75
    ax.fill_between(x, qmat[2], qmat[6], alpha=0.15)  # 10-90
    ax.fill_between(x, qmat[1], qmat[7], alpha=0.10)  # 5-95
    ax.fill_between(x, qmat[0], qmat[8], alpha=0.07)  # 1-99

    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])


def _cohort_survival_full_horizon_from_q(
    q_paths: np.ndarray,
    ages: np.ndarray,
    *,
    age0: float,
    horizon: int,
    age_fit_min: int = 80,
    age_fit_max: int = 95,
    m_floor: float = 1e-12,
) -> np.ndarray:
    """Build cohort survival S(t) for a cohort aged age0 at projection start,
    for the FULL horizon even if age0+k exceeds the last age in `ages`.

    Uses:
      - observed q_paths for ages within `ages`
      - Gompertz tail extrapolation beyond max(ages) per (scenario, time step),
        fitted on ages [age_fit_min, age_fit_max] using m ≈ -log(1-q).
    Returns: S_paths shape (N, horizon)
    """
    q_paths = np.asarray(q_paths, dtype=float)  # (N, A, H)
    ages = np.asarray(ages, dtype=float)
    N, A, H = q_paths.shape
    Ht = int(horizon)
    Ht = min(Ht, H)  # can't exceed available time steps

    # snap age0 to nearest age in grid
    a0_idx = int(np.argmin(np.abs(ages - float(age0))))
    age0_snap = ages[a0_idx]
    max_age = float(ages.max())

    # fit window indices (on available age grid)
    fit_mask = (ages >= float(age_fit_min)) & (ages <= float(age_fit_max))
    if not np.any(fit_mask):
        raise ValueError("Gompertz fit window not available in age grid.")
    x = ages[fit_mask]  # (Af,)
    Af = x.size
    x_mean = float(x.mean())
    x_var = float(np.mean((x - x_mean) ** 2))
    if x_var <= 0:
        raise ValueError("Degenerate Gompertz fit window (x_var=0).")

    # output
    q_diag = np.empty((N, Ht), dtype=float)

    for k in range(Ht):
        age_k = float(age0_snap + k)

        if age_k <= max_age:
            # inside grid
            idx = a0_idx + k
            if idx >= A:
                # safety: if we run out of age grid earlier than expected, use tail
                age_k = float(age0_snap + k)  # continue below to tail
            else:
                q_diag[:, k] = q_paths[:, idx, k]
                continue

        # tail: fit Gompertz on this time step k, per scenario, using ages in window
        q_fit = q_paths[:, fit_mask, k]  # (N, Af)

        # m ≈ -log(1-q) (stable for 1y)
        q_fit = np.clip(q_fit, 0.0, 1.0 - 1e-12)
        m_fit = -np.log(1.0 - q_fit)
        m_fit = np.clip(m_fit, m_floor, None)
        y = np.log(m_fit)  # (N, Af)

        # OLS slope/intercept per scenario (vectorized)
        y_mean = np.mean(y, axis=1, keepdims=True)  # (N,1)
        cov = np.mean((x[None, :] - x_mean) * (y - y_mean), axis=1)  # (N,)
        b = cov / x_var  # (N,)
        a = y_mean[:, 0] - b * x_mean  # (N,)

        # extrapolate m(age_k)
        m_k = np.exp(a + b * age_k)
        # convert back to q: q = 1 - exp(-m)
        q_k = 1.0 - np.exp(-m_k)
        q_diag[:, k] = np.clip(q_k, 0.0, 1.0 - 1e-12)

    # cohort survival
    S = np.cumprod(1.0 - q_diag, axis=1)
    return np.clip(S, 0.0, 1.0)


colL, colR = st.columns(2)

req_H = int(st.session_state.get("projP_cfg", {}).get("requested_horizon", H))
start_year = int(years_proj[0])

years_full = np.arange(start_year, start_year + req_H, dtype=int)
xlim_full = (years_full[0], years_full[-1])


def _pad_to_horizon(y2d: np.ndarray, H_target: int, *, fill_value=np.nan) -> np.ndarray:
    # y2d shape (N, H_current)
    N0, H0 = y2d.shape
    if H_target <= H0:
        out = y2d[:, :H_target].copy()
    else:
        out = np.full((N0, H_target), fill_value, dtype=float)
        out[:, :H0] = y2d
    return out


yQ = _pad_to_horizon(q_paths[:, age_idx, :], req_H)

# IMPORTANT: survival for plotting must be rebuilt to avoid NaN->0 drop at age>max_age
yS_plot = _cohort_survival_full_horizon_from_q(
    q_paths=q_paths,
    ages=age_list,
    age0=age_list[age_idx],
    horizon=req_H,
    age_fit_min=80,
    age_fit_max=int(st.session_state.get("slice_cfg", {}).get("age_max", 95)),
)

with colL:
    st.subheader("Survival fan (S)")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _fanplot(
        ax=ax,
        x=years_full,
        y_paths=yS_plot,
        title=f"Survival S(t) — age {int(age_list[age_idx])}",
        ylabel="S(t)",
        ylim=(0.0, 1.0),
    )
    st.pyplot(fig, use_container_width=True)

with colR:
    st.subheader("Mortality fan (q)")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _fanplot(
        ax=ax,
        x=years_full,
        y_paths=yQ,
        title=f"Mortality q(t) — age {int(age_list[age_idx])}",
        ylabel="q(t)",
    )
    st.pyplot(fig, use_container_width=True)

with st.expander("Metadata"):
    st.json(scen_P.metadata)

st.success("Next: go to **Risk-neutral (Q)** once you're happy with P scenarios.")
