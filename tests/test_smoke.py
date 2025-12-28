from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# make src/ importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pymort.analysis import (
    bootstrap_from_m,
    rmse_aic_bic,
    smooth_mortality_with_cpsplines,
    time_split_backtest_apc_m3,
    time_split_backtest_cbd_m5,
    time_split_backtest_cbd_m6,
    time_split_backtest_cbd_m7,
    time_split_backtest_lc_m1,
    time_split_backtest_lc_m2,
)
from pymort.analysis.projections import project_mortality_from_bootstrap
from pymort.lifetables import (
    load_m_from_excel,
    m_to_q,
    survival_from_q,
    validate_q,
    validate_survival_monotonic,
)
from pymort.models.apc_m3 import APCM3
from pymort.models.cbd_m5 import CBDM5, _logit
from pymort.models.cbd_m6 import CBDM6
from pymort.models.cbd_m7 import CBDM7
from pymort.models.lc_m1 import LCM1
from pymort.models.lc_m2 import LCM2
from pymort.pipeline import build_mortality_scenarios_for_pricing
from pymort.pricing.hedging import compute_min_variance_hedge
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


def main(argv: list[str] | None = None) -> int:
    default_excel = os.path.join(ROOT_DIR, "Data/data_france.xlsx")
    p = argparse.ArgumentParser(description="PYMORT smoke test (LC, CBD, CBD+cohort)")
    p.add_argument(
        "--excel", default=default_excel, help=f"Excel path (default: {default_excel})"
    )
    p.add_argument("--sex", default="Total", choices=["Total", "Female", "Male"])
    p.add_argument("--age-min", type=int, default=60)
    p.add_argument("--age-max", type=int, default=100)
    p.add_argument("--year-min", type=int, default=1970)
    p.add_argument(
        "--year-max",
        type=int,
        default=2019,
        help="set to >=2019 if you want 2016–2019 test",
    )
    p.add_argument(
        "--train-end",
        type=int,
        default=2015,
        help="last year in training set (e.g., 2015 → test starts 2016)",
    )
    p.add_argument(
        "--sims",
        type=int,
        default=100,
        help="Number of MC sims for k_t / kappa_t forecasts (default: 100)",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Random seed (None = random each run)"
    )
    p.add_argument(
        "--horizon", type=int, default=50, help="Forecast horizon (default: 50)"
    )
    p.add_argument(
        "--bootstraps",
        type=int,
        default=100,
        help="Number of bootstrap replications (default: 100)",
    )
    args, _unknown = p.parse_known_args(argv)
    if _unknown:
        print("Ignoring unknown args:", _unknown)

    # ================== Load data ===================

    print("=== Loading data ===")
    data = load_m_from_excel(
        args.excel,
        sex=args.sex,
        age_min=args.age_min,
        age_max=args.age_max,
        year_min=args.year_min,
        year_max=args.year_max,
    )
    ages, years, m = data["m"]
    print(
        f"m shape: {m.shape}  ages[{ages[0]}..{ages[-1]}]  years[{years[0]}..{years[-1]}]"
    )
    print(f"m min/max: {m.min():.6f} / {m.max():.6f}")

    # ================== LC backtest (explicit split) ===================

    if args.train_end < years[0] or args.train_end >= years[-1]:
        raise ValueError(f"--train-end must be within [{years[0]}, {years[-1]-1}]")
    te_start = args.train_end + 1
    if te_start not in years:
        raise ValueError(
            f"Test start year {te_start} not in data (have {years[0]}..{years[-1]}). "
            f"Load a file up to at least {te_start}."
        )

    tr_mask = years <= args.train_end
    te_mask = years >= te_start
    m_tr, m_te = m[:, tr_mask], m[:, te_mask]
    yrs_tr, yrs_te = years[tr_mask], years[te_mask]

    print("\n=== Backtest (explicit, Lee–Carter) ===")
    res = time_split_backtest_lc_m1(years, m, train_end=args.train_end)
    yrs_tr_bt = res["train_years"]
    yrs_te_bt = res["test_years"]
    rmse_log = res["rmse_log_forecast"]

    print(
        f"Train: {int(yrs_tr_bt[0])}–{int(yrs_tr_bt[-1])} | "
        f"Test: {int(yrs_te_bt[0])}–{int(yrs_te_bt[-1])}"
    )
    print(f"Out-of-sample RMSE (log m): {rmse_log:.6f}")

    # ================== Monte Carlo LC k_t + survival check ===================

    if args.sims > 0:
        print("\n=== Forecast k_t (Monte Carlo) & survival check (LC) ===")
        model = LCM1().fit(m_tr)  # fit on train only
        mu2, sigma2 = model.estimate_rw()
        print(f"Estimated (train) mu={mu2:.6f}, sigma={sigma2:.6f}")

        H_mc = args.horizon
        te_start = args.train_end + 1
        yrs_future = np.arange(te_start, te_start + H_mc)
        k_paths = model.simulate_k(horizon=H_mc, n_sims=args.sims, seed=args.seed)
        print(f"k_paths shape: {k_paths.shape} (n_sims, horizon={H_mc})")

        a, b = model.params.a, model.params.b
        ln_m_paths = (
            a[:, None][None, :, :] + b[:, None][None, :, :] * k_paths[:, None, :]
        )
        m_future_age0 = np.exp(ln_m_paths[:, 0, :])
        q = m_to_q(m_future_age0)
        validate_q(q)
        S = survival_from_q(q)
        validate_survival_monotonic(S)
        print("Survival monotonicity check: OK")

    # ================== LC Plots ===================

    print("\n=== LC diagnostic plots ===")

    age_star = 80
    i_age = (
        int(np.where(ages == age_star)[0][0]) if age_star in ages else len(ages) // 2
    )

    q_full = m_to_q(m)

    # ================== CBD Model (baseline) ===================

    print("\n=== CBD model: fit & basic plots ===")

    q_full = m_to_q(m)
    cbd = CBDM5().fit(q_full, ages)
    q_hat_full = cbd.predict_q()

    logit_q = _logit(q_full)
    logit_q_hat = _logit(q_hat_full)
    rmse_logit = float(np.sqrt(np.mean((logit_q - logit_q_hat) ** 2)))
    print(f"CBD in-sample RMSE (logit q): {rmse_logit:.6f}")

    params_cbd = cbd.params
    assert params_cbd is not None
    print("\n=== CBD RW parameters and Monte Carlo (kappa1) ===")
    mu1, sigma1, mu2, sigma2 = cbd.estimate_rw()
    print(f"kappa1: mu={mu1:.6f}, sigma={sigma1:.6f}")
    print(f"kappa2: mu={mu2:.6f}, sigma={sigma2:.6f}")
    # ================== CBD Cohort Model (M6) ===================

    print("\n=== CBD + cohort model: fit & comparison ===")

    cbd_co = CBDM6().fit(q_full, ages, years)
    q_hat_co = cbd_co.predict_q()

    logit_q_hat_co = _logit(q_hat_co)
    rmse_logit_co = float(np.sqrt(np.mean((logit_q - logit_q_hat_co) ** 2)))

    print(f"CBD (no cohort)  RMSE (logit q): {rmse_logit:.6f}")
    print(f"CBD + cohort     RMSE (logit q): {rmse_logit_co:.6f}")
    print("\n=== CBD + cohort: RW params and MC on kappa1_t ===")
    mu1_co, sigma1_co, mu2_co, sigma2_co = cbd_co.estimate_rw()
    print(f"[CBD+cohort] kappa1: mu={mu1_co:.6f}, sigma={sigma1_co:.6f}")
    print(f"[CBD+cohort] kappa2: mu={mu2_co:.6f}, sigma={sigma2_co:.6f}")

    # ================== CBD M7 (quadratic + cohort) ===================
    """
    m7 = CBDM7().fit(q_full, ages, years)
    q_hat_m7_full = m7.predict_q()  # (A, T)

    # --------- Plot 1 : observed vs CBD vs CBD+cohort vs M7 ---------
    q_obs_age = q_full[i_age, :]  # observed
    q_cbd_age = q_hat_full[i_age, :]  # CBD simple
    q_cbd_co_age = q_hat_co[i_age, :]  # CBD + cohort (M6)
    q_m7_age = q_hat_m7_full[i_age, :]  # CBD M7

    # ========== 1) Fits complets pour lignes historiques (comparaison) ==========
    
    par_lc_full = fit_lee_carter(m)
    q_lc_full = m_to_q(np.exp(reconstruct_log_m(par_lc_full)))
    q_lc_age = q_lc_full[i_age, :]

    lcm2_full = LCM2().fit(m, ages, years)
    q_lcm2_full = m_to_q(lcm2_full.predict_m())
    q_lcm2_age = q_lcm2_full[i_age, :]

    apc_full = APCM3().fit(m, ages, years)
    q_apc_full = m_to_q(apc_full.predict_m())
    q_apc_age = q_apc_full[i_age, :]

    # ========== 2) Bootstrap des paramètres du modèle M7 ==========
    B = args.bootstraps  # ex: 500
    seed_bs = args.seed

    bs_m7 = bootstrap_from_m(
        CBDM7,
        m,
        ages,
        years,
        B=B,
        seed=seed_bs,
        resample="year_block",
    )

    # ========== 3) Projections futures (param uncertainty × process uncertainty) ==========
    horizon_m7 = args.horizon
    n_process = args.sims
    seed_proj = None if args.seed is None else args.seed + 123

    proj_m7 = project_mortality_from_bootstrap(
        CBDM7,
        ages,
        years,
        m,
        bs_m7,
        horizon=horizon_m7,
        n_process=n_process,
        seed=seed_proj,
        include_last=True,
    )

    q_paths_all = proj_m7.q_paths
    years_forecast = proj_m7.years  # (H,)

    q_paths_m7 = q_paths_all[:, i_age, :]

    q_low_m7, q_med_m7, q_high_m7 = np.percentile(q_paths_m7, [5, 50, 95], axis=0)

    # ========== 4) Historique OBS pour âge i_age ==========
    split_year = int(years[-1])

    q_obs_age = q_full[i_age, :]

    if q_full.shape[1] > len(years):
        extra_years = years[len(years) :]
        extra_obs = q_obs_age[len(years) :]
    else:
        extra_years = years_forecast
        extra_obs = np.full_like(extra_years, np.nan, dtype=float)

    years_obs_extended = np.concatenate([years, extra_years])
    q_obs_extended = np.concatenate([q_obs_age, extra_obs])

    q_m7_age = q_hat_m7_full[i_age, :]
    q_cbd_age = q_hat_full[i_age, :]
    q_cbd_co_age = q_hat_co[i_age, :]

    # ========== 5) Plot ==========
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    ax.plot(
        years_obs_extended,
        q_obs_extended,
        color="black",
        linewidth=1,
        label=f"Observed q(age={ages[i_age]})",
    )

    ax.plot(
        years, q_lc_age, "--", color="tab:brown", linewidth=1, label="LC M1 fitted q"
    )
    ax.plot(
        years, q_lcm2_age, "--", color="tab:cyan", linewidth=1, label="LC M2 fitted q"
    )
    ax.plot(
        years, q_apc_age, "--", color="tab:gray", linewidth=1, label="APC M3 fitted q"
    )
    ax.plot(
        years, q_cbd_age, "--", color="tab:orange", linewidth=1, label="CBD M5 fitted q"
    )
    ax.plot(
        years,
        q_cbd_co_age,
        "--",
        color="tab:purple",
        linewidth=1,
        label="CBD M6 fitted q",
    )
    ax.plot(
        years, q_m7_age, "--", color="tab:green", linewidth=1, label="CBD M7 fitted q"
    )

    ax.axvspan(split_year, years_forecast[-1], color="grey", alpha=0.08)
    ax.axvline(split_year, color="black", linestyle="--", linewidth=1)

    N_total = q_paths_m7.shape[0]
    n_plot = min(300, N_total)  # évite de saturer le graph
    idx_plot = np.linspace(0, N_total - 1, n_plot, dtype=int)

    for i in idx_plot:
        ax.plot(
            years_forecast,
            q_paths_m7[i, :],
            color="tab:blue",
            alpha=0.06,
            linewidth=0.7,
        )

    ax.fill_between(
        years_forecast,
        q_low_m7,
        q_high_m7,
        alpha=0.25,
        label="90% band (M7 bootstrap forecast)",
    )
    ax.plot(
        years_forecast,
        q_med_m7,
        color="tab:red",
        linewidth=1.5,
        label="Median forecast (M7 bootstrap)",
    )

    ax.set_title(
        f"CBD M7 forecast with bootstrap * RW process risk (age={ages[i_age]})"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Death probability q")
    ax.set_ylim(0, max(q_obs_age.max(), q_high_m7.max()) * 1.1)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()
    """
    # ------------------- CPsplines smoothing (fit once) -------------------
    print("\n=== CPsplines smoothing (for comparison tables) ===")
    cp_res = smooth_mortality_with_cpsplines(
        m,
        ages,
        years,
        k=None,
        horizon=args.horizon,
        verbose=True,
    )
    m_cp_fit = cp_res["m_fitted"]
    m_cp_forecast = cp_res["m_forecast"]
    years_cp_forecast = cp_res["years_forecast"]
    cp_model = cp_res["model"]
    print("CPsplines fitted m shape:", m_cp_fit.shape)
    print("CPsplines forecast m shape:", m_cp_forecast.shape)
    print("Forecast years (CPsplines):", years_cp_forecast[:5], "...")

    # ------------------- RMSE / AIC / BIC + forecast table --------------------------
    def collect_metrics(
        label: str,
        m_fit_surface: np.ndarray,
        m_eval_surface: np.ndarray,
    ) -> list[dict[str, float | str]]:
        """
        Compute in-sample (RMSE, AIC, BIC) and forecast RMSE (log m / logit q)
        for one data source, using the backtest helpers in pymort.validation.

        m_fit_surface  : surface utilisée pour fitter les modèles
        m_eval_surface : surface de vérité pour les RMSE in-sample / AIC/BIC
                     (pour le scénario smoothed, c'est le m "raw").
        """
        A = len(ages)
        T = len(years)

        q_eval_surface = m_to_q(m_eval_surface)
        logit_true = _logit(q_eval_surface)

        rows: list[dict[str, float | str]] = []

        # === LC M1 ===
        lcm1_full = LCM1().fit(m_fit_surface)
        ln_hat_lc = lcm1_full.predict_log_m()
        q_hat_lc = m_to_q(np.exp(ln_hat_lc))

        rmse_lc_logm_full = float(
            np.sqrt(np.mean((np.log(m_eval_surface) - ln_hat_lc) ** 2))
        )
        rmse_lc_logitq, aic_lc, bic_lc = rmse_aic_bic(
            logit_true,
            _logit(q_hat_lc),
            n_params=2 * A + T,
        )

        bt_lc = time_split_backtest_lc_m1(
            years=years,
            m=m_fit_surface,
            train_end=args.train_end,
        )
        start_te = int(bt_lc["test_years"][0])
        end_te = int(bt_lc["test_years"][-1])

        rmse_lc_forecast_logm = bt_lc["rmse_log_forecast"]
        rmse_lc_forecast_logit = bt_lc["rmse_logit_forecast"]

        rows.append(
            {
                "Data": label,
                "Model": "LC (M1)",
                "RMSE in-sample (log m)": rmse_lc_logm_full,
                "RMSE in-sample (logit q)": rmse_lc_logitq,
                f"RMSE forecast {start_te}–{end_te} (log m)": rmse_lc_forecast_logm,
                f"RMSE forecast {start_te}–{end_te} (logit q)": rmse_lc_forecast_logit,
                "AIC": aic_lc,
                "BIC": bic_lc,
            }
        )

        # === LC M2 ===
        lcm2_full = LCM2().fit(m_fit_surface, ages, years)
        params_lcm2 = lcm2_full.params
        assert params_lcm2 is not None
        m_hat_lcm2 = lcm2_full.predict_m()
        q_hat_lcm2 = m_to_q(m_hat_lcm2)

        rmse_lcm2_logm_full = float(
            np.sqrt(np.mean((np.log(m_eval_surface) - np.log(m_hat_lcm2)) ** 2))
        )
        rmse_lcm2_logitq, aic_lcm2, bic_lcm2 = rmse_aic_bic(
            logit_true,
            _logit(q_hat_lcm2),
            n_params=2 * A + T + len(params_lcm2.cohorts),
        )

        bt_lcm2 = time_split_backtest_lc_m2(
            ages=ages,
            years=years,
            m=m_fit_surface,
            train_end=args.train_end,
        )
        rmse_lcm2_forecast_logm = bt_lcm2["rmse_log_forecast"]
        rmse_lcm2_forecast_logit = bt_lcm2["rmse_logit_forecast"]

        rows.append(
            {
                "Data": label,
                "Model": "LC+cohort (M2)",
                "RMSE in-sample (log m)": rmse_lcm2_logm_full,
                "RMSE in-sample (logit q)": rmse_lcm2_logitq,
                f"RMSE forecast {start_te}–{end_te} (log m)": rmse_lcm2_forecast_logm,
                f"RMSE forecast {start_te}–{end_te} (logit q)": rmse_lcm2_forecast_logit,
                "AIC": aic_lcm2,
                "BIC": bic_lcm2,
            }
        )

        # === APC M3 ===
        apc_full = APCM3().fit(m_fit_surface, ages, years)
        params_apc = apc_full.params
        assert params_apc is not None
        m_hat_apc = apc_full.predict_m()
        q_hat_apc = m_to_q(m_hat_apc)

        rmse_apc_logm_full = float(
            np.sqrt(np.mean((np.log(m_eval_surface) - np.log(m_hat_apc)) ** 2))
        )
        rmse_apc_logitq, aic_apc, bic_apc = rmse_aic_bic(
            logit_true,
            _logit(q_hat_apc),
            n_params=A + T + len(params_apc.cohorts),
        )

        bt_apc = time_split_backtest_apc_m3(
            ages=ages,
            years=years,
            m=m_fit_surface,
            train_end=args.train_end,
        )
        rmse_apc_forecast_logm = bt_apc["rmse_log_forecast"]
        rmse_apc_forecast_logit = bt_apc["rmse_logit_forecast"]

        rows.append(
            {
                "Data": label,
                "Model": "APC (M3)",
                "RMSE in-sample (log m)": rmse_apc_logm_full,
                "RMSE in-sample (logit q)": rmse_apc_logitq,
                f"RMSE forecast {start_te}–{end_te} (log m)": rmse_apc_forecast_logm,
                f"RMSE forecast {start_te}–{end_te} (logit q)": rmse_apc_forecast_logit,
                "AIC": aic_apc,
                "BIC": bic_apc,
            }
        )

        q_fit_surface = m_to_q(m_fit_surface)

        # === CBD M5 ===
        cbd_m5_full = CBDM5().fit(q_fit_surface, ages)
        q_hat_m5 = cbd_m5_full.predict_q()
        rmse_m5_logitq, aic_m5, bic_m5 = rmse_aic_bic(
            logit_true,
            _logit(q_hat_m5),
            n_params=2 * T,
        )

        bt_m5 = time_split_backtest_cbd_m5(
            ages=ages,
            years=years,
            q=q_fit_surface,
            train_end=args.train_end,
        )
        rmse_m5_forecast_logit = bt_m5["rmse_logit_forecast"]

        rows.append(
            {
                "Data": label,
                "Model": "CBD (M5)",
                "RMSE in-sample (log m)": np.nan,
                "RMSE in-sample (logit q)": rmse_m5_logitq,
                f"RMSE forecast {start_te}–{end_te} (log m)": np.nan,
                f"RMSE forecast {start_te}–{end_te} (logit q)": rmse_m5_forecast_logit,
                "AIC": aic_m5,
                "BIC": bic_m5,
            }
        )

        # === CBD M6 ===
        cbd_m6_full = CBDM6().fit(q_fit_surface, ages, years)
        params_m6_full = cbd_m6_full.params
        assert params_m6_full is not None
        q_hat_m6 = cbd_m6_full.predict_q()
        rmse_m6_logitq, aic_m6, bic_m6 = rmse_aic_bic(
            logit_true,
            _logit(q_hat_m6),
            n_params=2 * T + len(params_m6_full.cohorts),
        )

        bt_m6 = time_split_backtest_cbd_m6(
            ages=ages,
            years=years,
            q=q_fit_surface,
            train_end=args.train_end,
        )
        rmse_m6_forecast_logit = bt_m6["rmse_logit_forecast"]

        rows.append(
            {
                "Data": label,
                "Model": "CBD+cohort (M6)",
                "RMSE in-sample (log m)": np.nan,
                "RMSE in-sample (logit q)": rmse_m6_logitq,
                f"RMSE forecast {start_te}–{end_te} (log m)": np.nan,
                f"RMSE forecast {start_te}–{end_te} (logit q)": rmse_m6_forecast_logit,
                "AIC": aic_m6,
                "BIC": bic_m6,
            }
        )

        # === CBD M7 ===
        cbd_m7_full = CBDM7().fit(q_fit_surface, ages, years)
        params_m7_full = cbd_m7_full.params
        assert params_m7_full is not None
        q_hat_m7 = cbd_m7_full.predict_q()
        rmse_m7_logitq, aic_m7, bic_m7 = rmse_aic_bic(
            logit_true,
            _logit(q_hat_m7),
            n_params=3 * T + len(params_m7_full.cohorts),
        )

        bt_m7 = time_split_backtest_cbd_m7(
            ages=ages,
            years=years,
            q=q_fit_surface,
            train_end=args.train_end,
        )
        rmse_m7_forecast_logit = bt_m7["rmse_logit_forecast"]

        rows.append(
            {
                "Data": label,
                "Model": "CBD+quadratic+cohort (M7)",
                "RMSE in-sample (log m)": np.nan,
                "RMSE in-sample (logit q)": rmse_m7_logitq,
                f"RMSE forecast {start_te}–{end_te} (log m)": np.nan,
                f"RMSE forecast {start_te}–{end_te} (logit q)": rmse_m7_forecast_logit,
                "AIC": aic_m7,
                "BIC": bic_m7,
            }
        )

        return rows

    scenarios = [
        ("Observed (raw)", m, m),
        ("Smoothed (CPsplines fit, eval on raw)", m_cp_fit, m),
    ]

    all_rows: list[dict[str, float | str]] = []
    for lbl, m_fit_surface, m_eval_surface in scenarios:
        all_rows.extend(collect_metrics(lbl, m_fit_surface, m_eval_surface))

    comparison_df = pd.DataFrame(all_rows)
    print(
        "\n=== Model comparison (RMSE in-sample / forecast, AIC, BIC — raw vs smoothed) ===\n"
    )
    print(comparison_df.to_string(index=False))

    # ------------------------- BOOTSTRAP --------------------------------
    comp_map = {
        2: ["mu", "sigma"],
        4: ["mu1", "sigma1", "mu2", "sigma2"],
        6: ["mu1", "sigma1", "mu2", "sigma2", "mu3", "sigma3"],
    }
    bootstrap_rows: list[dict[str, str]] = []
    bootstrap_results: dict[tuple[str, str], object] = {}

    for lbl, m_fit_surface, _m_eval_surface in scenarios:
        for model_label, model_cls in [
            ("LC (M1)", LCM1),
            ("LC+cohort (M2)", LCM2),
            ("APC (M3)", APCM3),
            ("CBD (M5)", CBDM5),
            ("CBD+cohort (M6)", CBDM6),
            ("CBD+quadratic+cohort (M7)", CBDM7),
        ]:
            bs_res = bootstrap_from_m(
                model_cls,
                m_fit_surface,
                ages,
                years,
                B=args.bootstraps,
                seed=args.seed,
                resample="year_block",
            )
            bootstrap_results[(lbl, model_label)] = bs_res
            names = comp_map.get(bs_res.mu_sigma.shape[1], [])
            means = bs_res.mu_sigma.mean(axis=0)
            stds = bs_res.mu_sigma.std(axis=0)
            summary_parts = []
            for name, mean_v, std_v in zip(names, means, stds):
                summary_parts.append(f"{name}={mean_v:.4f}±{std_v:.4f}")
            summary = ", ".join(summary_parts) if summary_parts else "n/a"
            bootstrap_rows.append(
                {
                    "Data": lbl,
                    "Model": model_label,
                    f"Bootstrap drift/vol (B={args.bootstraps})": summary,
                }
            )

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    print("\n=== Bootstrap drift/vol summary ===\n")
    print(bootstrap_df.to_string(index=False))

    """
    # Quick visuals for two reference models on raw data
    boot_lcm2 = bootstrap_results.get(("Observed (raw)", "LC+cohort (M2)"))
    boot_m7 = bootstrap_results.get(("Observed (raw)", "CBD+quadratic+cohort (M7)"))
    if boot_lcm2 is not None:
        plt.hist(boot_lcm2.mu_sigma[:, 0], bins=30)
        plt.title("LCM2 – mu (bootstrap)")
        plt.show()
    if boot_m7 is not None:
        plt.hist(boot_m7.mu_sigma[:, 0], bins=30)
        plt.title("M7 – mu1 (bootstrap)")
        plt.show()
    """
    # ---------------------------------------------------------

    # ================== Smoothing diagnostic plots (raw vs CPsplines) ===================

    print("\n=== CPsplines smoothing diagnostics (external cpsplines) ===")
    m_hist = m
    years_hist = years
    """
    # -------- Plot 1 : Observed vs CPsplines-smoothed m pour un âge --------
    age_star = 80
    if age_star in ages:
        age_idx = int(np.where(ages == age_star)[0][0])
    else:
        age_idx = len(ages) // 2

    years_obs = years_hist
    m_obs_row = m_hist[age_idx, :]
    m_fit_row = m_cp_fit[age_idx, :]

    plt.figure(figsize=(10, 4))
    plt.scatter(years_obs, m_obs_row, s=30, label="Observed m")
    plt.plot(years_obs, m_fit_row, lw=3, label="CPsplines smoothed m")
    plt.yscale("log")
    plt.xlabel("Year")
    plt.ylabel("Central death rate m (log scale)")
    plt.legend()
    plt.title(f"Observed vs CPsplines smoothed m (age={ages[age_idx]})")
    plt.tight_layout()
    plt.show()

    # -------- Plot 2 : Heatmaps observed vs smoothed --------
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    im0 = ax[0].imshow(
        np.log(m_hist),
        aspect="auto",
        origin="lower",
        extent=[years_hist[0], years_hist[-1], ages[0], ages[-1]],
    )
    ax[0].set_title("Observed log m")
    ax[0].set_ylabel("Age")
    ax[0].set_xlabel("Year")

    im1 = ax[1].imshow(
        np.log(m_cp_fit),
        aspect="auto",
        origin="lower",
        extent=[years_hist[0], years_hist[-1], ages[0], ages[-1]],
    )
    ax[1].set_title("CPsplines smoothed log m")
    ax[1].set_xlabel("Year")

    plt.colorbar(im1, ax=ax, fraction=0.025, pad=0.02)
    plt.tight_layout()
    plt.show()
    """

    print("\n=== CPsplines internal model information ===")
    print("Selected smoothing parameters (sp):", cp_model.sp_args)
    print("Degrees:", cp_model.deg)
    print("Order of differences:", cp_model.ord_d)
    print("Internal number of basis (k):", cp_model.k)

    # ================== Forecast M2 & M7 (CPsplines -> bootstrap -> projections -> plot) ===================
    print(
        "\n=== Forecast fan chart for LC+cohort (M2) and CBD M7 on CPsplines (age_star=80) ==="
    )

    if args.sims > 0 and args.bootstraps > 0:
        if age_star in ages:
            idx_age = int(np.where(ages == age_star)[0][0])
        else:
            idx_age = len(ages) // 2
            print(
                f"Warning: age_star={age_star} not in ages, using ages[{idx_age}]={ages[idx_age]} instead."
            )
            age_star = int(ages[idx_age])

        years_obs = years
        last_obs_year = int(years_obs[-1])

        # ---------- 1) Observed data in q ----------
        q_obs_age = m_to_q(m[idx_age, :])  # (T,)

        # ---------- 2) Central fits on CPsplines (historical part) ----------
        # M2: LC+cohort on smoothed m
        lcm2_cp = LCM2().fit(m_cp_fit, ages, years)
        m_hat_lcm2_cp = lcm2_cp.predict_m()
        q_hat_lcm2_cp = m_to_q(m_hat_lcm2_cp)
        q_lcm2_hist_age = q_hat_lcm2_cp[idx_age, :]

        # M7: CBD+quadratic+cohort on smoothed (use q from m_cp_fit)
        q_cp_fit = m_to_q(m_cp_fit)
        m7_cp = CBDM7().fit(q_cp_fit, ages, years)
        q_hat_m7_cp = m7_cp.predict_q()
        q_m7_hist_age = q_hat_m7_cp[idx_age, :]

        # ---------- 3) Bootstrap on CPsplines-fitted surfaces ----------
        print(
            f"Bootstrap on CPsplines data: B={args.bootstraps}, n_process={args.sims}"
        )

        # LC+cohort (M2) bootstrap (on smoothed m)
        bs_lcm2_cp = bootstrap_from_m(
            LCM2,
            m_cp_fit,
            ages,
            years,
            B=args.bootstraps,
            seed=args.seed,
            resample="year_block",
        )

        # CBD M7 bootstrap (on smoothed m, conversion interne vers q)
        bs_m7_cp = bootstrap_from_m(
            CBDM7,
            m_cp_fit,
            ages,
            years,
            B=args.bootstraps,
            seed=None,
            resample="year_block",
        )

        # ---------- 4) Stochastic projections (param + process uncertainty) ----------
        horizon = args.horizon
        seed_proj_lcm2 = None
        seed_proj_m7 = None

        proj_lcm2 = project_mortality_from_bootstrap(
            model_cls=LCM2,
            ages=ages,
            years=years,
            m=m_cp_fit,
            bootstrap_result=bs_lcm2_cp,
            horizon=horizon,
            n_process=args.sims,
            seed=seed_proj_lcm2,
            include_last=True,
        )

        proj_m7 = project_mortality_from_bootstrap(
            model_cls=CBDM7,
            ages=ages,
            years=years,
            m=m_cp_fit,
            bootstrap_result=bs_m7_cp,
            horizon=horizon,
            n_process=args.sims,
            seed=seed_proj_m7,
            include_last=True,
        )

        years_future = proj_lcm2.years  # (H,)
        # Sanity check: both projections should have same future years
        if not np.array_equal(years_future, proj_m7.years):
            print(
                "Warning: proj_lcm2.years and proj_m7.years differ; using proj_lcm2.years for plotting."
            )

        # ---------- 5) Extract age_star paths and quantiles (in q) ----------
        # LC+cohort (M2)
        q_paths_lcm2 = proj_lcm2.q_paths[:, idx_age, :]  # (N, H)
        q_lcm2_med = np.percentile(q_paths_lcm2, 50, axis=0)
        q_lcm2_low = np.percentile(q_paths_lcm2, 2.5, axis=0)
        q_lcm2_high = np.percentile(q_paths_lcm2, 97.5, axis=0)

        # CBD M7
        q_paths_m7 = proj_m7.q_paths[:, idx_age, :]  # (N, H)
        q_m7_med = np.percentile(q_paths_m7, 50, axis=0)
        q_m7_low = np.percentile(q_paths_m7, 2.5, axis=0)
        q_m7_high = np.percentile(q_paths_m7, 97.5, axis=0)

        # ---------- 6) Plot: observed vs M2 / M7 (fit on CPsplines) + fan charts ----------
        plt.figure(figsize=(11, 5), dpi=150)

        # Observed q
        plt.plot(
            years_obs,
            q_obs_age,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Observed q(x={age_star})",
        )

        # Historical fitted (on CPsplines) for M2 & M7
        plt.plot(
            years_obs,
            q_lcm2_hist_age,
            color="tab:blue",
            linewidth=2,
            label="LC+cohort (M2) fitted on CPsplines",
        )
        plt.plot(
            years_obs,
            q_m7_hist_age,
            color="tab:red",
            linewidth=2,
            label="CBD M7 fitted on CPsplines",
        )

        # Vertical line at last observed year
        plt.axvline(last_obs_year, color="grey", linestyle=":", linewidth=1)

        # Fan chart for M2
        plt.fill_between(
            years_future,
            q_lcm2_low,
            q_lcm2_high,
            alpha=0.20,
            color="tab:blue",
            label=f"M2 forecast 95% band (CPsplines, B={args.bootstraps})",
        )
        plt.plot(
            years_future,
            q_lcm2_med,
            color="tab:blue",
            linestyle="--",
            linewidth=2,
            label="M2 median forecast",
        )

        # Fan chart for M7
        plt.fill_between(
            years_future,
            q_m7_low,
            q_m7_high,
            alpha=0.20,
            color="tab:red",
            label=f"M7 forecast 95% band (CPsplines, B={args.bootstraps})",
        )
        plt.plot(
            years_future,
            q_m7_med,
            color="tab:red",
            linestyle="--",
            linewidth=2,
            label="M7 median forecast",
        )

        plt.xlabel("Year")
        plt.ylabel("Death probability q")
        plt.title(
            f"Observed vs CPsplines-fitted M2 & M7 with bootstrap forecast\n(age={age_star})"
        )
        plt.ylim(bottom=0)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
    else:
        print(
            "Skipping M2/M7 forecast fan chart: need positive --sims and --bootstraps."
        )

    fitted_best, proj, scen_set = build_mortality_scenarios_for_pricing(
        ages=ages,
        years=years,
        m=m,
        train_end=args.train_end,
        selection_metric="logit_q",
        cpsplines_kwargs={"k": None, "horizon": 0, "verbose": False},
        B_bootstrap=args.bootstraps,
        horizon=args.horizon,
        n_process=args.sims,
        seed=args.seed,
        include_last=True,
    )

    print("Selected model for pricing:", fitted_best.name)
    print("Scenarios shape (q_paths):", scen_set.q_paths.shape)
    print("Survival shape (S_paths):", scen_set.S_paths.shape)

    # === Sanity check de la pipeline pricing end-to-end ===

    assert fitted_best.name in {
        "LCM1",
        "LCM2",
        "APCM3",
        "CBDM5",
        "CBDM6",
        "CBDM7",
    }, f"Unexpected selected model for pricing: {fitted_best.name}"

    A = len(ages)
    N_expected = args.bootstraps * args.sims
    H_out = proj.q_paths.shape[2]

    assert proj.q_paths.shape == (N_expected, A, H_out), (
        "proj.q_paths has wrong shape: "
        f"{proj.q_paths.shape} != ({N_expected}, {A}, {H_out})"
    )

    assert scen_set.q_paths.shape == (N_expected, A, H_out), (
        "scen_set.q_paths has wrong shape: "
        f"{scen_set.q_paths.shape} != ({N_expected}, {A}, {H_out})"
    )
    assert scen_set.S_paths.shape == (N_expected, A, H_out), (
        "scen_set.S_paths has wrong shape: "
        f"{scen_set.S_paths.shape} != ({N_expected}, {A}, {H_out})"
    )
    assert scen_set.years.shape[0] == H_out, (
        "Length of scen_set.years inconsistent with horizon: "
        f"{scen_set.years.shape[0]} != {H_out}"
    )

    validate_q(scen_set.q_paths)
    validate_survival_monotonic(scen_set.S_paths)

    diff_S = np.diff(scen_set.S_paths, axis=2)  # (N, A, H-1)
    mask = np.isfinite(diff_S)
    assert np.all(
        diff_S[mask] <= 1e-10
    ), "Survival curves must be non-increasing over time (ignoring NaNs)."

    print("End-to-end pricing pipeline checks: OK ✅")

    # ================== Longevity bond pricing smoke test ===================

    print("\n=== Longevity bond pricing smoke test ===")

    # On prend un cohort age raisonnable dans la grille
    issue_age = 80
    if issue_age in ages:
        issue_age_used = issue_age
    else:
        # si 65 n'est pas dispo, on prend l'âge le plus proche
        issue_age_used = float(ages[np.argmin(np.abs(ages - issue_age))])
        print(
            f"Warning: issue_age={issue_age} not in ages; using nearest age {issue_age_used}."
        )

    spec = LongevityBondSpec(
        issue_age=issue_age_used,
        notional=100.0,
        include_principal=True,
        maturity_years=min(20, scen_set.horizon()),  # au max 20 ans ou horizon dispo
    )

    res = price_simple_longevity_bond(
        scen_set=scen_set,
        spec=spec,
        short_rate=0.02,  # taux plat 2% en continu
    )

    price = res["price"]
    pv_paths = res["pv_paths"]
    df = res["discount_factors"]

    print(f"Longevity bond price (MC): {price:.4f}")
    print(
        f"PV paths: mean={pv_paths.mean():.4f}, "
        f"std={pv_paths.std():.4f}, "
        f"min={pv_paths.min():.4f}, max={pv_paths.max():.4f}"
    )

    # === Assertions smoke-test ===

    # 1) Prix strictement positif
    assert price > 0.0, "Longevity bond price should be positive."

    # 2) Nombre de chemins PV cohérent avec le nombre de scénarios
    assert pv_paths.shape[0] == scen_set.n_scenarios(), (
        f"pv_paths has wrong length: {pv_paths.shape[0]} "
        f"!= {scen_set.n_scenarios()} (n_scenarios)."
    )

    # 3) Longueur des discount factors cohérente avec la maturité
    df = np.asarray(res["discount_factors"])

    H_bond = spec.maturity_years or scen_set.horizon()

    if df.ndim == 1:
        assert (
            df.shape[0] == H_bond
        ), f"discount_factors length mismatch: {df.shape[0]} != {H_bond}"
    elif df.ndim == 2:
        # accepte (H,) en colonne ou en ligne, ou (N,H) si DF par scénario
        assert (df.shape[0] == H_bond) or (
            df.shape[1] == H_bond
        ), f"discount_factors shape mismatch: {df.shape} does not match H_bond={H_bond}"
    else:
        raise AssertionError(
            f"discount_factors has unexpected ndim={df.ndim}, shape={df.shape}"
        )

    # 4) Survie moyenne de la cohorte décroissante (sanity check produit)
    age_idx = res["age_index"]
    S_age = scen_set.S_paths[:, age_idx, :H_bond]  # (N, H_bond)
    S_mean = S_age.mean(axis=0, keepdims=True)  # (1, H_bond)

    validate_survival_monotonic(S_mean)

    diff_S_mean = np.diff(S_mean, axis=1)
    assert np.all(
        diff_S_mean <= 1e-10
    ), "Mean survival curve for the cohort must be non-increasing."

    # ================== Mortality derivatives pricing smoke tests ===================

    print(
        "\n=== Mortality derivatives smoke tests (q-forward, s-forward, survivor swap) ==="
    )

    # On réutilise la cohorte issue_age_used et la maturité max disponible
    max_T = min(20, scen_set.horizon())
    if max_T <= 1:
        print("Horizon trop court pour tester les dérivés : skipping.")
    else:
        # ---------- Q-forward ATM (strike=None) ----------
        qf_spec_atm = QForwardSpec(
            age=issue_age_used,
            maturity_years=max_T,  # dernier point de la courbe S_x(t)
            notional=100.0,
            # strike=None -> ATM
        )
        qf_res_atm = price_q_forward(
            scen_set=scen_set,
            spec=qf_spec_atm,
            short_rate=0.02,
        )

        qf_price_atm = qf_res_atm["price"]
        qf_pv_paths = qf_res_atm["pv_paths"]
        qf_strike_atm = qf_res_atm["strike"]

        print(f"Q-forward ATM price: {qf_price_atm:.6f}, strike={qf_strike_atm:.6f}")

        # 1) ATM -> prix proche de 0 (tolérance MC)
        tol_qf = 0.05 * qf_spec_atm.notional
        assert abs(qf_price_atm) < tol_qf, (
            f"ATM q-forward price too far from 0: {qf_price_atm:.6f}, "
            f"tol={tol_qf:.6f}"
        )
        # 2) Nombre de chemins cohérent
        assert qf_pv_paths.shape[0] == scen_set.n_scenarios(), (
            f"q-forward pv_paths length {qf_pv_paths.shape[0]} "
            f"!= {scen_set.n_scenarios()} (n_scenarios)."
        )

        # ---------- Q-forward off-market (strike plus élevé) ----------
        qf_spec_rich = QForwardSpec(
            age=issue_age_used,
            maturity_years=max_T,
            notional=100.0,
            strike=qf_strike_atm + 0.05,  # on fixe un strike plus élevé
        )
        qf_res_rich = price_q_forward(
            scen_set=scen_set,
            spec=qf_spec_rich,
            short_rate=0.02,
        )
        qf_price_rich = qf_res_rich["price"]
        print(f"Q-forward off-market price (higher strike): {qf_price_rich:.6f}")

        # On vérifie juste que le prix change bien quand on bouge le strike
        assert (
            qf_price_rich != qf_price_atm
        ), "Q-forward price should move when strike changes."

        # ---------- S-forward ATM (s-index forward sur la survie) ----------
        sf_spec_atm = SForwardSpec(
            age=issue_age_used,
            maturity_years=max_T,
            notional=100.0,
            # strike=None -> ATM (zero value)
        )
        sf_res_atm = price_s_forward(
            scen_set=scen_set,
            spec=sf_spec_atm,
            short_rate=0.02,
        )
        sf_price_atm = sf_res_atm["price"]
        sf_pv_paths = sf_res_atm["pv_paths"]
        sf_strike_atm = sf_res_atm["strike"]

        print(f"S-forward ATM price: {sf_price_atm:.6f}, strike={sf_strike_atm:.6f}")

        tol_sf = 0.05 * sf_spec_atm.notional
        assert abs(sf_price_atm) < tol_sf, (
            f"ATM s-forward price too far from 0: {sf_price_atm:.6f}, "
            f"tol={tol_sf:.6f}"
        )
        assert sf_pv_paths.shape[0] == scen_set.n_scenarios(), (
            f"s-forward pv_paths length {sf_pv_paths.shape[0]} "
            f"!= {scen_set.n_scenarios()} (n_scenarios)."
        )

        # ---------- S-forward off-market ----------
        sf_spec_rich = SForwardSpec(
            age=issue_age_used,
            maturity_years=max_T,
            notional=100.0,
            strike=sf_strike_atm - 0.05,  # strike plus bas
        )
        sf_res_rich = price_s_forward(
            scen_set=scen_set,
            spec=sf_spec_rich,
            short_rate=0.02,
        )
        sf_price_rich = sf_res_rich["price"]
        print(f"S-forward off-market price (lower strike): {sf_price_rich:.6f}")

        assert (
            sf_price_rich != sf_price_atm
        ), "S-forward price should move when strike changes."

        # ---------- Survivor swap ATM (K choisi pour PV=0) ----------
        swap_spec_atm = SurvivorSwapSpec(
            age=issue_age_used,
            maturity_years=max_T,
            notional=100.0,
            strike=None,  # ATM : on laisse la fonction calculer K
            payer="fixed",  # paie le fixe, reçoit la jambe survie
        )
        swap_res_atm = price_survivor_swap(
            scen_set=scen_set,
            spec=swap_spec_atm,
            short_rate=0.02,
        )
        swap_price_atm = swap_res_atm["price"]
        swap_pv_paths = swap_res_atm["pv_paths"]
        swap_strike_atm = swap_res_atm["strike"]

        print(
            f"Survivor swap ATM price: {swap_price_atm:.6f}, "
            f"strike={swap_strike_atm:.6f}"
        )

        # ATM -> PV ≈ 0
        tol_swap = 0.05 * swap_spec_atm.notional
        assert abs(swap_price_atm) < tol_swap, (
            f"ATM survivor swap price too far from 0: {swap_price_atm:.6f}, "
            f"tol={tol_swap:.6f}"
        )
        assert swap_pv_paths.shape[0] == scen_set.n_scenarios(), (
            f"survivor swap pv_paths length {swap_pv_paths.shape[0]} "
            f"!= {scen_set.n_scenarios()} (n_scenarios)."
        )

        # ---------- Survivor swap off-market (strike modifié) ----------
        swap_spec_rich = SurvivorSwapSpec(
            age=issue_age_used,
            maturity_years=max_T,
            notional=100.0,
            strike=swap_strike_atm
            - 0.05,  # on diminue le fixe -> payer fixed devient plus content
            payer="fixed",
        )
        swap_res_rich = price_survivor_swap(
            scen_set=scen_set,
            spec=swap_spec_rich,
            short_rate=0.02,
        )
        swap_price_rich = swap_res_rich["price"]
        print(f"Survivor swap off-market price (lower strike): {swap_price_rich:.6f}")

        # Ici, comme payer="fixed", baisser K (fixe) devrait augmenter la valeur pour le payer -> prix plus grand
        assert (
            swap_price_rich > swap_price_atm
        ), "For payer='fixed', lowering strike should increase swap value."

        # ================== Cohort life annuity & Hedging smoke tests ===================

        print("\n=== Cohort life annuity & hedging smoke tests ===")

        ann_spec = CohortLifeAnnuitySpec(
            issue_age=issue_age_used,
            payment_per_survivor=1.0,
            maturity_years=max_T,
        )
        ann_res = price_cohort_life_annuity(
            scen_set=scen_set,
            spec=ann_spec,
            short_rate=0.02,
        )

        liab_pv_paths = ann_res["pv_paths"]
        print(f"Cohort life annuity price (MC): {ann_res['price']:.4f}")
        print(
            f"Annuity PV paths: mean={liab_pv_paths.mean():.4f}, "
            f"std={liab_pv_paths.std():.4f}"
        )

        instruments_pv_paths = np.column_stack(
            [
                pv_paths,  # longevity bond PV paths
                swap_pv_paths,  # survivor swap PV paths (ATM)
            ]
        )
        instrument_names = ["LongevityBond", "SurvivorSwap"]

        hedge_res = compute_min_variance_hedge(
            liability_pv_paths=liab_pv_paths,
            instruments_pv_paths=instruments_pv_paths,
            instrument_names=instrument_names,
        )

        print("\n=== Hedging result (annuity hedged with bond + swap) ===")
        for name, w in zip(hedge_res.instrument_names, hedge_res.weights):
            print(f"  weight[{name}] = {w:.4f}")

        std_L = hedge_res.summary["std_liability"]
        std_net = hedge_res.summary["std_net"]
        var_red = hedge_res.summary["var_reduction"] * 100.0

        print(
            f"Std liability = {std_L:.4f}, "
            f"Std net = {std_net:.4f}, "
            f"Variance reduction = {var_red:.2f}%"
        )

        # Sanity checks
        assert std_net <= std_L + 1e-8, "Hedge should not increase liability std."
        assert var_red > 10.0, "Hedge should achieve non-trivial variance reduction."

    print("\nDONE ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
