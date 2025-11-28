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
from pymort.models import (
    APCM3,
    CBDM5,
    CBDM6,
    CBDM7,
    LCM1,
    LCM2,
    _logit,
)
from pymort.pipeline import build_mortality_scenarios_for_pricing


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
        default=1000,
        help="Number of MC sims for k_t / kappa_t forecasts",
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
        default=200,
        help="Number of bootstrap replications (default: 200)",
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

    diff_S = np.diff(scen_set.S_paths, axis=2)  # (N, A, H_out-1)
    assert np.all(
        diff_S <= 1e-10
    ), "Survival curves must be non-increasing over time in each scenario."

    print("End-to-end pricing pipeline checks: OK ✅")

    print("\nDONE ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
