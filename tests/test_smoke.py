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

from pymort.analysis.validation import (
    rmse_aic_bic,
    time_split_backtest_apc,
    time_split_backtest_cbd_m5,
    time_split_backtest_cbd_m6,
    time_split_backtest_cbd_m7,
    time_split_backtest_lc_m1,
    time_split_backtest_lc_m2,
)
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
    estimate_rw_params,
    fit_lee_carter,
    reconstruct_log_m,
)


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
    rmse_log = res["rmse_log"]

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

    mask_tr = years <= args.train_end
    par_tr = fit_lee_carter(m[:, mask_tr])
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(ages, par_tr.a)
    axs[0].set_title("a_x (level by age)")
    axs[0].set_xlabel("Age")
    axs[0].set_ylabel("log m")

    axs[1].plot(ages, par_tr.b)
    axs[1].set_title("b_x (sensitivity by age)")
    axs[1].set_xlabel("Age")

    axs[2].plot(years[mask_tr], par_tr.k)
    axs[2].set_title(f"k_t (train {int(years[mask_tr][0])}–{int(years[mask_tr][-1])})")
    axs[2].set_xlabel("Year")
    plt.tight_layout()
    plt.show()
    """
    mu_tr, _sigma_tr = estimate_rw_params(par_tr.k)
    mask_te = years > args.train_end
    yrs_te_det = years[mask_te]
    H_det = len(yrs_te_det)
    k_det = par_tr.k[-1] + mu_tr * np.arange(1, H_det + 1)

    ln_pred = par_tr.a[:, None] + np.outer(par_tr.b, k_det)
    m_pred = np.exp(ln_pred)
    m_obs = m[:, mask_te]

    age_star = 80
    i_age = (
        int(np.where(ages == age_star)[0][0]) if age_star in ages else len(ages) // 2
    )

    # --- LC: observed vs fitted death probabilities q_x,t on full historical period ---

    # LC fitted on the whole historical m[age, year]
    par_lc_full = fit_lee_carter(m)
    ln_m_hat_full = reconstruct_log_m(par_lc_full)
    m_hat_full = np.exp(ln_m_hat_full)

    # Convert to death probabilities q
    q_full = m_to_q(m)
    q_hat_lc_full = m_to_q(m_hat_full)

    # Plot for chosen age
    plt.figure(figsize=(8, 4))
    plt.plot(
        years,
        q_full[i_age, :],
        label=f"Observed q(age={ages[i_age]})",
    )
    plt.plot(
        years,
        q_hat_lc_full[i_age, :],
        "--",
        label="LC fitted q",
    )
    plt.title(f"Lee–Carter: observed vs fitted q_x,t (age={ages[i_age]})")
    plt.xlabel("Year")
    plt.ylabel("Death probability q")
    plt.legend()
    plt.tight_layout()
    plt.show()

    """
    plt.figure(figsize=(7, 4))
    plt.plot(yrs_te_det, m_obs[i_age, :], label=f"Observed m(age={ages[i_age]})")
    plt.plot(yrs_te_det, m_pred[i_age, :], "--", label="LC deterministic forecast")
    plt.yscale("log")
    plt.title(f"Observed vs LC forecast (age {ages[i_age]})")
    plt.xlabel("Year")
    plt.ylabel("Death rate m (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    if args.sims > 0:
        # fan chart on k_t (LC)
        k_last = model.params.k[-1]
        k_paths_plot = np.concatenate(
            [np.full((k_paths.shape[0], 1), k_last), k_paths],
            axis=1,
        )
        years_plot = np.concatenate([[args.train_end], yrs_future])

        lo, med, hi = np.percentile(k_paths_plot, [5, 50, 95], axis=0)
    """
        plt.figure(figsize=(8, 4))
        plt.fill_between(years_plot, lo, hi, alpha=0.3, label="90% band")
        plt.plot(years_plot, med, linewidth=2.0, label="Median $k_t$")
        plt.title(f"Monte Carlo fan chart for $k_t$ (H = {H_mc} years)")
        plt.xlabel("Year")
        plt.ylabel("$k_t$")
        plt.legend()
        plt.tight_layout()
        plt.show()
    """
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
    """
    # kappa1_t (level)
    plt.figure(figsize=(8, 4))
    plt.plot(years, params_cbd.kappa1, linewidth=2)
    plt.title("CBD factor kappa1_t (level)")
    plt.xlabel("Year")
    plt.ylabel("kappa1_t")
    plt.tight_layout()
    plt.show()
    
    # kappa2_t (slope)
    plt.figure(figsize=(8, 4))
    plt.plot(years, params_cbd.kappa2, linewidth=2)
    plt.title("CBD factor kappa2_t (slope)")
    plt.xlabel("Year")
    plt.ylabel("kappa2_t")
    plt.tight_layout()
    plt.show()

    # observed vs fitted q for age_star
    plt.figure(figsize=(8, 4))
    plt.plot(years, q_full[i_age, :], label=f"Observed q(age={ages[i_age]})")
    plt.plot(years, q_hat_full[i_age, :], "--", label="CBD fitted q")
    plt.title(f"CBD: observed vs fitted q_x,t (age={ages[i_age]})")
    plt.xlabel("Year")
    plt.ylabel("Death probability q")
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    # RW params + MC on kappa1 (CBD)
    print("\n=== CBD RW parameters and Monte Carlo (kappa1) ===")
    mu1, sigma1, mu2, sigma2 = cbd.estimate_rw()
    print(f"kappa1: mu={mu1:.6f}, sigma={sigma1:.6f}")
    print(f"kappa2: mu={mu2:.6f}, sigma={sigma2:.6f}")

    horizon_cbd = args.horizon
    n_sims_cbd = args.sims

    kappa1_paths = cbd.simulate_kappa(
        "kappa1",
        horizon=horizon_cbd,
        n_sims=n_sims_cbd,
        seed=args.seed,
        include_last=True,
    )  # (n_sims, H+1)

    years_future_cbd = np.arange(years[-1], years[-1] + horizon_cbd + 1)

    lo1, med1, hi1 = np.percentile(kappa1_paths, [5, 50, 95], axis=0)
    """
    plt.figure(figsize=(8, 4))
    plt.fill_between(years_future_cbd, lo1, hi1, alpha=0.3, label="90% band")
    plt.plot(years_future_cbd, med1, linewidth=2.0, label="Median kappa1_t")
    plt.title("CBD – Monte Carlo fan chart for kappa1_t")
    plt.xlabel("Year")
    plt.ylabel("kappa1_t")
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    # ================== CBD Cohort Model (M6) ===================

    print("\n=== CBD + cohort model: fit & comparison ===")

    cbd_co = CBDM6().fit(q_full, ages, years)
    q_hat_co = cbd_co.predict_q()

    logit_q_hat_co = _logit(q_hat_co)
    rmse_logit_co = float(np.sqrt(np.mean((logit_q - logit_q_hat_co) ** 2)))

    print(f"CBD (no cohort)  RMSE (logit q): {rmse_logit:.6f}")
    print(f"CBD + cohort     RMSE (logit q): {rmse_logit_co:.6f}")
    """
    # Observed vs CBD vs CBD+cohort for same age_star
    plt.figure(figsize=(9, 5))
    plt.plot(
        years, q_full[i_age, :], color="black", label=f"Observed q(age={ages[i_age]})"
    )
    plt.plot(
        years, q_hat_full[i_age, :], "--", color="tab:orange", label="CBD fitted q"
    )
    plt.plot(
        years, q_hat_co[i_age, :], "-.", color="tab:green", label="CBD+cohort fitted q"
    )
    plt.title(f"CBD vs CBD+cohort: observed vs fitted q_x,t (age={ages[i_age]})")
    plt.xlabel("Year")
    plt.ylabel("Death probability q")
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    # RW params + MC trên kappa1_t (CBD+cohort)
    print("\n=== CBD + cohort: RW params and MC on kappa1_t ===")
    mu1_co, sigma1_co, mu2_co, sigma2_co = cbd_co.estimate_rw()
    print(f"[CBD+cohort] kappa1: mu={mu1_co:.6f}, sigma={sigma1_co:.6f}")
    print(f"[CBD+cohort] kappa2: mu={mu2_co:.6f}, sigma={sigma2_co:.6f}")

    k1_paths_co = cbd_co.simulate_kappa(
        "kappa1",
        horizon=horizon_cbd,
        n_sims=n_sims_cbd,
        seed=args.seed,
        include_last=True,
    )

    years_future_cbd_co = np.arange(years[-1], years[-1] + horizon_cbd + 1)
    lo1_co, med1_co, hi1_co = np.percentile(k1_paths_co, [5, 50, 95], axis=0)
    """
    plt.figure(figsize=(8, 4))
    plt.fill_between(
        years_future_cbd_co, lo1_co, hi1_co, alpha=0.3, label="90% band (CBD+cohort)"
    )
    plt.plot(
        years_future_cbd_co,
        med1_co,
        linewidth=2.0,
        label="Median kappa1_t (CBD+cohort)",
    )
    plt.title("CBD + cohort – Monte Carlo fan chart for kappa1_t")
    plt.xlabel("Year")
    plt.ylabel("kappa1_t")
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    # ================== CBD M7 (quadratic + cohort) ===================

    # Fit M7 on toute la surface q_full
    m7 = CBDM7().fit(q_full, ages, years)
    q_hat_m7_full = m7.predict_q()  # (A, T)

    # --------- Plot 1 : observed vs CBD vs CBD+cohort vs M7 ---------
    q_obs_age = q_full[i_age, :]  # observed
    q_cbd_age = q_hat_full[i_age, :]  # CBD simple
    q_cbd_co_age = q_hat_co[i_age, :]  # CBD + cohort (M6)
    q_m7_age = q_hat_m7_full[i_age, :]  # CBD M7

    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(years, q_obs_age, color="black", label=f"Observed q(age={ages[i_age]})")
    plt.plot(years, q_cbd_age, "--", color="tab:orange", label="CBD M5 fitted q")
    plt.plot(
        years,
        q_cbd_co_age,
        "--",
        color="tab:green",
        linestyle=(0, (5, 2)),
        label="CBD M6 fitted q",
    )
    plt.plot(
        years,
        q_m7_age,
        "--",
        color="tab:red",
        linestyle=(0, (3, 1, 1, 1)),
        label="CBD M7 fitted q",
    )

    plt.title(f"CBD M5 vs M6 vs M7: observed vs fitted q_x,t (age={ages[i_age]})")
    plt.xlabel("Year")
    plt.ylabel("Death probability q")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------- Plot 2 : forecast fan chart avec MC du modèle M7 ---------

    par_lc_full = fit_lee_carter(m)  # fit LC sur TOUT l’historique
    ln_m_lc_full = reconstruct_log_m(par_lc_full)
    m_lc_full = np.exp(ln_m_lc_full)
    q_lc_full = m_to_q(m_lc_full)  # conversion m -> q
    q_lc_age = q_lc_full[i_age, :]

    # LC M2 (LC + cohort) sur tout l’historique
    lcm2_full = LCM2().fit(m, ages, years)
    m_lcm2_full = lcm2_full.predict_m()
    q_lcm2_full = m_to_q(m_lcm2_full)
    q_lcm2_age = q_lcm2_full[i_age, :]

    # APC M3 sur tout l’historique
    apc_full = APCM3().fit(m, ages, years)
    m_apc_full = apc_full.predict_m()
    q_apc_full = m_to_q(m_apc_full)
    q_apc_age = q_apc_full[i_age, :]

    # Estimation RW sur les trois kappa_t
    mu1, _sig1, mu2, _sig2, mu3, _sig3 = m7.estimate_rw()
    params_m7 = m7.params
    assert params_m7 is not None

    horizon_m7 = args.horizon
    n_sims_m7 = args.sims
    seed_m7 = args.seed

    # Simule kappa1, kappa2, kappa3 à partir de la dernière année observée
    k1_paths = m7.simulate_kappa(
        "kappa1",
        horizon=horizon_m7,
        n_sims=n_sims_m7,
        seed=seed_m7,
        include_last=True,
    )  # (n_sims, H+1)

    k2_paths = m7.simulate_kappa(
        "kappa2",
        horizon=horizon_m7,
        n_sims=n_sims_m7,
        seed=None if seed_m7 is None else seed_m7 + 1,
        include_last=True,
    )

    k3_paths = m7.simulate_kappa(
        "kappa3",
        horizon=horizon_m7,
        n_sims=n_sims_m7,
        seed=None if seed_m7 is None else seed_m7 + 2,
        include_last=True,
    )

    # Grille des années pour la partie forecast (inclut la dernière année historique)
    split_year = int(years[-1])
    years_forecast = np.arange(split_year, split_year + horizon_m7 + 1)

    # Termes en z pour l'âge étudié
    z_vec = params_m7.ages - params_m7.x_bar
    var_z = float(np.mean(z_vec**2))  # \hat\sigma_x^2
    z_star = float(ages[i_age] - params_m7.x_bar)
    z2c_star = float(z_star**2 - var_z)

    # Effet de cohorte : on gèle gamma_{t-x} à sa valeur pour la dernière année observée
    gamma_last = params_m7.gamma_for_age_at_last_year(float(ages[i_age]))

    # Construction des trajectoires logit(q) puis q pour l’âge étudié
    # shape (n_sims, H+1)
    logit_q_paths_m7 = k1_paths + k2_paths * z_star + k3_paths * z2c_star + gamma_last
    q_paths_m7 = 1.0 / (1.0 + np.exp(-logit_q_paths_m7))

    q_low_m7, q_med_m7, q_high_m7 = np.percentile(q_paths_m7, [5, 50, 95], axis=0)

    # Historique complet
    q_obs_age = q_full[i_age, :]

    # Si le fichier Excel contient des années > 2019
    if q_full.shape[1] > len(years):
        extra_years = years[len(years) :]
        extra_obs = q_obs_age[len(years) :]
    else:
        # sinon, remplir les observations futures par NaN
        extra_years = years_forecast[1:]  # enlever split_year
        extra_obs = np.full_like(extra_years, np.nan, dtype=float)

    years_obs_extended = np.concatenate([years, extra_years])
    q_obs_extended = np.concatenate([q_obs_age, extra_obs])

    q_m7_age = q_hat_m7_full[i_age, :]
    q_cbd_age = q_hat_full[i_age, :]  # pour comparaison dans la légende

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    # Historique (fond blanc)
    ax.plot(
        years_obs_extended,
        q_obs_extended,
        color="black",
        linewidth=1,
        label=f"Observed q(age={ages[i_age]})",
    )
    ax.plot(
        years,
        q_lc_age,
        "--",
        color="tab:brown",
        linewidth=1,
        label="LC M1 fitted q",
    )
    ax.plot(
        years,
        q_lcm2_age,
        "--",
        color="tab:cyan",
        linewidth=1,
        label="LC M2 fitted q",
    )
    ax.plot(
        years,
        q_apc_age,
        "--",
        color="tab:gray",
        linewidth=1,
        label="APC M3 fitted q",
    )
    ax.plot(
        years,
        q_cbd_age,
        "--",
        color="tab:orange",
        linewidth=1,
        label="CBD M5 fitted q",
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
        years,
        q_m7_age,
        "--",
        color="tab:green",
        linewidth=1,
        label="CBD M7 fitted q",
    )

    # Zone future en gris clair
    ax.axvspan(split_year, years_forecast[-1], color="grey", alpha=0.08)
    ax.axvline(split_year, color="black", linestyle="--", linewidth=1)

    # Quelques trajectoires MC (M7) pour visualiser la dispersion
    n_plot = min(1000, n_sims_m7)
    for i in range(n_plot):
        ax.plot(
            years_forecast,
            q_paths_m7[i, :],
            color="tab:blue",
            alpha=0.08,
            linewidth=0.7,
        )

    # Bande 90 % + médiane
    ax.fill_between(
        years_forecast,
        q_low_m7,
        q_high_m7,
        color="tab:blue",
        alpha=0.25,
        label="90% band (M7 forecast)",
    )
    ax.plot(
        years_forecast,
        q_med_m7,
        color="tab:red",
        linewidth=1.5,
        label="Median forecast (M7)",
    )

    ax.set_title(f"CBD M7 forecast with MC paths for q (age={ages[i_age]})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Death probability q")
    ax.set_ylim(0, max(q_obs_age.max(), q_high_m7.max()) * 1.1)
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()

    # ------------------- RMSE / AIC / BIC + forecast table --------------------------

    # Dimensions
    A = len(ages)
    T = len(years)

    # Vérité sur toute la surface en logit(q)
    q_full = m_to_q(m)
    logit_true = _logit(q_full)
    ln_m_true = np.log(m)

    # ---------- LC M1 : in-sample (full) ----------
    m_lc_full = np.exp(ln_m_lc_full)
    q_lc_full = m_to_q(m_lc_full)

    rmse_lc_logm_full = float(np.sqrt(np.mean((ln_m_true - ln_m_lc_full) ** 2)))

    rmse_lc_logitq, aic_lc, bic_lc = rmse_aic_bic(
        logit_true,
        _logit(q_lc_full),
        n_params=2 * A + T,  # a_x, b_x, k_t
    )

    # ---------- LC M1 : forecast RMSE (logit q) ----------

    res_lc1_bt = time_split_backtest_lc_m1(years, m, train_end=args.train_end)
    rmse_lc_forecast_logm = float(res_lc1_bt["rmse_log"])
    rmse_lc_forecast_logit = float(res_lc1_bt["rmse_logit_forecast"])

    # ---------- LC M2 : in-sample (full) + forecast ----------

    lcm2_full = LCM2().fit(m, ages, years)
    params_lcm2 = lcm2_full.params
    assert params_lcm2 is not None
    m_lcm2_full = lcm2_full.predict_m()
    q_lcm2_full = m_to_q(m_lcm2_full)

    ln_m_lcm2_full = np.log(m_lcm2_full)
    rmse_lcm2_logm_full = float(np.sqrt(np.mean((ln_m_true - ln_m_lcm2_full) ** 2)))

    C_lcm2 = len(params_lcm2.cohorts)
    n_params_lcm2 = 2 * A + T + C_lcm2  # a_x, b_x, k_t, gamma_c
    rmse_lcm2_logitq, aic_lcm2, bic_lcm2 = rmse_aic_bic(
        logit_true,
        _logit(q_lcm2_full),
        n_params=n_params_lcm2,
    )

    res_lcm2_bt = time_split_backtest_lc_m2(ages, years, m, train_end=args.train_end)
    rmse_lcm2_forecast_logm = float(res_lcm2_bt["rmse_log_forecast"])
    rmse_lcm2_forecast_logit = float(res_lcm2_bt["rmse_logit_forecast"])

    # ---------- APC M3 : in-sample (full) + forecast ----------

    apc_full = APCM3().fit(m, ages, years)
    params_apc = apc_full.params
    assert params_apc is not None
    m_apc_full = apc_full.predict_m()
    q_apc_full = m_to_q(m_apc_full)

    ln_m_apc_full = np.log(m_apc_full)
    rmse_apc_logm_full = float(np.sqrt(np.mean((ln_m_true - ln_m_apc_full) ** 2)))

    C_apc = len(params_apc.cohorts)
    n_params_apc = A + T + C_apc  # beta_x, kappa_t, gamma_c
    rmse_apc_logitq, aic_apc, bic_apc = rmse_aic_bic(
        logit_true,
        _logit(q_apc_full),
        n_params=n_params_apc,
    )

    res_apc_bt = time_split_backtest_apc(ages, years, m, train_end=args.train_end)
    rmse_apc_forecast_logm = float(res_apc_bt["rmse_log_forecast"])
    rmse_apc_forecast_logit = float(res_apc_bt["rmse_logit_forecast"])

    # ---------- CBD M5 : backtest + AIC/BIC ----------

    res_m5 = time_split_backtest_cbd_m5(ages, years, q_full, train_end=args.train_end)
    rmse_m5_forecast_logit = float(res_m5["rmse_logit_forecast"])

    rmse_m5_logitq, aic_m5, bic_m5 = rmse_aic_bic(
        logit_true,
        _logit(q_hat_full),  # q_hat_full : fit M5 sur tout l’échantillon
        n_params=2 * T,  # kappa1_t, kappa2_t
    )

    # ---------- CBD M6 : backtest + AIC/BIC ----------
    C_cbd = len(params_m7.cohorts)  # nombre de cohortes distinctes pour M6/M7

    res_m6 = time_split_backtest_cbd_m6(ages, years, q_full, train_end=args.train_end)
    rmse_m6_forecast_logit = float(res_m6["rmse_logit_forecast"])

    rmse_m6_logitq, aic_m6, bic_m6 = rmse_aic_bic(
        logit_true,
        _logit(q_hat_co),  # q_hat_co : fit M6 sur tout l’échantillon
        n_params=2 * T + C_cbd,  # kappa1_t, kappa2_t + gamma_c
    )

    # ---------- CBD M7 : backtest + AIC/BIC ----------

    res_m7_bt = time_split_backtest_cbd_m7(
        ages, years, q_full, train_end=args.train_end
    )
    rmse_m7_forecast_logit = float(res_m7_bt["rmse_logit_forecast"])

    rmse_m7_logitq, aic_m7, bic_m7 = rmse_aic_bic(
        logit_true,
        _logit(q_hat_m7_full),  # q_hat_m7_full : fit M7 sur tout l’échantillon
        n_params=3 * T + C_cbd,  # kappa1_t, kappa2_t, kappa3_t + gamma_c
    )

    # ---------- Tableau récapitulatif ----------

    start_te = int(args.train_end + 1)
    end_te = int(years[-1])

    results_df = pd.DataFrame(
        {
            "Model": [
                "LC (M1)",
                "LC+cohort (M2)",
                "APC (M3)",
                "CBD (M5)",
                "CBD+cohort (M6)",
                "CBD+quadratic+cohort (M7)",
            ],
            # RMSE log m (full in-sample)
            "RMSE full (log m)": [
                rmse_lc_logm_full,
                rmse_lcm2_logm_full,
                rmse_apc_logm_full,
                np.nan,
                np.nan,
                np.nan,
            ],
            # RMSE logit q (full in-sample)
            "RMSE full (logit q)": [
                rmse_lc_logitq,
                rmse_lcm2_logitq,
                rmse_apc_logitq,
                rmse_m5_logitq,
                rmse_m6_logitq,
                rmse_m7_logitq,
            ],
            # RMSE forecast log m (LC family only)
            f"RMSE forecast {start_te}–{end_te} (log m)": [
                rmse_lc_forecast_logm,
                rmse_lcm2_forecast_logm,
                rmse_apc_forecast_logm,
                np.nan,
                np.nan,
                np.nan,
            ],
            # RMSE forecast logit q (all models)
            f"RMSE forecast {start_te}–{end_te} (logit q)": [
                rmse_lc_forecast_logit,
                rmse_lcm2_forecast_logit,
                rmse_apc_forecast_logit,
                rmse_m5_forecast_logit,
                rmse_m6_forecast_logit,
                rmse_m7_forecast_logit,
            ],
            "AIC": [
                aic_lc,
                aic_lcm2,
                aic_apc,
                aic_m5,
                aic_m6,
                aic_m7,
            ],
            "BIC": [
                bic_lc,
                bic_lcm2,
                bic_apc,
                bic_m5,
                bic_m6,
                bic_m7,
            ],
        }
    )

    print(
        "\n=== Model comparison (RMSE log m, RMSE logit q, AIC, BIC, forecast RMSE) ===\n"
    )
    print(results_df.to_string(index=False))
    # ---------------------------------------------------------

    print("\nDONE ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
