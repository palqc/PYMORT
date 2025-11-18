#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# make src/ importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pymort.lifetables import (
    load_m_from_excel,
    m_to_q,
    survival_from_q,
    validate_q,
    validate_survival_monotonic,
)
from pymort.models import (
    CBDModel,
    LeeCarter,
    _logit,
    estimate_rw_params,
    fit_lee_carter,
    reconstruct_log_m,
)
from pymort.analysis.validation import time_split_backtest_lc


def main(argv: list[str] | None = None) -> int:
    default_excel = os.path.join(ROOT_DIR, "Data/data_france.xlsx")
    p = argparse.ArgumentParser(description="PYMORT smoke test (explicit split)")
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
        help="Number of MC sims for k_t forecast, default 1000",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Random seed (None = random each run)"
    )
    p.add_argument(
        "--horizon", type=int, default=30, help="Forecast horizon (default: 30)"
    )
    args, _unknown = p.parse_known_args(argv)
    if _unknown:
        print("Ignoring unknown args:", _unknown)

    # load
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

        # explicit split: train_end → test starts train_end+1
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

    print("\n=== Backtest (explicit) ===")
    res = time_split_backtest_lc(ages, years, m, train_end=args.train_end)
    yrs_tr = res["train_years"]
    yrs_te = res["test_years"]
    rmse_log = res["rmse_log"]

    print(
    f"Train: {int(yrs_tr[0])}–{int(yrs_tr[-1])} | "
    f"Test: {int(yrs_te[0])}–{int(yrs_te[-1])}"
    )
    print(f"Out-of-sample RMSE (log): {rmse_log:.6f}")

    # MC forecast & survival sanity
    if args.sims > 0:
        print("\n=== Forecast k_t (Monte Carlo) & survival check ===")
        model = LeeCarter().fit(
            m_tr
        )  # fit on train only, then simulate forward H years
        mu2, sigma2 = model.estimate_rw()
        print(f"Estimated (train) mu={mu2:.6f}, sigma={sigma2:.6f}")
        # simulate exactly horizon H (length of test)
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

    # ================== Plots LC ===================

    print("\n=== Optional plots ===")

    mask_tr = years <= 2015
    par_tr = fit_lee_carter(m[:, mask_tr])

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(ages, par_tr.a)
    axs[0].set_title("a_x (level by age)")
    axs[0].set_xlabel("Age")
    axs[0].set_ylabel("log m")

    axs[1].plot(ages, par_tr.b)
    axs[1].set_title("b_x (sensitivity by age)")
    axs[1].set_xlabel("Age")

    axs[2].plot(years[mask_tr], par_tr.k)
    axs[2].set_title("k_t (train 1970–2015)")
    axs[2].set_xlabel("Year")
    plt.tight_layout()
    plt.show()

    mu_tr, sigma_tr = estimate_rw_params(par_tr.k)
    mask_te = years >= 2016
    yrs_te = years[mask_te]
    H = len(yrs_te)
    k_det = par_tr.k[-1] + mu_tr * np.arange(1, H + 1)

    ln_pred = par_tr.a[:, None] + np.outer(par_tr.b, k_det)
    m_pred = np.exp(ln_pred)
    m_obs = m[:, mask_te]

    age_star = 80
    i_age = (
        int(np.where(ages == age_star)[0][0]) if age_star in ages else len(ages) // 2
    )

    plt.figure(figsize=(7, 4))
    plt.plot(yrs_te, m_obs[i_age, :], label=f"Observed m(age={ages[i_age]})")
    plt.plot(yrs_te, m_pred[i_age, :], "--", label="LC deterministic forecast")
    plt.yscale("log")
    plt.title(f"Observed vs forecast (age {ages[i_age]})")
    plt.xlabel("Year")
    plt.ylabel("Death rate m (log)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    k_last = model.params.k[-1]

    k_paths_plot = np.concatenate(
        [np.full((k_paths.shape[0], 1), k_last), k_paths],
        axis=1,
    )
    years_plot = np.concatenate([[args.train_end], yrs_future])

    lo, med, hi = np.percentile(k_paths_plot, [5, 50, 95], axis=0)

    plt.figure(figsize=(8, 4))
    plt.fill_between(years_plot, lo, hi, alpha=0.3, label="90% band")
    plt.plot(years_plot, med, linewidth=2.0, label="Median $k_t$")

    plt.title(f"Monte Carlo fan chart for $k_t$ (H = {H_mc} years)")
    plt.xlabel("Year")
    plt.ylabel("$k_t$")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ================== CBD Model & Plots ===================

    # full-sample q
    q_full = m_to_q(m)
    cbd = CBDModel().fit(q_full, ages)
    q_hat_full = cbd.predict_q()

    logit_q = _logit(q_full)
    logit_q_hat = _logit(q_hat_full)
    rmse_logit = float(np.sqrt(np.mean((logit_q - logit_q_hat) ** 2)))
    print(f"CBD in-sample RMSE (logit q): {rmse_logit:.6f}")

    params_cbd = cbd.params
    assert params_cbd is not None

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
    plt.plot(years, params_cbd.kappa2, linewidth=2, color="orange")
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

    # RW params + MC on kappa1, kappa2
    print("\n=== CBD RW parameters and Monte Carlo ===")
    mu1, sigma1, mu2, sigma2 = cbd.estimate_rw()
    print(f"kappa1: mu={mu1:.6f}, sigma={sigma1:.6f}")
    print(f"kappa2: mu={mu2:.6f}, sigma={sigma2:.6f}")

    horizon_cbd = args.horizon
    n_sims_cbd = args.sims

    kappa1_paths = cbd.simulate_kappa(
        "kappa1", horizon=horizon_cbd, n_sims=n_sims_cbd, seed=None, include_last=True
    )  # (n_sims, H+1)

    years_future_cbd = np.arange(years[-1], years[-1] + horizon_cbd + 1)

    lo1, med1, hi1 = np.percentile(kappa1_paths, [5, 50, 95], axis=0)

    plt.figure(figsize=(8, 4))
    plt.fill_between(years_future_cbd, lo1, hi1, alpha=0.3, label="90% band")
    plt.plot(years_future_cbd, med1, linewidth=2.0, label="Median kappa1_t")
    plt.title("CBD – Monte Carlo fan chart for kappa1_t")
    plt.xlabel("Year")
    plt.ylabel("kappa1_t")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------- Unified CBD plot: history + fitted + MC forecast ----------

    horizon_cbd = args.horizon
    n_sims_cbd = args.sims
    seed_cbd = args.seed

    # Age of interest (reuse i_age from before)
    age_star = int(ages[i_age])
    q_obs_age = q_full[i_age, :]  # observed q_x,t
    q_fit_age = q_hat_full[i_age, :]  # CBD fitted q_x,t

    split_year = int(years[-1])  # last historical year, e.g. 2019

    # Simulate kappa1 and kappa2 starting from last historical year,
    # including the last fitted value (year = split_year) as column 0.
    k1_paths = cbd.simulate_kappa(
        "kappa1",
        horizon=horizon_cbd,
        n_sims=n_sims_cbd,
        seed=seed_cbd,
        include_last=True,
    )  # shape: (n_sims, H+1)

    k2_paths = cbd.simulate_kappa(
        "kappa2",
        horizon=horizon_cbd,
        n_sims=n_sims_cbd,
        seed=None if seed_cbd is None else seed_cbd + 1,
        include_last=True,
    )

    # Years for the simulated paths: [split_year, split_year+1, ..., split_year+H]
    years_all = np.arange(split_year, split_year + horizon_cbd + 1)

    # Convert kappa-paths to q-paths for the chosen age
    z_star = age_star - params_cbd.x_bar
    logit_q_paths = k1_paths + k2_paths * z_star  # (n_sims, H+1)
    q_paths = 1.0 / (1.0 + np.exp(-logit_q_paths))  # (n_sims, H+1)

    # Fan chart statistics (including last fitted year)
    q_low, q_med, q_high = np.percentile(q_paths, [5, 50, 95], axis=0)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 5))

    # 1) Historical part (fond blanc)
    ax.plot(
        years,
        q_obs_age,
        color="tab:blue",
        label=f"Observed q(age={age_star})",
    )
    ax.plot(
        years,
        q_fit_age,
        "--",
        color="tab:orange",
        label="CBD fitted q",
    )

    # 2) Future zone: fond gris léger à partir de la première année forecast
    ax.axvspan(split_year, years_all[-1], color="grey", alpha=0.08)

    # Ligne verticale séparation historique / prévision
    ax.axvline(split_year, color="black", linestyle="--", linewidth=1)

    # 3) Monte Carlo sample paths (toutes partent de la régression au split_year)
    n_plot = min(100, n_sims_cbd)  # nombre de chemins visibles
    for i in range(n_plot):
        ax.plot(
            years_all,
            q_paths[i, :],
            color="tab:blue",
            alpha=0.15,
            linewidth=0.7,
        )

    # 4) Fan chart (médiane + bande 90 %)
    ax.fill_between(
        years_all,
        q_low,
        q_high,
        color="tab:blue",
        alpha=0.25,
        label="90% band",
    )
    ax.plot(
        years_all,
        q_med,
        color="tab:red",
        linewidth=2.0,
        label="Median forecast (CBD)",
    )

    ax.set_title(f"CBD forecast with MC paths for q(age={age_star})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Death probability q")
    ax.set_ylim(0, max(q_obs_age.max(), q_high.max()) * 1.1)
    ax.legend(loc="upper right")

    fig.tight_layout()
    plt.show()

    print("\nDONE ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
