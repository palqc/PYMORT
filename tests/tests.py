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
    LeeCarter,
    estimate_rw_params,
    fit_lee_carter,
    reconstruct_log_m,
)

# optional analysis
try:
    from pymort.analysis.validation import reconstruction_rmse_log

    HAVE_ANALYSIS = True
except Exception:
    HAVE_ANALYSIS = False


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
        default=1000,
        help="Number of MC sims for k_t forecast, default 1000",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Random seed (None = random each run)"
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

    # fit full-sample diagnostics
    print("\n=== Fit Lee–Carter (full-sample diagnostics) ===")
    params = fit_lee_carter(m)
    ln_hat = reconstruct_log_m(params)
    ln_true = np.log(m)
    rms_in = float(np.sqrt(np.mean((ln_true - ln_hat) ** 2)))
    print(f"In-sample RMSE (log): {rms_in:.6f}")
    print(f"sum(b): {params.b.sum():.12f}  mean(k): {params.k.mean():.3e}")

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
    print(
        f"Train: {int(yrs_tr[0])}–{int(yrs_tr[-1])} | Test: {int(yrs_te[0])}–{int(yrs_te[-1])}"
    )
    par_tr = fit_lee_carter(m_tr)
    mu, sigma = estimate_rw_params(par_tr.k)
    H = len(yrs_te)
    k_det = par_tr.k[-1] + mu * np.arange(1, H + 1)  # deterministic forecast for speed
    ln_pred = par_tr.a[:, None] + np.outer(par_tr.b, k_det)
    rmse_log = float(np.sqrt(np.mean((np.log(m_te) - ln_pred) ** 2)))
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
        k_paths = model.simulate_k(horizon=H, n_sims=args.sims, seed=args.seed)
        print(f"k_paths shape: {k_paths.shape} {args.seed}(n_sims, horizon={H})")
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
    """
    # 1) Refit LC sur train (1970–2015) et tracer a_x, b_x, k_t (train)
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

    # 2) Forecast déterministe 2016–2019 vs observé pour un âge (ex: 80 ans)
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

    # 3) Fan chart sur k_t (Monte Carlo) 2016–2019
    # Supposons que tu as déjà k_paths de shape (n_sims, H) provenant de model.simulate_k(...)
    lo, med, hi = np.percentile(k_paths, [5, 50, 95], axis=0)

    plt.figure(figsize=(7, 4))
    plt.plot(yrs_te, med, label="Median k_t (MC)")
    plt.fill_between(yrs_te, lo, hi, alpha=0.3, label="90% band")
    plt.title("Monte Carlo fan chart for k_t (2016–2019)")
    plt.xlabel("Year")
    plt.ylabel("k_t")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # (Option) Fan chart sur m pour l'âge choisi
    a_x = par_tr.a[i_age]
    b_x = par_tr.b[i_age]
    m_paths = np.exp(a_x + b_x * k_paths)  # (n_sims, H)
    lo, med, hi = np.percentile(m_paths, [5, 50, 95], axis=0)

    plt.figure(figsize=(7, 4))
    plt.plot(yrs_te, m_obs[i_age, :], label="Observed")
    plt.plot(yrs_te, med, "--", label="Median (MC)")
    plt.fill_between(yrs_te, lo, hi, alpha=0.3, label="90% band")
    plt.yscale("log")
    plt.title(f"Fan chart of m(age={ages[i_age]})")
    plt.xlabel("Year")
    plt.ylabel("Death rate m (log)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    print("\nDONE ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
