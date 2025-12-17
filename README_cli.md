PYMORT CLI Quickstart
=====================

This document highlights the production-style CLI implemented in `pymort/cli.py`. All commands share global options (`--config`, `--seed`, `--outdir`, `--format`, `--verbose/--quiet`, `--log-level`, `--overwrite`, `--save`, `--load`).

Key workflows
-------------

1) Validate and prep data
```bash
# validate HMD-style csv/parquet/npy
pymort data validate-m --m-path Data/data_france.xlsx

# clip tiny m values then convert to q
pymort data clip-m --m-path Data/data_france.xlsx -o outputs/m_clipped.npz
pymort data to-q --m-path outputs/m_clipped.npz -o outputs/q_surface.npz
```

2) Smooth, fit, and select models
```bash
# CPsplines smoothing (log m)
pymort smooth cpsplines --m-path Data/data_france.xlsx --horizon 0 -o outputs/cpsplines.npz

# fit one model
pymort fit one --model LCM2 --m-path Data/data_france.xlsx --smoothing cpsplines -o outputs/fitted_lcm2.pkl

# model selection then final fit for pricing
pymort fit select-and-fit --m-path Data/data_france.xlsx --train-end 2015 \
    --metric logit_q --cpsplines-horizon 0 -o outputs/fitted_best.pkl
```

3) Build scenarios
```bash
# real-world scenarios (bootstrap + RW)
pymort scen build-P --fitted-model outputs/fitted_best.pkl --horizon 40 -o outputs/scen_P.npz

# risk-neutral scenarios with Esscher lambda
pymort scen build-Q --m-path Data/data_france.xlsx --model-name CBDM7 --lambda-esscher 0.1 \
    --horizon 40 -o outputs/scen_Q.npz
```

4) Stress and summarize
```bash
pymort stress apply --scen-path outputs/scen_P.npz --shock-type long_life --magnitude 0.1 \
    -o outputs/scen_P_longlife.npz
pymort scen summarize --scen-path outputs/scen_P.npz
```

5) Pricing
```bash
pymort price longevity-bond --scen-path outputs/scen_Q.npz --issue-age 65 --maturity-years 20 \
    --rate 0.02 -o outputs/price_bond.json
pymort price survivor-swap --scen-path outputs/scen_Q.npz --age 70 --maturity-years 15 \
    --payer fixed --rate 0.02 -o outputs/price_swap.json
```

6) Sensitivities and hedging
```bash
pymort sens rate --scen-path outputs/scen_Q.npz --kind longevity_bond \
    --spec configs/bond_spec.yaml --base-short-rate 0.02
pymort hedge min-variance --liab-pv-path outputs/liab_pv.csv \
    --instr-pv-path outputs/hedge_pv.csv --names "Bond,Swap"
```

7) Risk reporting and plots
```bash
pymort report risk --pv-path outputs/price_paths.csv --name bond --var-level 0.99
pymort plot survival-fan --scen-path outputs/scen_Q.npz --age 70 -o outputs/survival_fan.png
```

Sample configs
--------------
Two sample YAML configs are provided:
- `configs/pricing-pipeline.yaml` – parameters for an end-to-end pricing run (data → fit → scenarios → pricing)
- `configs/hedge-pipeline.yaml` – parameters for pricing liability & hedge instruments and computing hedge weights

Notes
-----
- Mortality inputs can be csv/parquet/npy, wide (ages rows, years cols) or long (Age, Year, m).
- Scenario sets are stored as `.npz` with `q_paths`, `S_paths`, `ages`, `years`, `metadata`.
- Tables can be saved as csv/parquet/json using `--format`.
- Most commands accept `--seed` for reproducibility and `--outdir` for outputs.
