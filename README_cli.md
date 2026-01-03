# PYMORT CLI Manual

pymort is a command-line toolkit for mortality surfaces, scenario generation (P/Q measures), longevity instrument pricing, sensitivities, hedging, reporting, and plotting.

## Installation & entrypoint

Depending on how you package/install:

```bash
pymort --help
pymort version
```

Tip: Every command supports `--help`, e.g. `pymort scen build-P --help`.

---

## Global options (apply to all commands)

These options are defined at the CLI root and are available everywhere:

- `-c, --config PATH` : YAML/JSON config file providing default values.
- `--seed INT` : RNG seed (used by many pipelines).
- `--outdir PATH` : output directory (default: outputs).
- `--format [csv|json|parquet]` : tabular output format (default: csv).
- `-v, --verbose` : DEBUG logs.
- `--quiet` : ERROR-only logs.
- `--log-level [DEBUG|INFO|WARNING|ERROR]` : default INFO.
- `--overwrite` : allow overwriting (note: current helper does not actively delete contents; it mainly ensures outdir exists).
- `--save PATH` : optional “save result” path (pickle/npz) — currently not wired to every command, but kept as a global flag.
- `--load PATH` : load a previously saved object — used by some top-level commands that read pickle/npz paths explicitly.

**Example**

```bash
pymort --outdir runs/run1 --seed 123 --verbose data validate-m --m-path Data/france.csv
```

---

## Input file formats

### Mortality surface (m_path)

Supported formats:

1. Excel: `.xlsx/.xls`
   - Loaded via `load_m_from_excel_any(...)`.
   - You can choose a column via `--rate-column` (fit default) or `preferred_rate_col` in some pipelines:
     - Accepted rate columns: Total, Female, Male (case-insensitive).
   - Internally uses defaults age_min=60, age_max=110 for Excel loader in `_load_m_surface`.

2. CSV / Parquet: `.csv`, `.parquet`, `.pq`
   Two layouts:
   - Long format (must contain Age and Year columns, case-insensitive):
     - Must also contain one mortality-rate column among: m, mx, rate, Total, Male, Female.
     - If you pass a preferred column (e.g. Total), it will be used when present; else it picks Total if available, otherwise the first candidate.
   - Wide format:
     - First column must be Age (or Ages).
     - Remaining columns are years (must be parseable as ints).
     - Values are mortality rates.

3. NumPy wide arrays: `.npy/.npz`
   - Must contain a 2D array shaped (n_ages, n_years).
   - You must provide ages and years explicitly via:
     - `--ages "60,61,62"` and `--years "2000,2001"` (inline lists), or
     - `--ages-path PATH` and `--years-path PATH` (files containing 1D numeric arrays).

### Ages / Years selection: list vs range

Many commands accept:

- `--ages` and `--years` as either
  - list: `"60,61,62"` (comma-separated)
  - range: `"60-100"` or `"60:100"` (two numbers)
- For `.npy/.npz` mortality matrices, lists are required to define dimensions.
- For CSV/Parquet/Excel, ranges are commonly used to subset after loading.

---

## Scenario set format (.npz)

Scenario sets are stored as .npz and loaded with `_load_scenarios`. Expected keys:

- `q_paths` : mortality probability paths
- `S_paths` : survival paths
- `ages`, `years`
- optional: `m_paths`, `discount_factors`
- `metadata` : stored as JSON when possible

---

## Commands overview

Top-level groups:

- `data` : validate/clip/convert mortality surfaces
- `smooth` : CPsplines smoothing
- `fit` : fit models + selection
- `scen` : build scenario sets (P/Q) + summarize
- `forecast` : scenarios from a fitted model pickle
- `stress` : apply shocks / chain / bundle
- `price` : price instruments under a scenario set
- `price-bond` : convenience wrapper for longevity bond pricing from fitted model OR scenario set
- `rn` : risk-neutral calibration (lambda) and pricing under lambda
- `sens` : sensitivities (rate, convexity, mortality deltas, all)
- `hedge` : hedging (PV paths or end-to-end specs; min-variance; multihorizon)
- `report` : risk report on PV paths
- `plot` : plotting helpers (fan charts, lexis, histograms)
- `run` : one-click pipelines from config
- `version`, `echo`

---

## 1) data — Data utilities

### `pymort data validate-m`

Validate an m surface and write a JSON report.

**Options**

- `--m-path PATH` (required)
- `--ages`, `--years` (optional; list or range depending on format)
- `--ages-path`, `--years-path` (optional)

**Output**

- writes outputs/validation_m.json (or `--outdir/...`)

**Example**

```bash
pymort data validate-m --m-path Data/france.csv --ages 60-100 --years 1970-2019
```

### `pymort data clip-m`

Clip mortality rates at eps and save as .npz.

**Options**

- `--m-path PATH` (required)
- `--eps FLOAT` (default 1e-12)
- `--output PATH` (default: outdir/m_clipped.npz)
- `--ages/--years` + `--ages-path/--years-path` as above

**Example**

```bash
pymort data clip-m --m-path Data/france.csv --ages 60-100 --years 1970-2019 --output outputs/m_clipped.npz
```

### `pymort data to-q`

Convert m to q via m_to_q and save .npz.

**Output**

- default: outdir/q_surface.npz

**Example**

```bash
pymort data to-q --m-path outputs/m_clipped.npz --output outputs/q_surface.npz
```

---

## 2) smooth — Smoothing utilities

### `pymort smooth cpsplines`

Smooth an m surface with CPsplines.

**Key options**

- `--m-path PATH` (required)
- `--deg "3,3"` : spline degrees (age, year)
- `--ord-d "2,2"` : derivative penalty orders
- `--k "ka,kt"` or `--k auto` or omit
- `--sp-method grid_search` : smoothing parameter method
- `--sp-args JSON_STRING` : extra args for smoothing method
- `--horizon INT` : optional forecast horizon for cpsplines
- `--output PATH` : default outdir/cpsplines.npz
- plus `--ages/--years` selection

**Example**

```bash
pymort smooth cpsplines --m-path Data/france.csv --ages 60-100 --years 1970-2019 \
  --deg 3,3 --ord-d 2,2 --sp-method grid_search --output outputs/cpsplines.npz
```

---

## 3) fit — Model fitting & selection

There are two entry styles:

- `pymort fit ...` (default command when you call fit without subcommand)
- `pymort fit one|select|select-and-fit`

### `pymort fit` (default)

Fits a model from a long CSV/parquet mortality dataset (but in practice it loads via `_load_m_surface`, so wide formats may also work).

**Required**

- positional data path
- `--model/-m` model alias (lee-carter, cbd-m6, etc.)

**Optional**

- `--ages "60-100"` and `--years "1970-2019"` (ranges are expected here)
- `--rate-column Total|Male|Female|m`
- `--output` pickle path (default: outdir/fitted_<model>.pkl)
- `--summary` JSON path (default: outdir/fit_summary.json)

**Example**

```bash
pymort fit Data/france.csv --model lee-carter --ages 60-100 --years 1970-2019
```

### `pymort fit one`

Fit a specific ModelName directly.

**Options**

- `--model [LCM1|LCM2|APCM3|CBDM5|CBDM6|CBDM7]` (required)
- `--m-path PATH` (required)
- `--smoothing none|cpsplines` (default none)
- `--eval-on-raw BOOL` (default True)
- `--cpsplines-k INT`, `--cpsplines-horizon INT`
- `--output PATH` (default: outdir/fitted_<MODEL>.pkl)
- plus ages/years list-or-range + ages_path/years_path

**Example**

```bash
pymort fit one --model LCM2 --m-path Data/france.csv --ages 60-100 --years 1970-2019 \
  --smoothing cpsplines --cpsplines-horizon 0 --output outputs/fitted_lcm2.pkl
```

### `pymort fit select`

Compare models by forecast RMSE and select best.

**Required**

- `--m-path PATH`
- `--train-end YEAR` (last year included in training set)

**Optional**

- `--models/-m NAME` repeated (if empty: uses all default models)
- `--metric log_m|logit_q` (default logit_q)
- `--output PATH` (default: outdir/model_selection.csv but written using `--format`)

**Example**

```bash
pymort fit select --m-path Data/france.csv --train-end 2015 --metric logit_q --output outputs/model_selection.csv
```

### `pymort fit select-and-fit`

Select best model and fit it for pricing (also outputs selection table).

**Required**

- `--m-path PATH`
- `--train-end YEAR`

**Optional**

- `--models/-m ...`
- `--metric log_m|logit_q`
- `--cpsplines-k`, `--cpsplines-horizon`
- `--output PATH` (default outdir/fitted_best.pkl, saves a dict {selection, fitted})
- `--selection-output PATH` (default outdir/model_selection.csv)

**Example**

```bash
pymort fit select-and-fit --m-path Data/france.csv --train-end 2015 --metric logit_q \
  --cpsplines-horizon 0 --output outputs/fitted_best.pkl --selection-output outputs/selection.csv
```

---

## 4) scen — Scenario generation & summary

### `pymort scen build-P`

Build real-world (P-measure) scenarios using the projection pipeline.

**Required**

- `--m-path PATH`
- `--train-end YEAR`

**Key options**

- `--horizon INT` (default 50)
- `--n-scenarios INT` (default 1000)
- `--models/-m ...` subset (default: all supported)
- `--cpsplines-k`, `--cpsplines-horizon`
- `--seed` (local option; falls back to global `--seed`)
- `--output PATH` (default outdir/scenarios_P.npz)
- plus ages/years selection

**Example**

```bash
pymort scen build-P --m-path Data/france.csv --train-end 2015 --horizon 40 --n-scenarios 2000 \
  --ages 60-100 --years 1970-2019 --output outputs/scenarios_P.npz
```

### `pymort scen build-Q`

Build risk-neutral (Q-measure) scenarios under an Esscher tilt lambda_esscher.

**Required**

- `--m-path PATH`
- `--lambda-esscher FLOAT`

**Key options**

- `--model-name` (default CBDM7)
- `--B-bootstrap`, `--n-process`, `--horizon`
- `--scale-sigma` (default 1.0)
- `--include-last`
- `--seed` (local option; falls back to global)
- `--output PATH` (default outdir/scenarios_Q.npz)
- plus ages/years selection

**Example**

```bash
pymort scen build-Q --m-path Data/france.csv --lambda-esscher 0.1 --horizon 40 --output outputs/scenarios_Q.npz
```

### `pymort scen summarize`

Summarize a scenario set (writes JSON summary).

**Example**

```bash
pymort scen summarize --scen-path outputs/scenarios_P.npz --output outputs/scen_summary.json
```

---

## 5) forecast — Scenarios from a fitted model

### `pymort forecast MODEL_PATH`

Generate scenarios from a pickled fitted model.

**MODEL_PATH**

- must be a pickled FittedModel, or a dict with key "fitted" (as produced by fit select-and-fit).

**Options**

- `--horizon INT` (default 50)
- `--scenarios INT` (default 1000)
- `--resample cell|year_block` (default year_block)
- `--include-last BOOL` (default True)
- `--output PATH` (default outdir/scenarios_forecast.npz)

**Example**

```bash
pymort forecast outputs/fitted_best.pkl --horizon 40 --scenarios 2000 --output outputs/scenarios_forecast.npz
```

---

## 6) stress — Stress testing

### `pymort stress apply`

Apply a single built-in shock type.

**Options**

- `--scen-path PATH` (required)
- `--shock-type` (default long_life)
- `--magnitude FLOAT` (default 0.1)
- optional pandemic/plateau/acceleration timing knobs:
  - `--pandemic-year`, `--pandemic-duration`
  - `--plateau-start-year`
  - `--accel-start-year`
- `--output PATH` (default outdir/scenarios_<shock>.npz)

**Example**

```bash
pymort stress apply --scen-path outputs/scenarios_P.npz --shock-type long_life --magnitude 0.1
```

### `pymort stress chain`

Apply a sequence of shocks from a YAML/JSON list.

**chain-spec format**

A list of dicts like:

```yaml
- name: pandemic
  shock_type: pandemic
  params:
    magnitude: 0.2
    pandemic_year: 2020
    pandemic_duration: 2
- name: longlife
  shock_type: long_life
  params:
    magnitude: 0.05
```

**Example**

```bash
pymort stress chain --scen-path outputs/scenarios_P.npz --chain-spec configs/shock_chain.yaml --output outputs/scenarios_chain.npz
```

### `pymort stress bundle`

Create a bundle directory containing:

- base.npz
- optimistic.npz (long-life bump)
- pessimistic.npz (short-life bump)
- manifest.json

**Example**

```bash
pymort stress bundle --scen-path outputs/scenarios_P.npz --long-life-bump 0.1 --short-life-bump 0.1 --output outputs/bundle
```

---

## 7) price — Pricing instruments

All price ... commands take a scenario set:

- `--scen-path PATH` (required, .npz)
- optional `--short-rate` (flat rate; if omitted, code uses 0.0)

### `pymort price longevity-bond`

**Options**: `--issue-age`, `--maturity-years`, `--notional`, `--include-principal/--no-include-principal`, `--short-rate`, `--output`

**Example**

```bash
pymort price longevity-bond --scen-path outputs/scenarios_Q.npz --issue-age 65 --maturity-years 20 \
  --short-rate 0.02 --output outputs/price_longevity_bond.json
```

### `pymort price survivor-swap`

**Options**: `--age`, `--maturity-years`, `--notional`, `--strike`, `--payer fixed|floating`, `--short-rate`, `--output`

**Example**

```bash
pymort price survivor-swap --scen-path outputs/scenarios_Q.npz --age 70 --maturity-years 15 \
  --payer fixed --short-rate 0.02 --output outputs/price_survivor_swap.json
```

### `pymort price q-forward`

**Options**: `--age`, `--maturity-years`, `--strike`, `--settlement-years`, `--notional`, `--short-rate`

### `pymort price s-forward`

Same idea as q-forward but for S.

### `pymort price life-annuity`

**Options**: `--issue-age`, optional `--maturity-years`, `--payment-per-survivor`, `--defer-years`,
`--exposure-at-issue`, `--include-terminal`, `--terminal-notional`, `--short-rate`

---

## 8) price-bond — Convenience wrapper

### `pymort price-bond`

Prices a longevity bond from either:

- a scenario set .npz, or
- a fitted model pickle (it will project scenarios internally)

**Key options**

- `--model PATH` (required): .npz or .pkl
- `--maturity INT` (required)
- `--coupon survivor` (only supported value)
- `--issue-age FLOAT` (optional; if omitted, uses median age of scenario set)
- `--notional`, `--include-principal/--no-include-principal`
- `--short-rate FLOAT` (optional; default internally 0.0)
- `--scenarios INT` (only if projecting from fitted model)
- `--output PATH`

**Examples**

```bash
# from scenarios
pymort price-bond --model outputs/scenarios_Q.npz --maturity 20 --short-rate 0.02

# from fitted model (projects scenarios up to maturity)
pymort price-bond --model outputs/fitted_best.pkl --maturity 20 --scenarios 2000 --short-rate 0.02
```

---

## 9) rn — Risk-neutral calibration & pricing under lambda

### `pymort rn calibrate-lambda`

Calibrate Esscher lambda to match market quotes.

**Required**

- `--quotes-path PATH` : YAML/JSON list of quotes
- `--m-path PATH`

**Quote format**

A list of dicts; each item must have:

- `kind` : longevity_bond | survivor_swap | s_forward | q_forward | life_annuity (hyphens tolerated)
- `spec` : dict matching the dataclass fields for that instrument
- `market_price` : float

Optional:

- `name`
- `weight`

**Example quotes YAML**

```yaml
- name: bond_20y
  kind: longevity_bond
  market_price: 0.93
  weight: 1.0
  spec:
    issue_age: 65
    maturity_years: 20
    notional: 1.0
    include_principal: true
```

**Other options**

- `--model-name` (default CBDM7)
- `--lambda0`, `--bounds "-5,5"`
- `--B-bootstrap`, `--n-process`
- `--short-rate`, optional `--horizon`
- `--include-last`, `--seed`
- plus ages/years selection

**Outputs (fixed paths inside outdir)**

- outdir/lambda_calibration.json
- outdir/calibration_cache.pkl
- outdir/scenarios_Q_calibrated.npz

**Example**

```bash
pymort rn calibrate-lambda --quotes-path configs/quotes.yaml --m-path Data/france.csv \
  --short-rate 0.02 --bounds -5,5 --B-bootstrap 50 --n-process 200
```

### `pymort rn price-under-lambda`

Price a basket of instruments under a given lambda (builds cache + scenarios internally).

**Required**

- `--lambda-val FLOAT`
- `--m-path PATH`
- `--specs PATH` : YAML/JSON mapping name -> {kind, spec}

**Example specs YAML**

```yaml
bond:
  kind: longevity_bond
  spec:
    issue_age: 65
    maturity_years: 20
    notional: 1.0
    include_principal: true
swap:
  kind: survivor_swap
  spec:
    age: 70
    maturity_years: 15
    notional: 1.0
    payer: fixed
```

**Example**

```bash
pymort rn price-under-lambda --lambda-val 0.1 --m-path Data/france.csv --specs configs/specs.yaml \
  --short-rate 0.02 --horizon 50 --output outputs/prices_under_lambda.json
```

---

## 10) sens — Sensitivities

All commands here price via pricing_pipeline on a scenario set.

### `pymort sens rate`

Computes first-order sensitivity to rate bump.

**Required**:

- `--scen-path PATH`
- `--kind KIND`
- `--spec-path PATH` (YAML/JSON for the instrument spec)
- `--base-short-rate FLOAT`

**Optional**:

- `--bump FLOAT` (default 1e-4)
- `--output PATH`

### `pymort sens convexity`

Same inputs as sens rate, returns convexity.

### `pymort sens delta-by-age`

Computes delta to mortality bump by age.

**Options**:

- `--rel-bump` (default 0.01)
- `--ages "60,65,70"` optional subset list
- `--short-rate` (default 0.0)

### `pymort sens all`

Runs risk_analysis_pipeline and returns a bundle:

- base prices
- vega via sigma scaling
- delta by age
- rate sensitivity + convexity

**Required**:

- `--scen-path PATH`
- `--specs-path PATH` mapping of specs

**Optional**:

- `--short-rate`, `--sigma-rel-bump`, `--q-rel-bump`, `--rate-bump`, `--output`

---

## 11) hedge — Hedging

There are two modes:

### A) PV paths mode (no scenarios)

- liability PV paths: 1D array (csv/parquet/npy)
- hedge PV paths: 2D matrix (N,M)

**Command**

- `pymort hedge min-variance --liab-pv-path ... --instr-pv-path ...`

### B) End-to-end mode (specs + scenarios)

You provide:

- `--scenarios` scenario set .npz
- liability specs file (YAML/JSON)
- instrument specs file (YAML/JSON)

The CLI prices PV paths internally, then hedges.

**Command**

- `pymort hedge end-to-end ...`

### `pymort hedge` (default alias)

Calling `pymort hedge --liabilities ... --instruments ...`:

- If either file looks like a spec file (.json/.yaml/.yml) or you provide `--scenarios`, it routes to end-to-end.
- Otherwise it expects PV paths and routes to min-variance.

### `pymort hedge end-to-end`

**Options**:

- `--scenarios PATH` (required)
- `--liabilities PATH` (required) YAML/JSON specs
- `--instruments PATH` (required) YAML/JSON specs
- `--method` (default min_variance)
- `--short-rate` optional (used if scenario set has no discount factors)
- `--output` (default: outdir/hedge_end_to_end.json)

### `pymort hedge min-variance`

**Options**:

- `--liab-pv-path PATH` (required)
- `--instr-pv-path PATH` (required)
- `--output` (default outdir/hedge_min_variance.json)

### `pymort hedge multihorizon`

Multi-horizon hedging using CF paths:

- liability CF paths: (N,T)
- instrument CF paths: (N,M,T) (requires a true 3D .npy/.npz)

Optional constraints:

- `--discount-factors-path` 1D
- `--time-weights-path` 1D

---

## 12) report — Risk reporting

### `pymort report risk`

Compute risk metrics (VaR etc.) on PV paths.

**Options**:

- `--pv-path PATH` (required)
- `--name` (optional; defaults to file stem)
- `--var-level FLOAT` (default 0.95)
- `--ref-pv-path PATH` (optional)
- `--output PATH` (default outdir/risk_<name>.json)

**Example**

```bash
pymort report risk --pv-path outputs/bond_pv.npy --var-level 0.99
```

---

## 13) plot — Plotting helpers

These commands require matplotlib.

### `pymort plot survival-fan`

- `--scen-path`, `--age`, `--quantiles "5,50,95"`, `--output PNG`

### `pymort plot price-dist`

Histogram from PV paths:

- `--pv-path`, `--bins`, `--output PNG`

### `pymort plot lexis`

Lexis heatmap-ish plot from scenario set:

- `--scen-path`
- `--value m|q|S`
- `--statistic mean|median`
- `--cohorts "1950,1960,1970"` (optional)
- `--output PNG`

### `pymort plot fan`

General fan chart:

- `--value S|q`
- `--age`, `--quantiles`

---

## 14) run — One-click pipelines

### `pymort run pricing-pipeline --config PATH`

Runs: load data → build scenarios (P or Q depending on config) → price instruments → save outputs.

**Config structure (high level):**

- `outputs`: outdir, format
- `data`: m_path, optional sex, optional age_min/max, year_min/max
- `fit`: optional models, train_end, cpsplines
- `scenarios`: measure = P or Q, horizon, n_scenarios / B_bootstrap / n_process, etc.
- `pricing`: short_rate, instruments mapping

### `pymort run hedge-pipeline --config PATH`

Loads scenario set + prices liability and hedge specs + computes hedge weights.

**Config:**

- `data`: scen_path, optional short_rate
- `liability`: spec dict
- `hedge_instruments`: mapping
- `hedging`: method, output

---

## 15) Utilities

### `pymort version`

Print installed package version (or 0.0.dev).

### `pymort echo "message"`

Echo utility.

---

## Practical end-to-end examples

### A) From raw mortality to P scenarios + stress + summary

```bash
pymort data validate-m --m-path Data/france.csv --ages 60-100 --years 1970-2019
pymort scen build-P --m-path Data/france.csv --train-end 2015 --horizon 40 --n-scenarios 2000 \
  --ages 60-100 --years 1970-2019 --output outputs/scenarios_P.npz
pymort stress apply --scen-path outputs/scenarios_P.npz --shock-type long_life --magnitude 0.1 \
  --output outputs/scenarios_P_longlife.npz
pymort scen summarize --scen-path outputs/scenarios_P.npz --output outputs/scen_summary.json
```

### B) Fit → forecast → price a bond

```bash
pymort fit select-and-fit --m-path Data/france.csv --train-end 2015 --metric logit_q --output outputs/fitted_best.pkl
pymort forecast outputs/fitted_best.pkl --horizon 20 --scenarios 2000 --output outputs/scenarios_forecast.npz
pymort price longevity-bond --scen-path outputs/scenarios_forecast.npz --issue-age 65 --maturity-years 20 \
  --short-rate 0.02 --output outputs/price_bond.json
```

### C) Risk-neutral scenarios (Q) + pricing

```bash
pymort scen build-Q --m-path Data/france.csv --lambda-esscher 0.1 --horizon 40 --output outputs/scenarios_Q.npz
pymort price survivor-swap --scen-path outputs/scenarios_Q.npz --age 70 --maturity-years 15 \
  --short-rate 0.02 --output outputs/price_swap.json
```

---

## Notes / gotchas

- For .npy/.npz mortality matrices (m_path), you must provide `--ages` and `--years` as lists (or via `--ages-path/--years-path`), because shape checks rely on them.
- For CSV/Parquet/Excel, `--ages` and `--years` can be ranges like 60-100 / 1970-2019 to slice after loading.
- Scenario sets must be .npz (the loader rejects other formats).
