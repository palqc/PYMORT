<h1 align="center">🧮 PyMORT — Longevity Bond Pricing & Mortality Modeling</h1>

<p align="center">
  <em>A teaching-size Python library and CLI for pricing longevity-linked securities and modeling mortality risk.</em><br>

  <a href="https://github.com/palqc/PYMORT/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" />
  </a>

  <a href="https://github.com/palqc/PYMORT/actions/workflows/ci.yml">
    <img src="https://github.com/palqc/PYMORT/actions/workflows/ci.yml/badge.svg" />
  </a>

  <a href="https://github.com/palqc/PYMORT/actions/workflows/release.yml">
    <img src="https://github.com/palqc/PYMORT/actions/workflows/release.yml/badge.svg?branch=main" />
  </a>

  <a href="https://codecov.io/gh/palqc/PYMORT">
    <img src="https://codecov.io/gh/palqc/PYMORT/branch/main/graph/badge.svg" />
  </a>

  <a href="https://pypi.org/project/pymort-palqc/">
    <img src="https://img.shields.io/pypi/v/pymort?style=flat-square" />
  </a>

  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" />
</p>


📦 **PyPI package**: https://pypi.org/project/pymort-palqc/

---

## ✨ Overview
**PyMORT** provides a compact yet extensible framework for **mortality modeling** and **longevity-linked security pricing**.  
It is designed for educational and research purposes within the *Data Science & Advanced Programming* MSc course (HEC Lausanne, Winter 2025).


---


## 📦 Key Features

### Mortality Models
- Lee-Carter model for mortality forecasting
- Cairns-Blake-Dowd (CBD) model extensions
- Age-Period-Cohort models
- Stochastic mortality projections

### Pricing Instruments
- Longevity bonds (survivor-linked coupons)
- Survivor swaps and forwards
- q-forwards (mortality derivatives)
- Annuity valuations

### Risk Analysis
- Scenario analysis and stress testing
- Sensitivity to mortality parameters
- Hedging strategy optimization
- Mortality surface visualization

### Core Tools
- CLI and Python package modes
- Full test coverage (80%+) with pytest and hypothesis
- Type safety via strict mypy configuration
- Reproducible builds using Makefile targets

---

## 📊 Project Structure

```
pymort/
├── src/
│   └── pymort/
│       ├── analysis/                 # Mortality analysis & risk tools
│       │   ├── bootstrap.py
│       │   ├── fitting.py
│       │   ├── projections.py
│       │   ├── reporting.py
│       │   ├── risk_tools.py
│       │   ├── scenario.py
│       │   ├── scenario_analysis.py
│       │   ├── sensitivities.py
│       │   ├── smoothing.py           # CPsplines-based smoothing (optional)
│       │   └── validation.py
│       │
│       ├── interest_rates/            # Interest-rate models
│       │   └── hull_white.py
│       │
│       ├── models/                    # Mortality models
│       │   ├── apc_m3.py
│       │   ├── cbd_m5.py
│       │   ├── cbd_m6.py
│       │   ├── cbd_m7.py
│       │   ├── gompertz.py
│       │   ├── lc_m1.py
│       │   ├── lc_m2.py
│       │   └── utils.py
│       │
│       ├── pricing/                   # Pricing of longevity-linked instruments
│       │   ├── hedging.py
│       │   ├── liabilities.py
│       │   ├── longevity_bonds.py
│       │   ├── mortality_derivatives.py
│       │   ├── risk_neutral.py
│       │   ├── survivor_swaps.py
│       │   └── utils.py
│       │
│       ├── visualization/             # Plotting & diagnostics
│       │   ├── fans.py
│       │   └── lexis.py
│       │
│       ├── cli.py                     # Command-line interface
│       ├── lifetables.py
│       ├── pipeline.py                # High-level pricing & sensitivity pipeline
│       ├── utils.py
│       ├── _types.py
│       └── py.typed                   # PEP 561 typing marker
│
├── streamlit_app/                     # Interactive Streamlit application
│   ├── App.py
│   ├── pages/
│   │   ├── 1_Data_Upload.py
│   │   ├── 2_Data_Slicing.py
│   │   ├── 3_Fit_Select.py
│   │   ├── 4_Projection_P.py
│   │   ├── 5_Risk_Neutral_Q.py
│   │   ├── 6_Pricing.py
│   │   ├── 7_Hedging.py
│   │   ├── 8_Scenario_Analysis.py
│   │   ├── 9_Sensitivities.py
│   │   └── 10_Report_Export.py
│   ├── assets/
│   │   └── logo.png
│   └── .streamlit/
│       ├── config.toml
│       └── secrets.toml
│
├── cpsplines/                         # External CPsplines dependency (optional)
│   └── README.md                      # Install notes & Python ≥ 3.12 requirement
│
├── tests/                             # Pytest suite (≥80% coverage)
│
├── validation_against_StMoMo/         # External validation vs R (StMoMo)
│   ├── stmomo_fit_cbd.R
│   ├── stmomo_fit_lc.R
│   ├── validate_vs_stmomo.py
│   └── outputs/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                     # CI: tests, coverage, ruff, mypy
│       └── release.yml                # Build & PyPI release
│
├── .coverage                          # Local coverage database (gitignored)
├── coverage.xml                       # Coverage report (CI / Codecov)
│
├── .editorconfig
├── .gitignore
├── .pre-commit-config.yaml            # Pre-commit hooks (ruff, mypy, etc.)
├── .secrets.baseline                  # Secret scanning baseline
│
├── CONTRIBUTING.md                    # Contribution guidelines
├── PROJECT_SPECIFICATION.md           # Technical & academic specification
├── README.md                          # Main project README
├── README_cli.md                      # CLI documentation
├── LICENSE                            # MIT license
├── Makefile                           # Developer shortcuts
├── pyproject.toml                     # Build, deps, tooling config
└── requirements.txt
```
---

## 🛠️ Development Workflow

```bash
make install-dev    # Set up development environment
make check          # Run all quality checks
make test           # Run tests with coverage
```

---

## 📖 Documentation

See [PROJECT_SPECIFICATION.md](PROJECT_SPECIFICATION.md) for full project requirements.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

See [README_cli.md](README_cli.md) for CLI documentation.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Author

Developed and maintained by Pierre-Antoine Le Quellec (@palqc)
MSc Finance – HEC Lausanne | Focus: Financial Data Science & Risk Analytics.