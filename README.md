<h1 align="center">🧮 PyMORT — Longevity Bond Pricing & Mortality Modeling</h1>

<p align="center">
  <em>A teaching-size Python library and CLI for pricing longevity-linked securities and modeling mortality risk.</em><br>

  <a href="https://github.com/palqc/PYMORT/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" />
  </a>

  <a href="https://github.com/palqc/PYMORT/actions/workflows/ci.yml">
    <img src="https://github.com/palqc/PYMORT/actions/workflows/ci.yml/badge.svg" />
  </a>
  <a href="https://codecov.io/gh/palqc/PYMORT">
    <img src="https://codecov.io/gh/palqc/PYMORT/branch/main/graph/badge.svg" />
  </a>

  <a href="https://github.com/palqc/PYMORT/actions/workflows/release.yml">
    <img src="https://github.com/palqc/PYMORT/actions/workflows/release.yml/badge.svg?branch=main" />
  </a>

  <img src="https://img.shields.io/badge/coverage-%E2%89%A580%25-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/ruff-strict-blueviolet?style=flat-square" />
  <img src="https://img.shields.io/badge/mypy-strict-blue?style=flat-square" />

  <a href="https://pypi.org/project/pymort/">
    <img src="https://img.shields.io/pypi/v/pymort?style=flat-square" />
  </a>

  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" />
</p>

---

## ✨ Overview
**PyMORT** provides a compact yet extensible framework for **mortality modeling** and **longevity-linked security pricing**.  
It is designed for educational and research purposes within the *Data Science & Advanced Programming* MSc course (HEC Lausanne, Winter 2025).


---

## 🚀 Quick demo

```bash
# editable install
pip install -e .[dev]

# fit Lee-Carter mortality model
pymort fit data/mortality.csv --model lee-carter --forecast 30
# > Fitted Lee-Carter model
# > kt parameter: -0.023 (annual improvement)
# > 30-year mortality forecast generated

# price longevity bond
pymort price-bond data/mortality.csv --maturity 20 --coupon survivor
# > Bond price: 87.3
# > Duration: 14.2 years
# > Convexity: 243.5
```

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
src/pymort/        # Main package
├── __init__.py    # Public API exports
├── cli.py         # CLI interface
├── models.py      # Mortality models (Lee-Carter, CBD, etc.)
├── lifetable.py   # Life table operations
├── pricing.py     # Bond and derivative pricing
└── analysis.py    # Risk analysis and scenarios

tests/             # Test suite
└── test_*.py      # Test modules
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

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Author

Developed and maintained by Pierre-Antoine Le Quellec (@palqc)
MSc Finance – HEC Lausanne | Focus: Financial Data Science & Risk Analytics.