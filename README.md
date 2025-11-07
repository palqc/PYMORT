# PyMort – Longevity Bond Pricing & Mortality Modeling

_"A teaching-size library and CLI for pricing longevity-linked securities and modeling mortality risk."_

This repository is a **template** for the Winter 2025 MSc final project.
Fork or use as a GitHub Template → complete the TODOs → tag **`v1.0.0`**.

---

## ✨ Quick demo

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

## 📦 What's included

- **Mortality models**: Lee-Carter, Cairns-Blake-Dowd, stochastic projections
- **Pricing engine**: Longevity bonds, survivor swaps, mortality derivatives
- **Risk analysis**: Scenario testing, stress analysis, hedging strategies
- **Data handling**: Life tables, cohort effects, data smoothing
- **CLI and library**: Use from command line or as a Python package
- **Full test coverage**: 80%+ with pytest and hypothesis
- **Type safety**: Strict MyPy configuration

---

## 🛠️ Development Workflow

```bash
make install-dev    # Set up development environment
make check          # Run all quality checks
make test           # Run tests with coverage
```

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

## 🎯 Key Features

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

---

## 📖 Documentation

See [PROJECT_SPECIFICATION.md](PROJECT_SPECIFICATION.md) for full project requirements.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
