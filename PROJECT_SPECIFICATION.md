# PYMORT: Longevity Bond Pricing & Mortality Modeling
## Project Specification for Individual Project

---

## Project Overview

In this project, you will develop PYMORT, a sophisticated longevity bond pricing engine that models mortality risk and prices longevity-linked securities. This project combines actuarial science with advanced financial mathematics, tackling one of the most challenging problems in modern finance: how to price and hedge longevity risk.

Longevity risk—the risk that people live longer than expected—poses significant challenges for pension funds, insurers, and governments. Your PYMORT package will enable users to model mortality dynamics, project future mortality rates, and price financial instruments that transfer this risk.

## Problem Statement

Consider a pension fund with obligations to pay benefits for life. If beneficiaries live longer than actuarially expected, the fund faces unexpected liabilities. Longevity bonds—securities whose payments depend on realized survival rates—allow hedging this risk. But how should these bonds be priced?

Your PYMORT package must enable users to:
- Model mortality rates and their stochastic evolution
- Fit sophisticated mortality models to historical data
- Project future mortality with quantified uncertainty
- Price longevity bonds, survivor swaps, and mortality derivatives
- Analyze risk and develop hedging strategies

## Technical Requirements

### Core Architecture@

**Mortality Models**:

**Lee-Carter Model**: The benchmark model for mortality forecasting
- Represents log mortality rate as: ln(m_x,t) = a_x + b_x * k_t + ε_x,t
- a_x: average age-specific mortality pattern
- b_x: age-specific sensitivity to mortality improvement
- k_t: time-varying mortality index (often modeled as random walk with drift)
- Fit via Singular Value Decomposition (SVD)

**Cairns-Blake-Dowd (CBD) Model**: Focused on higher ages
- logit(q_x,t) = κ₁(t) + κ₂(t) * (x - x̄)
- Two time-varying parameters for level and slope
- Better for pension ages (typically 60+)
- Extensions include cohort effects and quadratic terms

**Stochastic Projections**:
- Simulate future mortality paths using fitted models
- Account for parameter uncertainty via bootstrap
- Generate mortality fans (probability distributions over time)
- Validate against out-of-sample data

**Pricing Engine**:

**Longevity Bonds**: Bonds with survivor-linked payments
- Coupons proportional to realized survival rates
- Principal repayment linked to cohort survival
- Require risk-neutral pricing framework

**Survivor Swaps**: Exchange fixed payments for floating survivor-linked payments
- Used by pension funds to hedge longevity risk
- Pricing requires mortality market price of risk

**Mortality Derivatives**: q-forwards, s-forwards
- Bet on realized mortality rates at specific ages
- Building blocks for complex hedging strategies

**Risk-Neutral Valuation**:
- Calibrate market price of longevity risk
- Transform real-world to risk-neutral measure
- Price via expected discounted cashflows under Q-measure

**Analysis Tools**:

**Scenario Analysis**: Test pricing under different mortality scenarios
- Base case, optimistic (low mortality), pessimistic (high mortality)
- Stress test with extreme longevity improvements

**Sensitivity Analysis**: Measure impact of parameter changes
- Delta: sensitivity to mortality rates at each age
- Vega: sensitivity to mortality volatility
- Duration: sensitivity to discount rates

**Hedging**: Construct portfolios to offset longevity exposure
- Match durations and convexities
- Minimize basis risk from model mismatch

### Command-Line Interface

```bash
# Fit mortality model to data
pymort fit mortality_data.csv --model lee-carter --ages 60-100

# Generate mortality forecasts
pymort forecast fitted_model.pkl --horizon 50 --scenarios 1000

# Price longevity bond
pymort price-bond --model fitted_model.pkl --maturity 20 --coupon survivor

# Analyze hedging strategy
pymort hedge --liabilities pension_obligations.csv --instruments bond_universe.csv
```

### Implementation Challenges

**Model Calibration**: Fit complex models to sparse and noisy mortality data. Handle cohort effects and structural breaks (e.g., COVID-19).

**Numerical Stability**: Mortality rates span many orders of magnitude (from 10⁻⁴ for young ages to 0.5+ for old ages). Use log-scale computations.

**Data Quality**: Historical mortality data has gaps, inconsistencies, and varies by jurisdiction. Implement robust data cleaning and smoothing.

**Performance**: Pricing via Monte Carlo requires thousands of simulations of 50+ year mortality trajectories across 100+ ages. Optimize with NumPy vectorization.

**Validation**: Compare model outputs with published life tables (e.g., Social Security Administration, ONS UK) and academic benchmarks.

### Software Engineering Standards

**Type Safety**: Complete type hints including NumPy array shapes where possible. MyPy strict mode.

**Testing**: 80% coverage minimum. Validate Lee-Carter against published implementations. Test CBD model reproduces paper results. Property-based tests (e.g., survival probabilities decrease with age).

**Code Quality**: Ruff with extensive rules. Google-style docstrings explaining actuarial concepts for non-experts.

**CI/CD**: GitHub Actions across Python 3.10-3.12, multiple OS.

## Extensions and Enhancements

**Advanced Models**:
- Age-Period-Cohort (APC) models
- Functional data analysis approaches
- Multi-population models (males/females, multiple countries)
- Non-parametric mortality forecasting

**Calibration to Market Data**:
- Fit models to observed longevity bond prices
- Extract implied market price of longevity risk
- Develop arbitrage-free frameworks

**Stochastic Interest Rates**:
- Joint modeling of mortality and interest rate risk
- Affine term structure models
- Hull-White or CIR dynamics

**Machine Learning**:
- Neural networks for mortality forecasting
- Incorporating economic and health covariates
- Deep learning for mortality surfaces

**Visualization**:
- Animated mortality surfaces over time
- Interactive mortality fans
- Lexis diagrams for cohort analysis
- Dashboard for pricing and risk metrics

## Deliverables

1. **Tagged GitHub repository** (v1.0.0) with all code, tests, and documentation
2. **Comprehensive README** serving as both user guide and technical documentation
3. **Published package on TestPyPI** installable via pip
4. **Technical report** (10 pages):
   - Model validation against published data
   - Pricing example with real mortality data
   - Comparison of models (Lee-Carter vs. CBD)
   - Discussion of longevity risk management
5. **Live demonstration**: Price a longevity bond using real mortality data (e.g., Human Mortality Database)

## Getting Started

1. **Learn mortality modeling**: Study Lee-Carter (1992) and Cairns-Blake-Dowd (2006) papers
2. **Understand life tables**: Explore Human Mortality Database (HMD) for various countries
3. **Review longevity bonds**: Read about EIB/BNP Paribas longevity bond (2004)
4. **Start simple**: Fit Lee-Carter to single country, validate against published forecasts
5. **Expand**: Add CBD model, implement pricing, develop analysis tools

### Validation Strategy

**Lee-Carter Model**:
- Fit to US mortality 1950-2000
- Compare forecasts 2000-2020 with realized mortality
- Reproduce published k_t trajectories and forecast errors

**Pricing Validation**:
- Price simple survivor bond with deterministic mortality
- Compare with manual calculation
- Add stochasticity and verify via Monte Carlo variance

## Domain Knowledge Required

**Actuarial Science**:
- Life tables and survival functions
- Mortality rate definitions (q_x vs. m_x)
- Basic demography concepts

**Financial Mathematics**:
- Bond pricing and yield curves
- Risk-neutral valuation
- Derivatives pricing fundamentals

**Statistics**:
- Time series models
- Principal component analysis
- Maximum likelihood estimation

**No prior actuarial experience needed**—all concepts will be learned through implementation!

## Assessment Focus

**Mathematical Rigor**: Correct implementation of Lee-Carter, CBD, and pricing formulas.

**Model Validation**: Careful comparison with published results and real data.

**Innovation**: Creative approaches to mortality modeling and risk analysis.

**Practical Value**: Tools that practicing actuaries could actually use.

**Code Quality**: Production-grade code with comprehensive tests.

---

## Resources

### Essential Reading
- 📖 **Lee, R. & Carter, L.** (1992) "Modeling and Forecasting U.S. Mortality" - The foundational paper
- 📖 **Cairns, A., Blake, D., & Dowd, K.** (2006) "A Two-Factor Model for Stochastic Mortality" - CBD model
- 📖 **Pitacco, E. et al.** (2009) *Modelling Longevity Dynamics for Pensions and Annuity Business* - Comprehensive textbook
- 📖 **Blake, D. et al.** (2006) "Living with Mortality: Longevity Bonds and Other Mortality-Linked Securities" - Longevity bond primer

### Data Sources
- 🔧 [Human Mortality Database](https://www.mortality.org/) - High-quality mortality data for 40+ countries
- 🔧 [Human Fertility Database](https://www.humanfertility.org/) - Population data
- 🔧 [Social Security Administration](https://www.ssa.gov/oact/HistEst/index.html) - US cohort life tables

### Software
- 🔧 [StMoMo (R)](https://cran.r-project.org/web/packages/StMoMo/) - Stochastic mortality models in R
- 🔧 [lifecontingencies (R)](https://cran.r-project.org/web/packages/lifecontingencies/) - Actuarial computations
- 🔧 [lifelines](https://github.com/CamDavidsonPilon/lifelines) - Survival analysis in Python

### Academic Resources
- 📺 [Institute and Faculty of Actuaries](https://www.actuaries.org.uk/) - Professional body with resources
- 📺 [Society of Actuaries](https://www.soa.org/) - US actuarial organization
- 📺 [Longevity Risk Working Party](https://www.actuaries.org.uk/learn-and-develop/continuous-professional-development/research-and-knowledge/longevity-risk) - Latest research

---

## Summary

PYMORT challenges you to master one of the most sophisticated areas in actuarial finance. You're building a tool for pricing securities whose value depends on whether people live or die—a profoundly important but technically demanding problem.

Success requires:
- Understanding demographic and mortality dynamics
- Implementing advanced statistical models
- Applying financial mathematics rigorously
- Validating carefully against real data
- Writing production-quality code

Tackle one of the most challenging problems in actuarial finance with this cutting-edge project!

---

*Questions? Contact course instructors during office hours or via email*
