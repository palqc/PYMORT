🧬 PyMort Streamlit App

Longevity Modeling, Projection & Pricing Interface

This Streamlit application provides an interactive end-to-end interface for mortality modeling, longevity risk projection, risk-neutral valuation, pricing, hedging, and scenario analysis, powered by the pymort Python library.

The app is designed for actuarial, quantitative finance, and longevity risk research, with a clear step-by-step workflow mirroring a real-world modeling pipeline.

⸻

🚀 Features
	•	📥 Data upload & validation (mortality rates, exposures, years, ages)
	•	✂️ Data slicing & preprocessing
	•	🧠 Model selection & fitting
	•	Lee-Carter family
	•	APC / CBD variants
	•	📈 Stochastic projections under the physical measure (P)
	•	⚖️ Risk-neutral transformation (Q)
	•	💰 Pricing of longevity-linked instruments
	•	Survivor swaps
	•	Longevity bonds
	•	Life annuities
	•	🛡️ Hedging analysis
	•	🌪️ Scenario & stress testing
	•	📊 Sensitivities & risk metrics
	•	📤 Report export

⸻

🗂️ App Structure

streamlit_app/
├── App.py               
├── pages/
│   ├── 1_Data_Upload.py
│   ├── 2_Data_Slicing.py
│   ├── 3_Fit_Select.py
│   ├── 4_Projection_P.py
│   ├── 5_Risk_Neutral_Q.py
│   ├── 6_Pricing.py
│   ├── 7_Hedging.py
│   ├── 8_Scenario_Analysis.py
│   ├── 9_Sensitivities.py
│   └── 10_Report_Export.py
├── assets/
│   └── logo.png
└── README.md

The app follows a linear and transparent workflow, allowing users to move sequentially from raw data to pricing and risk outputs.

⸻

▶️ Running the App Locally

1️⃣ Install dependencies

From the root of the repository:

pip install -e .[dev]

2️⃣ Launch Streamlit

cd streamlit_app
streamlit run App.py


⸻

🧭 Workflow Overview
	1.	Data Upload
Load mortality surfaces (rates or log-rates), ages, and calendar years.
	2.	Data Slicing
Restrict age ranges, calendar windows, or cohorts.
	3.	Fit & Model Selection
Fit stochastic mortality models and inspect parameters.
	4.	Projection (P-measure)
Generate stochastic mortality paths under the real-world measure.
	5.	Risk-Neutral Measure (Q)
Apply Esscher / pricing kernel transformations.
	6.	Pricing
Price longevity-linked liabilities and instruments.
	7.	Hedging
Analyze hedge effectiveness and residual risk.
	8.	Scenario Analysis
Stress longevity improvements or shocks.
	9.	Sensitivities
Compute deltas, quantiles, and risk metrics.
	10.	Report Export
Export structured outputs for further analysis.

⸻

🎯 Target Audience
	•	Actuarial science students & researchers
	•	Quantitative finance practitioners
	•	Longevity risk & insurance analytics
	•	Academic projects and demonstrations

⸻

⚠️ Notes
	•	The app is research-oriented, not intended for production pricing without independent validation.
	•	All computations rely on the underlying pymort library.
	•	The ui/ folder is intentionally modular and may evolve as the app grows.

⸻

👤 Author

Pierre-Antoine Le Quellec
Master’s in Finance – HEC Lausanne
Longevity Risk & Quantitative Finance