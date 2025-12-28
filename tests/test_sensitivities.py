from __future__ import annotations

import numpy as np

from pymort.analysis.sensitivities import (
    mortality_delta_by_age,
    rate_convexity,
    rate_sensitivity,
)
from pymort.pipeline import risk_analysis_pipeline
from pymort.pricing.liabilities import CohortLifeAnnuitySpec
from pymort.pricing.longevity_bonds import LongevityBondSpec, price_simple_longevity_bond

try:
    from pymort.analysis.scenario import MortalityScenarioSet
except Exception:  # pragma: no cover - fallback if import path changes
    from pymort.analysis import MortalityScenarioSet


def _make_toy_scenarios(N: int = 2, T: int = 4) -> MortalityScenarioSet:
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.arange(2020, 2020 + T, dtype=int)
    q_paths = np.full((N, ages.size, T), 0.01, dtype=float)
    # small variation to avoid symmetric cancellations
    for n in range(N):
        q_paths[n] += 0.0005 * n
    S_paths = np.ones_like(q_paths)
    for t in range(T):
        if t == 0:
            S_paths[:, :, t] = 1.0 - q_paths[:, :, t]
        else:
            S_paths[:, :, t] = S_paths[:, :, t - 1] * (1.0 - q_paths[:, :, t])
    return MortalityScenarioSet(
        years=years,
        ages=ages,
        q_paths=q_paths,
        S_paths=S_paths,
        metadata={"name": "toy"},
    )


def test_rate_sensitivity_sign_and_scale():
    scen = _make_toy_scenarios()
    spec = LongevityBondSpec(issue_age=60.0, maturity_years=4, include_principal=True, notional=1.0)

    def pricer(**kwargs: object) -> float:
        res = price_simple_longevity_bond(
            scen_set=kwargs["scen_set"], spec=spec, short_rate=kwargs["short_rate"]
        )
        return float(res["price"])

    sens = rate_sensitivity(pricer, scen, base_short_rate=0.02, bump=1e-4)
    assert np.isfinite(sens.dP_dr)
    assert sens.dP_dr < 0.0  # higher rates -> lower PV
    assert sens.dv01 < 0.0
    assert sens.price_base > 0.0


def test_rate_convexity_non_negative():
    scen = _make_toy_scenarios()
    spec = LongevityBondSpec(issue_age=60.0, maturity_years=3, include_principal=True, notional=1.0)

    def pricer(**kwargs: object) -> float:
        res = price_simple_longevity_bond(
            scen_set=kwargs["scen_set"], spec=spec, short_rate=kwargs["short_rate"]
        )
        return float(res["price"])

    conv = rate_convexity(pricer, scen, base_short_rate=0.03, bump=1e-4)
    assert np.isfinite(conv.convexity)
    assert conv.convexity >= -abs(conv.price_base) * 1e-3  # allow tiny relative noise
    assert conv.price_base > 0.0


def test_mortality_delta_by_age_sign():
    scen = _make_toy_scenarios()
    spec = CohortLifeAnnuitySpec(issue_age=60.0, maturity_years=4, payment_per_survivor=1.0)

    def price_annuity(s: MortalityScenarioSet) -> float:
        from pymort.pricing.liabilities import price_cohort_life_annuity

        return float(price_cohort_life_annuity(scen_set=s, spec=spec, short_rate=0.02)["price"])

    # ✅ On ne calcule la delta que pour l âge pertinent (issue_age)
    delta = mortality_delta_by_age(price_annuity, scen, ages=[60.0], rel_bump=0.01)

    assert delta.deltas.shape == (1,)
    assert delta.deltas[0] < 0.0  # higher mortality -> lower annuity PV
    assert np.isfinite(delta.base_price)


def test_sensitivities_stability_no_nan():
    scen_base = _make_toy_scenarios()
    specs = {
        "bond": LongevityBondSpec(issue_age=60.0, maturity_years=4, include_principal=True),
        "annuity": CohortLifeAnnuitySpec(
            issue_age=60.0, maturity_years=4, payment_per_survivor=1.0
        ),
    }

    def build_scen(scale_sigma: float):
        # ignore scale for this deterministic test; ensures pipeline path works
        return scen_base

    res = risk_analysis_pipeline(
        scen_Q=scen_base,
        specs=specs,
        short_rate=0.02,
        bumps={
            "build_scenarios_func": build_scen,
            "sigma_rel_bump": 0.05,
            "q_rel_bump": 0.01,
            "rate_bump": 1e-4,
        },
    )

    # No NaNs across outputs
    for price in res.prices_base.values():
        assert np.isfinite(price)
    for vega in res.vega_sigma_scale.values():
        assert np.isfinite(vega)
    for delta in res.delta_by_age.values():
        assert np.isfinite(delta.deltas).all()
    for rs in res.rate_sensitivity.values():
        assert np.isfinite(rs.dP_dr)
    for rc in res.rate_convexity.values():
        assert np.isfinite(rc.convexity)

    # Shapes and keys
    assert set(res.prices_base.keys()) == set(specs.keys())
    for key in specs:
        assert res.delta_by_age[key].deltas.shape[0] == scen_base.ages.shape[0]
