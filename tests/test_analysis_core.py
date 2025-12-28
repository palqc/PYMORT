from __future__ import annotations

import numpy as np

from pymort.analysis.bootstrap import bootstrap_logitq_model, bootstrap_logm_model
from pymort.analysis.risk_tools import summarize_pv_paths, summarize_survival_scenarios
from pymort.analysis.scenario import MortalityScenarioSet
from pymort.analysis.scenario_analysis import (
    apply_cohort_trend_shock,
    apply_life_expectancy_shift,
    apply_mortality_shock,
)
from pymort.lifetables import (
    m_to_q,
    q_to_m,
    survival_from_q,
    validate_survival_monotonic,
)
from pymort.models.cbd_m5 import CBDM5
from pymort.models.lc_m1 import LCM1


def test_lifetable_round_trip_and_survival_monotonic():
    m = np.array([[0.01, 0.011, 0.012], [0.013, 0.014, 0.015]], dtype=float)
    q = m_to_q(m)
    m_back = q_to_m(q)
    assert np.allclose(m_back, m, rtol=1e-3, atol=1e-4)
    S = survival_from_q(q)
    validate_survival_monotonic(S)
    assert np.all((q > 0) & (q < 1))


def test_bootstrap_small_samples_shapes_and_finiteness():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2000, 2001, 2002], dtype=int)
    m = np.array([[0.01, 0.011, 0.012], [0.0125, 0.013, 0.0135]], dtype=float)
    res_logm = bootstrap_logm_model(
        LCM1,
        m=m,
        ages=ages,
        years=years,
        B=2,
        seed=123,
        resample="year_block",
    )
    assert len(res_logm.params_list) == 2
    assert res_logm.mu_sigma.shape == (2, 2)
    assert np.isfinite(res_logm.mu_sigma).all()

    q = m_to_q(m)
    res_logit = bootstrap_logitq_model(
        CBDM5,
        q=q,
        ages=ages,
        years=years,
        B=2,
        seed=123,
        resample="cell",
    )
    assert len(res_logit.params_list) == 2
    assert res_logit.mu_sigma.shape == (2, 4)
    assert np.isfinite(res_logit.mu_sigma).all()


def test_scenario_analysis_shocks_keep_shapes_and_monotonicity():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2020, 2021, 2022, 2023], dtype=int)
    q = np.full((3, ages.size, years.size), 0.01, dtype=float)
    S = survival_from_q(q)
    base = MortalityScenarioSet(years=years, ages=ages, q_paths=q, S_paths=S, metadata={})

    shocked = apply_mortality_shock(base, shock_type="plateau", plateau_start_year=2022)
    assert shocked.q_paths.shape == q.shape
    validate_survival_monotonic(shocked.S_paths)

    cohort_shocked = apply_cohort_trend_shock(
        base,
        cohort_start=1960,
        cohort_end=1962,
        magnitude=0.05,
        direction="favorable",
        ramp=True,
    )
    assert cohort_shocked.q_paths.shape == q.shape
    validate_survival_monotonic(cohort_shocked.S_paths)

    shifted = apply_life_expectancy_shift(base, age=60.0, delta_years=0.05, bracket=(0.0, 0.95))
    assert shifted.q_paths.shape == q.shape
    validate_survival_monotonic(shifted.S_paths)


def test_risk_tool_summaries_shapes():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2030, 2031], dtype=int)
    q_paths = np.stack(
        [
            np.array([[0.01, 0.011], [0.012, 0.013]]),
            np.array([[0.009, 0.0105], [0.0115, 0.0125]]),
        ],
        axis=0,
    )
    S_paths = survival_from_q(q_paths)
    scen = MortalityScenarioSet(
        years=years,
        ages=ages,
        q_paths=q_paths,
        S_paths=S_paths,
        metadata={},
    )
    summary = summarize_survival_scenarios(scen, percentiles=(5, 50, 95))
    assert summary.S_mean.shape == (ages.size, years.size)
    assert 5 in summary.S_quantiles and 95 in summary.q_quantiles
    pv_summary = summarize_pv_paths(np.array([1.0, 1.1, 0.9]))
    assert pv_summary.n_scenarios == 3
    assert pv_summary.min <= pv_summary.mean <= pv_summary.max
