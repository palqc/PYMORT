from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pymort.analysis.bootstrap import (
    _resample_residuals,
    bootstrap_from_m,
    bootstrap_logitq_model,
    bootstrap_logm_model,
)
from pymort.analysis.projections import (
    project_mortality_from_bootstrap,
    simulate_random_walk_paths,
    simulate_random_walk_paths_with_eps,
)
from pymort.analysis.reporting import generate_risk_report
from pymort.lifetables import survival_from_q, validate_survival_monotonic
from pymort.models.cbd_m5 import CBDM5, fit_cbd
from pymort.models.lc_m1 import LCM1, fit_lee_carter


def test_project_mortality_from_bootstrap_lc_shapes_and_finiteness():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2000, 2001, 2002], dtype=int)
    m = np.array([[0.01, 0.011, 0.012], [0.012, 0.013, 0.014]], dtype=float)
    params = fit_lee_carter(m)
    bs = SimpleNamespace(params_list=[params], mu_sigma=np.array([[0.0, 0.05]]))
    eps_rw = np.zeros((1, 2, 2))

    proj = project_mortality_from_bootstrap(
        model_cls=LCM1,
        ages=ages,
        years=years,
        m=m,
        bootstrap_result=bs,
        horizon=2,
        n_process=2,
        seed=123,
        include_last=False,
        eps_rw=eps_rw,
    )
    assert proj.q_paths.shape == (2, ages.size, 2)
    assert proj.m_paths is not None and proj.m_paths.shape == proj.q_paths.shape
    assert np.isfinite(proj.q_paths).all()
    validate_survival_monotonic(survival_from_q(proj.q_paths))


def test_project_mortality_from_bootstrap_cbd_shapes_and_finiteness():
    ages = np.array([68.0, 70.0], dtype=float)
    years = np.array([2000, 2001, 2002], dtype=int)
    q = np.array([[0.1, 0.11, 0.12], [0.15, 0.16, 0.17]], dtype=float)
    params = fit_cbd(q, ages)
    bs = SimpleNamespace(params_list=[params], mu_sigma=np.array([[0.0, 0.02, 0.0, 0.03]]))
    eps1 = np.zeros((1, 1, 2))
    eps2 = np.zeros((1, 1, 2))

    proj = project_mortality_from_bootstrap(
        model_cls=CBDM5,
        ages=ages,
        years=years,
        m=q,  # unused for CBD, but required signature
        bootstrap_result=bs,
        horizon=2,
        n_process=1,
        seed=123,
        include_last=True,
        eps1=eps1,
        eps2=eps2,
    )
    assert proj.q_paths.shape == (1, ages.size, 3)
    assert proj.m_paths is None  # CBD logit-q models return None m_paths
    assert np.isfinite(proj.q_paths).all()
    validate_survival_monotonic(survival_from_q(proj.q_paths))


def test_generate_risk_report_basic_and_reference():
    pv = np.array([1.0, 1.2, 0.8, 1.1], dtype=float)
    ref = pv + 0.5  # higher variance
    report = generate_risk_report(pv, name="Test", var_level=0.9, ref_pv_paths=ref)
    assert report.n_scenarios == pv.size
    assert np.isfinite(report.mean_pv)
    assert np.isfinite(report.std_pv)
    assert report.pv_min <= report.mean_pv <= report.pv_max
    assert np.isfinite(report.var) and np.isfinite(report.cvar)
    assert report.hedge_var_reduction is not None
    report_no_ref = generate_risk_report(pv, name="NoRef")
    assert report_no_ref.hedge_var_reduction is None


def test_bootstrap_helpers_return_finite_params():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2000, 2001], dtype=int)
    m = np.array([[0.01, 0.011], [0.012, 0.013]], dtype=float)
    res_logm = bootstrap_logm_model(
        model_cls=LCM1,
        m=m,
        ages=ages,
        years=years,
        B=2,
        seed=42,
        resample="year_block",
    )
    assert len(res_logm.params_list) == 2
    assert res_logm.mu_sigma.shape == (2, 2)
    assert np.isfinite(res_logm.mu_sigma).all()

    q = np.array([[0.1, 0.11], [0.12, 0.13]], dtype=float)
    res_logit = bootstrap_logitq_model(
        model_cls=CBDM5,
        q=q,
        ages=ages,
        years=years,
        B=2,
        seed=24,
        resample="cell",
    )
    assert len(res_logit.params_list) == 2
    assert res_logit.mu_sigma.shape == (2, 4)
    assert np.isfinite(res_logit.mu_sigma).all()


# ----------------------------
# RW helpers coverage
# ----------------------------


def test_simulate_random_walk_paths_basic_and_include_last_and_errors():
    rng = np.random.default_rng(0)

    # basic
    paths = simulate_random_walk_paths(
        k_last=1.0, mu=0.1, sigma=0.2, horizon=3, n_sims=4, rng=rng, include_last=False
    )
    assert paths.shape == (4, 3)
    assert np.isfinite(paths).all()

    # include_last adds one column
    rng2 = np.random.default_rng(0)
    paths2 = simulate_random_walk_paths(
        k_last=1.0, mu=0.1, sigma=0.2, horizon=3, n_sims=4, rng=rng2, include_last=True
    )
    assert paths2.shape == (4, 4)
    assert np.allclose(paths2[:, 0], 1.0)

    # sigma < 0 is abs'ed (should not raise)
    rng3 = np.random.default_rng(0)
    paths_neg = simulate_random_walk_paths(
        k_last=1.0, mu=0.1, sigma=-0.2, horizon=3, n_sims=4, rng=rng3
    )
    assert paths_neg.shape == (4, 3)

    # errors
    with pytest.raises(ValueError):
        simulate_random_walk_paths(k_last=1.0, mu=0.0, sigma=0.1, horizon=0, n_sims=1, rng=rng)
    with pytest.raises(ValueError):
        simulate_random_walk_paths(k_last=1.0, mu=0.0, sigma=0.1, horizon=1, n_sims=0, rng=rng)
    with pytest.raises(ValueError):
        simulate_random_walk_paths(k_last=1.0, mu=np.nan, sigma=0.1, horizon=1, n_sims=1, rng=rng)


def test_simulate_random_walk_paths_with_eps_basic_and_validation():
    eps = np.zeros((2, 3), dtype=float)

    # deterministic with eps=0
    paths = simulate_random_walk_paths_with_eps(
        k_last=1.0, mu=0.1, sigma=0.2, eps=eps, include_last=False
    )
    assert paths.shape == (2, 3)
    assert np.allclose(paths, 1.0 + np.cumsum(0.1 + 0.2 * eps, axis=1))

    # include_last
    paths2 = simulate_random_walk_paths_with_eps(
        k_last=1.0, mu=0.1, sigma=0.2, eps=eps, include_last=True
    )
    assert paths2.shape == (2, 4)
    assert np.allclose(paths2[:, 0], 1.0)

    # sigma < 0 abs'ed
    paths3 = simulate_random_walk_paths_with_eps(k_last=1.0, mu=0.1, sigma=-0.2, eps=eps)
    assert paths3.shape == (2, 3)

    # validation errors
    with pytest.raises(ValueError):
        simulate_random_walk_paths_with_eps(
            k_last=1.0, mu=0.1, sigma=0.2, eps=np.zeros((3,)), include_last=False
        )
    bad_eps = eps.copy()
    bad_eps[0, 0] = np.inf
    with pytest.raises(ValueError):
        simulate_random_walk_paths_with_eps(
            k_last=1.0, mu=0.1, sigma=0.2, eps=bad_eps, include_last=False
        )


# ----------------------------
# project_mortality_from_bootstrap error branches
# ----------------------------


def test_project_mortality_mu_sigma_length_mismatch_raises():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2000, 2001, 2002], dtype=int)
    m = np.array([[0.01, 0.011, 0.012], [0.012, 0.013, 0.014]], dtype=float)
    params = fit_lee_carter(m)

    # params_list length 1 but mu_sigma has 2 rows
    bs = SimpleNamespace(params_list=[params], mu_sigma=np.array([[0.0, 0.05], [0.0, 0.05]]))
    with pytest.raises(ValueError):
        project_mortality_from_bootstrap(
            model_cls=LCM1,
            ages=ages,
            years=years,
            m=m,
            bootstrap_result=bs,
            horizon=2,
            n_process=1,
            seed=0,
        )


def test_project_mortality_horizon_and_n_process_must_be_positive():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2000, 2001, 2002], dtype=int)
    m = np.array([[0.01, 0.011, 0.012], [0.012, 0.013, 0.014]], dtype=float)
    params = fit_lee_carter(m)
    bs = SimpleNamespace(params_list=[params], mu_sigma=np.array([[0.0, 0.05]]))

    with pytest.raises(ValueError):
        project_mortality_from_bootstrap(LCM1, ages, years, m, bs, horizon=0, n_process=1)
    with pytest.raises(ValueError):
        project_mortality_from_bootstrap(LCM1, ages, years, m, bs, horizon=2, n_process=0)


def test_project_mortality_params_none_in_bootstrap_raises_runtimeerror():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2000, 2001, 2002], dtype=int)
    m = np.array([[0.01, 0.011, 0.012], [0.012, 0.013, 0.014]], dtype=float)

    bs = SimpleNamespace(params_list=[None], mu_sigma=np.array([[0.0, 0.05]]))
    with pytest.raises(RuntimeError):
        project_mortality_from_bootstrap(
            model_cls=LCM1,
            ages=ages,
            years=years,
            m=m,
            bootstrap_result=bs,
            horizon=2,
            n_process=1,
            seed=0,
        )


def test_project_mortality_drift_overrides_validation_lc_and_cbd():
    # LC: drift_overrides must have length 1 and finite
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2000, 2001, 2002], dtype=int)
    m = np.array([[0.01, 0.011, 0.012], [0.012, 0.013, 0.014]], dtype=float)
    params_lc = fit_lee_carter(m)
    bs_lc = SimpleNamespace(params_list=[params_lc], mu_sigma=np.array([[0.0, 0.05]]))
    eps_rw = np.zeros((1, 1, 2))

    with pytest.raises(ValueError):
        project_mortality_from_bootstrap(
            LCM1,
            ages,
            years,
            m,
            bs_lc,
            horizon=2,
            n_process=1,
            eps_rw=eps_rw,
            drift_overrides=np.array([0.0, 0.1]),
        )
    with pytest.raises(ValueError):
        project_mortality_from_bootstrap(
            LCM1,
            ages,
            years,
            m,
            bs_lc,
            horizon=2,
            n_process=1,
            eps_rw=eps_rw,
            drift_overrides=np.array([np.nan]),
        )

    # CBD: drift_overrides must have length 2 (M5/M6) and finite
    ages2 = np.array([68.0, 70.0], dtype=float)
    q = np.array([[0.1, 0.11, 0.12], [0.15, 0.16, 0.17]], dtype=float)
    params_cbd = fit_cbd(q, ages2)
    bs_cbd = SimpleNamespace(params_list=[params_cbd], mu_sigma=np.array([[0.0, 0.02, 0.0, 0.03]]))
    eps1 = np.zeros((1, 1, 2))
    eps2 = np.zeros((1, 1, 2))

    with pytest.raises(ValueError):
        project_mortality_from_bootstrap(
            CBDM5,
            ages2,
            years,
            m=q,
            bootstrap_result=bs_cbd,
            horizon=2,
            n_process=1,
            eps1=eps1,
            eps2=eps2,
            drift_overrides=np.array([0.1]),
        )
    with pytest.raises(ValueError):
        project_mortality_from_bootstrap(
            CBDM5,
            ages2,
            years,
            m=q,
            bootstrap_result=bs_cbd,
            horizon=2,
            n_process=1,
            eps1=eps1,
            eps2=eps2,
            drift_overrides=np.array([0.1, np.nan]),
        )


def test_project_mortality_scale_sigma_validation_lc_and_cbd():
    # LC: scale_sigma must be scalar/len1 and >0
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2000, 2001, 2002], dtype=int)
    m = np.array([[0.01, 0.011, 0.012], [0.012, 0.013, 0.014]], dtype=float)
    params = fit_lee_carter(m)
    bs = SimpleNamespace(params_list=[params], mu_sigma=np.array([[0.0, 0.05]]))
    eps_rw = np.zeros((1, 1, 2))

    with pytest.raises(ValueError):
        project_mortality_from_bootstrap(
            LCM1,
            ages,
            years,
            m,
            bs,
            horizon=2,
            n_process=1,
            eps_rw=eps_rw,
            scale_sigma=np.array([1.0, 2.0]),
        )
    with pytest.raises(ValueError):
        project_mortality_from_bootstrap(
            LCM1,
            ages,
            years,
            m,
            bs,
            horizon=2,
            n_process=1,
            eps_rw=eps_rw,
            scale_sigma=0.0,
        )

    # CBD: scale_sigma must be len1 or len2 and >0
    ages2 = np.array([68.0, 70.0], dtype=float)
    q = np.array([[0.1, 0.11, 0.12], [0.15, 0.16, 0.17]], dtype=float)
    params_cbd = fit_cbd(q, ages2)
    bs_cbd = SimpleNamespace(params_list=[params_cbd], mu_sigma=np.array([[0.0, 0.02, 0.0, 0.03]]))
    eps1 = np.zeros((1, 1, 2))
    eps2 = np.zeros((1, 1, 2))

    with pytest.raises(ValueError):
        project_mortality_from_bootstrap(
            CBDM5,
            ages2,
            years,
            m=q,
            bootstrap_result=bs_cbd,
            horizon=2,
            n_process=1,
            eps1=eps1,
            eps2=eps2,
            scale_sigma=np.array([1.0, 2.0, 3.0]),
        )
    with pytest.raises(ValueError):
        project_mortality_from_bootstrap(
            CBDM5,
            ages2,
            years,
            m=q,
            bootstrap_result=bs_cbd,
            horizon=2,
            n_process=1,
            eps1=eps1,
            eps2=eps2,
            scale_sigma=np.array([1.0, -2.0]),
        )


def test_project_mortality_sigma_overrides_validation_lc_and_cbd():
    # LC: sigma_overrides must be len1 and >0
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2000, 2001, 2002], dtype=int)
    m = np.array([[0.01, 0.011, 0.012], [0.012, 0.013, 0.014]], dtype=float)
    params = fit_lee_carter(m)
    bs = SimpleNamespace(params_list=[params], mu_sigma=np.array([[0.0, 0.05]]))
    eps_rw = np.zeros((1, 1, 2))

    with pytest.raises(ValueError):
        project_mortality_from_bootstrap(
            LCM1,
            ages,
            years,
            m,
            bs,
            horizon=2,
            n_process=1,
            eps_rw=eps_rw,
            sigma_overrides=np.array([0.1, 0.2]),
        )
    with pytest.raises(ValueError):
        project_mortality_from_bootstrap(
            LCM1,
            ages,
            years,
            m,
            bs,
            horizon=2,
            n_process=1,
            eps_rw=eps_rw,
            sigma_overrides=np.array([0.0]),
        )

    # CBD: sigma_overrides must be len2 and >0 for M5/M6
    ages2 = np.array([68.0, 70.0], dtype=float)
    q = np.array([[0.1, 0.11, 0.12], [0.15, 0.16, 0.17]], dtype=float)
    params_cbd = fit_cbd(q, ages2)
    bs_cbd = SimpleNamespace(params_list=[params_cbd], mu_sigma=np.array([[0.0, 0.02, 0.0, 0.03]]))
    eps1 = np.zeros((1, 1, 2))
    eps2 = np.zeros((1, 1, 2))

    with pytest.raises(ValueError):
        project_mortality_from_bootstrap(
            CBDM5,
            ages2,
            years,
            m=q,
            bootstrap_result=bs_cbd,
            horizon=2,
            n_process=1,
            eps1=eps1,
            eps2=eps2,
            sigma_overrides=np.array([0.1]),
        )
    with pytest.raises(ValueError):
        project_mortality_from_bootstrap(
            CBDM5,
            ages2,
            years,
            m=q,
            bootstrap_result=bs_cbd,
            horizon=2,
            n_process=1,
            eps1=eps1,
            eps2=eps2,
            sigma_overrides=np.array([0.1, -0.2]),
        )


def test_project_mortality_crn_validation_errors_lc_and_cbd():
    # LC: eps_rw wrong shape / not finite
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2000, 2001, 2002], dtype=int)
    m = np.array([[0.01, 0.011, 0.012], [0.012, 0.013, 0.014]], dtype=float)
    params = fit_lee_carter(m)
    bs = SimpleNamespace(params_list=[params], mu_sigma=np.array([[0.0, 0.05]]))

    with pytest.raises(ValueError):
        project_mortality_from_bootstrap(
            LCM1, ages, years, m, bs, horizon=2, n_process=1, eps_rw=np.zeros((1, 1, 3))
        )  # H mismatch
    bad_eps = np.zeros((1, 1, 2))
    bad_eps[0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        project_mortality_from_bootstrap(
            LCM1, ages, years, m, bs, horizon=2, n_process=1, eps_rw=bad_eps
        )

    # CBD: must provide eps1 and eps2 together
    ages2 = np.array([68.0, 70.0], dtype=float)
    q = np.array([[0.1, 0.11, 0.12], [0.15, 0.16, 0.17]], dtype=float)
    params_cbd = fit_cbd(q, ages2)
    bs_cbd = SimpleNamespace(params_list=[params_cbd], mu_sigma=np.array([[0.0, 0.02, 0.0, 0.03]]))

    with pytest.raises(ValueError):
        project_mortality_from_bootstrap(
            CBDM5,
            ages2,
            years,
            m=q,
            bootstrap_result=bs_cbd,
            horizon=2,
            n_process=1,
            eps1=np.zeros((1, 1, 2)),
            eps2=None,
        )
    with pytest.raises(ValueError):
        project_mortality_from_bootstrap(
            CBDM5,
            ages2,
            years,
            m=q,
            bootstrap_result=bs_cbd,
            horizon=2,
            n_process=1,
            eps1=None,
            eps2=np.zeros((1, 1, 2)),
        )


def test_project_mortality_lc_params_age_mismatch_raises():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2000, 2001, 2002], dtype=int)
    m = np.array([[0.01, 0.011, 0.012], [0.012, 0.013, 0.014]], dtype=float)
    params = fit_lee_carter(m)

    # Tamper with params.a to create mismatch vs ages length (A=2)
    params_bad = SimpleNamespace(a=np.array([0.1, 0.2, 0.3]), b=params.b, k=params.k)
    bs = SimpleNamespace(params_list=[params_bad], mu_sigma=np.array([[0.0, 0.05]]))
    eps_rw = np.zeros((1, 1, 2))

    with pytest.raises(ValueError):
        project_mortality_from_bootstrap(
            model_cls=LCM1,
            ages=ages,
            years=years,
            m=m,
            bootstrap_result=bs,
            horizon=2,
            n_process=1,
            seed=0,
            eps_rw=eps_rw,
        )


def test_resample_residuals_year_block_and_invalid_mode():
    rng = np.random.default_rng(0)
    resid = np.arange(12, dtype=float).reshape(3, 4)

    r = _resample_residuals(resid, mode="year_block", rng=rng)
    assert r.shape == resid.shape
    # year_block resamples columns => each column equals some original column
    for j in range(r.shape[1]):
        assert any(np.allclose(r[:, j], resid[:, k]) for k in range(resid.shape[1]))

    with pytest.raises(ValueError):
        _resample_residuals(resid, mode="nope", rng=rng)  # type: ignore[arg-type]


def test_bootstrap_logm_model_errors():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2000, 2001], dtype=int)
    m = np.array([[0.01, 0.011], [0.012, 0.013]], dtype=float)

    with pytest.raises(ValueError):
        bootstrap_logm_model(LCM1, m, ages, years, B=0)

    class Dummy:
        pass

    with pytest.raises(ValueError, match="does not support"):
        bootstrap_logm_model(Dummy, m, ages, years, B=1)

    m_bad = m.copy()
    m_bad[0, 0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        bootstrap_logm_model(LCM1, m_bad, ages, years, B=1)


def test_bootstrap_logitq_model_errors():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2000, 2001], dtype=int)
    q = np.array([[0.1, 0.11], [0.12, 0.13]], dtype=float)

    with pytest.raises(ValueError):
        bootstrap_logitq_model(CBDM5, q, ages, years, B=0)

    class Dummy:
        pass

    with pytest.raises(ValueError, match="does not support"):
        bootstrap_logitq_model(Dummy, q, ages, years, B=1)

    q_bad = q.copy()
    q_bad[0, 0] = 0.0  # validate_q should reject
    with pytest.raises(ValueError):
        bootstrap_logitq_model(CBDM5, q_bad, ages, years, B=1)


def test_bootstrap_from_m_routes_to_correct_bootstrap():
    ages = np.array([60.0, 61.0], dtype=float)
    years = np.array([2000, 2001], dtype=int)
    m = np.array([[0.01, 0.011], [0.012, 0.013]], dtype=float)

    # non-CBD => logm bootstrap
    res_lc = bootstrap_from_m(LCM1, m, ages, years, B=1, seed=0)
    assert res_lc.mu_sigma.shape == (1, 2)

    # CBD => logitq bootstrap (via m_to_q)
    res_cbd = bootstrap_from_m(CBDM5, m, ages, years, B=1, seed=0)
    assert res_cbd.mu_sigma.shape[0] == 1
    assert res_cbd.mu_sigma.shape[1] == 4


def test_bootstrap_logm_rejects_bad_B_and_bad_m():
    ages = np.array([60.0, 61.0])
    years = np.array([2000, 2001])
    m = np.array([[0.01, 0.011], [0.012, 0.013]])

    with pytest.raises(ValueError, match="B must be strictly positive"):
        bootstrap_logm_model(LCM1, m=m, ages=ages, years=years, B=0)

    m_bad = m.copy()
    m_bad[0, 0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        bootstrap_logm_model(LCM1, m=m_bad, ages=ages, years=years, B=1)


def test_bootstrap_logm_rejects_unsupported_model_cls():
    class Dummy:
        pass

    ages = np.array([60.0, 61.0])
    years = np.array([2000, 2001])
    m = np.array([[0.01, 0.011], [0.012, 0.013]])

    with pytest.raises(ValueError, match="does not support"):
        bootstrap_logm_model(Dummy, m=m, ages=ages, years=years, B=1)


def test_bootstrap_logitq_rejects_bad_B_and_bad_q():
    from pymort.models.cbd_m5 import CBDM5

    ages = np.array([60.0, 61.0])
    years = np.array([2000, 2001])
    q = np.array([[0.1, 0.11], [0.12, 0.13]])

    with pytest.raises(ValueError, match="B must be strictly positive"):
        bootstrap_logitq_model(CBDM5, q=q, ages=ages, years=years, B=0)

    q_bad = q.copy()
    q_bad[0, 0] = 1.0
    with pytest.raises(ValueError, match="strictly in"):
        bootstrap_logitq_model(CBDM5, q=q_bad, ages=ages, years=years, B=1)
