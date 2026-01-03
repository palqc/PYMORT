from __future__ import annotations

import numpy as np
import pytest

from pymort.analysis.smoothing import smooth_mortality_with_cpsplines

pytest.importorskip("cpsplines")


def _toy_surface(A: int = 6, T: int = 8):
    ages = np.arange(60, 60 + A, dtype=float)
    years = np.arange(2000, 2000 + T, dtype=int)
    base = (0.01 + 0.0005 * (ages - ages[0]))[:, None]
    trend = np.linspace(1.0, 0.8, T)[None, :]
    m = base * trend
    return ages, years, m


def test_cpsplines_smoke_returns_fitted_surface():
    ages, years, m = _toy_surface()
    res = smooth_mortality_with_cpsplines(
        m=m,
        ages=ages,
        years=years,
        deg=(3, 3),
        ord_d=(2, 2),
        k=None,
        sp_method="grid_search",
        horizon=0,
        verbose=False,
    )
    assert "m_fitted" in res
    m_fit = res["m_fitted"]
    assert m_fit.shape == m.shape
    assert np.isfinite(m_fit).all()
    assert np.all(m_fit > 0.0)


def test_cpsplines_forecast_horizon_adds_years_and_surface():
    ages, years, m = _toy_surface()
    horizon = 3
    res = smooth_mortality_with_cpsplines(
        m=m,
        ages=ages,
        years=years,
        deg=(3, 3),
        ord_d=(2, 2),
        k=None,
        sp_method="grid_search",
        horizon=horizon,
        verbose=False,
    )
    assert "m_forecast" in res and "years_forecast" in res
    m_forecast = res["m_forecast"]
    years_forecast = res["years_forecast"]
    assert m_forecast.shape == (ages.size, horizon)
    assert years_forecast.shape == (horizon,)
    assert np.isfinite(m_forecast).all()
    assert np.all(m_forecast > 0.0)


def test_cpsplines_is_deterministic_given_same_inputs():
    ages, years, m = _toy_surface()
    res1 = smooth_mortality_with_cpsplines(
        m=m, ages=ages, years=years, k=None, horizon=2, verbose=False
    )
    res2 = smooth_mortality_with_cpsplines(
        m=m, ages=ages, years=years, k=None, horizon=2, verbose=False
    )
    assert np.allclose(res1["m_fitted"], res2["m_fitted"])
    assert np.allclose(res1["m_forecast"], res2["m_forecast"])


def test_cpsplines_constant_surface_stays_close():
    A, T = 5, 6
    ages = np.arange(60, 60 + A, dtype=float)
    years = np.arange(2000, 2000 + T, dtype=int)
    m_const = np.full((A, T), 0.02, dtype=float)
    res = smooth_mortality_with_cpsplines(
        m=m_const, ages=ages, years=years, k=None, horizon=0, verbose=False
    )
    m_fit = res["m_fitted"]
    rel_err = np.max(np.abs(m_fit - m_const) / m_const)
    assert rel_err < 0.05  # stays close to constant


def test_cpsplines_raises_on_bad_inputs():
    ages, years, m = _toy_surface()
    m_bad = m.copy()
    m_bad[0, 0] = -1.0
    with pytest.raises(ValueError):
        smooth_mortality_with_cpsplines(m=m_bad, ages=ages, years=years)

    m_nan = m.copy()
    m_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        smooth_mortality_with_cpsplines(m=m_nan, ages=ages, years=years)

    with pytest.raises(ValueError):
        smooth_mortality_with_cpsplines(m=m, ages=ages[:2], years=years)

    with pytest.raises(ValueError):
        smooth_mortality_with_cpsplines(m=m[0], ages=ages, years=years)


def test_cpsplines_knots_auto_and_manual():
    ages, years, m = _toy_surface(A=5, T=7)
    # auto k
    res_auto = smooth_mortality_with_cpsplines(
        m=m, ages=ages, years=years, k=None, horizon=0, verbose=False
    )
    assert res_auto["m_fitted"].shape == m.shape
    # manual small k
    res_manual = smooth_mortality_with_cpsplines(
        m=m, ages=ages, years=years, k=(4, 4), horizon=0, verbose=False
    )
    assert res_manual["m_fitted"].shape == m.shape


def test_cpsplines_raises_if_k_not_greater_than_deg():
    ages, years, m = _toy_surface(A=6, T=6)
    # deg par défaut (3,3) => il faut k <= 3 pour déclencher
    with pytest.raises(ValueError):
        smooth_mortality_with_cpsplines(m=m, ages=ages, years=years, k=(3, 4), horizon=0)

    with pytest.raises(ValueError):
        smooth_mortality_with_cpsplines(m=m, ages=ages, years=years, k=(4, 3), horizon=0)


def test_cpsplines_raises_on_negative_horizon():
    ages, years, m = _toy_surface()
    with pytest.raises(ValueError):
        smooth_mortality_with_cpsplines(m=m, ages=ages, years=years, horizon=-1)


def test_cpsplines_raises_if_ages_or_years_not_1d():
    ages, years, m = _toy_surface()
    with pytest.raises(ValueError):
        smooth_mortality_with_cpsplines(m=m, ages=ages.reshape(-1, 1), years=years)

    with pytest.raises(ValueError):
        smooth_mortality_with_cpsplines(m=m, ages=ages, years=years.reshape(-1, 1))


def test_cpsplines_default_sp_args_path():
    ages, years, m = _toy_surface()
    res = smooth_mortality_with_cpsplines(
        m=m, ages=ages, years=years, sp_args=None, horizon=0, verbose=False
    )
    assert res["m_fitted"].shape == m.shape
