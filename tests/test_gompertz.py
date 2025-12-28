from __future__ import annotations

import numpy as np
import pytest

from pymort.models.gompertz import (
    _safe_log_m,
    extrapolate_gompertz_surface,
    fit_gompertz_per_year,
)


def _toy_gompertz_surface(ages: np.ndarray, years: np.ndarray, a_t, b_t) -> np.ndarray:
    """Build m[a,t] = exp(a_t[t] + b_t[t]*age[a])."""
    a_t = np.asarray(a_t, dtype=float).reshape(-1)
    b_t = np.asarray(b_t, dtype=float).reshape(-1)
    assert a_t.shape[0] == years.shape[0]
    assert b_t.shape[0] == years.shape[0]
    return np.exp(a_t[None, :] + b_t[None, :] * ages[:, None])


def test_safe_log_m_clips_floor():
    m = np.array([[0.0, -1.0, 1e-20, 1.0]], dtype=float)
    out = _safe_log_m(m, m_floor=1e-12)
    assert np.isfinite(out).all()
    # first three should be log(1e-12)
    assert np.allclose(out[0, 0], np.log(1e-12))
    assert np.allclose(out[0, 1], np.log(1e-12))
    assert np.allclose(out[0, 2], np.log(1e-12))
    assert np.allclose(out[0, 3], 0.0)  # log(1)


def test_fit_gompertz_per_year_happy_path_recovers_params_approximately():
    ages = np.arange(80, 101, dtype=float)  # inclusive window
    years = np.array([2000, 2001, 2002], dtype=int)
    a_true = np.array([-7.0, -7.2, -7.1])
    b_true = np.array([0.09, 0.095, 0.10])
    m = _toy_gompertz_surface(ages, years, a_true, b_true)

    res = fit_gompertz_per_year(
        ages=ages,
        years=years,
        m=m,
        age_fit_min=80,
        age_fit_max=100,
        m_floor=1e-12,
        compute_fitted_surface=True,
    )

    assert res.a_t.shape == years.shape
    assert res.b_t.shape == years.shape
    assert res.m_fitted is not None
    assert res.m_fitted.shape == m.shape
    assert res.meta["model"] == "gompertz_per_year"
    # OLS on perfect data => should be very close
    assert np.allclose(res.a_t, a_true, rtol=1e-10, atol=1e-10)
    assert np.allclose(res.b_t, b_true, rtol=1e-10, atol=1e-10)
    # fitted surface matches original
    assert np.allclose(res.m_fitted, m, rtol=1e-10, atol=1e-10)


def test_fit_gompertz_per_year_compute_fitted_surface_false():
    ages = np.arange(75, 111, dtype=float)
    years = np.array([2010, 2011], dtype=int)
    a_true = np.array([-8.0, -8.1])
    b_true = np.array([0.08, 0.082])
    m = _toy_gompertz_surface(ages, years, a_true, b_true)

    res = fit_gompertz_per_year(
        ages=ages,
        years=years,
        m=m,
        age_fit_min=80,
        age_fit_max=100,
        compute_fitted_surface=False,
    )
    assert res.m_fitted is None
    assert np.isfinite(res.a_t).all()
    assert np.isfinite(res.b_t).all()


def test_fit_gompertz_per_year_errors_on_bad_shapes():
    ages = np.array([80.0, 81.0, 82.0])
    years = np.array([2000, 2001], dtype=int)

    # m not 2D
    with pytest.raises(ValueError, match="m must be 2D"):
        fit_gompertz_per_year(ages=ages, years=years, m=np.array([1.0, 2.0]))

    # ages length mismatch A
    m = np.ones((4, 2), dtype=float)
    with pytest.raises(ValueError, match="ages length"):
        fit_gompertz_per_year(ages=ages, years=years, m=m)

    # years length mismatch T
    m2 = np.ones((3, 3), dtype=float)
    with pytest.raises(ValueError, match="years length"):
        fit_gompertz_per_year(ages=ages, years=years, m=m2)


def test_fit_gompertz_per_year_errors_on_invalid_fit_window():
    ages = np.arange(60, 70, dtype=float)
    years = np.array([2000, 2001], dtype=int)
    m = np.exp(-9.0 + 0.08 * ages[:, None]) * np.ones((ages.size, years.size), dtype=float)

    # min >= max
    with pytest.raises(ValueError, match="age_fit_min must be < age_fit_max"):
        fit_gompertz_per_year(ages=ages, years=years, m=m, age_fit_min=80, age_fit_max=80)

    # window outside available ages
    with pytest.raises(ValueError, match="No ages in fit window"):
        fit_gompertz_per_year(ages=ages, years=years, m=m, age_fit_min=80, age_fit_max=100)


def test_extrapolate_gompertz_surface_happy_path_and_errors():
    ages = np.arange(80, 91, dtype=float)
    years = np.array([2000, 2001], dtype=int)
    a_true = np.array([-7.0, -7.1])
    b_true = np.array([0.09, 0.095])
    m = _toy_gompertz_surface(ages, years, a_true, b_true)

    fit = fit_gompertz_per_year(
        ages=ages,
        years=years,
        m=m,
        age_fit_min=80,
        age_fit_max=90,
        compute_fitted_surface=False,
    )

    ages_ext, years_out, m_ext = extrapolate_gompertz_surface(fit, age_max=100)
    assert years_out.shape == years.shape
    assert np.array_equal(years_out, years)
    assert ages_ext[0] == ages.min()
    assert ages_ext[-1] == 100.0
    assert m_ext.shape == (ages_ext.size, years.size)
    assert np.isfinite(m_ext).all()
    assert (m_ext > 0).all()

    # explicit age_min
    ages_ext2, _, _ = extrapolate_gompertz_surface(fit, age_min=85, age_max=90)
    assert ages_ext2[0] == 85.0
    assert ages_ext2[-1] == 90.0

    # error: age_max <= age_min
    with pytest.raises(ValueError, match="age_max must be > age_min"):
        extrapolate_gompertz_surface(fit, age_min=90, age_max=90)
