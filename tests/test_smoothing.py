from __future__ import annotations

import builtins
import sys
import types
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from pymort.analysis.smoothing import smooth_mortality_with_cpsplines


def _install_stub_cpsplines(monkeypatch):
    class DummyCPsplines:
        last_init: ClassVar[dict[str, object]] = {}

        def __init__(self, **kwargs):
            DummyCPsplines.last_init = dict(kwargs)

        def fit(self, data, y_col="y"):
            self._data = data
            self._y_col = y_col

        def predict(self, df):
            return np.zeros(len(df), dtype=float)

    def grid_to_scatter(*, x, y):
        ages, years = x
        A = len(ages)
        T = len(years)
        return pd.DataFrame(
            {
                "x0": np.repeat(ages, T),
                "x1": np.tile(years, A),
                "y": y.reshape(-1),
            }
        )

    mod_cpsplines = types.ModuleType("cpsplines")
    mod_cpsplines.__path__ = []
    mod_fittings = types.ModuleType("cpsplines.fittings")
    mod_fittings.__path__ = []
    mod_fit = types.ModuleType("cpsplines.fittings.fit_cpsplines")
    mod_utils = types.ModuleType("cpsplines.utils")
    mod_utils.__path__ = []
    mod_rearrange = types.ModuleType("cpsplines.utils.rearrange_data")

    mod_fit.CPsplines = DummyCPsplines
    mod_rearrange.grid_to_scatter = grid_to_scatter

    mod_cpsplines.fittings = mod_fittings
    mod_cpsplines.utils = mod_utils
    mod_fittings.fit_cpsplines = mod_fit
    mod_utils.rearrange_data = mod_rearrange

    monkeypatch.setitem(sys.modules, "cpsplines", mod_cpsplines)
    monkeypatch.setitem(sys.modules, "cpsplines.fittings", mod_fittings)
    monkeypatch.setitem(sys.modules, "cpsplines.fittings.fit_cpsplines", mod_fit)
    monkeypatch.setitem(sys.modules, "cpsplines.utils", mod_utils)
    monkeypatch.setitem(sys.modules, "cpsplines.utils.rearrange_data", mod_rearrange)

    return DummyCPsplines


def test_smoothing_raises_when_cpsplines_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("cpsplines"):
            raise ModuleNotFoundError("No module named 'cpsplines'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    ages = np.array([60, 61], dtype=float)
    years = np.array([2000, 2001], dtype=int)
    m = np.full((ages.size, years.size), 0.01, dtype=float)

    with pytest.raises(ModuleNotFoundError, match="cpsplines is required"):
        smooth_mortality_with_cpsplines(m=m, ages=ages, years=years, horizon=0)


def test_smoothing_runs_with_stubbed_cpsplines(monkeypatch):
    DummyCPsplines = _install_stub_cpsplines(monkeypatch)

    ages = np.array([60, 61, 62], dtype=float)
    years = np.array([2000, 2001, 2002, 2003], dtype=int)
    m = np.full((ages.size, years.size), 0.01, dtype=float)

    res = smooth_mortality_with_cpsplines(
        m=m,
        ages=ages,
        years=years,
        horizon=2,
        sp_args=None,
    )

    assert res["m_fitted"].shape == m.shape
    assert res["m_forecast"].shape == (ages.size, 2)
    assert np.array_equal(res["years_forecast"], np.array([2004, 2005], dtype=int))

    A, T = m.shape
    deg_age = max(1, min(3, A - 1))
    deg_year = max(1, min(3, T - 1))
    ord_age = max(0, min(2, deg_age - 1))
    ord_year = max(0, min(2, deg_year - 1))
    expected_deg = (deg_age, deg_year)
    expected_ord_d = (ord_age, ord_year)
    expected_k = (max(deg_age + 1, A), max(deg_year + 1, T))

    init = DummyCPsplines.last_init
    assert init["deg"] == expected_deg
    assert init["ord_d"] == expected_ord_d
    assert init["k"] == expected_k
    assert init["sp_args"] == {"top_n": 5, "parallel": False}
