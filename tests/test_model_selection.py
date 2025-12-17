from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pymort.analysis.fitting import (
    model_selection_by_forecast_rmse,
    select_and_fit_best_model_for_pricing,
)


def _toy_surface():
    ages = np.array([60.0, 61.0, 62.0, 63.0], dtype=float)
    years = np.arange(2000, 2015, dtype=int)  # T=15
    base_age = np.array([0.01, 0.011, 0.012, 0.013], dtype=float)[:, None]
    trend = np.linspace(1.0, 0.7, years.size)[None, :]
    m = base_age * trend
    return ages, years, m


def test_model_selection_returns_dataframe_and_best_model():
    ages, years, m = _toy_surface()
    df, best = model_selection_by_forecast_rmse(
        ages=ages,
        years=years,
        m=m,
        train_end=2010,
        model_names=("LCM1", "LCM2"),
        metric="logit_q",
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Model" in df.columns
    assert "RMSE forecast (logit q)" in df.columns
    assert best in df["Model"].values
    assert np.all(np.isfinite(df["RMSE forecast (logit q)"]))
    assert np.all(df["RMSE forecast (logit q)"] >= 0.0)


def test_model_selection_deterministic_with_seed():
    ages, years, m = _toy_surface()
    df1, best1 = model_selection_by_forecast_rmse(
        ages=ages,
        years=years,
        m=m,
        train_end=2010,
        model_names=("LCM1", "LCM2"),
        metric="logit_q",
    )
    df2, best2 = model_selection_by_forecast_rmse(
        ages=ages,
        years=years,
        m=m,
        train_end=2010,
        model_names=("LCM1", "LCM2"),
        metric="logit_q",
    )
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))
    assert best1 == best2


def test_model_selection_respects_train_end():
    ages, years, m = _toy_surface()
    df_early, _ = model_selection_by_forecast_rmse(
        ages=ages,
        years=years,
        m=m,
        train_end=2005,
        model_names=("LCM1", "LCM2"),
        metric="logit_q",
    )
    df_late, _ = model_selection_by_forecast_rmse(
        ages=ages,
        years=years,
        m=m,
        train_end=2010,
        model_names=("LCM1", "LCM2"),
        metric="logit_q",
    )
    assert not df_early.empty and not df_late.empty
    assert np.all(np.isfinite(df_early["RMSE forecast (logit q)"]))
    assert np.all(np.isfinite(df_late["RMSE forecast (logit q)"]))


def test_select_and_fit_best_model_matches_selection_best():
    ages, years, m = _toy_surface()
    selection_df, fitted = select_and_fit_best_model_for_pricing(
        ages=ages,
        years=years,
        m=m,
        train_end=2010,
        model_names=("LCM1", "LCM2"),
        metric="logit_q",
        cpsplines_kwargs={"k": None, "horizon": 0, "verbose": False},
    )
    df_ref, best_ref = model_selection_by_forecast_rmse(
        ages=ages,
        years=years,
        m=m,
        train_end=2010,
        model_names=("LCM1", "LCM2"),
        metric="logit_q",
    )
    assert fitted.name == best_ref
    assert set(selection_df["Model"]) == set(df_ref["Model"])


def test_model_selection_raises_on_bad_inputs():
    ages, years, m = _toy_surface()
    m_bad = m.copy()
    m_bad[0, 0] = -1.0
    with pytest.raises(ValueError):
        model_selection_by_forecast_rmse(ages=ages, years=years, m=m_bad, train_end=2010)

    m_nan = m.copy()
    m_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        model_selection_by_forecast_rmse(ages=ages, years=years, m=m_nan, train_end=2010)

    with pytest.raises(ValueError):
        model_selection_by_forecast_rmse(ages=ages[:2], years=years, m=m, train_end=2010)

    with pytest.raises(ValueError):
        model_selection_by_forecast_rmse(ages=ages, years=years, m=m, train_end=1990)
