from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import pymort.analysis.fitting as fitmod


# -------------------------
# helpers
# -------------------------
def _toy_m(A: int = 3, T: int = 4):
    ages = np.arange(60, 60 + A, dtype=float)
    years = np.arange(2000, 2000 + T, dtype=int)
    base = (0.01 + 0.001 * np.arange(A))[:, None]
    trend = np.linspace(1.0, 0.9, T)[None, :]
    m = base * trend
    return ages, years, m


def _patch_rmse(monkeypatch):
    # rmse_aic_bic(logit_true, logit_hat, n_params=...) -> (rmse, aic, bic)
    def fake_rmse_aic_bic(_true, _hat, n_params: int):
        # deterministic, depends on n_params to ensure branch executed
        return (float(n_params), float(n_params + 1), float(n_params + 2))

    monkeypatch.setattr(fitmod, "rmse_aic_bic", fake_rmse_aic_bic)


class _DummyLCM1:
    def fit(self, m_fit):
        self._A, self._T = m_fit.shape
        return self

    def predict_log_m(self):
        # return log(m_hat)
        return np.log(np.full((self._A, self._T), 0.01))


class _DummyLCM2:
    class _Params:
        def __init__(self):
            self.cohorts = np.array([1, 2, 3])

    def __init__(self, params_none: bool = False):
        self._params_none = params_none
        self.params = None

    def fit(self, m_fit, ages, years):
        self._A, self._T = m_fit.shape
        self.params = None if self._params_none else self._Params()
        return self

    def predict_m(self):
        return np.full((self._A, self._T), 0.012)


class _DummyAPCM3:
    class _Params:
        def __init__(self):
            self.cohorts = np.array([10, 11])

    def __init__(self, params_none: bool = False):
        self._params_none = params_none
        self.params = None

    def fit(self, m_fit, ages, years):
        self._A, self._T = m_fit.shape
        self.params = None if self._params_none else self._Params()
        return self

    def predict_m(self):
        return np.full((self._A, self._T), 0.013)


class _DummyCBD:
    class _Params:
        def __init__(self):
            self.cohorts = np.array([7])

    def __init__(self, params_none: bool = False):
        self._params_none = params_none
        self.params = None

    def fit(self, q_fit, *args):
        # q_fit shape (A,T)
        self._A, self._T = q_fit.shape
        self.params = None if self._params_none else self._Params()
        return self

    def predict_q(self):
        # always valid q
        return np.clip(np.full((self._A, self._T), 0.02), 1e-6, 0.5)


def _patch_models(
    monkeypatch,
    *,
    lcm2_params_none=False,
    apc_params_none=False,
    cbd6_params_none=False,
    cbd7_params_none=False,
):
    monkeypatch.setattr(fitmod, "LCM1", lambda: _DummyLCM1())
    monkeypatch.setattr(fitmod, "LCM2", lambda: _DummyLCM2(params_none=lcm2_params_none))
    monkeypatch.setattr(fitmod, "APCM3", lambda: _DummyAPCM3(params_none=apc_params_none))
    monkeypatch.setattr(fitmod, "CBDM5", lambda: _DummyCBD(params_none=False))
    monkeypatch.setattr(fitmod, "CBDM6", lambda: _DummyCBD(params_none=cbd6_params_none))
    monkeypatch.setattr(fitmod, "CBDM7", lambda: _DummyCBD(params_none=cbd7_params_none))


# -------------------------
# _fit_single_model: input checks
# -------------------------
def test_fit_single_model_rejects_shape_mismatch():
    ages, years, m = _toy_m()
    with pytest.raises(ValueError, match="same shape"):
        fitmod._fit_single_model("LCM1", ages, years, m_fit=m, m_eval=m[:, :-1])


def test_fit_single_model_rejects_inconsistent_grids():
    ages, years, m = _toy_m()
    with pytest.raises(ValueError, match="consistent with ages and years"):
        fitmod._fit_single_model("LCM1", ages[:-1], years, m_fit=m, m_eval=m)


def test_fit_single_model_rejects_unknown_model():
    ages, years, m = _toy_m()
    with pytest.raises(ValueError, match="Unknown model_name"):
        fitmod._fit_single_model("LCM9", ages, years, m_fit=m, m_eval=m)  # type: ignore[arg-type]


# -------------------------
# _fit_single_model: cover each branch
# -------------------------
@pytest.mark.parametrize("name", ["LCM1", "LCM2", "APCM3", "CBDM5", "CBDM6", "CBDM7"])
def test_fit_single_model_branches_return_diagnostics(monkeypatch, name):
    _patch_rmse(monkeypatch)
    _patch_models(monkeypatch)

    ages, years, m = _toy_m()
    out = fitmod._fit_single_model(name, ages, years, m_fit=m, m_eval=m)

    assert out.name == name
    assert out.m_fit_surface is not None and out.m_eval_surface is not None
    assert np.isfinite(out.rmse_logitq)  # always set
    assert np.isfinite(out.aic) and np.isfinite(out.bic)

    if name in ("LCM1", "LCM2", "APCM3"):
        assert np.isfinite(out.rmse_logm)
    else:
        assert out.rmse_logm is None


def test_fit_single_model_lcm2_params_none_raises(monkeypatch):
    _patch_rmse(monkeypatch)
    _patch_models(monkeypatch, lcm2_params_none=True)
    ages, years, m = _toy_m()
    with pytest.raises(RuntimeError, match=re.escape("LCM2.fit() returned None params")):
        fitmod._fit_single_model("LCM2", ages, years, m_fit=m, m_eval=m)


def test_fit_single_model_apcm3_params_none_raises(monkeypatch):
    _patch_rmse(monkeypatch)
    _patch_models(monkeypatch, apc_params_none=True)
    ages, years, m = _toy_m()
    with pytest.raises(RuntimeError, match=re.escape("APCM3.fit() returned None params")):
        fitmod._fit_single_model("APCM3", ages, years, m_fit=m, m_eval=m)


def test_fit_single_model_cbdm6_params_none_raises(monkeypatch):
    _patch_rmse(monkeypatch)
    _patch_models(monkeypatch, cbd6_params_none=True)
    ages, years, m = _toy_m()
    with pytest.raises(RuntimeError, match=re.escape("CBDM6.fit() returned None params")):
        fitmod._fit_single_model("CBDM6", ages, years, m_fit=m, m_eval=m)


def test_fit_single_model_cbdm7_params_none_raises(monkeypatch):
    _patch_rmse(monkeypatch)
    _patch_models(monkeypatch, cbd7_params_none=True)
    ages, years, m = _toy_m()
    with pytest.raises(RuntimeError, match=re.escape("CBDM7.fit() returned None params")):
        fitmod._fit_single_model("CBDM7", ages, years, m_fit=m, m_eval=m)


# -------------------------
# fit_mortality_model: smoothing branches
# -------------------------
def test_fit_mortality_model_none_sets_metadata(monkeypatch):
    _patch_rmse(monkeypatch)
    _patch_models(monkeypatch)

    ages, years, m = _toy_m()
    fm = fitmod.fit_mortality_model("LCM1", ages, years, m, smoothing="none")

    assert fm.metadata["data_source"] == "raw"
    assert fm.metadata["smoothing"] == "none"
    assert fm.metadata["eval_on_raw"] is True


def test_fit_mortality_model_cpsplines_success_eval_on_raw(monkeypatch):
    _patch_rmse(monkeypatch)
    _patch_models(monkeypatch)

    def fake_cps(m, ages, years, k, horizon, verbose):
        return {"m_fitted": np.full_like(m, 0.02)}

    monkeypatch.setattr(fitmod, "smooth_mortality_with_cpsplines", fake_cps)

    ages, years, m = _toy_m()
    fm = fitmod.fit_mortality_model(
        "LCM1",
        ages,
        years,
        m,
        smoothing="cpsplines",
        cpsplines_kwargs={"k": None, "horizon": 0, "verbose": False},
        eval_on_raw=True,
    )

    assert fm.metadata["data_source"] == "cpsplines_fit_eval_on_raw"
    assert fm.m_fit_surface is not None
    assert np.allclose(fm.m_fit_surface, 0.02)  # fitted on smooth
    assert np.allclose(fm.m_eval_surface, m)  # eval on raw


def test_fit_mortality_model_cpsplines_success_eval_on_smooth(monkeypatch):
    _patch_rmse(monkeypatch)
    _patch_models(monkeypatch)

    def fake_cps(m, ages, years, k, horizon, verbose):
        return {"m_fitted": np.full_like(m, 0.02)}

    monkeypatch.setattr(fitmod, "smooth_mortality_with_cpsplines", fake_cps)

    ages, years, m = _toy_m()
    fm = fitmod.fit_mortality_model(
        "LCM1",
        ages,
        years,
        m,
        smoothing="cpsplines",
        cpsplines_kwargs={"k": None, "horizon": 0, "verbose": False},
        eval_on_raw=False,
    )

    assert fm.metadata["data_source"] == "cpsplines_fit_eval_on_smooth"
    assert np.allclose(fm.m_eval_surface, 0.02)  # eval on smooth


def test_fit_mortality_model_cpsplines_fallback_on_exception(monkeypatch):
    _patch_rmse(monkeypatch)
    _patch_models(monkeypatch)

    def fake_cps(*args, **kwargs):
        raise RuntimeError("solver exploded")

    monkeypatch.setattr(fitmod, "smooth_mortality_with_cpsplines", fake_cps)

    ages, years, m = _toy_m()
    fm = fitmod.fit_mortality_model(
        "LCM1",
        ages,
        years,
        m,
        smoothing="cpsplines",
        cpsplines_kwargs={"verbose": False},
        eval_on_raw=True,
    )

    assert fm.metadata["data_source"] == "raw_fallback_cpsplines_failed"
    assert np.allclose(fm.m_fit_surface, m)


def test_fit_mortality_model_rejects_unknown_smoothing(monkeypatch):
    _patch_rmse(monkeypatch)
    _patch_models(monkeypatch)

    ages, years, m = _toy_m()
    with pytest.raises(ValueError, match="Unknown smoothing option"):
        fitmod.fit_mortality_model("LCM1", ages, years, m, smoothing="wat")  # type: ignore[arg-type]


def test_fit_mortality_model_shape_mismatch_raises(monkeypatch):
    _patch_rmse(monkeypatch)
    _patch_models(monkeypatch)

    ages, years, m = _toy_m()
    with pytest.raises(ValueError, match="Shape mismatch"):
        fitmod.fit_mortality_model("LCM1", ages, years, m[:, :-1])


# -------------------------
# model_selection_by_forecast_rmse: validations
# -------------------------
def test_model_selection_validation_train_end_bounds():
    ages, years, m = _toy_m()
    with pytest.raises(ValueError, match="train_end must lie"):
        fitmod.model_selection_by_forecast_rmse(ages, years, m, train_end=1999)

    with pytest.raises(ValueError, match="train_end must lie"):
        fitmod.model_selection_by_forecast_rmse(ages, years, m, train_end=int(years[-1]))


def test_model_selection_validation_m_finite_positive():
    ages, years, m = _toy_m()
    m2 = m.copy()
    m2[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fitmod.model_selection_by_forecast_rmse(ages, years, m2, train_end=int(years[1]))

    m3 = m.copy()
    m3[0, 0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        fitmod.model_selection_by_forecast_rmse(ages, years, m3, train_end=int(years[1]))


def test_model_selection_validation_years_unique_increasing():
    ages, years, m = _toy_m()

    # duplicates
    years_dup = years.copy()
    years_dup[1] = years_dup[0]
    with pytest.raises(ValueError, match=r"years must not contain duplicates"):
        fitmod.model_selection_by_forecast_rmse(ages, years_dup, m, train_end=int(years_dup[0]))

    # not strictly increasing (NO duplicates, and years[0] < years[-1])
    years_bad = years.copy()
    years_bad[1], years_bad[2] = years_bad[2], years_bad[1]  # e.g. 2000,2002,2001,2003
    with pytest.raises(ValueError, match=r"years must be strictly increasing"):
        fitmod.model_selection_by_forecast_rmse(
            ages,
            years_bad,
            m,
            train_end=int(years_bad[2]),  # 2001, in years_bad
        )


def test_model_selection_train_end_must_be_in_years():
    ages = np.array([60.0, 61.0, 62.0], dtype=float)
    years = np.array([2000, 2002, 2004, 2006], dtype=int)  # gaps
    base = np.array([0.01, 0.011, 0.012], dtype=float)[:, None]
    trend = np.linspace(1.0, 0.9, years.size)[None, :]
    m = base * trend

    # 2003 is within [2000, 2005] but NOT in years
    with pytest.raises(ValueError, match=r"train_end must be one of the provided years"):
        fitmod.model_selection_by_forecast_rmse(ages, years, m, train_end=2003)


# -------------------------
# model_selection_by_forecast_rmse: robust failure recording + metric branches
# -------------------------
def test_model_selection_records_failed_models_and_selects_best(monkeypatch):
    ages, years, m = _toy_m()
    train_end = int(years[1])

    # backtest stubs
    def ok_backtest(**kwargs):
        return {
            "rmse_log_forecast": 0.9,
            "rmse_logit_forecast": 0.8,
            "train_years": np.array([years[0], train_end]),
            "test_years": np.array([train_end + 1, years[-1]]),
        }

    def ok_cbd(**kwargs):
        return {
            "rmse_logit_forecast": 0.7,
            "train_years": np.array([years[0], train_end]),
            "test_years": np.array([train_end + 1, years[-1]]),
        }

    def boom(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(fitmod, "time_split_backtest_lc_m1", ok_backtest)
    monkeypatch.setattr(fitmod, "time_split_backtest_lc_m2", boom)  # fails
    monkeypatch.setattr(fitmod, "time_split_backtest_apc_m3", ok_backtest)
    monkeypatch.setattr(fitmod, "time_split_backtest_cbd_m5", ok_cbd)
    monkeypatch.setattr(fitmod, "time_split_backtest_cbd_m6", boom)  # fails
    monkeypatch.setattr(fitmod, "time_split_backtest_cbd_m7", ok_cbd)

    df, best = fitmod.model_selection_by_forecast_rmse(
        ages,
        years,
        m,
        train_end=train_end,
        model_names=("LCM1", "LCM2", "APCM3", "CBDM5", "CBDM6", "CBDM7"),
        metric="logit_q",
    )

    assert isinstance(df, pd.DataFrame)
    assert "status" in df.columns
    assert any(df["status"].str.startswith("failed:"))
    assert best in df["Model"].tolist()
    # best should be the minimum finite RMSE forecast (logit q)
    col = "RMSE forecast (logit q)"
    mask = np.isfinite(df[col].to_numpy())
    assert df.loc[mask, col].min() == df.loc[df["Model"] == best, col].iloc[0]


def test_model_selection_metric_log_m_requires_finite(monkeypatch):
    ages, years, m = _toy_m()
    train_end = int(years[1])

    # make all models return NaN on log_m (simulate CBD-only scenario)
    def cbd_like(**kwargs):
        return {
            "rmse_logit_forecast": 0.5,
            "train_years": np.array([years[0], train_end]),
            "test_years": np.array([train_end + 1, years[-1]]),
        }

    monkeypatch.setattr(fitmod, "time_split_backtest_cbd_m5", cbd_like)
    monkeypatch.setattr(fitmod, "time_split_backtest_cbd_m6", cbd_like)
    monkeypatch.setattr(fitmod, "time_split_backtest_cbd_m7", cbd_like)
    monkeypatch.setattr(
        fitmod, "time_split_backtest_lc_m1", lambda **k: (_ for _ in ()).throw(RuntimeError("x"))
    )
    monkeypatch.setattr(
        fitmod, "time_split_backtest_lc_m2", lambda **k: (_ for _ in ()).throw(RuntimeError("x"))
    )
    monkeypatch.setattr(
        fitmod, "time_split_backtest_apc_m3", lambda **k: (_ for _ in ()).throw(RuntimeError("x"))
    )

    with pytest.raises(ValueError, match="No model has a defined forecast RMSE on log m"):
        fitmod.model_selection_by_forecast_rmse(
            ages,
            years,
            m,
            train_end=train_end,
            model_names=("LCM1", "LCM2", "APCM3", "CBDM5", "CBDM6", "CBDM7"),
            metric="log_m",
        )


def test_model_selection_metric_logit_q_all_failed_has_details(monkeypatch):
    ages, years, m = _toy_m()
    train_end = int(years[1])

    def boom(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(fitmod, "time_split_backtest_lc_m1", boom)
    monkeypatch.setattr(fitmod, "time_split_backtest_lc_m2", boom)
    monkeypatch.setattr(fitmod, "time_split_backtest_apc_m3", boom)
    monkeypatch.setattr(fitmod, "time_split_backtest_cbd_m5", boom)
    monkeypatch.setattr(fitmod, "time_split_backtest_cbd_m6", boom)
    monkeypatch.setattr(fitmod, "time_split_backtest_cbd_m7", boom)

    with pytest.raises(ValueError, match="Details:"):
        fitmod.model_selection_by_forecast_rmse(
            ages,
            years,
            m,
            train_end=train_end,
            model_names=("LCM1", "LCM2", "APCM3", "CBDM5", "CBDM6", "CBDM7"),
            metric="logit_q",
        )


def test_model_selection_rejects_unknown_metric(monkeypatch):
    ages, years, m = _toy_m()
    train_end = int(years[1])

    # minimal stubs so it reaches metric check
    monkeypatch.setattr(
        fitmod,
        "time_split_backtest_cbd_m5",
        lambda **k: {
            "rmse_logit_forecast": 0.5,
            "train_years": np.array([years[0], train_end]),
            "test_years": np.array([train_end + 1, years[-1]]),
        },
    )

    with pytest.raises(ValueError, match="Unknown metric"):
        fitmod.model_selection_by_forecast_rmse(
            ages,
            years,
            m,
            train_end=train_end,
            model_names=("CBDM5",),
            metric="nope",  # type: ignore[arg-type]
        )


# -------------------------
# select_and_fit_best_model_for_pricing: orchestration
# -------------------------
def test_select_and_fit_best_model_for_pricing_wires_calls(monkeypatch):
    ages, years, m = _toy_m()
    train_end = int(years[1])

    fake_df = pd.DataFrame([{"Model": "LCM1", "RMSE forecast (logit q)": 0.1, "status": "ok"}])

    def fake_selection(**kwargs):
        return fake_df, "LCM1"

    def fake_fit(**kwargs):
        return fitmod.FittedModel(
            name="LCM1",
            ages=np.asarray(kwargs["ages"], dtype=float),
            years=np.asarray(kwargs["years"], dtype=int),
            model=object(),
            m_fit_surface=np.asarray(kwargs["m"]),
            m_eval_surface=np.asarray(kwargs["m"]),
            rmse_logm=0.0,
            rmse_logitq=0.0,
            aic=0.0,
            bic=0.0,
            metadata={},
        )

    monkeypatch.setattr(fitmod, "model_selection_by_forecast_rmse", fake_selection)
    monkeypatch.setattr(fitmod, "fit_mortality_model", fake_fit)

    out_df, fitted = fitmod.select_and_fit_best_model_for_pricing(
        ages=ages,
        years=years,
        m=m,
        train_end=train_end,
        model_names=("LCM1", "LCM2"),
        metric="logit_q",
        cpsplines_kwargs={"k": None, "horizon": 0, "verbose": False},
    )

    assert out_df is fake_df
    assert fitted.metadata["selection_metric"] == "logit_q"
    assert fitted.metadata["selection_train_end"] == train_end
    assert fitted.metadata["selected_model"] == "LCM1"
