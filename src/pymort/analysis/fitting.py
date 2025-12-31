"""Model fitting and selection helpers for mortality models.

This module fits LC/CBD/APC models, computes diagnostics, and helps select
models for projection or pricing workflows.

Note:
    Docstrings follow Google style for clarity and spec alignment.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from pymort._types import AnyArray, FloatArray, IntArray
from pymort.analysis import (
    rmse_aic_bic,
    smooth_mortality_with_cpsplines,
    time_split_backtest_apc_m3,
    time_split_backtest_cbd_m5,
    time_split_backtest_cbd_m6,
    time_split_backtest_cbd_m7,
    time_split_backtest_lc_m1,
    time_split_backtest_lc_m2,
)
from pymort.lifetables import m_to_q
from pymort.models.apc_m3 import APCM3
from pymort.models.cbd_m5 import CBDM5, _logit
from pymort.models.cbd_m6 import CBDM6
from pymort.models.cbd_m7 import CBDM7
from pymort.models.lc_m1 import LCM1
from pymort.models.lc_m2 import LCM2

ModelName = Literal["LCM1", "LCM2", "APCM3", "CBDM5", "CBDM6", "CBDM7"]


@dataclass
class FittedModel:
    """Container for a fitted mortality model and its diagnostics.

    This object bridges the modeling layer (LC/CBD/APC fitted on raw or
    smoothed data) and the projection/pricing layers.

    Attributes:
        name (str): Model name, e.g. "LCM2" or "CBDM7".
        ages (np.ndarray): Age grid, shape (A,).
        years (np.ndarray): Year grid for the calibration window, shape (T,).
        model (object): Fitted model instance (LCM2, CBDM7, ...).
        m_fit_surface (np.ndarray | None): Surface used for fitting, shape (A, T).
            For CBD models (logit-q), this is the m-surface associated with the
            fitted q via `m_to_q`.
        m_eval_surface (np.ndarray | None): "Truth" surface used for diagnostics,
            typically the raw m surface, shape (A, T).
        rmse_logm (float | None): In-sample RMSE on log m for LC/APC models.
        rmse_logitq (float | None): In-sample RMSE on logit(q) for all models.
        aic (float | None): Akaike Information Criterion on logit(q) errors.
        bic (float | None): Bayesian Information Criterion on logit(q) errors.
        metadata (dict[str, Any]): Extra info such as data source and notes.
    """

    name: str
    ages: AnyArray
    years: AnyArray
    model: object

    m_fit_surface: FloatArray | None = None
    m_eval_surface: FloatArray | None = None

    rmse_logm: float | None = None
    rmse_logitq: float | None = None
    aic: float | None = None
    bic: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


def _fit_single_model(
    model_name: ModelName,
    ages: AnyArray,
    years: AnyArray,
    m_fit: FloatArray,
    m_eval: FloatArray,
) -> FittedModel:
    """Fit one model on m_fit and evaluate diagnostics on m_eval.

    Args:
        model_name: Model identifier (LCM1, LCM2, APCM3, CBDM5/6/7).
        ages: Age grid, shape (A,).
        years: Year grid, shape (T,).
        m_fit: Fitting surface m_{x,t}, shape (A, T).
        m_eval: Evaluation surface m_{x,t}, shape (A, T).

    Returns:
        FittedModel with diagnostics computed on m_eval.
    """
    ages = np.asarray(ages, dtype=float)
    years = np.asarray(years, dtype=int)
    m_fit = np.asarray(m_fit, dtype=float)
    m_eval = np.asarray(m_eval, dtype=float)

    if m_fit.shape != m_eval.shape:
        raise ValueError(
            f"m_fit and m_eval must have the same shape; got {m_fit.shape} vs {m_eval.shape}."
        )

    A, T = m_fit.shape
    if ages.shape[0] != A or years.shape[0] != T:
        raise ValueError("m_fit shape must be consistent with ages and years grids.")

    m_eval_safe = np.clip(m_eval, 1e-12, None)

    # "Truth" for diagnostics on q/logit q: always from m_eval (usually raw).
    q_eval = m_to_q(m_eval_safe)
    logit_true = _logit(q_eval)

    rmse_logm: float | None = None
    rmse_logitq: float | None = None
    aic: float | None = None
    bic: float | None = None
    model: object

    # -------- LC / APC family: fit on log m --------
    if model_name == "LCM1":
        model_lcm1 = LCM1().fit(m_fit)
        ln_hat = model_lcm1.predict_log_m()  # (A, T)
        m_hat = np.exp(ln_hat)
        q_hat = m_to_q(m_hat)

        rmse_logm = float(np.sqrt(np.mean((np.log(m_eval_safe) - ln_hat) ** 2)))
        rmse_logitq, aic, bic = rmse_aic_bic(
            logit_true,
            _logit(q_hat),
            n_params=2 * A + T,
        )
        model = model_lcm1

    elif model_name == "LCM2":
        model_lcm2 = LCM2().fit(m_fit, ages, years)
        params_lcm2 = model_lcm2.params
        if params_lcm2 is None:
            raise RuntimeError("LCM2.fit() returned None params")
        m_hat = model_lcm2.predict_m()
        q_hat = m_to_q(m_hat)

        rmse_logm = float(np.sqrt(np.mean((np.log(m_eval_safe) - np.log(m_hat)) ** 2)))
        rmse_logitq, aic, bic = rmse_aic_bic(
            logit_true,
            _logit(q_hat),
            n_params=2 * A + T + len(params_lcm2.cohorts),
        )
        model = model_lcm2

    elif model_name == "APCM3":
        model_apcm3 = APCM3().fit(m_fit, ages, years)
        params_apcm3 = model_apcm3.params
        if params_apcm3 is None:
            raise RuntimeError("APCM3.fit() returned None params")
        m_hat = model_apcm3.predict_m()
        q_hat = m_to_q(m_hat)

        rmse_logm = float(np.sqrt(np.mean((np.log(m_eval_safe) - np.log(m_hat)) ** 2)))
        rmse_logitq, aic, bic = rmse_aic_bic(
            logit_true,
            _logit(q_hat),
            n_params=A + T + len(params_apcm3.cohorts),
        )
        model = model_apcm3

    # -------- CBD family: fit on logit q --------
    elif model_name == "CBDM5":
        q_fit = m_to_q(m_fit)
        model_cbdm5 = CBDM5().fit(q_fit, ages)
        q_hat = model_cbdm5.predict_q()

        rmse_logitq, aic, bic = rmse_aic_bic(
            logit_true,
            _logit(q_hat),
            n_params=2 * T,
        )
        model = model_cbdm5

    elif model_name == "CBDM6":
        q_fit = m_to_q(m_fit)
        model_cbdm6 = CBDM6().fit(q_fit, ages, years)
        params_cbdm6 = model_cbdm6.params
        if params_cbdm6 is None:
            raise RuntimeError("CBDM6.fit() returned None params")
        q_hat = model_cbdm6.predict_q()

        rmse_logitq, aic, bic = rmse_aic_bic(
            logit_true,
            _logit(q_hat),
            n_params=2 * T + len(params_cbdm6.cohorts),
        )
        model = model_cbdm6

    elif model_name == "CBDM7":
        q_fit = m_to_q(m_fit)
        model_cbdm7 = CBDM7().fit(q_fit, ages, years)
        params_cbdm7 = model_cbdm7.params
        if params_cbdm7 is None:
            raise RuntimeError("CBDM7.fit() returned None params")
        q_hat = model_cbdm7.predict_q()

        rmse_logitq, aic, bic = rmse_aic_bic(
            logit_true,
            _logit(q_hat),
            n_params=3 * T + len(params_cbdm7.cohorts),
        )
        model = model_cbdm7

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return FittedModel(
        name=model_name,
        ages=ages,
        years=years,
        model=model,
        m_fit_surface=m_fit,
        m_eval_surface=m_eval,
        rmse_logm=rmse_logm,
        rmse_logitq=rmse_logitq,
        aic=aic,
        bic=bic,
        metadata={},
    )


def fit_mortality_model(
    model_name: ModelName,
    ages: AnyArray,
    years: AnyArray,
    m: FloatArray,
    *,
    smoothing: Literal["none", "cpsplines"] = "none",
    cpsplines_kwargs: dict[str, Any] | None = None,
    eval_on_raw: bool = True,
) -> FittedModel:
    """Fit a single mortality model with optional CPsplines smoothing.

    Args:
        model_name: One of {"LCM1", "LCM2", "APCM3", "CBDM5", "CBDM6", "CBDM7"}.
        ages: Age grid, shape (A,).
        years: Year grid, shape (T,).
        m: Observed central death rates m_{x,t}, shape (A, T).
        smoothing: "none" or "cpsplines".
        cpsplines_kwargs: Extra options forwarded to
            `smooth_mortality_with_cpsplines`.
        eval_on_raw: If True, compute diagnostics against the raw m surface.
            If False, use the same smoothed surface as the fit.

    Returns:
        FittedModel with diagnostics and metadata.

    Notes:
        Comparing models fitted on raw vs. smoothed data using RMSE/AIC/BIC can
        be misleading. Models fitted on raw data often achieve smaller in-sample
        errors because they partly fit noise. Using `eval_on_raw=True` keeps the
        evaluation surface fixed (raw) and isolates the impact of smoothing on
        the systematic component.
    """
    ages = np.asarray(ages, dtype=float)
    years = np.asarray(years, dtype=int)
    m = np.asarray(m, dtype=float)

    if m.shape != (ages.shape[0], years.shape[0]):
        raise ValueError(
            f"Shape mismatch: m has shape {m.shape}, expected ({ages.shape[0]}, {years.shape[0]})."
        )

    if smoothing == "none":
        m_fit = m
        m_eval = m
        data_source = "raw"
    elif smoothing == "cpsplines":
        kw = {"k": None, "horizon": 0, "verbose": False}
        if cpsplines_kwargs is not None:
            kw.update(cpsplines_kwargs)

        k = cast(tuple[int, int] | None, kw["k"])
        horizon = cast(int, kw["horizon"])
        verbose = cast(bool, kw["verbose"])

        try:
            cp_res = smooth_mortality_with_cpsplines(
                m=m,
                ages=ages,
                years=years,
                k=k,
                horizon=horizon,
                verbose=verbose,
            )
            m_fit = cast(FloatArray, cp_res["m_fitted"])
            data_source = (
                "cpsplines_fit_eval_on_raw" if eval_on_raw else "cpsplines_fit_eval_on_smooth"
            )
        except Exception as exc:
            # Tiny grids / solver issues (e.g. mosek fusion DimensionError): fall back gracefully
            if verbose:
                print(
                    f"[fit_mortality_model] CPsplines failed ({type(exc).__name__}: {exc}); "
                    f"falling back to raw m."
                )
            m_fit = m
            data_source = "raw_fallback_cpsplines_failed"

        m_eval = m if eval_on_raw else m_fit
    else:
        raise ValueError(f"Unknown smoothing option: {smoothing!r}")

    fitted = _fit_single_model(
        model_name=model_name,
        ages=ages,
        years=years,
        m_fit=m_fit,
        m_eval=m_eval,
    )

    fitted.metadata.update(
        {
            "data_source": data_source,
            "smoothing": smoothing,
            "eval_on_raw": bool(eval_on_raw),
        }
    )

    return fitted


def model_selection_by_forecast_rmse(
    ages: AnyArray,
    years: AnyArray,
    m: FloatArray,
    *,
    train_end: int,
    model_names: Iterable[ModelName] = (
        "LCM1",
        "LCM2",
        "APCM3",
        "CBDM5",
        "CBDM6",
        "CBDM7",
    ),
    metric: Literal["log_m", "logit_q"] = "logit_q",
) -> tuple[pd.DataFrame, ModelName]:
    """Select a mortality model based on out-of-sample forecast RMSE.

    If a candidate model fails to fit or backtest, it is recorded as failed
    and skipped from selection.

    Args:
        ages: Age grid, shape (A,).
        years: Year grid, shape (T,).
        m: Mortality surface m_{x,t}, shape (A, T).
        train_end: Last year in the training window.
        model_names: Candidate model names.
        metric: Selection metric ("log_m" or "logit_q").

    Returns:
        Tuple of (selection table, best model name).
    """
    ages = np.asarray(ages, dtype=float)
    years = np.asarray(years, dtype=int)
    m = np.asarray(m, dtype=float)

    if m.shape != (ages.shape[0], years.shape[0]):
        raise ValueError(
            f"Shape mismatch: m has shape {m.shape}, expected ({ages.shape[0]}, {years.shape[0]})."
        )

    if train_end < years[0] or train_end >= years[-1]:
        raise ValueError(
            f"train_end must lie in [{int(years[0])}, {int(years[-1]) - 1}], got {train_end}."
        )
    if not np.isfinite(m).all():
        raise ValueError("m must contain only finite values.")
    if (m <= 0.0).any():
        raise ValueError("m must be strictly positive everywhere.")
    years_unique = np.unique(years)
    if years_unique.shape[0] != years.shape[0]:
        raise ValueError("years must not contain duplicates.")
    if not np.all(np.diff(years) > 0):
        raise ValueError("years must be strictly increasing.")

    if train_end not in set(years.tolist()):
        raise ValueError("train_end must be one of the provided years.")

    q_full = m_to_q(m)

    rows: list[dict[str, Any]] = []

    for name in model_names:
        # Defaults
        rmse_forecast_logm = np.nan
        rmse_forecast_logit = np.nan
        train_start = int(years[0])
        train_end_eff = int(train_end)
        test_start = int(train_end + 1)
        test_end = int(years[-1])
        status = "ok"

        try:
            if name == "LCM1":
                res = time_split_backtest_lc_m1(
                    years=years,
                    m=m,
                    train_end=train_end,
                )
                rmse_forecast_logm = float(res["rmse_log_forecast"])
                rmse_forecast_logit = float(res["rmse_logit_forecast"])
                train_years = cast(IntArray, res["train_years"])
                test_years = cast(IntArray, res["test_years"])
                train_start = int(train_years[0])
                train_end_eff = int(train_years[-1])
                test_start = int(test_years[0])
                test_end = int(test_years[-1])

            elif name == "LCM2":
                res = time_split_backtest_lc_m2(
                    ages=ages,
                    years=years,
                    m=m,
                    train_end=train_end,
                )
                rmse_forecast_logm = float(res["rmse_log_forecast"])
                rmse_forecast_logit = float(res["rmse_logit_forecast"])
                train_years = cast(IntArray, res["train_years"])
                test_years = cast(IntArray, res["test_years"])
                train_start = int(train_years[0])
                train_end_eff = int(train_years[-1])
                test_start = int(test_years[0])
                test_end = int(test_years[-1])

            elif name == "APCM3":
                res = time_split_backtest_apc_m3(
                    ages=ages,
                    years=years,
                    m=m,
                    train_end=train_end,
                )
                rmse_forecast_logm = float(res["rmse_log_forecast"])
                rmse_forecast_logit = float(res["rmse_logit_forecast"])
                train_years = cast(IntArray, res["train_years"])
                test_years = cast(IntArray, res["test_years"])
                train_start = int(train_years[0])
                train_end_eff = int(train_years[-1])
                test_start = int(test_years[0])
                test_end = int(test_years[-1])

            elif name == "CBDM5":
                res = time_split_backtest_cbd_m5(
                    ages=ages,
                    years=years,
                    q=q_full,
                    train_end=train_end,
                )
                rmse_forecast_logit = float(res["rmse_logit_forecast"])
                train_years = cast(IntArray, res["train_years"])
                test_years = cast(IntArray, res["test_years"])
                train_start = int(train_years[0])
                train_end_eff = int(train_years[-1])
                test_start = int(test_years[0])
                test_end = int(test_years[-1])

            elif name == "CBDM6":
                res = time_split_backtest_cbd_m6(
                    ages=ages,
                    years=years,
                    q=q_full,
                    train_end=train_end,
                )
                rmse_forecast_logit = float(res["rmse_logit_forecast"])
                train_years = cast(IntArray, res["train_years"])
                test_years = cast(IntArray, res["test_years"])
                train_start = int(train_years[0])
                train_end_eff = int(train_years[-1])
                test_start = int(test_years[0])
                test_end = int(test_years[-1])

            elif name == "CBDM7":
                res = time_split_backtest_cbd_m7(
                    ages=ages,
                    years=years,
                    q=q_full,
                    train_end=train_end,
                )
                rmse_forecast_logit = float(res["rmse_logit_forecast"])
                train_years = cast(IntArray, res["train_years"])
                test_years = cast(IntArray, res["test_years"])
                train_start = int(train_years[0])
                train_end_eff = int(train_years[-1])
                test_start = int(test_years[0])
                test_end = int(test_years[-1])

            else:
                raise ValueError(f"Unknown model name in model_selection: {name!r}")

        except Exception as e:
            # Record failure, keep RMSE as NaN, move on
            status = f"failed: {e}"

        rows.append(
            {
                "Model": name,
                "RMSE forecast (log m)": rmse_forecast_logm,
                "RMSE forecast (logit q)": rmse_forecast_logit,
                "train_start": train_start,
                "train_end": train_end_eff,
                "test_start": test_start,
                "test_end": test_end,
                "status": status,
            }
        )

    df = pd.DataFrame(rows)

    # Selection among finite RMSE only
    if metric == "log_m":
        col = "RMSE forecast (log m)"
        mask = np.isfinite(df[col].to_numpy())
        if not mask.any():
            raise ValueError(
                "No model has a defined forecast RMSE on log m; "
                "cannot select best model with metric='log_m'."
            )
        idx = df.loc[mask, col].idxmin()

    elif metric == "logit_q":
        col = "RMSE forecast (logit q)"
        mask = np.isfinite(df[col].to_numpy())
        if not mask.any():
            # Helpful debug: show who failed
            failed = df[["Model", "status"]].to_dict(orient="records")
            raise ValueError(
                "No model has a defined forecast RMSE on logit(q); "
                f"all candidates failed or returned NaN. Details: {failed}"
            )
        idx = df.loc[mask, col].idxmin()

    else:
        raise ValueError(f"Unknown metric: {metric!r} (expected 'log_m' or 'logit_q').")

    best_model_name = df.loc[idx, "Model"]
    return df, best_model_name


def select_and_fit_best_model_for_pricing(
    ages: AnyArray,
    years: AnyArray,
    m: FloatArray,
    *,
    train_end: int,
    model_names: Iterable[ModelName] = (
        "LCM1",
        "LCM2",
        "APCM3",
        "CBDM5",
        "CBDM6",
        "CBDM7",
    ),
    metric: Literal["log_m", "logit_q"] = "logit_q",
    cpsplines_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, FittedModel]:
    """Select the best model and refit it on the full dataset.

    Workflow:
        1) Use forecast RMSE on raw data to choose the structural model.
        2) Smooth the full mortality surface with CPsplines.
        3) Fit the selected model on the smoothed surface (full window),
           while evaluating diagnostics against raw m.

    Args:
        ages: Age grid, shape (A,).
        years: Year grid, shape (T,).
        m: Raw central death rates surface, shape (A, T).
        train_end: Last year in training set used for forecast backtests.
        model_names: Candidate models for selection.
        metric: Selection metric ("log_m" or "logit_q").
        cpsplines_kwargs: Optional CPsplines settings for the final smoothing
            step (e.g., {"k": None, "horizon": 0, "verbose": False}).

    Returns:
        Tuple of (selection table, fitted best model).
    """
    # 1) Model selection on raw data via forecast RMSE
    selection_df, best_model_name = model_selection_by_forecast_rmse(
        ages=ages,
        years=years,
        m=m,
        train_end=train_end,
        model_names=model_names,
        metric=metric,
    )

    # 2) Fit best model on CPsplines-smoothed data (full window)
    #    horizon=0 ici: on ne se sert pas du forecast CPsplines, seulement
    #    de la surface lissée historique pour le fit structurel.
    default_cp = {"k": None, "horizon": 0, "verbose": False}
    if cpsplines_kwargs is not None:
        default_cp.update(cpsplines_kwargs)

    fitted_best = fit_mortality_model(
        model_name=best_model_name,
        ages=ages,
        years=years,
        m=m,
        smoothing="cpsplines",
        cpsplines_kwargs=default_cp,
        eval_on_raw=True,  # diagnostics toujours versus données brutes
    )

    # Ajouter un peu de métadonnées utiles
    fitted_best.metadata.setdefault("selection_metric", metric)
    fitted_best.metadata.setdefault("selection_train_end", int(train_end))
    fitted_best.metadata.setdefault("selected_from_models", tuple(model_names))
    fitted_best.metadata["selected_model"] = best_model_name

    return selection_df, fitted_best
