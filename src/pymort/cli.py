from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd
import typer

from pymort.analysis import MortalityScenarioSet, smooth_mortality_with_cpsplines
from pymort.analysis.fitting import (
    ModelName,
    fit_mortality_model,
    model_selection_by_forecast_rmse,
    select_and_fit_best_model_for_pricing,
)
from pymort.analysis.risk_tools import summarize_pv_paths
from pymort.analysis.scenario_analysis import (
    ShockSpec,
    apply_mortality_shock,
    apply_shock_chain,
)
from pymort.analysis.sensitivities import (
    mortality_delta_by_age,
    rate_convexity,
    rate_sensitivity,
)
from pymort.interest_rates.hull_white import build_interest_rate_scenarios
from pymort.lifetables import m_to_q
from pymort.pipeline import (
    build_joint_scenarios,
    build_projection_pipeline,
    build_risk_neutral_pipeline,
    hedging_pipeline,
    pricing_pipeline,
    reporting_pipeline,
    risk_analysis_pipeline,
    stress_testing_pipeline,
)
from pymort.pricing.liabilities import CohortLifeAnnuitySpec
from pymort.pricing.longevity_bonds import LongevityBondSpec
from pymort.pricing.mortality_derivatives import QForwardSpec, SForwardSpec
from pymort.pricing.risk_neutral import (
    MultiInstrumentQuote,
    build_calibration_cache,
    build_scenarios_under_lambda_fast,
)
from pymort.pricing.survivor_swaps import SurvivorSwapSpec
from pymort.visualization import (
    animate_mortality_surface,
    animate_survival_curves,
    plot_lexis,
    plot_mortality_fan,
    plot_survival_fan,
)

app = typer.Typer(help="PYMORT – Longevity bond & mortality toolkit")


# ---------------------------------------------------------------------------
# Helpers and shared options
# ---------------------------------------------------------------------------


def _setup_logging(level: str, verbose: bool, quiet: bool) -> None:
    log_level = level.upper()
    if verbose:
        log_level = "DEBUG"
    if quiet:
        log_level = "ERROR"
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(levelname)s | %(message)s",
    )


def _load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise typer.BadParameter(f"Config file {path} does not exist.")
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise typer.BadParameter(
                f"Could not parse config {path} as JSON; install pyyaml for YAML support."
            ) from exc
        return yaml.safe_load(text)


def _ensure_outdir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(path.iterdir()):
        return


def _parse_number_list(spec: Optional[str]) -> Optional[np.ndarray]:
    if spec is None:
        return None
    if spec.strip() == "":
        return None
    items = [float(x) for x in spec.replace(" ", "").split(",") if x]
    return np.asarray(items, dtype=float)


def _read_numeric_series(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".npz":
        data = np.load(path)
        first_key = list(data.keys())[0]
        return np.asarray(data[first_key]).reshape(-1)
    df = (
        pd.read_parquet(path)
        if ext in {".parquet", ".pq"}
        else pd.read_csv(path, header=None)
    )
    arr = df.to_numpy().reshape(-1)
    return np.asarray(arr, dtype=float)


def _read_numeric_matrix(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext in {".npy", ".npz"}:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[list(arr.keys())[0]]
        return np.asarray(arr, dtype=float)
    df = (
        pd.read_parquet(path)
        if ext in {".parquet", ".pq"}
        else pd.read_csv(path, header=None)
    )
    return np.asarray(df.to_numpy(), dtype=float)


def _read_numeric_cube(path: Path) -> np.ndarray:
    arr = _read_numeric_matrix(path)
    if arr.ndim == 3:
        return arr
    raise typer.BadParameter(
        f"Expected 3D array at {path}; use .npy/.npz with shape (N,M,T)."
    )


def _load_m_surface(
    m_path: Path,
    ages_inline: Optional[str],
    years_inline: Optional[str],
    ages_path: Optional[Path],
    years_path: Optional[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ext = m_path.suffix.lower()
    ages = _parse_number_list(ages_inline) if ages_inline else None
    years = _parse_number_list(years_inline) if years_inline else None
    if ages_path is not None:
        ages = _read_numeric_series(ages_path)
    if years_path is not None:
        years = _read_numeric_series(years_path)

    if ext in {".npy", ".npz"}:
        data = np.load(m_path)
        if isinstance(data, np.lib.npyio.NpzFile):
            m = np.asarray(data[list(data.keys())[0]], dtype=float)
        else:
            m = np.asarray(data, dtype=float)
        if m.ndim != 2:
            raise typer.BadParameter("m-path .npy must contain a 2D array (A,T).")
        if ages is None or years is None:
            raise typer.BadParameter(
                "Provide --ages/--years when loading .npy wide arrays."
            )
        if m.shape != (ages.shape[0], years.shape[0]):
            raise typer.BadParameter(
                f"m shape {m.shape} incompatible with ages {ages.shape} and years {years.shape}."
            )
        return ages, years, m

    df = pd.read_parquet(m_path) if ext in {".parquet", ".pq"} else pd.read_csv(m_path)
    cols_lower = {c.lower() for c in df.columns}
    if {"age", "year"} <= cols_lower:
        # long format
        age_col = [c for c in df.columns if c.lower() == "age"][0]
        year_col = [c for c in df.columns if c.lower() == "year"][0]
        rate_col = [c for c in df.columns if c.lower() in {"m", "mx", "rate"}]
        if not rate_col:
            raise typer.BadParameter("Long format requires column named 'm' or 'rate'.")
        rate_col = rate_col[0]
        ages = df[age_col].unique()
        years = df[year_col].unique()
        ages = np.sort(ages.astype(float))
        years = np.sort(years.astype(int))
        pivot = (
            df.pivot_table(index=age_col, columns=year_col, values=rate_col)
            .reindex(index=ages, columns=years)
            .to_numpy()
        )
        return ages, years, np.asarray(pivot, dtype=float)

    # wide format: first column age, others years
    first_col = df.columns[0]
    if first_col.lower() not in {"age", "ages"}:
        raise typer.BadParameter(
            "Wide format expected first column 'Age' followed by year columns."
        )
    ages = df[first_col].to_numpy(dtype=float)
    years = np.asarray([int(c) for c in df.columns[1:]], dtype=int)
    m = df.iloc[:, 1:].to_numpy(dtype=float)
    return ages, years, m


def _save_table(obj: Any, path: Path, fmt: str) -> None:
    if isinstance(obj, pd.DataFrame):
        df = obj
    elif isinstance(obj, dict):
        df = pd.DataFrame([obj])
    else:
        df = pd.DataFrame(obj)

    fmt_l = fmt.lower()
    if fmt_l == "json":
        df.to_json(path, orient="records", lines=True)
    elif fmt_l == "csv":
        df.to_csv(path, index=False)
    elif fmt_l == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise typer.BadParameter(f"Unsupported format: {fmt}")


def _save_npz(data: Dict[str, Any], path: Path) -> None:
    # convert metadata to json-serializable
    meta = data.get("metadata")
    if meta is not None and not isinstance(meta, (str, bytes)):
        try:
            meta_enc = json.dumps(meta)
        except TypeError:
            meta_enc = json.dumps(str(meta))
        data = {**data, "metadata": meta_enc}
    np.savez_compressed(path, **data)


def _load_scenarios(path: Path) -> MortalityScenarioSet:
    ext = path.suffix.lower()
    if ext in {".npz"}:
        data = np.load(path, allow_pickle=True)
        q = np.asarray(data["q_paths"])
        s = np.asarray(data["S_paths"])
        ages = np.asarray(data["ages"])
        years = np.asarray(data["years"])
        meta_raw = data.get("metadata")
        metadata: Dict[str, Any] = {}
        if meta_raw is not None:
            if isinstance(meta_raw, (bytes, str)):
                try:
                    metadata = json.loads(meta_raw)
                except Exception:
                    metadata = {}
            else:
                try:
                    metadata = json.loads(str(meta_raw))
                except Exception:
                    metadata = meta_raw.item() if hasattr(meta_raw, "item") else {}
        return MortalityScenarioSet(
            years=years,
            ages=ages,
            q_paths=q,
            S_paths=s,
            m_paths=data.get("m_paths") if "m_paths" in data else None,
            discount_factors=(
                data.get("discount_factors") if "discount_factors" in data else None
            ),
            metadata=metadata,
        )
    raise typer.BadParameter(f"Unsupported scenario format for {path} (use .npz).")


def _save_scenarios(scen_set: MortalityScenarioSet, path: Path) -> None:
    payload = {
        "q_paths": scen_set.q_paths,
        "S_paths": scen_set.S_paths,
        "ages": scen_set.ages,
        "years": scen_set.years,
        "metadata": scen_set.metadata,
    }
    if scen_set.m_paths is not None:
        payload["m_paths"] = scen_set.m_paths
    if scen_set.discount_factors is not None:
        payload["discount_factors"] = scen_set.discount_factors
    _save_npz(payload, path)


def _maybe_pickle(obj: Any, path: Optional[Path]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _to_spec(kind: str, cfg: Dict[str, Any]) -> Union[
    LongevityBondSpec,
    SurvivorSwapSpec,
    SForwardSpec,
    QForwardSpec,
    CohortLifeAnnuitySpec,
]:
    k = kind.lower()
    if k == "longevity_bond":
        return LongevityBondSpec(**cfg)
    if k == "survivor_swap":
        return SurvivorSwapSpec(**cfg)
    if k == "s_forward":
        return SForwardSpec(**cfg)
    if k == "q_forward":
        return QForwardSpec(**cfg)
    if k == "life_annuity":
        return CohortLifeAnnuitySpec(**cfg)
    raise typer.BadParameter(f"Unknown instrument kind '{kind}'.")


@dataclass
class CLIContext:
    outdir: Path
    output_format: str
    seed: Optional[int]
    verbose: bool
    quiet: bool
    log_level: str
    overwrite: bool
    config: Dict[str, Any]
    save_path: Optional[Path]
    load_path: Optional[Path]


def _ctx(ctx: typer.Context) -> CLIContext:
    return cast(CLIContext, ctx.obj)


# ---------------------------------------------------------------------------
# Global options
# ---------------------------------------------------------------------------


@app.callback()
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="YAML/JSON config file with defaults."
    ),
    seed: Optional[int] = typer.Option(None, help="RNG seed."),
    outdir: Path = typer.Option(Path("outputs"), help="Output directory."),
    format: str = typer.Option("csv", "--format", help="Tabular output format."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
    quiet: bool = typer.Option(False, "--quiet", help="Quiet logging."),
    log_level: str = typer.Option(
        "INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)."
    ),
    overwrite: bool = typer.Option(False, help="Allow overwriting output files."),
    save: Optional[Path] = typer.Option(
        None, help="Optional pickle/npz path to save result."
    ),
    load: Optional[Path] = typer.Option(None, help="Load a previously saved object."),
) -> None:
    cfg = _load_config(config)
    _setup_logging(log_level, verbose, quiet)
    outdir.mkdir(parents=True, exist_ok=True)
    ctx.obj = CLIContext(
        outdir=outdir,
        output_format=format,
        seed=seed,
        verbose=verbose,
        quiet=quiet,
        log_level=log_level,
        overwrite=overwrite,
        config=cfg,
        save_path=save,
        load_path=load,
    )


# ---------------------------------------------------------------------------
# DATA subcommands
# ---------------------------------------------------------------------------

data_app = typer.Typer(help="Data utilities (validation, clipping, conversion).")


@data_app.command("validate-m")
def data_validate_m(
    ctx: typer.Context,
    m_path: Path = typer.Option(
        ..., help="Path to mortality surface (csv/parquet/npy)."
    ),
    ages: Optional[str] = typer.Option(None, help="Inline ages list, e.g. '60,61,62'."),
    years: Optional[str] = typer.Option(
        None, help="Inline years list, e.g. '2000,2001'."
    ),
    ages_path: Optional[Path] = typer.Option(
        None, help="Path to ages (csv/parquet/npy)."
    ),
    years_path: Optional[Path] = typer.Option(
        None, help="Path to years (csv/parquet/npy)."
    ),
) -> None:
    c = _ctx(ctx)
    ages_arr, years_arr, m = _load_m_surface(m_path, ages, years, ages_path, years_path)
    report = {
        "shape": m.shape,
        "ages_min": float(ages_arr.min()),
        "ages_max": float(ages_arr.max()),
        "years_min": int(years_arr.min()),
        "years_max": int(years_arr.max()),
        "finite": bool(np.isfinite(m).all()),
        "non_negative": bool((m >= 0.0).all()),
        "min_m": float(np.nanmin(m)),
        "max_m": float(np.nanmax(m)),
        "has_nan": bool(np.isnan(m).any()),
    }
    out = c.outdir / "validation_m.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    typer.echo(f"Validation report saved to {out}")


@data_app.command("clip-m")
def data_clip_m(
    ctx: typer.Context,
    m_path: Path = typer.Option(
        ..., help="Path to mortality surface (csv/parquet/npy)."
    ),
    eps: float = typer.Option(1e-12, help="Minimum value for clipping."),
    ages: Optional[str] = typer.Option(None, help="Inline ages."),
    years: Optional[str] = typer.Option(None, help="Inline years."),
    ages_path: Optional[Path] = typer.Option(None, help="Path to ages."),
    years_path: Optional[Path] = typer.Option(None, help="Path to years."),
    output: Optional[Path] = typer.Option(None, help="Output path (.npz or .npy)."),
) -> None:
    c = _ctx(ctx)
    ages_arr, years_arr, m = _load_m_surface(m_path, ages, years, ages_path, years_path)
    m_clipped = np.clip(m, eps, None)
    out = output or c.outdir / "m_clipped.npz"
    _save_npz({"m": m_clipped, "ages": ages_arr, "years": years_arr}, out)
    typer.echo(f"Clipped m saved to {out}")


@data_app.command("to-q")
def data_to_q(
    ctx: typer.Context,
    m_path: Path = typer.Option(
        ..., help="Path to mortality surface (csv/parquet/npy)."
    ),
    ages: Optional[str] = typer.Option(None),
    years: Optional[str] = typer.Option(None),
    ages_path: Optional[Path] = typer.Option(None),
    years_path: Optional[Path] = typer.Option(None),
    output: Optional[Path] = typer.Option(None, help="Output path (.npz)."),
) -> None:
    c = _ctx(ctx)
    ages_arr, years_arr, m = _load_m_surface(m_path, ages, years, ages_path, years_path)
    q = m_to_q(m)
    out = output or c.outdir / "q_surface.npz"
    _save_npz({"q": q, "ages": ages_arr, "years": years_arr}, out)
    typer.echo(f"q saved to {out}")


app.add_typer(data_app, name="data")


# ---------------------------------------------------------------------------
# SMOOTH subcommands
# ---------------------------------------------------------------------------

smooth_app = typer.Typer(help="Smoothing utilities (CPsplines).")


@smooth_app.command("cpsplines")
def smooth_cpsplines_cmd(
    ctx: typer.Context,
    m_path: Path = typer.Option(..., help="Path to mortality surface."),
    ages: Optional[str] = typer.Option(None),
    years: Optional[str] = typer.Option(None),
    ages_path: Optional[Path] = typer.Option(None),
    years_path: Optional[Path] = typer.Option(None),
    deg: str = typer.Option("3,3", help="Degrees for (age,year) splines."),
    ord_d: str = typer.Option("2,2", help="Orders of derivative penalties."),
    k: Optional[str] = typer.Option(None, help="Knots as 'ka,kt' or 'auto'."),
    sp_method: str = typer.Option("grid_search", help="Smoothing parameter method."),
    sp_args: Optional[str] = typer.Option(None, help="JSON string for smoothing args."),
    horizon: int = typer.Option(0, help="Forecast horizon for CPsplines."),
    output: Optional[Path] = typer.Option(
        None, help="Output npz path for fitted surface."
    ),
) -> None:
    c = _ctx(ctx)
    ages_arr, years_arr, m = _load_m_surface(m_path, ages, years, ages_path, years_path)
    deg_tuple = tuple(int(x) for x in deg.split(","))
    ord_tuple = tuple(int(x) for x in ord_d.split(","))
    k_tuple = None
    if k not in (None, "auto"):
        k_tuple = tuple(int(x) for x in str(k).split(","))
    sp_kwargs = json.loads(sp_args) if sp_args else None
    res = smooth_mortality_with_cpsplines(
        m=m,
        ages=ages_arr,
        years=years_arr,
        deg=deg_tuple,
        ord_d=ord_tuple,
        k=k_tuple,
        sp_method=sp_method,
        sp_args=sp_kwargs,
        horizon=horizon,
        verbose=c.verbose,
    )
    out = output or c.outdir / "cpsplines.npz"
    payload = {
        "m_fitted": res["m_fitted"],
        "ages": ages_arr,
        "years": years_arr,
        "metadata": {"horizon": horizon, "deg": deg_tuple, "ord_d": ord_tuple},
    }
    if "m_forecast" in res:
        payload["m_forecast"] = res["m_forecast"]
    if "years_forecast" in res:
        payload["years_forecast"] = res["years_forecast"]
    _save_npz(payload, out)
    typer.echo(f"CPsplines result saved to {out}")


app.add_typer(smooth_app, name="smooth")


# ---------------------------------------------------------------------------
# FIT subcommands
# ---------------------------------------------------------------------------

fit_app = typer.Typer(help="Model fitting and selection.")


@fit_app.command("one")
def fit_one_cmd(
    ctx: typer.Context,
    model: ModelName = typer.Option(..., help="Model name."),
    m_path: Path = typer.Option(...),
    ages: Optional[str] = typer.Option(None),
    years: Optional[str] = typer.Option(None),
    ages_path: Optional[Path] = typer.Option(None),
    years_path: Optional[Path] = typer.Option(None),
    smoothing: str = typer.Option("none", help="none or cpsplines"),
    eval_on_raw: bool = typer.Option(True, help="Evaluate diagnostics on raw m."),
    cpsplines_k: Optional[int] = typer.Option(None),
    cpsplines_horizon: int = typer.Option(0),
    output: Optional[Path] = typer.Option(None, help="Pickle path for fitted model."),
) -> None:
    c = _ctx(ctx)
    ages_arr, years_arr, m = _load_m_surface(m_path, ages, years, ages_path, years_path)
    cps_kwargs = (
        {"k": cpsplines_k, "horizon": cpsplines_horizon, "verbose": c.verbose}
        if smoothing == "cpsplines"
        else None
    )
    fitted = fit_mortality_model(
        model_name=model,
        ages=ages_arr,
        years=years_arr,
        m=m,
        smoothing=smoothing,
        cpsplines_kwargs=cps_kwargs,
        eval_on_raw=eval_on_raw,
    )
    out = output or c.outdir / f"fitted_{model}.pkl"
    _maybe_pickle(fitted, out)
    typer.echo(f"Fitted {model} saved to {out}")


@fit_app.command("select")
def fit_select_cmd(
    ctx: typer.Context,
    m_path: Path = typer.Option(...),
    train_end: int = typer.Option(..., help="Last year in training set."),
    models: List[str] = typer.Option(
        [], "--models", "-m", help="Comma-separated model list (default all)."
    ),
    metric: str = typer.Option("logit_q", help="Selection metric (log_m or logit_q)."),
    ages: Optional[str] = typer.Option(None),
    years: Optional[str] = typer.Option(None),
    ages_path: Optional[Path] = typer.Option(None),
    years_path: Optional[Path] = typer.Option(None),
    output: Optional[Path] = typer.Option(None, help="Selection table path."),
) -> None:
    c = _ctx(ctx)
    ages_arr, years_arr, m = _load_m_surface(m_path, ages, years, ages_path, years_path)
    model_names: Sequence[ModelName] = (
        tuple(name.upper() for name in models) if models else ("LCM1", "LCM2", "APCM3", "CBDM5", "CBDM6", "CBDM7")  # type: ignore[assignment]
    )
    df, best = model_selection_by_forecast_rmse(
        ages=ages_arr,
        years=years_arr,
        m=m,
        train_end=train_end,
        model_names=model_names,
        metric=metric,
    )
    out = output or c.outdir / "model_selection.csv"
    _save_table(df, out, c.output_format)
    typer.echo(f"Selection table saved to {out}; best={best}")


@fit_app.command("select-and-fit")
def fit_select_and_fit_cmd(
    ctx: typer.Context,
    m_path: Path = typer.Option(...),
    train_end: int = typer.Option(...),
    models: List[str] = typer.Option([], "--models", "-m"),
    metric: str = typer.Option("logit_q"),
    cpsplines_k: Optional[int] = typer.Option(None),
    cpsplines_horizon: int = typer.Option(0),
    ages: Optional[str] = typer.Option(None),
    years: Optional[str] = typer.Option(None),
    ages_path: Optional[Path] = typer.Option(None),
    years_path: Optional[Path] = typer.Option(None),
    output: Optional[Path] = typer.Option(None, help="Pickle path for fitted model."),
    selection_output: Optional[Path] = typer.Option(None, help="Selection table path."),
) -> None:
    c = _ctx(ctx)
    ages_arr, years_arr, m = _load_m_surface(m_path, ages, years, ages_path, years_path)
    model_names: Sequence[ModelName] = (
        tuple(name.upper() for name in models) if models else ("LCM1", "LCM2", "APCM3", "CBDM5", "CBDM6", "CBDM7")  # type: ignore[assignment]
    )
    cp_kwargs = {"k": cpsplines_k, "horizon": cpsplines_horizon, "verbose": c.verbose}
    selection_df, fitted = select_and_fit_best_model_for_pricing(
        ages=ages_arr,
        years=years_arr,
        m=m,
        train_end=train_end,
        model_names=model_names,
        metric=metric,
        cpsplines_kwargs=cp_kwargs,
    )
    sel_out = selection_output or c.outdir / "model_selection.csv"
    _save_table(selection_df, sel_out, c.output_format)
    out = output or c.outdir / "fitted_best.pkl"
    _maybe_pickle({"selection": selection_df, "fitted": fitted}, out)
    typer.echo(f"Best model {fitted.name} saved to {out}; selection table {sel_out}")


app.add_typer(fit_app, name="fit")


# ---------------------------------------------------------------------------
# SCENARIO subcommands
# ---------------------------------------------------------------------------

scen_app = typer.Typer(help="Scenario generation and summaries.")


@scen_app.command("build-P")
def scen_build_p_cmd(
    ctx: typer.Context,
    m_path: Path = typer.Option(..., help="Mortality surface (csv/parquet/npy)."),
    train_end: int = typer.Option(..., help="Last year in training set for backtest."),
    horizon: int = typer.Option(50, help="Projection horizon."),
    n_scenarios: int = typer.Option(1000, help="Target number of scenarios."),
    models: List[str] = typer.Option(
        [], "--models", "-m", help="Subset of models to consider."
    ),
    cpsplines_k: Optional[int] = typer.Option(None),
    cpsplines_horizon: int = typer.Option(0),
    seed: Optional[int] = typer.Option(None),
    ages: Optional[str] = typer.Option(None),
    years: Optional[str] = typer.Option(None),
    ages_path: Optional[Path] = typer.Option(None),
    years_path: Optional[Path] = typer.Option(None),
    output: Optional[Path] = typer.Option(None, help="Output scenarios npz."),
) -> None:
    """
    End-to-end projection pipeline (P-measure) → MortalityScenarioSet.
    """
    c = _ctx(ctx)
    ages_arr, years_arr, m = _load_m_surface(m_path, ages, years, ages_path, years_path)
    model_names: Sequence[str] = (
        tuple(name.upper() for name in models)
        if models
        else ("LCM1", "LCM2", "APCM3", "CBDM5", "CBDM6", "CBDM7")
    )
    scen_set = build_projection_pipeline(
        ages=ages_arr,
        years=years_arr,
        m=m,
        train_end=train_end,
        horizon=horizon,
        n_scenarios=n_scenarios,
        model_names=model_names,
        cpsplines_kwargs={
            "k": cpsplines_k,
            "horizon": cpsplines_horizon,
            "verbose": c.verbose,
        },
        bootstrap_kwargs={"include_last": True},
        seed=seed if seed is not None else c.seed,
    )
    out = output or c.outdir / "scenarios_P.npz"
    _save_scenarios(scen_set, out)
    typer.echo(
        f"Scenarios (P) saved to {out} | N={scen_set.n_scenarios()} horizon={scen_set.horizon()}"
    )


@scen_app.command("build-Q")
def scen_build_q_cmd(
    ctx: typer.Context,
    m_path: Path = typer.Option(..., help="Raw mortality surface for calibration."),
    model_name: str = typer.Option(
        "CBDM7", help="Model for lambda calibration (LCM2 or CBDM7)."
    ),
    lambda_esscher: float = typer.Option(
        ..., help="Lambda Esscher tilt (single value or first component)."
    ),
    B_bootstrap: int = typer.Option(100),
    n_process: int = typer.Option(200),
    horizon: int = typer.Option(50),
    seed: Optional[int] = typer.Option(None),
    scale_sigma: float = typer.Option(1.0, help="Scale factor for sigma (vega)."),
    include_last: bool = typer.Option(False),
    ages: Optional[str] = typer.Option(None),
    years: Optional[str] = typer.Option(None),
    ages_path: Optional[Path] = typer.Option(None),
    years_path: Optional[Path] = typer.Option(None),
    output: Optional[Path] = typer.Option(None, help="Output scenarios npz."),
) -> None:
    c = _ctx(ctx)
    ages_arr, years_arr, m = _load_m_surface(m_path, ages, years, ages_path, years_path)
    cache = build_calibration_cache(
        ages=ages_arr,
        years=years_arr,
        m=m,
        model_name=model_name,
        B_bootstrap=B_bootstrap,
        n_process=n_process,
        horizon=horizon,
        seed=seed if seed is not None else c.seed,
        include_last=include_last,
    )
    scen_set_q = build_scenarios_under_lambda_fast(
        cache=cache, lambda_esscher=lambda_esscher, scale_sigma=scale_sigma
    )
    out = output or c.outdir / "scenarios_Q.npz"
    _save_scenarios(scen_set_q, out)
    typer.echo(
        f"Scenarios (Q) saved to {out} | N={scen_set_q.n_scenarios()} horizon={scen_set_q.horizon()}"
    )


@scen_app.command("summarize")
def scen_summarize_cmd(
    ctx: typer.Context,
    scen_path: Path = typer.Option(..., help="Scenario set npz."),
    percentiles: str = typer.Option("5,50,95"),
    output: Optional[Path] = typer.Option(None, help="Summary table path."),
) -> None:
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    perc = [int(x) for x in percentiles.split(",") if x]
    summary = summarize_pv_paths(
        scen_set.q_paths.reshape(scen_set.q_paths.shape[0], -1).mean(axis=1)
    )
    out = output or c.outdir / "scen_summary.json"
    out.write_text(json.dumps(asdict(summary), indent=2))
    typer.echo(f"Scenario summary saved to {out}")


app.add_typer(scen_app, name="scen")


# ---------------------------------------------------------------------------
# STRESS subcommands
# ---------------------------------------------------------------------------

stress_app = typer.Typer(help="Scenario stress testing.")


@stress_app.command("apply")
def stress_apply_cmd(
    ctx: typer.Context,
    scen_path: Path = typer.Option(..., help="Scenario set npz."),
    shock_type: str = typer.Option("long_life"),
    magnitude: float = typer.Option(0.1),
    pandemic_year: Optional[int] = typer.Option(None),
    pandemic_duration: int = typer.Option(1),
    plateau_start_year: Optional[int] = typer.Option(None),
    accel_start_year: Optional[int] = typer.Option(None),
    output: Optional[Path] = typer.Option(None, help="Output stressed scenarios npz."),
) -> None:
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    stressed_map = stress_testing_pipeline(
        scen_set,
        shock_specs=[
            {
                "name": shock_type,
                "shock_type": shock_type,
                "params": {
                    "magnitude": magnitude,
                    "pandemic_year": pandemic_year,
                    "pandemic_duration": pandemic_duration,
                    "plateau_start_year": plateau_start_year,
                    "accel_start_year": accel_start_year,
                },
            }
        ],
    )
    stressed = stressed_map[shock_type]
    out = output or c.outdir / f"scenarios_{shock_type}.npz"
    _save_scenarios(stressed, out)
    typer.echo(f"Stressed scenarios saved to {out}")


@stress_app.command("chain")
def stress_chain_cmd(
    ctx: typer.Context,
    scen_path: Path = typer.Option(...),
    chain_spec: Path = typer.Option(..., help="JSON/YAML list of shocks."),
    output: Optional[Path] = typer.Option(None),
) -> None:
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    spec_list = _load_config(chain_spec)
    if not isinstance(spec_list, list):
        raise typer.BadParameter("chain-spec must be a list of shock dictionaries.")
    chain: list[ShockSpec] = []
    for spec in spec_list:
        if not isinstance(spec, dict) or "shock_type" not in spec:
            raise typer.BadParameter(
                "Each shock must be a dict with shock_type and params."
            )
        chain.append(
            ShockSpec(
                name=str(spec.get("name", spec["shock_type"])),
                shock_type=str(spec["shock_type"]),
                params=spec.get("params", {}),
            )
        )
    current = apply_shock_chain(scen_set, chain)
    out = output or c.outdir / "scenarios_chain.npz"
    _save_scenarios(current, out)
    typer.echo(f"Chained scenarios saved to {out}")


@stress_app.command("bundle")
def stress_bundle_cmd(
    ctx: typer.Context,
    scen_path: Path = typer.Option(...),
    long_life_bump: float = typer.Option(0.1),
    short_life_bump: float = typer.Option(0.1),
    output: Optional[Path] = typer.Option(None, help="Output directory for bundle."),
) -> None:
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    outdir = output or (c.outdir / "bundle")
    outdir.mkdir(parents=True, exist_ok=True)
    base_path = outdir / "base.npz"
    _save_scenarios(scen_set, base_path)
    long_life = apply_mortality_shock(
        scen_set, shock_type="long_life", magnitude=long_life_bump
    )
    short_life = apply_mortality_shock(
        scen_set, shock_type="short_life", magnitude=short_life_bump
    )
    _save_scenarios(long_life, outdir / "optimistic.npz")
    _save_scenarios(short_life, outdir / "pessimistic.npz")
    manifest = {
        "base": str(base_path),
        "optimistic": str(outdir / "optimistic.npz"),
        "pessimistic": str(outdir / "pessimistic.npz"),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    typer.echo(f"Bundle saved under {outdir}")


app.add_typer(stress_app, name="stress")


# ---------------------------------------------------------------------------
# PRICING subcommands
# ---------------------------------------------------------------------------

price_app = typer.Typer(help="Pricing of longevity instruments.")


def _load_scen_and_age(ctx: typer.Context, scen_path: Path) -> MortalityScenarioSet:
    return _load_scenarios(scen_path)


@price_app.command("longevity-bond")
def price_longevity_bond_cmd(
    ctx: typer.Context,
    scen_path: Path = typer.Option(...),
    issue_age: float = typer.Option(...),
    maturity_years: int = typer.Option(...),
    notional: float = typer.Option(1.0),
    include_principal: bool = typer.Option(
        True, "--include-principal/--no-include-principal"
    ),
    short_rate: Optional[float] = typer.Option(None, help="Flat short rate."),
    output: Optional[Path] = typer.Option(None, help="Result JSON/CSV."),
) -> None:
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    spec = LongevityBondSpec(
        issue_age=issue_age,
        notional=notional,
        include_principal=include_principal,
        maturity_years=maturity_years,
    )
    prices = pricing_pipeline(
        scen_Q=scen_set,
        specs={"bond": spec},
        short_rate=float(short_rate) if short_rate is not None else 0.0,
    )
    out = output or c.outdir / "price_longevity_bond.json"
    out.write_text(json.dumps(prices, indent=2))
    typer.echo(f"Price={prices['bond']:.6f} saved to {out}")


@price_app.command("survivor-swap")
def price_survivor_swap_cmd(
    ctx: typer.Context,
    scen_path: Path = typer.Option(...),
    age: float = typer.Option(...),
    maturity_years: int = typer.Option(...),
    notional: float = typer.Option(1.0),
    strike: Optional[float] = typer.Option(None),
    payer: str = typer.Option("fixed", help="fixed or floating"),
    short_rate: Optional[float] = typer.Option(None),
    output: Optional[Path] = typer.Option(None),
) -> None:
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    spec = SurvivorSwapSpec(
        age=age,
        maturity_years=maturity_years,
        notional=notional,
        strike=strike,
        payer=payer,
    )
    prices = pricing_pipeline(
        scen_Q=scen_set,
        specs={"swap": spec},
        short_rate=float(short_rate) if short_rate is not None else 0.0,
    )
    out = output or c.outdir / "price_survivor_swap.json"
    out.write_text(json.dumps(prices, indent=2))
    typer.echo(f"Price={prices['swap']:.6f} saved to {out}")


@price_app.command("q-forward")
def price_q_forward_cmd(
    ctx: typer.Context,
    scen_path: Path = typer.Option(...),
    age: float = typer.Option(...),
    maturity_years: int = typer.Option(...),
    strike: Optional[float] = typer.Option(None),
    settlement_years: Optional[int] = typer.Option(None),
    notional: float = typer.Option(1.0),
    short_rate: Optional[float] = typer.Option(None),
    output: Optional[Path] = typer.Option(None),
) -> None:
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    spec = QForwardSpec(
        age=age,
        maturity_years=maturity_years,
        strike=strike,
        settlement_years=settlement_years,
        notional=notional,
    )
    prices = pricing_pipeline(
        scen_Q=scen_set,
        specs={"q_forward": spec},
        short_rate=float(short_rate) if short_rate is not None else 0.0,
    )
    out = output or c.outdir / "price_q_forward.json"
    out.write_text(json.dumps(prices, indent=2))
    typer.echo(f"Price={prices['q_forward']:.6f} saved to {out}")


@price_app.command("s-forward")
def price_s_forward_cmd(
    ctx: typer.Context,
    scen_path: Path = typer.Option(...),
    age: float = typer.Option(...),
    maturity_years: int = typer.Option(...),
    strike: Optional[float] = typer.Option(None),
    settlement_years: Optional[int] = typer.Option(None),
    notional: float = typer.Option(1.0),
    short_rate: Optional[float] = typer.Option(None),
    output: Optional[Path] = typer.Option(None),
) -> None:
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    spec = SForwardSpec(
        age=age,
        maturity_years=maturity_years,
        strike=strike,
        settlement_years=settlement_years,
        notional=notional,
    )
    prices = pricing_pipeline(
        scen_Q=scen_set,
        specs={"s_forward": spec},
        short_rate=float(short_rate) if short_rate is not None else 0.0,
    )
    out = output or c.outdir / "price_s_forward.json"
    out.write_text(json.dumps(prices, indent=2))
    typer.echo(f"Price={prices['s_forward']:.6f} saved to {out}")


@price_app.command("life-annuity")
def price_life_annuity_cmd(
    ctx: typer.Context,
    scen_path: Path = typer.Option(...),
    issue_age: float = typer.Option(...),
    maturity_years: Optional[int] = typer.Option(None),
    payment_per_survivor: float = typer.Option(1.0),
    defer_years: int = typer.Option(0),
    exposure_at_issue: float = typer.Option(1.0),
    include_terminal: bool = typer.Option(False),
    terminal_notional: float = typer.Option(0.0),
    short_rate: Optional[float] = typer.Option(None),
    output: Optional[Path] = typer.Option(None),
) -> None:
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    spec = CohortLifeAnnuitySpec(
        issue_age=issue_age,
        maturity_years=maturity_years,
        payment_per_survivor=payment_per_survivor,
        defer_years=defer_years,
        exposure_at_issue=exposure_at_issue,
        include_terminal=include_terminal,
        terminal_notional=terminal_notional,
    )
    prices = pricing_pipeline(
        scen_Q=scen_set,
        specs={"life_annuity": spec},
        short_rate=float(short_rate) if short_rate is not None else 0.0,
    )
    out = output or c.outdir / "price_life_annuity.json"
    out.write_text(json.dumps(prices, indent=2))
    typer.echo(f"Price={prices['life_annuity']:.6f} saved to {out}")


app.add_typer(price_app, name="price")


# ---------------------------------------------------------------------------
# RISK-NEUTRAL calibration subcommands
# ---------------------------------------------------------------------------

rn_app = typer.Typer(help="Risk-neutral calibration and pricing under lambda.")


@rn_app.command("calibrate-lambda")
def rn_calibrate_lambda_cmd(
    ctx: typer.Context,
    quotes_path: Path = typer.Option(..., help="JSON/YAML with quotes list."),
    m_path: Path = typer.Option(..., help="Mortality surface."),
    model_name: str = typer.Option("CBDM7"),
    lambda0: float = typer.Option(0.0),
    bounds: str = typer.Option("-5,5"),
    B_bootstrap: int = typer.Option(50),
    n_process: int = typer.Option(200),
    short_rate: float = typer.Option(0.02),
    horizon: Optional[int] = typer.Option(None),
    seed: Optional[int] = typer.Option(None),
    include_last: bool = typer.Option(False),
    output: Optional[Path] = typer.Option(None, help="Calibration result pickle."),
    ages: Optional[str] = typer.Option(None),
    years: Optional[str] = typer.Option(None),
    ages_path: Optional[Path] = typer.Option(None),
    years_path: Optional[Path] = typer.Option(None),
) -> None:
    c = _ctx(ctx)
    quotes_cfg = _load_config(quotes_path)
    if not isinstance(quotes_cfg, list):
        raise typer.BadParameter("quotes file must be a list of quote dicts.")

    def _mk_quote(d: Dict[str, Any]) -> MultiInstrumentQuote:
        if "kind" not in d or "spec" not in d:
            raise typer.BadParameter("Each quote must have 'kind' and 'spec' fields.")
        kind_norm = str(d["kind"]).replace("-", "_")
        spec = _to_spec(kind_norm, d["spec"])
        return MultiInstrumentQuote(
            kind=kind_norm,
            spec=spec,
            market_price=float(d["market_price"]),
            weight=float(d.get("weight", 1.0)),
        )

    ages_arr, years_arr, m = _load_m_surface(m_path, ages, years, ages_path, years_path)
    instruments: Dict[str, Any] = {}
    market_prices: Dict[str, float] = {}
    for i, d in enumerate(quotes_cfg):
        name = str(d.get("name", f"inst_{i}"))
        q = _mk_quote(d)
        instruments[name] = {
            "kind": q.kind,
            "spec": q.spec,
            "weight": float(d.get("weight", 1.0)),
        }
        market_prices[name] = float(d["market_price"])

    lo, hi = (float(x) for x in bounds.split(","))
    scen_Q, calib_summary, cache = build_risk_neutral_pipeline(
        scen_P=None,
        instruments=instruments,
        market_prices=market_prices,
        short_rate=short_rate,
        calibration_kwargs={
            "ages": ages_arr,
            "years": years_arr,
            "m": m,
            "model_name": model_name,
            "B_bootstrap": B_bootstrap,
            "n_process": n_process,
            "horizon": horizon if horizon is not None else years_arr.shape[0],
            "seed": seed if seed is not None else c.seed,
            "include_last": include_last,
            "lambda0": lambda0,
            "bounds": (lo, hi),
        },
    )

    out_json = output or c.outdir / "lambda_calibration.json"
    out_json.write_text(json.dumps(calib_summary, indent=2))
    cache_path = c.outdir / "calibration_cache.pkl"
    _maybe_pickle(cache, cache_path)
    scen_path = c.outdir / "scenarios_Q_calibrated.npz"
    _save_scenarios(scen_Q, scen_path)
    rmse = calib_summary.get("rmse_pricing_error")
    if rmse is not None:
        typer.echo(f"Calibration RMSE: {rmse}")
    typer.echo(
        f"Lambda*={calib_summary['lambda_star']} | summary -> {out_json} | cache -> {cache_path} | scen -> {scen_path}"
    )


@rn_app.command("price-under-lambda")
def rn_price_under_lambda_cmd(
    ctx: typer.Context,
    lambda_val: float = typer.Option(..., help="Lambda Esscher tilt."),
    m_path: Path = typer.Option(...),
    model_name: str = typer.Option("CBDM7"),
    B_bootstrap: int = typer.Option(50),
    n_process: int = typer.Option(200),
    horizon: int = typer.Option(50),
    short_rate: float = typer.Option(0.02),
    specs: Path = typer.Option(..., help="Specs file for instruments."),
    seed: Optional[int] = typer.Option(None),
    include_last: bool = typer.Option(False),
    output: Optional[Path] = typer.Option(None, help="Prices table path."),
    ages: Optional[str] = typer.Option(None),
    years: Optional[str] = typer.Option(None),
    ages_path: Optional[Path] = typer.Option(None),
    years_path: Optional[Path] = typer.Option(None),
) -> None:
    c = _ctx(ctx)
    spec_cfg = _load_config(specs)
    ages_arr, years_arr, m = _load_m_surface(m_path, ages, years, ages_path, years_path)
    cache = build_calibration_cache(
        ages=ages_arr,
        years=years_arr,
        m=m,
        model_name=model_name,
        B_bootstrap=B_bootstrap,
        n_process=n_process,
        horizon=horizon,
        seed=seed if seed is not None else c.seed,
        include_last=include_last,
    )
    scen_set_q = build_scenarios_under_lambda_fast(
        cache=cache, lambda_esscher=lambda_val
    )

    specs_norm: Dict[str, Any] = {}
    for name, item in spec_cfg.items():
        if not isinstance(item, dict) or "kind" not in item or "spec" not in item:
            raise typer.BadParameter("Specs file must map name -> {kind, spec}.")
        specs_norm[name] = _to_spec(str(item["kind"]).replace("-", "_"), item["spec"])

    prices = pricing_pipeline(
        scen_Q=scen_set_q, specs=specs_norm, short_rate=short_rate
    )
    out = output or c.outdir / "prices_under_lambda.json"
    out.write_text(json.dumps(prices, indent=2))
    typer.echo(f"Prices saved to {out}")


app.add_typer(rn_app, name="rn")


# ---------------------------------------------------------------------------
# SENSITIVITIES subcommands
# ---------------------------------------------------------------------------

sens_app = typer.Typer(help="Sensitivity analysis (rate, convexity, mortality).")


@sens_app.command("rate")
def sens_rate_cmd(
    ctx: typer.Context,
    scen_path: Path = typer.Option(...),
    kind: str = typer.Option(..., help="Instrument kind."),
    spec_path: Path = typer.Option(..., help="JSON/YAML spec for instrument."),
    base_short_rate: float = typer.Option(...),
    bump: float = typer.Option(1e-4),
    output: Optional[Path] = typer.Option(None),
) -> None:
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    spec_cfg = _load_config(spec_path)
    kind_norm = kind.replace("-", "_")
    spec = _to_spec(kind_norm, spec_cfg)

    def price_func(*, scen_set: MortalityScenarioSet, short_rate: float) -> float:
        return float(
            pricing_pipeline(
                scen_Q=scen_set, specs={"inst": spec}, short_rate=short_rate
            )["inst"]
        )

    res = rate_sensitivity(
        price_func, scen_set, base_short_rate=base_short_rate, bump=bump
    )
    out = output or c.outdir / "rate_sensitivity.json"
    out.write_text(json.dumps(asdict(res), indent=2))
    typer.echo(f"Rate sensitivity saved to {out}")


@sens_app.command("convexity")
def sens_convexity_cmd(
    ctx: typer.Context,
    scen_path: Path = typer.Option(...),
    kind: str = typer.Option(...),
    spec_path: Path = typer.Option(...),
    base_short_rate: float = typer.Option(...),
    bump: float = typer.Option(1e-4),
    output: Optional[Path] = typer.Option(None),
) -> None:
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    spec_cfg = _load_config(spec_path)
    kind_norm = kind.replace("-", "_")
    spec = _to_spec(kind_norm, spec_cfg)

    def price_func(*, scen_set: MortalityScenarioSet, short_rate: float) -> float:
        return float(
            pricing_pipeline(
                scen_Q=scen_set, specs={"inst": spec}, short_rate=short_rate
            )["inst"]
        )

    res = rate_convexity(
        price_func, scen_set, base_short_rate=base_short_rate, bump=bump
    )
    out = output or c.outdir / "rate_convexity.json"
    out.write_text(json.dumps(asdict(res), indent=2))
    typer.echo(f"Rate convexity saved to {out}")


@sens_app.command("delta-by-age")
def sens_delta_by_age_cmd(
    ctx: typer.Context,
    scen_path: Path = typer.Option(...),
    kind: str = typer.Option(...),
    spec_path: Path = typer.Option(...),
    rel_bump: float = typer.Option(0.01),
    ages: Optional[str] = typer.Option(None, help="Subset ages, comma-separated."),
    short_rate: float = typer.Option(0.0, help="Short rate for pricing."),
    output: Optional[Path] = typer.Option(None),
) -> None:
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    spec_cfg = _load_config(spec_path)
    kind_norm = kind.replace("-", "_")
    spec = _to_spec(kind_norm, spec_cfg)

    def price_func(scen: MortalityScenarioSet) -> float:
        return float(
            pricing_pipeline(scen_Q=scen, specs={"inst": spec}, short_rate=short_rate)[
                "inst"
            ]
        )

    ages_sel = _parse_number_list(ages) if ages else None
    res = mortality_delta_by_age(price_func, scen_set, ages=ages_sel, rel_bump=rel_bump)
    out = output or c.outdir / "delta_by_age.json"
    out.write_text(
        json.dumps(
            {
                "ages": res.ages.tolist(),
                "deltas": res.deltas.tolist(),
                "base_price": res.base_price,
                "rel_bump": res.rel_bump,
            },
            indent=2,
        )
    )
    typer.echo(f"Delta-by-age saved to {out}")


@sens_app.command("all")
def sens_all_cmd(
    ctx: typer.Context,
    scen_path: Path = typer.Option(..., help="Scenario set npz (Q)."),
    specs_path: Path = typer.Option(..., help="JSON/YAML specs mapping."),
    short_rate: float = typer.Option(0.02, help="Short rate for pricing."),
    sigma_rel_bump: float = typer.Option(0.05),
    q_rel_bump: float = typer.Option(0.01),
    rate_bump: float = typer.Option(1e-4),
    output: Optional[Path] = typer.Option(None),
) -> None:
    """
    Compute all sensitivities via pipeline.risk_analysis_pipeline.
    """
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    specs_cfg = _load_config(specs_path)
    bumps = {
        "build_scenarios_func": lambda scale_sigma: scen_set,
        "sigma_rel_bump": sigma_rel_bump,
        "q_rel_bump": q_rel_bump,
        "rate_bump": rate_bump,
    }
    res = risk_analysis_pipeline(
        scen_Q=scen_set,
        specs=specs_cfg,
        short_rate=short_rate,
        bumps=bumps,
    )
    out = output or c.outdir / "sensitivities.json"
    payload = {
        "prices_base": res.prices_base,
        "vega_sigma_scale": res.vega_sigma_scale,
        "delta_by_age": {
            k: {"ages": v.ages.tolist(), "deltas": v.deltas.tolist()}
            for k, v in res.delta_by_age.items()
        },
        "rate_sensitivity": {k: asdict(v) for k, v in res.rate_sensitivity.items()},
        "rate_convexity": {k: asdict(v) for k, v in res.rate_convexity.items()},
        "meta": res.meta,
    }
    out.write_text(json.dumps(payload, indent=2))
    typer.echo(f"Sensitivities saved to {out}")


app.add_typer(sens_app, name="sens")


# ---------------------------------------------------------------------------
# HEDGE subcommands
# ---------------------------------------------------------------------------

hedge_app = typer.Typer(help="Hedging utilities.")


@hedge_app.command("min-variance")
def hedge_min_variance_cmd(
    ctx: typer.Context,
    liab_pv_path: Path = typer.Option(..., help="Liability PV paths npy/csv/parquet."),
    instr_pv_path: Path = typer.Option(..., help="Instrument PV paths (N,M)."),
    names: Optional[str] = typer.Option(None, help="Comma-separated instrument names."),
    output: Optional[Path] = typer.Option(None),
) -> None:
    c = _ctx(ctx)
    liab = _read_numeric_series(liab_pv_path).reshape(-1)
    H = _read_numeric_matrix(instr_pv_path)
    res = hedging_pipeline(
        liability_pv_paths=liab,
        hedge_pv_paths=H,
        method="min_variance",
    )
    out = output or c.outdir / "hedge_min_variance.json"
    # HedgeResult
    out.write_text(
        json.dumps(
            {
                "weights": res.weights.tolist(),
                "summary": getattr(res, "summary", {}),
            },
            indent=2,
        )
    )
    typer.echo(f"Hedge weights saved to {out}")


@hedge_app.command("multihorizon")
def hedge_multihorizon_cmd(
    ctx: typer.Context,
    liab_cf_path: Path = typer.Option(..., help="Liability CF paths (N,T)."),
    instr_cf_path: Path = typer.Option(..., help="Instrument CF paths (N,M,T)."),
    discount_factors_path: Optional[Path] = typer.Option(None),
    time_weights_path: Optional[Path] = typer.Option(None),
    output: Optional[Path] = typer.Option(None),
) -> None:
    c = _ctx(ctx)
    L = _read_numeric_matrix(liab_cf_path)
    H_cf = _read_numeric_cube(instr_cf_path)
    df = _read_numeric_series(discount_factors_path) if discount_factors_path else None
    tw = _read_numeric_series(time_weights_path) if time_weights_path else None
    res = hedging_pipeline(
        liability_pv_paths=L,
        hedge_pv_paths=H_cf,
        method="multihorizon",
        constraints={"discount_factors": df, "time_weights": tw},
    )
    out = output or c.outdir / "hedge_multihorizon.json"
    out.write_text(
        json.dumps(
            {
                "weights": res.weights.tolist(),
                "summary": getattr(res, "summary", {}),
            },
            indent=2,
        )
    )
    typer.echo(f"Hedge weights saved to {out}")


app.add_typer(hedge_app, name="hedge")


# ---------------------------------------------------------------------------
# REPORT subcommands
# ---------------------------------------------------------------------------

report_app = typer.Typer(help="Risk reporting utilities.")


@report_app.command("risk")
def report_risk_cmd(
    ctx: typer.Context,
    pv_path: Path = typer.Option(..., help="PV paths csv/parquet/npy."),
    name: Optional[str] = typer.Option(None, help="Name for the report."),
    var_level: float = typer.Option(0.95, help="VaR level."),
    ref_pv_path: Optional[Path] = typer.Option(
        None, help="Optional reference PV paths."
    ),
    output: Optional[Path] = typer.Option(None),
) -> None:
    c = _ctx(ctx)
    pv = _read_numeric_series(pv_path)
    ref = _read_numeric_series(ref_pv_path) if ref_pv_path else None
    report = reporting_pipeline(
        pv_paths=pv,
        ref_pv_paths=ref,
        name=name or pv_path.stem,
        var_level=var_level,
    )
    out = output or c.outdir / f"risk_{report.name}.json"
    out.write_text(json.dumps(report.to_dict(), indent=2))
    typer.echo(f"Risk report saved to {out}")


app.add_typer(report_app, name="report")


# ---------------------------------------------------------------------------
# PLOT subcommands (lightweight; require matplotlib)
# ---------------------------------------------------------------------------

plot_app = typer.Typer(help="Plotting helpers.")


@plot_app.command("survival-fan")
def plot_survival_fan_cmd(
    ctx: typer.Context,
    scen_path: Path = typer.Option(...),
    age: float = typer.Option(...),
    quantiles: str = typer.Option("5,50,95"),
    output: Optional[Path] = typer.Option(None, help="PNG path."),
) -> None:
    import matplotlib.pyplot as plt

    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    qs = [int(x) for x in quantiles.split(",") if x]
    plot_survival_fan(scen_set, age=age, quantiles=qs)
    out = output or c.outdir / "survival_fan.png"
    plt.tight_layout()
    plt.savefig(out)
    typer.echo(f"Plot saved to {out}")


@plot_app.command("price-dist")
def plot_price_dist_cmd(
    ctx: typer.Context,
    pv_path: Path = typer.Option(...),
    bins: int = typer.Option(30),
    output: Optional[Path] = typer.Option(None),
) -> None:
    import matplotlib.pyplot as plt

    c = _ctx(ctx)
    pv = _read_numeric_series(pv_path)
    plt.figure(figsize=(6, 4))
    plt.hist(pv, bins=bins, alpha=0.7)
    plt.xlabel("PV")
    plt.ylabel("Frequency")
    plt.title("Price distribution")
    out = output or c.outdir / "price_dist.png"
    plt.tight_layout()
    plt.savefig(out)
    typer.echo(f"Plot saved to {out}")


@plot_app.command("lexis")
def plot_lexis_cmd(
    ctx: typer.Context,
    scen_path: Path = typer.Option(..., help="Scenario set npz."),
    value: str = typer.Option("q", help="m, q, or S"),
    statistic: str = typer.Option("median", help="mean or median"),
    cohorts: Optional[str] = typer.Option(
        None, help="Comma-separated cohort birth years."
    ),
    output: Optional[Path] = typer.Option(None, help="PNG path."),
) -> None:
    import matplotlib.pyplot as plt

    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    coh_list = None
    if cohorts:
        coh_list = [int(x) for x in cohorts.split(",") if x]
    plot_lexis(
        scen_set,
        value=value,
        statistic=statistic,
        cohorts=coh_list,
    )
    out = output or c.outdir / "lexis.png"
    plt.tight_layout()
    plt.savefig(out)
    typer.echo(f"Lexis plot saved to {out}")


@plot_app.command("fan")
def plot_fan_cmd(
    ctx: typer.Context,
    scen_path: Path = typer.Option(...),
    age: float = typer.Option(...),
    value: str = typer.Option("S", help="S or q"),
    quantiles: str = typer.Option("5,25,50,75,95"),
    output: Optional[Path] = typer.Option(None),
) -> None:
    import matplotlib.pyplot as plt

    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    qs = [int(x) for x in quantiles.split(",") if x]
    if value.lower() == "s":
        plot_survival_fan(scen_set, age=age, quantiles=qs)
    else:
        plot_mortality_fan(scen_set, age=age, quantiles=qs)
    out = output or c.outdir / f"{value.lower()}_fan.png"
    plt.tight_layout()
    plt.savefig(out)
    typer.echo(f"Fan plot saved to {out}")


@plot_app.command("animate")
def plot_animate_cmd(
    ctx: typer.Context,
    scen_path: Path = typer.Option(...),
    type: str = typer.Option("surface", help="surface or survival"),
    value: str = typer.Option("q", help="surface value q or S"),
    output: Path = typer.Option(..., help="Output file (.mp4 or .gif)"),
    statistic: str = typer.Option("median", help="mean or median"),
    interval: int = typer.Option(200, help="Frame interval ms"),
) -> None:
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    if type == "surface":
        animate_mortality_surface(
            scen_set,
            value=value,
            statistic=statistic,
            interval=interval,
            save_path=str(output),
        )
    else:
        animate_survival_curves(
            scen_set,
            statistic=statistic,
            interval=interval,
            save_path=str(output),
        )
    typer.echo(f"Animation saved to {output}")


app.add_typer(plot_app, name="plot")


# ---------------------------------------------------------------------------
# RUN pipelines (lightweight wrappers)
# ---------------------------------------------------------------------------

run_app = typer.Typer(help="One-click pipelines.")


@run_app.command("pricing-pipeline")
def run_pricing_pipeline_cmd(
    ctx: typer.Context,
    config: Path = typer.Option(..., help="YAML/JSON config file."),
) -> None:
    cfg = _load_config(config)
    typer.echo(
        "Pricing pipeline not fully automated; please chain commands manually using config."
    )
    typer.echo(json.dumps(cfg, indent=2))


@run_app.command("hedge-pipeline")
def run_hedge_pipeline_cmd(
    ctx: typer.Context,
    config: Path = typer.Option(..., help="YAML/JSON config file."),
) -> None:
    cfg = _load_config(config)
    typer.echo(
        "Hedge pipeline not fully automated; please chain commands manually using config."
    )
    typer.echo(json.dumps(cfg, indent=2))


app.add_typer(run_app, name="run")


# ---------------------------------------------------------------------------
# Version and echo
# ---------------------------------------------------------------------------


@app.command("version")
def version_cmd() -> None:
    try:
        from importlib.metadata import version as _pkg_version
    except ImportError:
        typer.echo("0.0.dev")
        return
    try:
        typer.echo(_pkg_version("pymort"))
    except Exception:
        typer.echo("0.0.dev")


@app.command("echo")
def echo_cmd(msg: str) -> None:
    """Echo a message."""
    typer.echo(msg)


if __name__ == "__main__":
    app()
