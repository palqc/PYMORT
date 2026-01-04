"""Command-line interface for PYMORT.

Note:
    Docstrings follow Google style to align with project standards.
"""

from __future__ import annotations

import json
import logging
import pickle
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import typer

from pymort._types import AnyArray, FloatArray, IntArray
from pymort.analysis import MortalityScenarioSet, smooth_mortality_with_cpsplines
from pymort.analysis.fitting import (
    FittedModel,
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
from pymort.lifetables import Sex, load_m_from_excel_any, m_to_q
from pymort.pipeline import (
    BumpsConfig,
    HedgeConstraints,
    _derive_bootstrap_params,
    _infer_kind,
    _normalize_spec,
    build_projection_pipeline,
    build_risk_neutral_pipeline,
    hedging_pipeline,
    pricing_pipeline,
    project_from_fitted_model,
    reporting_pipeline,
    risk_analysis_pipeline,
    stress_testing_pipeline,
)
from pymort.pricing.liabilities import CohortLifeAnnuitySpec, price_cohort_life_annuity
from pymort.pricing.longevity_bonds import (
    LongevityBondSpec,
    price_simple_longevity_bond,
)
from pymort.pricing.mortality_derivatives import (
    QForwardSpec,
    SForwardSpec,
    price_q_forward,
    price_s_forward,
)
from pymort.pricing.risk_neutral import (
    MultiInstrumentQuote,
    build_calibration_cache,
    build_scenarios_under_lambda_fast,
)
from pymort.pricing.survivor_swaps import SurvivorSwapSpec, price_survivor_swap
from pymort.visualization import plot_lexis, plot_mortality_fan, plot_survival_fan

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


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise typer.BadParameter(f"Config file {path} does not exist.")
    text = path.read_text()
    try:
        return cast(dict[str, Any], json.loads(text))
    except json.JSONDecodeError:
        try:
            import yaml
        except Exception as exc:
            raise typer.BadParameter(
                f"Could not parse config {path} as JSON; install pyyaml for YAML support."
            ) from exc
        return cast(dict[str, Any], yaml.safe_load(text))


def _ensure_outdir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(path.iterdir()):
        return


def _parse_number_list(spec: str | None) -> FloatArray | None:
    if spec is None:
        return None
    if spec.strip() == "":
        return None
    items = [float(x) for x in spec.replace(" ", "").split(",") if x]
    return np.asarray(items, dtype=float)


def _parse_range_spec(spec: str | None) -> tuple[float, float] | None:
    if spec is None:
        return None
    txt = spec.replace(" ", "")
    if txt == "":
        return None
    if "-" in txt:
        parts = txt.split("-")
    elif ":" in txt:
        parts = txt.split(":")
    else:
        parts = txt.split(",")
    if len(parts) != 2:
        raise typer.BadParameter(f"Range '{spec}' must be like 60-100.")
    lo, hi = float(parts[0]), float(parts[1])
    if hi < lo:
        raise typer.BadParameter(f"Range '{spec}' must have min <= max.")
    return lo, hi


def _slice_surface(
    ages: AnyArray,
    years: AnyArray,
    m: AnyArray,
    *,
    age_range: tuple[float, float] | None = None,
    year_range: tuple[float, float] | None = None,
) -> tuple[FloatArray, IntArray, FloatArray]:
    ages_arr = np.asarray(ages, dtype=float)
    years_arr = np.asarray(years, dtype=int)
    m_arr = np.asarray(m, dtype=float)

    age_mask = np.ones_like(ages_arr, dtype=bool)
    if age_range is not None:
        age_mask = (ages_arr >= age_range[0]) & (ages_arr <= age_range[1])
    year_mask = np.ones_like(years_arr, dtype=bool)
    if year_range is not None:
        year_mask = (years_arr >= year_range[0]) & (years_arr <= year_range[1])

    return ages_arr[age_mask], years_arr[year_mask], m_arr[np.ix_(age_mask, year_mask)]


def _read_numeric_series(path: Path) -> FloatArray:
    ext = path.suffix.lower()
    if ext == ".npy":
        return cast(FloatArray, np.load(path))
    if ext == ".npz":
        data = np.load(path)
        first_key = next(iter(data.keys()))
        return np.asarray(data[first_key]).reshape(-1)
    df = pd.read_parquet(path) if ext in {".parquet", ".pq"} else pd.read_csv(path, header=None)
    arr = df.to_numpy().reshape(-1)
    return np.asarray(arr, dtype=float)


def _read_numeric_matrix(path: Path) -> FloatArray:
    ext = path.suffix.lower()
    if ext in {".npy", ".npz"}:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[next(iter(arr.keys()))]
        return np.asarray(arr, dtype=float)
    df = pd.read_parquet(path) if ext in {".parquet", ".pq"} else pd.read_csv(path, header=None)
    return np.asarray(df.to_numpy(), dtype=float)


def _read_numeric_cube(path: Path) -> FloatArray:
    arr = _read_numeric_matrix(path)
    if arr.ndim == 3:
        return arr
    raise typer.BadParameter(f"Expected 3D array at {path}; use .npy/.npz with shape (N,M,T).")


def _load_m_surface(
    m_path: Path,
    ages_inline: str | None,
    years_inline: str | None,
    ages_path: Path | None,
    years_path: Path | None,
    preferred_rate_col: str | None = None,
) -> tuple[AnyArray, AnyArray, FloatArray]:
    def _normalize_excel_sex(value: str | None) -> Sex:
        if value is None:
            return "Total"
        key = value.strip().lower()
        if key in {"total", "female", "male"}:
            return cast(Sex, key.capitalize())
        raise typer.BadParameter(
            "Excel mortality tables accept rate columns: Total, Female, or Male."
        )

    ext = m_path.suffix.lower()
    ages = _parse_number_list(ages_inline) if ages_inline else None
    years = _parse_number_list(years_inline) if years_inline else None
    if ages_path is not None:
        ages = _read_numeric_series(ages_path)
    if years_path is not None:
        years = _read_numeric_series(years_path)

    if ext in {".xlsx", ".xls"}:
        sex = _normalize_excel_sex(preferred_rate_col)
        out = load_m_from_excel_any(
            str(m_path),
            sex=sex,
            age_min=60,
            age_max=110,
            year_min=None,
            year_max=None,
        )
        ages_xlsx, years_xlsx, m_xlsx = out["m"]
        return (
            ages_xlsx.astype(float),
            years_xlsx.astype(int),
            np.asarray(m_xlsx, dtype=float),
        )
    if ext in {".npy", ".npz"}:
        data = np.load(m_path)
        if isinstance(data, np.lib.npyio.NpzFile):
            m = np.asarray(data[next(iter(data.keys()))], dtype=float)
        else:
            m = np.asarray(data, dtype=float)
        if m.ndim != 2:
            raise typer.BadParameter("m-path .npy must contain a 2D array (A,T).")
        if ages is None or years is None:
            raise typer.BadParameter("Provide --ages/--years when loading .npy wide arrays.")
        if m.shape != (ages.shape[0], years.shape[0]):
            raise typer.BadParameter(
                f"m shape {m.shape} incompatible with ages {ages.shape} and years {years.shape}."
            )
        return ages, years, m

    df = pd.read_parquet(m_path) if ext in {".parquet", ".pq"} else pd.read_csv(m_path)
    cols_lower = {c.lower() for c in df.columns}
    if {"age", "year"} <= cols_lower:
        # long format
        age_col = next(c for c in df.columns if c.lower() == "age")
        year_col = next(c for c in df.columns if c.lower() == "year")
        rate_candidates = [
            c for c in df.columns if c.lower() in {"m", "mx", "rate", "total", "male", "female"}
        ]
        rate_col = None
        if preferred_rate_col is not None:
            rate_col = next(
                (c for c in df.columns if c.lower() == preferred_rate_col.lower()),
                None,
            )
        if rate_col is None and rate_candidates:
            # prefer Total if available, else first candidate
            rate_col = next(
                (c for c in rate_candidates if c.lower() == "total"),
                rate_candidates[0],
            )
        if rate_col is None:
            raise typer.BadParameter(
                "Long format requires a mortality column (m, rate, Total, Male, Female)."
            )
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


def _save_npz(data: dict[str, Any], path: Path) -> None:
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
        metadata: dict[str, Any] = {}
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
            discount_factors=(data.get("discount_factors") if "discount_factors" in data else None),
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


def _maybe_pickle(obj: Any, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)  # nosec: B301


def _normalize_model_flag(model: str) -> ModelName:
    alias = model.strip().lower().replace("_", "-")
    if alias in {"lee-carter", "lc", "lcm1", "lc-m1", "lee-carter-m1"}:
        return "LCM1"
    if alias in {"lee-carter-m2", "lc-m2", "lcm2"}:
        return "LCM2"
    if alias in {"apc-m3", "apcm3"}:
        return "APCM3"
    if alias in {"cbd-m5", "cbdm5"}:
        return "CBDM5"
    if alias in {"cbd-m6", "cbdm6"}:
        return "CBDM6"
    if alias in {"cbd-m7", "cbdm7"}:
        return "CBDM7"
    return cast(ModelName, model.upper())  # fallback to raw ModelName


def _price_paths_for_spec(
    scen_set: MortalityScenarioSet,
    spec_obj: object,
    *,
    short_rate: float | None,
) -> tuple[float, FloatArray]:
    spec = _normalize_spec(spec_obj)
    kind = _infer_kind(spec)
    if kind == "longevity_bond":
        res = price_simple_longevity_bond(
            scen_set=scen_set,
            spec=cast(LongevityBondSpec, spec),
            short_rate=short_rate,
        )
    elif kind == "s_forward":
        res = price_s_forward(
            scen_set=scen_set,
            spec=cast(SForwardSpec, spec),
            short_rate=short_rate,
        )
    elif kind == "q_forward":
        res = price_q_forward(
            scen_set=scen_set,
            spec=cast(QForwardSpec, spec),
            short_rate=short_rate,
        )
    elif kind == "survivor_swap":
        res = price_survivor_swap(
            scen_set=scen_set,
            spec=cast(SurvivorSwapSpec, spec),
            short_rate=short_rate,
        )
    elif kind == "life_annuity":
        res = price_cohort_life_annuity(
            scen_set=scen_set,
            spec=cast(CohortLifeAnnuitySpec, spec),
            short_rate=short_rate,
            discount_factors=None,
        )
    else:
        raise typer.BadParameter(f"Unsupported instrument kind '{kind}'.")

    return float(res["price"]), np.asarray(res["pv_paths"], dtype=float).reshape(-1)


def _to_spec(
    kind: str, cfg: dict[str, Any]
) -> LongevityBondSpec | SurvivorSwapSpec | SForwardSpec | QForwardSpec | CohortLifeAnnuitySpec:
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


def _is_range_flag(s: str | None) -> bool:
    if s is None:
        return False
    txt = s.strip()
    return (("-" in txt) or (":" in txt)) and ("," not in txt)


def _is_list_flag(s: str | None) -> bool:
    if s is None:
        return False
    return "," in s


@dataclass
class CLIContext:
    outdir: Path
    output_format: str
    seed: int | None
    verbose: bool
    quiet: bool
    log_level: str
    overwrite: bool
    config: dict[str, Any]
    save_path: Path | None
    load_path: Path | None


def _ctx(ctx: typer.Context) -> CLIContext:
    return cast(CLIContext, ctx.obj)


# ---------------------------------------------------------------------------
# Global options
# ---------------------------------------------------------------------------


@app.callback()
def main(
    ctx: typer.Context,
    config: Path | None = typer.Option(
        None, "--config", "-c", help="YAML/JSON config file with defaults."
    ),
    seed: int | None = typer.Option(None, help="RNG seed."),
    outdir: Path = typer.Option(Path("outputs"), help="Output directory."),
    format: str = typer.Option("csv", "--format", help="Tabular output format."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging."),
    quiet: bool = typer.Option(False, "--quiet", help="Quiet logging."),
    log_level: str = typer.Option("INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)."),
    overwrite: bool = typer.Option(False, help="Allow overwriting output files."),
    save: Path | None = typer.Option(None, help="Optional pickle/npz path to save result."),
    load: Path | None = typer.Option(None, help="Load a previously saved object."),
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
    m_path: Path = typer.Option(..., help="Path to mortality surface (csv/parquet/npy)."),
    ages: str | None = typer.Option(None, help="Inline ages list, e.g. '60,61,62'."),
    years: str | None = typer.Option(None, help="Inline years list, e.g. '2000,2001'."),
    ages_path: Path | None = typer.Option(None, help="Path to ages (csv/parquet/npy)."),
    years_path: Path | None = typer.Option(None, help="Path to years (csv/parquet/npy)."),
) -> None:
    c = _ctx(ctx)
    ages_inline = ages if _is_list_flag(ages) else None
    years_inline = years if _is_list_flag(years) else None
    ages_arr, years_arr, m = _load_m_surface(
        m_path, ages_inline, years_inline, ages_path, years_path
    )
    ages_arr, years_arr, m = _slice_surface(
        ages_arr,
        years_arr,
        m,
        age_range=_parse_range_spec(ages) if _is_range_flag(ages) else None,
        year_range=_parse_range_spec(years) if _is_range_flag(years) else None,
    )
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
    m_path: Path = typer.Option(..., help="Path to mortality surface (csv/parquet/npy)."),
    eps: float = typer.Option(1e-12, help="Minimum value for clipping."),
    ages: str | None = typer.Option(None, help="Inline ages."),
    years: str | None = typer.Option(None, help="Inline years."),
    ages_path: Path | None = typer.Option(None, help="Path to ages."),
    years_path: Path | None = typer.Option(None, help="Path to years."),
    output: Path | None = typer.Option(None, help="Output path (.npz or .npy)."),
) -> None:
    c = _ctx(ctx)
    ages_inline = ages if _is_list_flag(ages) else None
    years_inline = years if _is_list_flag(years) else None
    ages_arr, years_arr, m = _load_m_surface(
        m_path, ages_inline, years_inline, ages_path, years_path
    )
    ages_arr, years_arr, m = _slice_surface(
        ages_arr,
        years_arr,
        m,
        age_range=_parse_range_spec(ages) if _is_range_flag(ages) else None,
        year_range=_parse_range_spec(years) if _is_range_flag(years) else None,
    )
    m_clipped = np.clip(m, eps, None)
    out = output or c.outdir / "m_clipped.npz"
    _save_npz({"m": m_clipped, "ages": ages_arr, "years": years_arr}, out)
    typer.echo(f"Clipped m saved to {out}")


@data_app.command("to-q")
def data_to_q(
    ctx: typer.Context,
    m_path: Path = typer.Option(..., help="Path to mortality surface (csv/parquet/npy)."),
    ages: str | None = typer.Option(None),
    years: str | None = typer.Option(None),
    ages_path: Path | None = typer.Option(None),
    years_path: Path | None = typer.Option(None),
    output: Path | None = typer.Option(None, help="Output path (.npz)."),
) -> None:
    c = _ctx(ctx)
    ages_inline = ages if _is_list_flag(ages) else None
    years_inline = years if _is_list_flag(years) else None
    ages_arr, years_arr, m = _load_m_surface(
        m_path, ages_inline, years_inline, ages_path, years_path
    )
    ages_arr, years_arr, m = _slice_surface(
        ages_arr,
        years_arr,
        m,
        age_range=_parse_range_spec(ages) if _is_range_flag(ages) else None,
        year_range=_parse_range_spec(years) if _is_range_flag(years) else None,
    )
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
    ages: str | None = typer.Option(None),
    years: str | None = typer.Option(None),
    ages_path: Path | None = typer.Option(None),
    years_path: Path | None = typer.Option(None),
    deg: str = typer.Option("3,3", help="Degrees for (age,year) splines."),
    ord_d: str = typer.Option("2,2", help="Orders of derivative penalties."),
    k: str | None = typer.Option(None, help="Knots as 'ka,kt' or 'auto'."),
    sp_method: str = typer.Option("grid_search", help="Smoothing parameter method."),
    sp_args: str | None = typer.Option(None, help="JSON string for smoothing args."),
    horizon: int = typer.Option(0, help="Forecast horizon for CPsplines."),
    output: Path | None = typer.Option(None, help="Output npz path for fitted surface."),
) -> None:
    c = _ctx(ctx)
    ages_inline = ages if _is_list_flag(ages) else None
    years_inline = years if _is_list_flag(years) else None
    ages_arr, years_arr, m = _load_m_surface(
        m_path, ages_inline, years_inline, ages_path, years_path
    )
    ages_arr, years_arr, m = _slice_surface(
        ages_arr,
        years_arr,
        m,
        age_range=_parse_range_spec(ages) if _is_range_flag(ages) else None,
        year_range=_parse_range_spec(years) if _is_range_flag(years) else None,
    )
    deg_tuple = cast(tuple[int, int], tuple(int(x) for x in deg.split(",")))
    ord_tuple = cast(tuple[int, int], tuple(int(x) for x in ord_d.split(",")))
    k_tuple: tuple[int, int] | None = None
    if k not in (None, "auto"):
        k_tuple = cast(tuple[int, int], tuple(int(x) for x in str(k).split(",")))
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


@fit_app.callback(invoke_without_command=True)
def fit_default_cmd(
    ctx: typer.Context,
    data: Path | None = typer.Argument(None, help="Long CSV/parquet mortality data."),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model alias (lee-carter, cbd-m6, etc.).",
    ),
    ages: str | None = typer.Option(None, help="Age range e.g. '60-100'."),
    years: str | None = typer.Option(None, help="Year range e.g. '1970-2019'."),
    rate_column: str | None = typer.Option(
        None, help="Optional column to use (Total, Male, Female, m)."
    ),
    output: Path | None = typer.Option(None, help="Pickle path for fitted model."),
    summary: Path | None = typer.Option(None, help="JSON summary path."),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    if data is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()
    if model is None:
        raise typer.BadParameter("Provide --model for the fit command.")

    c = _ctx(ctx)
    ages_arr, years_arr, m = _load_m_surface(
        data,
        None,
        None,
        None,
        None,
        preferred_rate_col=rate_column,
    )
    ages_arr, years_arr, m = _slice_surface(
        ages_arr,
        years_arr,
        m,
        age_range=_parse_range_spec(ages),
        year_range=_parse_range_spec(years),
    )
    model_name = _normalize_model_flag(model)
    fitted = fit_mortality_model(
        model_name=model_name,
        ages=ages_arr,
        years=years_arr,
        m=m,
        smoothing="none",
        cpsplines_kwargs=None,
        eval_on_raw=True,
    )

    out = output or c.outdir / f"fitted_{model_name.lower()}.pkl"
    _maybe_pickle(fitted, out)

    summary_path = summary or c.outdir / "fit_summary.json"
    summary_payload = {
        "model": fitted.name,
        "ages": {"min": float(ages_arr.min()), "max": float(ages_arr.max())},
        "years": {"min": int(years_arr.min()), "max": int(years_arr.max())},
        "n_ages": int(ages_arr.shape[0]),
        "n_years": int(years_arr.shape[0]),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    typer.echo(f"Fitted {fitted.name} saved to {out}; summary to {summary_path}")


@fit_app.command("one")
def fit_one_cmd(
    ctx: typer.Context,
    model: ModelName = typer.Option(..., help="Model name."),
    m_path: Path = typer.Option(...),
    ages: str | None = typer.Option(None),
    years: str | None = typer.Option(None),
    ages_path: Path | None = typer.Option(None),
    years_path: Path | None = typer.Option(None),
    smoothing: str = typer.Option("none", help="none or cpsplines"),
    eval_on_raw: bool = typer.Option(True, help="Evaluate diagnostics on raw m."),
    cpsplines_k: int | None = typer.Option(None),
    cpsplines_horizon: int = typer.Option(0),
    output: Path | None = typer.Option(None, help="Pickle path for fitted model."),
) -> None:
    c = _ctx(ctx)
    ages_inline = ages if _is_list_flag(ages) else None
    years_inline = years if _is_list_flag(years) else None

    ages_arr, years_arr, m = _load_m_surface(
        m_path, ages_inline, years_inline, ages_path, years_path
    )

    ages_arr, years_arr, m = _slice_surface(
        ages_arr,
        years_arr,
        m,
        age_range=_parse_range_spec(ages) if _is_range_flag(ages) else None,
        year_range=_parse_range_spec(years) if _is_range_flag(years) else None,
    )
    cps_kwargs = (
        {"k": cpsplines_k, "horizon": cpsplines_horizon, "verbose": c.verbose}
        if smoothing == "cpsplines"
        else None
    )
    smoothing_mode = cast(Literal["none", "cpsplines"], smoothing)
    fitted = fit_mortality_model(
        model_name=model,
        ages=ages_arr,
        years=years_arr,
        m=m,
        smoothing=smoothing_mode,
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
    models: list[str] = typer.Option(
        [], "--models", "-m", help="Comma-separated model list (default all)."
    ),
    metric: str = typer.Option("logit_q", help="Selection metric (log_m or logit_q)."),
    ages: str | None = typer.Option(None),
    years: str | None = typer.Option(None),
    ages_path: Path | None = typer.Option(None),
    years_path: Path | None = typer.Option(None),
    output: Path | None = typer.Option(None, help="Selection table path."),
) -> None:
    c = _ctx(ctx)
    ages_inline = ages if _is_list_flag(ages) else None
    years_inline = years if _is_list_flag(years) else None

    ages_arr, years_arr, m = _load_m_surface(
        m_path, ages_inline, years_inline, ages_path, years_path
    )

    ages_arr, years_arr, m = _slice_surface(
        ages_arr,
        years_arr,
        m,
        age_range=_parse_range_spec(ages) if _is_range_flag(ages) else None,
        year_range=_parse_range_spec(years) if _is_range_flag(years) else None,
    )
    model_names: Sequence[ModelName] = (
        tuple(cast(ModelName, name.upper()) for name in models)
        if models
        else (
            "LCM1",
            "LCM2",
            "APCM3",
            "CBDM5",
            "CBDM6",
            "CBDM7",
        )
    )
    metric_mode = cast(Literal["log_m", "logit_q"], metric)
    df, best = model_selection_by_forecast_rmse(
        ages=ages_arr,
        years=years_arr,
        m=m,
        train_end=train_end,
        model_names=model_names,
        metric=metric_mode,
    )
    out = output or c.outdir / "model_selection.csv"
    _save_table(df, out, c.output_format)
    typer.echo(f"Selection table saved to {out}; best={best}")


@fit_app.command("select-and-fit")
def fit_select_and_fit_cmd(
    ctx: typer.Context,
    m_path: Path = typer.Option(...),
    train_end: int = typer.Option(...),
    models: list[str] = typer.Option([], "--models", "-m"),
    metric: str = typer.Option("logit_q"),
    cpsplines_k: int | None = typer.Option(None),
    cpsplines_horizon: int = typer.Option(0),
    ages: str | None = typer.Option(None),
    years: str | None = typer.Option(None),
    ages_path: Path | None = typer.Option(None),
    years_path: Path | None = typer.Option(None),
    output: Path | None = typer.Option(None, help="Pickle path for fitted model."),
    selection_output: Path | None = typer.Option(None, help="Selection table path."),
) -> None:
    c = _ctx(ctx)
    ages_inline = ages if _is_list_flag(ages) else None
    years_inline = years if _is_list_flag(years) else None

    ages_arr, years_arr, m = _load_m_surface(
        m_path, ages_inline, years_inline, ages_path, years_path
    )

    ages_arr, years_arr, m = _slice_surface(
        ages_arr,
        years_arr,
        m,
        age_range=_parse_range_spec(ages) if _is_range_flag(ages) else None,
        year_range=_parse_range_spec(years) if _is_range_flag(years) else None,
    )
    model_names: Sequence[ModelName] = (
        tuple(cast(ModelName, name.upper()) for name in models)
        if models
        else (
            "LCM1",
            "LCM2",
            "APCM3",
            "CBDM5",
            "CBDM6",
            "CBDM7",
        )
    )
    cp_kwargs = {"k": cpsplines_k, "horizon": cpsplines_horizon, "verbose": c.verbose}
    metric_mode = cast(Literal["log_m", "logit_q"], metric)
    selection_df, fitted = select_and_fit_best_model_for_pricing(
        ages=ages_arr,
        years=years_arr,
        m=m,
        train_end=train_end,
        model_names=model_names,
        metric=metric_mode,
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
    models: list[str] = typer.Option([], "--models", "-m", help="Subset of models to consider."),
    cpsplines_k: int | None = typer.Option(None),
    cpsplines_horizon: int = typer.Option(0),
    seed: int | None = typer.Option(None),
    ages: str | None = typer.Option(None),
    years: str | None = typer.Option(None),
    ages_path: Path | None = typer.Option(None),
    years_path: Path | None = typer.Option(None),
    output: Path | None = typer.Option(None, help="Output scenarios npz."),
) -> None:
    """Run the P-measure projection pipeline.

    Args:
        ctx: Typer context.
        m_path: Mortality surface path.
        train_end: Last year in training set.
        horizon: Projection horizon in years.
        n_scenarios: Target number of scenarios.
        models: Optional subset of models.
        cpsplines_k: CPsplines knot count.
        cpsplines_horizon: CPsplines horizon.
        seed: Random seed for reproducibility.
        ages: Optional age range.
        years: Optional year range.
        ages_path: Optional ages array path.
        years_path: Optional years array path.
        output: Output scenarios path.
    """
    c = _ctx(ctx)
    ages_inline = ages if _is_list_flag(ages) else None
    years_inline = years if _is_list_flag(years) else None
    ages_arr, years_arr, m = _load_m_surface(
        m_path, ages_inline, years_inline, ages_path, years_path
    )
    ages_arr, years_arr, m = _slice_surface(
        ages_arr,
        years_arr,
        m,
        age_range=_parse_range_spec(ages) if _is_range_flag(ages) else None,
        year_range=_parse_range_spec(years) if _is_range_flag(years) else None,
    )
    model_names: Sequence[str] = (
        tuple(name.upper() for name in models)
        if models
        else ("LCM1", "LCM2", "APCM3", "CBDM5", "CBDM6", "CBDM7")
    )
    scen_set = cast(
        MortalityScenarioSet,
        build_projection_pipeline(
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
        ),
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
    model_name: str = typer.Option("CBDM7", help="Model for lambda calibration (LCM2 or CBDM7)."),
    lambda_esscher: float = typer.Option(
        ..., help="Lambda Esscher tilt (single value or first component)."
    ),
    B_bootstrap: int = typer.Option(100),
    n_process: int = typer.Option(200),
    horizon: int = typer.Option(50),
    seed: int | None = typer.Option(None),
    scale_sigma: float = typer.Option(1.0, help="Scale factor for sigma (vega)."),
    include_last: bool = typer.Option(False),
    ages: str | None = typer.Option(None),
    years: str | None = typer.Option(None),
    ages_path: Path | None = typer.Option(None),
    years_path: Path | None = typer.Option(None),
    output: Path | None = typer.Option(None, help="Output scenarios npz."),
) -> None:
    c = _ctx(ctx)
    ages_inline = ages if _is_list_flag(ages) else None
    years_inline = years if _is_list_flag(years) else None
    ages_arr, years_arr, m = _load_m_surface(
        m_path, ages_inline, years_inline, ages_path, years_path
    )
    ages_arr, years_arr, m = _slice_surface(
        ages_arr,
        years_arr,
        m,
        age_range=_parse_range_spec(ages) if _is_range_flag(ages) else None,
        year_range=_parse_range_spec(years) if _is_range_flag(years) else None,
    )
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
        f"Scenarios (Q) saved to {out} | N={scen_set_q.n_scenarios()} "
        f"horizon={scen_set_q.horizon()}"
    )


@scen_app.command("summarize")
def scen_summarize_cmd(
    ctx: typer.Context,
    scen_path: Path = typer.Option(..., help="Scenario set npz."),
    output: Path | None = typer.Option(None, help="Summary table path."),
) -> None:
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)

    summary = summarize_pv_paths(
        scen_set.q_paths.reshape(scen_set.q_paths.shape[0], -1).mean(axis=1)
    )

    out = output or c.outdir / "scen_summary.json"
    out.write_text(json.dumps(asdict(summary), indent=2))
    typer.echo(f"Scenario summary saved to {out}")


app.add_typer(scen_app, name="scen")


# ---------------------------------------------------------------------------
# SPEC aliases: forecast
# ---------------------------------------------------------------------------


@app.command("forecast")
def forecast_cmd(
    ctx: typer.Context,
    model_path: Path = typer.Argument(..., help="Pickled fitted model."),
    horizon: int = typer.Option(50, help="Projection horizon."),
    scenarios: int = typer.Option(1000, "--scenarios", help="Target number of scenarios."),
    resample: str = typer.Option("year_block", help="Bootstrap resampling mode."),
    include_last: bool = typer.Option(True, help="Include the last observed year."),
    output: Path | None = typer.Option(None, help="Output scenarios npz."),
) -> None:
    """Generate mortality forecasts from a fitted model.

    Args:
        ctx: Typer context.
        model_path: Pickled fitted model (or dict with key "fitted").
        horizon: Projection horizon in years.
        scenarios: Target number of scenarios.
        resample: Bootstrap resampling mode.
        include_last: Whether to include the last observed year.
        output: Output path for scenario set (.npz).
    """
    c = _ctx(ctx)
    obj = _load_pickle(model_path)
    fitted = obj.get("fitted") if isinstance(obj, dict) else obj
    if not isinstance(fitted, FittedModel):
        raise typer.BadParameter(
            "model_path must contain a FittedModel or a dict with key 'fitted'."
        )

    B_bootstrap, n_process, resample_mode = _derive_bootstrap_params(
        n_scenarios=int(scenarios),
        bootstrap_kwargs={"resample": resample},
    )
    resample_typed = cast(Literal["cell", "year_block"], resample_mode)
    _, scen_set, _cache = project_from_fitted_model(
        fitted=fitted,
        B_bootstrap=B_bootstrap,
        horizon=int(horizon),
        n_process=n_process,
        seed=c.seed,
        include_last=bool(include_last),
        resample=resample_typed,
    )
    out = output or c.outdir / "scenarios_forecast.npz"
    _save_scenarios(scen_set, out)
    typer.echo(
        f"Forecast scenarios saved to {out} | "
        f"N={scen_set.n_scenarios()} horizon={scen_set.horizon()}"
    )


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
    pandemic_year: int | None = typer.Option(None),
    pandemic_duration: int = typer.Option(1),
    plateau_start_year: int | None = typer.Option(None),
    accel_start_year: int | None = typer.Option(None),
    output: Path | None = typer.Option(None, help="Output stressed scenarios npz."),
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
    output: Path | None = typer.Option(None),
) -> None:
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    spec_list = _load_config(chain_spec)
    if not isinstance(spec_list, list):
        raise typer.BadParameter("chain-spec must be a list of shock dictionaries.")
    chain: list[ShockSpec] = []
    for spec in spec_list:
        if not isinstance(spec, dict) or "shock_type" not in spec:
            raise typer.BadParameter("Each shock must be a dict with shock_type and params.")
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
    output: Path | None = typer.Option(None, help="Output directory for bundle."),
) -> None:
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    outdir = output or (c.outdir / "bundle")
    outdir.mkdir(parents=True, exist_ok=True)
    base_path = outdir / "base.npz"
    _save_scenarios(scen_set, base_path)
    long_life = apply_mortality_shock(scen_set, shock_type="long_life", magnitude=long_life_bump)
    short_life = apply_mortality_shock(scen_set, shock_type="short_life", magnitude=short_life_bump)
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


@price_app.command("longevity-bond")
def price_longevity_bond_cmd(
    ctx: typer.Context,
    scen_path: Path = typer.Option(...),
    issue_age: float = typer.Option(...),
    maturity_years: int = typer.Option(...),
    notional: float = typer.Option(1.0),
    include_principal: bool = typer.Option(True, "--include-principal/--no-include-principal"),
    short_rate: float | None = typer.Option(None, help="Flat short rate."),
    output: Path | None = typer.Option(None, help="Result JSON/CSV."),
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
    strike: float | None = typer.Option(None),
    payer: str = typer.Option("fixed", help="fixed or floating"),
    short_rate: float | None = typer.Option(None),
    output: Path | None = typer.Option(None),
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
    strike: float | None = typer.Option(None),
    settlement_years: int | None = typer.Option(None),
    notional: float = typer.Option(1.0),
    short_rate: float | None = typer.Option(None),
    output: Path | None = typer.Option(None),
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
    strike: float | None = typer.Option(None),
    settlement_years: int | None = typer.Option(None),
    notional: float = typer.Option(1.0),
    short_rate: float | None = typer.Option(None),
    output: Path | None = typer.Option(None),
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
    maturity_years: int | None = typer.Option(None),
    payment_per_survivor: float = typer.Option(1.0),
    defer_years: int = typer.Option(0),
    exposure_at_issue: float = typer.Option(1.0),
    include_terminal: bool = typer.Option(False),
    terminal_notional: float = typer.Option(0.0),
    short_rate: float | None = typer.Option(None),
    output: Path | None = typer.Option(None),
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
# SPEC aliases: price-bond
# ---------------------------------------------------------------------------


@app.command("price-bond")
def price_bond_cmd(
    ctx: typer.Context,
    model: Path = typer.Option(..., "--model", help="Fitted model pickle or scenario set npz."),
    maturity: int = typer.Option(..., "--maturity", help="Bond maturity in years."),
    coupon: str = typer.Option("survivor", "--coupon", help="Coupon type (only 'survivor')."),
    issue_age: float | None = typer.Option(
        None, "--issue-age", help="Cohort age at valuation (defaults to median age)."
    ),
    notional: float = typer.Option(1.0, help="Bond notional."),
    include_principal: bool = typer.Option(
        True,
        "--include-principal/--no-include-principal",
        help="Include survival-linked principal.",
    ),
    short_rate: float | None = typer.Option(None, help="Flat short rate for pricing."),
    scenarios: int = typer.Option(
        1000, "--scenarios", help="Scenario count when projecting from a fitted model."
    ),
    output: Path | None = typer.Option(None, help="Output JSON path."),
) -> None:
    """Price a survival-linked bond from a fitted model or scenario set.

    Args:
        ctx: Typer context.
        model: Fitted model pickle or scenario set (.npz).
        maturity: Bond maturity in years.
        coupon: Coupon type (only "survivor" is supported).
        issue_age: Cohort age at valuation. Defaults to median age if omitted.
        notional: Bond notional.
        include_principal: Whether to include survival-linked principal.
        short_rate: Flat short rate used for pricing.
        scenarios: Scenario count when projecting from a fitted model.
        output: Output JSON path.
    """
    c = _ctx(ctx)
    if coupon.lower() != "survivor":
        raise typer.BadParameter("Only coupon='survivor' is supported for price-bond.")

    if model.suffix.lower() == ".npz":
        scen_set = _load_scenarios(model)
    else:
        obj = _load_pickle(model)
        fitted = obj.get("fitted") if isinstance(obj, dict) else obj
        if not isinstance(fitted, FittedModel):
            raise typer.BadParameter(
                "model must contain a FittedModel or a dict with key 'fitted'."
            )
        B_bootstrap, n_process, resample_mode = _derive_bootstrap_params(
            n_scenarios=int(scenarios),
            bootstrap_kwargs={"resample": "year_block"},
        )
        resample_typed = cast(Literal["cell", "year_block"], resample_mode)
        _, scen_set, _cache = project_from_fitted_model(
            fitted=fitted,
            B_bootstrap=B_bootstrap,
            horizon=int(maturity),
            n_process=n_process,
            seed=c.seed,
            include_last=True,
            resample=resample_typed,
        )

    if issue_age is None:
        issue_age = float(np.median(np.asarray(scen_set.ages, dtype=float)))

    spec = LongevityBondSpec(
        issue_age=float(issue_age),
        notional=float(notional),
        include_principal=bool(include_principal),
        maturity_years=int(maturity),
    )
    prices = pricing_pipeline(
        scen_Q=scen_set,
        specs={"bond": spec},
        short_rate=float(short_rate) if short_rate is not None else 0.0,
    )
    out = output or c.outdir / "price_bond.json"
    out.write_text(json.dumps(prices, indent=2))
    typer.echo(f"Price={prices['bond']:.6f} saved to {out}")


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
    horizon: int | None = typer.Option(None),
    seed: int | None = typer.Option(None),
    include_last: bool = typer.Option(False),
    output: Path | None = typer.Option(None, help="Calibration result pickle."),
    ages: str | None = typer.Option(None),
    years: str | None = typer.Option(None),
    ages_path: Path | None = typer.Option(None),
    years_path: Path | None = typer.Option(None),
) -> None:
    c = _ctx(ctx)
    quotes_cfg = _load_config(quotes_path)
    if not isinstance(quotes_cfg, list):
        raise typer.BadParameter("quotes file must be a list of quote dicts.")

    def _mk_quote(d: dict[str, Any]) -> MultiInstrumentQuote:
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

    ages_inline = ages if _is_list_flag(ages) else None
    years_inline = years if _is_list_flag(years) else None
    ages_arr, years_arr, m = _load_m_surface(
        m_path, ages_inline, years_inline, ages_path, years_path
    )
    ages_arr, years_arr, m = _slice_surface(
        ages_arr,
        years_arr,
        m,
        age_range=_parse_range_spec(ages) if _is_range_flag(ages) else None,
        year_range=_parse_range_spec(years) if _is_range_flag(years) else None,
    )
    instruments: dict[str, Any] = {}
    market_prices: dict[str, float] = {}
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
        f"Lambda*={calib_summary['lambda_star']} | summary -> {out_json} "
        f"| cache -> {cache_path} | scen -> {scen_path}"
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
    seed: int | None = typer.Option(None),
    include_last: bool = typer.Option(False),
    output: Path | None = typer.Option(None, help="Prices table path."),
    ages: str | None = typer.Option(None),
    years: str | None = typer.Option(None),
    ages_path: Path | None = typer.Option(None),
    years_path: Path | None = typer.Option(None),
) -> None:
    c = _ctx(ctx)
    spec_cfg = _load_config(specs)
    ages_inline = ages if _is_list_flag(ages) else None
    years_inline = years if _is_list_flag(years) else None
    ages_arr, years_arr, m = _load_m_surface(
        m_path, ages_inline, years_inline, ages_path, years_path
    )
    ages_arr, years_arr, m = _slice_surface(
        ages_arr,
        years_arr,
        m,
        age_range=_parse_range_spec(ages) if _is_range_flag(ages) else None,
        year_range=_parse_range_spec(years) if _is_range_flag(years) else None,
    )
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
    scen_set_q = build_scenarios_under_lambda_fast(cache=cache, lambda_esscher=lambda_val)

    specs_norm: dict[str, Any] = {}
    for name, item in spec_cfg.items():
        if not isinstance(item, dict) or "kind" not in item or "spec" not in item:
            raise typer.BadParameter("Specs file must map name -> {kind, spec}.")
        specs_norm[name] = _to_spec(str(item["kind"]).replace("-", "_"), item["spec"])

    prices = pricing_pipeline(scen_Q=scen_set_q, specs=specs_norm, short_rate=short_rate)
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
    output: Path | None = typer.Option(None),
) -> None:
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    spec_cfg = _load_config(spec_path)
    kind_norm = kind.replace("-", "_")
    spec = _to_spec(kind_norm, spec_cfg)

    def price_func(*, scen_set: MortalityScenarioSet, short_rate: float) -> float:
        return float(
            pricing_pipeline(scen_Q=scen_set, specs={"inst": spec}, short_rate=short_rate)["inst"]
        )

    res = rate_sensitivity(price_func, scen_set, base_short_rate=base_short_rate, bump=bump)
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
    output: Path | None = typer.Option(None),
) -> None:
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    spec_cfg = _load_config(spec_path)
    kind_norm = kind.replace("-", "_")
    spec = _to_spec(kind_norm, spec_cfg)

    def price_func(*, scen_set: MortalityScenarioSet, short_rate: float) -> float:
        return float(
            pricing_pipeline(scen_Q=scen_set, specs={"inst": spec}, short_rate=short_rate)["inst"]
        )

    res = rate_convexity(price_func, scen_set, base_short_rate=base_short_rate, bump=bump)
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
    ages: str | None = typer.Option(None, help="Subset ages, comma-separated."),
    short_rate: float = typer.Option(0.0, help="Short rate for pricing."),
    output: Path | None = typer.Option(None),
) -> None:
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    spec_cfg = _load_config(spec_path)
    kind_norm = kind.replace("-", "_")
    spec = _to_spec(kind_norm, spec_cfg)

    def price_func(scen: MortalityScenarioSet) -> float:
        return float(
            pricing_pipeline(scen_Q=scen, specs={"inst": spec}, short_rate=short_rate)["inst"]
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
    output: Path | None = typer.Option(None),
) -> None:
    """Compute all sensitivities via pipeline.risk_analysis_pipeline.

    Args:
        ctx: Typer context.
        scen_path: Scenario set npz (Q).
        specs_path: JSON/YAML specs mapping.
        short_rate: Short rate for pricing.
        sigma_rel_bump: Relative sigma bump.
        q_rel_bump: Relative q bump.
        rate_bump: Rate bump size.
        output: Output JSON path.
    """
    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    specs_cfg = _load_config(specs_path)
    bumps: BumpsConfig = {
        "build_scenarios_func": lambda _scale_sigma: scen_set,
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


@hedge_app.callback(invoke_without_command=True)
def hedge_default_cmd(
    ctx: typer.Context,
    liabilities: Path | None = typer.Option(
        None, "--liabilities", help="Liability PV paths or specs."
    ),
    instruments: Path | None = typer.Option(
        None, "--instruments", help="Instrument PV paths or specs."
    ),
    scenarios: Path | None = typer.Option(
        None, "--scenarios", help="Scenario set npz (required for spec files)."
    ),
    method: str = typer.Option("min_variance", help="Hedging method."),
    short_rate: float | None = typer.Option(None, help="Flat short rate for spec pricing."),
    output: Path | None = typer.Option(None, help="Output JSON."),
) -> None:
    """Alias for hedging with PV paths or scenario-based specs.

    Args:
        ctx: Typer context.
        liabilities: Liability PV paths or spec file (YAML/JSON).
        instruments: Instrument PV paths or spec file (YAML/JSON).
        scenarios: Scenario set npz when using spec files.
        method: Hedging method.
        short_rate: Flat short rate for pricing specs.
        output: Output JSON path.
    """
    if ctx.invoked_subcommand is not None:
        return
    if liabilities is None or instruments is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    spec_exts = {".json", ".yaml", ".yml"}
    is_spec_file = (
        liabilities.suffix.lower() in spec_exts or instruments.suffix.lower() in spec_exts
    )
    if scenarios is not None or is_spec_file:
        if scenarios is None:
            raise typer.BadParameter("Provide --scenarios when using spec files.")
        hedge_end_to_end_cmd(
            ctx=ctx,
            scenarios=scenarios,
            liabilities=liabilities,
            instruments=instruments,
            method=method,
            short_rate=short_rate,
            output=output,
        )
        return

    if method.lower() != "min_variance":
        raise typer.BadParameter("Only method='min_variance' is supported for PV paths.")
    cast(Callable[..., None], hedge_min_variance_cmd)(
        ctx=ctx,
        liab_pv_path=liabilities,
        instr_pv_path=instruments,
        names=None,
        output=output,
    )


@hedge_app.command("end-to-end")
def hedge_end_to_end_cmd(
    ctx: typer.Context,
    scenarios: Path = typer.Option(..., help="Mortality scenarios (.npz)."),
    liabilities: Path = typer.Option(..., help="Liability specs (YAML/JSON)."),
    instruments: Path = typer.Option(..., help="Hedge instrument specs (YAML/JSON)."),
    method: str = typer.Option("min_variance", help="Hedging method."),
    short_rate: float | None = typer.Option(None, help="Flat rate if no discount factors."),
    output: Path | None = typer.Option(None, help="Output JSON."),
) -> None:
    c = _ctx(ctx)
    scen_set = _load_scenarios(scenarios)
    liab_cfg = _load_config(liabilities)
    instr_cfg = _load_config(instruments)

    if not isinstance(liab_cfg, dict):
        raise typer.BadParameter("Liabilities file must contain a mapping of specs.")
    liab_specs = {"liability": liab_cfg} if "kind" in liab_cfg else liab_cfg
    if not isinstance(instr_cfg, dict):
        raise typer.BadParameter("Instruments file must contain a mapping of specs.")
    instr_specs = {"hedge": instr_cfg} if "kind" in instr_cfg else instr_cfg

    liab_paths: FloatArray | None = None
    for spec in liab_specs.values():
        _, pv = _price_paths_for_spec(scen_set, spec, short_rate=short_rate)
        liab_paths = pv if liab_paths is None else liab_paths + pv
    if liab_paths is None:
        raise typer.BadParameter("No liability specs provided.")

    hedge_cols: list[FloatArray] = []
    instr_names: list[str] = []
    for name, spec in instr_specs.items():
        _, pv = _price_paths_for_spec(scen_set, spec, short_rate=short_rate)
        hedge_cols.append(pv)
        instr_names.append(str(name))
    if not hedge_cols:
        raise typer.BadParameter("No hedge instrument specs provided.")

    hedge_matrix = np.column_stack(hedge_cols)
    res = hedging_pipeline(
        liability_pv_paths=liab_paths,
        hedge_pv_paths=hedge_matrix,
        method=method,
    )
    if hasattr(res, "instrument_names"):
        res.instrument_names = instr_names

    weights_map = {
        name: float(w)
        for name, w in zip(instr_names, np.asarray(res.weights).reshape(-1), strict=True)
    }
    payload = {
        "method": method,
        "weights": weights_map,
        "summary": getattr(res, "summary", {}),
    }
    out = output or c.outdir / "hedge_end_to_end.json"
    out.write_text(json.dumps(payload, indent=2))
    typer.echo(f"Hedge result saved to {out}")


@hedge_app.command("min-variance")
def hedge_min_variance_cmd(
    ctx: typer.Context,
    liab_pv_path: Path = typer.Option(..., help="Liability PV paths npy/csv/parquet."),
    instr_pv_path: Path = typer.Option(..., help="Instrument PV paths (N,M)."),
    output: Path | None = typer.Option(None),
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
    discount_factors_path: Path | None = typer.Option(None),
    time_weights_path: Path | None = typer.Option(None),
    output: Path | None = typer.Option(None),
) -> None:
    c = _ctx(ctx)
    L = _read_numeric_matrix(liab_cf_path)
    H_cf = _read_numeric_cube(instr_cf_path)
    df = _read_numeric_series(discount_factors_path) if discount_factors_path else None
    tw = _read_numeric_series(time_weights_path) if time_weights_path else None
    constraints = cast(HedgeConstraints, {"discount_factors": df, "time_weights": tw})
    res = hedging_pipeline(
        liability_pv_paths=L,
        hedge_pv_paths=H_cf,
        method="multihorizon",
        constraints=constraints,
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
    name: str | None = typer.Option(None, help="Name for the report."),
    var_level: float = typer.Option(0.95, help="VaR level."),
    ref_pv_path: Path | None = typer.Option(None, help="Optional reference PV paths."),
    output: Path | None = typer.Option(None),
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
    output: Path | None = typer.Option(None, help="PNG path."),
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
    output: Path | None = typer.Option(None),
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
    cohorts: str | None = typer.Option(None, help="Comma-separated cohort birth years."),
    output: Path | None = typer.Option(None, help="PNG path."),
) -> None:
    import matplotlib.pyplot as plt

    c = _ctx(ctx)
    scen_set = _load_scenarios(scen_path)
    coh_list = None
    if cohorts:
        coh_list = [int(x) for x in cohorts.split(",") if x]
    plot_lexis(
        scen_set,
        value=cast(Literal["m", "q", "S"], value),
        statistic=cast(Literal["mean", "median"], statistic),
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
    output: Path | None = typer.Option(None),
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
    c = _ctx(ctx)
    cfg = _load_config(config)
    out_cfg = cfg.get("outputs", {})
    outdir = Path(out_cfg.get("outdir", c.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg.get("data", {})
    m_path = Path(data_cfg["m_path"])
    ages_arr, years_arr, m = _load_m_surface(
        m_path,
        None,
        None,
        None,
        None,
        preferred_rate_col=data_cfg.get("sex"),
    )
    ages_arr, years_arr, m = _slice_surface(
        ages_arr,
        years_arr,
        m,
        age_range=(
            (
                float(data_cfg["age_min"]),
                float(data_cfg["age_max"]),
            )
            if "age_min" in data_cfg and "age_max" in data_cfg
            else None
        ),
        year_range=(
            (
                float(data_cfg["year_min"]),
                float(data_cfg["year_max"]),
            )
            if "year_min" in data_cfg and "year_max" in data_cfg
            else None
        ),
    )

    fit_cfg = cfg.get("fit", {})
    scen_cfg = cfg.get("scenarios", {})
    models = tuple(fit_cfg.get("models", ()))
    scen_measure = str(scen_cfg.get("measure", "P")).upper()
    horizon = int(scen_cfg.get("horizon", 50))
    n_scenarios = int(scen_cfg.get("n_scenarios", scen_cfg.get("B_bootstrap", 200)))

    if scen_measure == "P":
        scen_set = cast(
            MortalityScenarioSet,
            build_projection_pipeline(
                ages=ages_arr,
                years=years_arr,
                m=m,
                train_end=int(fit_cfg.get("train_end", years_arr.max())),
                horizon=horizon,
                n_scenarios=n_scenarios,
                model_names=(
                    tuple(m.upper() for m in models)
                    if models
                    else ("LCM1", "LCM2", "APCM3", "CBDM5", "CBDM6", "CBDM7")
                ),
                cpsplines_kwargs=fit_cfg.get("cpsplines"),
                seed=c.seed,
            ),
        )
    else:
        cache = build_calibration_cache(
            ages=ages_arr,
            years=years_arr,
            m=m,
            model_name=scen_cfg.get("model_name", "CBDM7"),
            B_bootstrap=int(scen_cfg.get("B_bootstrap", n_scenarios)),
            n_process=int(scen_cfg.get("n_process", n_scenarios)),
            horizon=horizon,
            seed=scen_cfg.get("seed", c.seed),
            include_last=bool(scen_cfg.get("include_last", False)),
        )
        scen_set = build_scenarios_under_lambda_fast(
            cache=cache,
            lambda_esscher=float(scen_cfg.get("lambda_esscher", 0.0)),
            scale_sigma=float(scen_cfg.get("scale_sigma", 1.0)),
            kappa_drift_shock=scen_cfg.get("kappa_drift_shock"),
            kappa_drift_shock_mode=scen_cfg.get("kappa_drift_shock_mode", "additive"),
            cohort_shock_type=scen_cfg.get("cohort_shock_type"),
            cohort_shock_magnitude=scen_cfg.get("cohort_shock_magnitude", 0.01),
            cohort_pivot_year=scen_cfg.get("cohort_pivot_year"),
        )
        scen_set.metadata.setdefault("measure", "Q")

    scen_path = outdir / f"scenarios_{scen_measure}.npz"
    _save_scenarios(scen_set, scen_path)

    pricing_cfg = cfg.get("pricing", {})
    instruments_cfg = pricing_cfg.get("instruments", {})
    prices = pricing_pipeline(
        scen_Q=scen_set,
        specs=instruments_cfg,
        short_rate=float(pricing_cfg.get("short_rate", 0.02)),
    )
    prices_path = outdir / "prices.csv"
    _save_table(
        [{"name": k, "price": v} for k, v in prices.items()],
        prices_path,
        out_cfg.get("format", c.output_format),
    )

    typer.echo(f"Scenarios saved to {scen_path}")
    typer.echo(f"Prices saved to {prices_path}")


@run_app.command("hedge-pipeline")
def run_hedge_pipeline_cmd(
    ctx: typer.Context,
    config: Path = typer.Option(..., help="YAML/JSON config file."),
) -> None:
    c = _ctx(ctx)
    cfg = _load_config(config)
    data_cfg = cfg.get("data", {})
    scen_set = _load_scenarios(Path(data_cfg["scen_path"]))
    short_rate_raw = data_cfg.get("short_rate", None)
    short_rate = None if short_rate_raw is None else float(short_rate_raw)

    liab_spec = cfg.get("liability", {})
    hedge_specs = cfg.get("hedge_instruments", {})
    if not hedge_specs:
        raise typer.BadParameter("hedge_instruments section is required.")

    _, liab_paths = _price_paths_for_spec(scen_set, liab_spec, short_rate=short_rate)
    hedge_cols: list[FloatArray] = []
    names: list[str] = []
    for name, spec in hedge_specs.items():
        _, pv = _price_paths_for_spec(scen_set, spec, short_rate=short_rate)
        hedge_cols.append(pv)
        names.append(str(name))
    hedge_matrix = np.column_stack(hedge_cols)

    hedge_cfg = cfg.get("hedging", {})
    res = hedging_pipeline(
        liability_pv_paths=liab_paths,
        hedge_pv_paths=hedge_matrix,
        method=hedge_cfg.get("method", "min_variance"),
    )
    res.instrument_names = names
    weights_map = {
        name: float(w) for name, w in zip(names, np.asarray(res.weights).reshape(-1), strict=True)
    }

    out_path = Path(hedge_cfg.get("output", c.outdir / "hedge_weights.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"weights": weights_map, "summary": getattr(res, "summary", {})}, indent=2)
    )
    typer.echo(f"Hedge weights saved to {out_path}")


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
