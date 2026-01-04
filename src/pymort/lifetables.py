"""Mortality table utilities and Excel loaders.

This module loads HMD-style period mortality tables and provides helpers to
convert between central death rates (m) and one-year death probabilities (q),
plus survival curve builders.

Note:
    Docstrings follow Google style and type hints use NDArray for clarity.
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from contextlib import suppress
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from pymort._types import FloatArray, IntArray

Sex = Literal["Total", "Female", "Male"]
MortalitySurface = tuple[IntArray, IntArray, FloatArray]


def _norm(s: str) -> str:
    """Normalize header tokens for robust matching."""
    return (
        str(s)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("\t", "")
        .replace("\n", "")
        .replace("-", "")
        .replace("_", "")
    )


_CANON = {
    "year": "Year",
    "age": "Age",
    "female": "Female",
    "male": "Male",
    "total": "Total",
}


def _find_header_and_map(
    sheet_df: pd.DataFrame, max_scan_rows: int = 50
) -> tuple[int | None, dict[str, int]]:
    """Scan top rows to find the header row and a mapping.

    Args:
        sheet_df: Raw sheet values without headers.
        max_scan_rows: Number of rows to scan from the top.

    Returns:
        Tuple of (header_row_index, column_map). If no header is found,
        returns (None, {}).
    """
    # work on raw values (no header)
    raw = sheet_df.copy()
    raw.columns = [f"col{j}" for j in range(raw.shape[1])]

    for r in range(min(max_scan_rows, len(raw))):
        # row r as candidate header
        row_vals = raw.iloc[r].tolist()
        normalized = [_norm(v) for v in row_vals]

        # try to map required keys
        colmap: dict[str, int] = {}
        for j, key in enumerate(normalized):
            if key in _CANON:
                canon = _CANON[key]
                # keep first occurrence
                colmap.setdefault(canon, j)

        # we require at least Year & Age and one of (Total, Female+Male or one of them)
        has_year_age = ("Year" in colmap) and ("Age" in colmap)
        has_any_rate = ("Total" in colmap) or ("Female" in colmap) or ("Male" in colmap)
        if has_year_age and has_any_rate:
            return r, colmap

    return None, {}


def _read_table_with_header(
    sheet_df: pd.DataFrame, header_row: int, colmap: dict[str, int]
) -> pd.DataFrame:
    """Build a tidy DataFrame with canonical columns present in the sheet.

    Args:
        sheet_df: Raw sheet values without headers.
        header_row: Row index of the header line.
        colmap: Mapping of canonical names to column indices.

    Returns:
        DataFrame with canonical column names and cleaned numeric values.
    """
    # Re-read the sheet row subset with header at header_row
    # (convert again to let pandas parse types under the header)
    data = sheet_df.iloc[header_row + 1 :].copy()
    header_vals = sheet_df.iloc[header_row].tolist()
    data.columns = header_vals

    # Keep only mapped columns
    keep_cols = [list(data.columns)[colmap[c]] for c in colmap]
    sub = data[keep_cols].copy()

    # Rename to canonical names
    rename_map = {list(data.columns)[colmap[c]]: c for c in colmap}
    sub = sub.rename(columns=rename_map)

    # Strip whitespace in string columns
    for c in sub.columns:
        if sub[c].dtype == object:
            sub[c] = sub[c].astype(str).str.strip()

    # Drop open age (e.g., "110+")
    if "Age" in sub.columns:
        sub = sub[sub["Age"] != "110+"]

    # Coerce numerics
    year_col = cast("pd.Series[Any]", sub.get("Year"))
    age_col = cast("pd.Series[Any]", sub.get("Age"))
    sub["Year"] = pd.to_numeric(year_col, errors="coerce")
    sub["Age"] = pd.to_numeric(age_col, errors="coerce")
    for sx in ("Total", "Female", "Male"):
        if sx in sub.columns:
            sub[sx] = pd.to_numeric(sub[sx], errors="coerce")

    # Keep only rows with Year & Age
    sub = sub.dropna(subset=["Year", "Age"]).copy()
    sub["Year"] = sub["Year"].astype(int)
    sub["Age"] = sub["Age"].astype(int)
    return sub


def load_m_from_excel(
    path: str,
    *,
    sex: Sex = "Total",
    age_min: int = 60,
    age_max: int = 100,
    year_min: int | None = None,
    year_max: int | None = None,
    m_floor: float = 1e-12,
    drop_years: Iterable[int] | None = None,
) -> dict[str, MortalitySurface]:
    """Load an HMD-style period mortality table from Excel.

    The loader expects a long-form sheet with columns for age, year, and one
    mortality rate column (Total, Male, or Female). It pivots the data into a
    rectangular surface m[age, year].

    Args:
        path: Excel file path.
        sex: Which rate column to use when multiple are available.
        age_min: Minimum age to include.
        age_max: Maximum age to include.
        year_min: Minimum year to include.
        year_max: Maximum year to include.
        m_floor: Floor applied to m for numerical stability.
        drop_years: Optional years to drop after loading.

    Returns:
        Mapping with key "m" and value (ages, years, m) where:
        - ages has shape (A,) and dtype int
        - years has shape (T,) and dtype int
        - m has shape (A, T) with central death rates

    Raises:
        ValueError: If no suitable sheet/columns are found or filters drop all rows.
    """
    # Read all sheets raw (no header) to allow header detection
    xls_dict = pd.read_excel(path, sheet_name=None, header=None, engine="openpyxl")
    found_df: pd.DataFrame | None = None
    _found_cols: dict[str, int] = {}
    _found_sheet: str | None = None
    _header_row: int | None = None

    # Try each sheet until we find Year & Age & (Total/Female/Male)
    for sheet_name, df in xls_dict.items():
        hrow, cmap = _find_header_and_map(df)
        if hrow is None:
            continue
        # Build a proper table for this sheet
        table = _read_table_with_header(df, hrow, cmap)
        # Must contain at least one rate column
        if {"Year", "Age"} - set(table.columns):
            continue
        if not ({"Total", "Female", "Male"} & set(table.columns)):
            continue
        # keep the first valid sheet (or prefer one that has the requested sex)
        if found_df is None:
            found_df, _found_cols, _found_sheet, _header_row = (
                table,
                cmap,
                sheet_name,
                hrow,
            )
        # Prefer sheet where requested sex is available
        if (sex in table.columns) and (found_df is not None) and (sex not in found_df.columns):
            found_df, _found_cols, _found_sheet, _header_row = (
                table,
                cmap,
                sheet_name,
                hrow,
            )

    if found_df is None:
        raise ValueError(
            "Could not find a sheet with Year & Age & (Total/Female/Male). "
            f"Scanned sheets: {list(xls_dict.keys())}."
        )

    df = found_df

    # Choose the rate column to use
    rate_col = sex if sex in df.columns else ("Total" if "Total" in df.columns else None)
    if rate_col is None:
        # If requested sex not present and no Total, fallback to Female or Male (whichever exists)
        rate_col = "Female" if "Female" in df.columns else "Male"

    # Filter ranges
    if year_min is not None:
        df = df[df["Year"] >= year_min]
    if year_max is not None:
        df = df[df["Year"] <= year_max]
    df = df[(df["Age"] >= age_min) & (df["Age"] <= age_max)]

    if df.empty:
        raise ValueError("No rows left after filtering age/year. Check filters.")

    # Build regular grids
    ages = cast(IntArray, np.arange(df["Age"].min(), df["Age"].max() + 1, dtype=int))
    years = cast(IntArray, np.arange(df["Year"].min(), df["Year"].max() + 1, dtype=int))

    # Pivot to (A, T)
    pivot = df.pivot(index="Age", columns="Year", values=rate_col).reindex(
        index=ages, columns=years
    )
    m = cast(FloatArray, pivot.to_numpy(dtype=float))

    if drop_years is not None:
        mask = ~np.isin(years, np.array(list(drop_years)))
        years = years[mask]
        m = m[:, mask]

    # Simple imputation if gaps exist (ffill/bfill along time then age)
    if np.isnan(m).any():
        m = (
            pd.DataFrame(m, index=ages, columns=years)
            .ffill(axis=1)
            .bfill(axis=1)
            .ffill(axis=0)
            .bfill(axis=0)
            .to_numpy()
        )
        if np.isnan(m).any():
            raise ValueError("Missing values remain after simple imputation.")

    # Ensure strictly positive
    m = cast(FloatArray, np.clip(m, m_floor, None))

    return {"m": (ages, years, m)}


def m_to_q(m: FloatArray) -> FloatArray:
    """Convert central death rates to one-year death probabilities.

    Uses the standard approximation q = m / (1 + 0.5 * m) and clips values to
    keep 0 < q < 1.

    Args:
        m: Central death rates. Shape (A, T) or any array-like shape.

    Returns:
        One-year death probabilities with the same shape as m.
    """
    q = m / (1.0 + 0.5 * m)
    return np.clip(q, 1e-10, 1 - 1e-10)


def q_to_m(q: FloatArray) -> FloatArray:
    """Convert one-year death probabilities back to central death rates.

    Uses m = 2q / (2 - q) and clips q for numerical stability.

    Args:
        q: One-year death probabilities. Shape (A, T) or any array-like shape.

    Returns:
        Central death rates with the same shape as q.
    """
    q = np.clip(q, 1e-10, 1 - 1e-10)
    return (2.0 * q) / (2.0 - q)


def survival_from_q(q: FloatArray) -> FloatArray:
    """Compute survival probabilities from one-year death probabilities.

    Survival is computed by cumulative multiplication of (1 - q) along the
    last axis.

    Args:
        q: One-year death probabilities. Shape (..., T).

    Returns:
        Survival probabilities with the same shape as q.
    """
    q = np.asarray(q, dtype=float)

    if q.ndim < 1:
        raise ValueError(f"q must have at least 1 dimension, got shape {q.shape}.")
    validate_q(q)

    return np.cumprod(1.0 - q, axis=-1)


def validate_q(q: FloatArray) -> None:
    """Validate that q values are strictly within (0, 1) and finite.

    Args:
        q: One-year death probabilities. Shape (..., T).

    Raises:
        ValueError: If q contains non-finite values or values outside (0, 1).
    """
    if not (np.all(q > 0) and np.all(q < 1)):
        raise ValueError("q must be strictly in (0,1).")
    if not np.isfinite(q).all():
        raise ValueError("q must contain finite values.")


def validate_survival_monotonic(S: FloatArray) -> None:
    """Check that survival curves are non-increasing over time.

    Comparisons where either side is not finite (e.g., NaN tails) are ignored.

    Args:
        S: Survival curves. Shape (..., T).

    Raises:
        AssertionError: If any finite segment increases over time.
    """
    S = np.asarray(S, dtype=float)
    if S.ndim < 1:
        raise ValueError(f"S must have at least 1 dimension, got shape {S.shape}.")

    dS = np.diff(S, axis=-1)
    finite_pairs = np.isfinite(S[..., :-1]) & np.isfinite(S[..., 1:])
    if np.any(finite_pairs & (dS > 1e-12)):
        raise AssertionError("S_x(t) must be non-increasing in t.")


def cohort_survival_from_q_paths(q_paths: FloatArray) -> FloatArray:
    """Build cohort survival surfaces from q paths.

    For each starting age index a, the cohort survival is:
    S[n, a, k] = prod_{j=0..k} (1 - q_paths[n, a + j, j]).

    Args:
        q_paths: One-year death probabilities. Shape (N, A, H).

    Returns:
        Cohort survival paths with shape (N, A, H). Entries that are not
        reachable because a + k >= A are filled with NaN.

    Raises:
        ValueError: If q_paths is not 3D or contains invalid probabilities.
    """
    q = np.asarray(q_paths, dtype=float)
    if q.ndim != 3:
        raise ValueError(f"q_paths must be (N,A,H), got {q.shape}.")
    validate_q(q)

    N, A, H = q.shape
    S = np.full((N, A, H), np.nan, dtype=float)

    for a in range(A):
        K = min(H, A - a)  # max horizon possible for this starting age
        if K <= 0:
            continue
        idx = np.arange(K)
        q_diag = q[:, a + idx, idx]  # (N, K)
        S[:, a, :K] = np.cumprod(1.0 - q_diag, axis=1)

    return cast(FloatArray, np.clip(S, 1e-12, 1.0))


def survival_paths_from_q_paths(q_paths: FloatArray) -> FloatArray:
    """Compute survival paths by cumulative product along the horizon.

    Args:
        q_paths: One-year death probabilities. Shape (N, A, H).

    Returns:
        Survival paths with shape (N, A, H).
    """
    q = np.asarray(q_paths, dtype=float)
    if q.ndim != 3:
        raise ValueError("q_paths must have shape (N, A, H).")
    q = np.clip(q, 0.0, 1.0)
    return np.cumprod(1.0 - q, axis=2)


def load_m_from_excel_any(
    source: str | bytes | bytearray | Any,
    *,
    sex: Sex = "Total",
    age_min: int = 60,
    age_max: int = 100,
    year_min: int | None = None,
    year_max: int | None = None,
    m_floor: float = 1e-12,
    drop_years: Iterable[int] | None = None,
) -> dict[str, MortalitySurface]:
    """Load an HMD-style table from multiple input types.

    Args:
        source: Excel file path, bytes, file-like object, or Streamlit UploadedFile.
        sex: Which rate column to use when multiple are available.
        age_min: Minimum age to include.
        age_max: Maximum age to include.
        year_min: Minimum year to include.
        year_max: Maximum year to include.
        m_floor: Floor applied to m for numerical stability.
        drop_years: Optional years to drop after loading.

    Returns:
        Mapping with key "m" and value (ages, years, m) where:
        - ages has shape (A,) and dtype int
        - years has shape (T,) and dtype int
        - m has shape (A, T) with central death rates
    """
    # --- Normalize to something pandas.read_excel can read ---
    excel_obj: Any = source

    # Streamlit UploadedFile: prefer getbuffer() to avoid consuming the stream
    if hasattr(source, "getbuffer"):
        excel_obj = io.BytesIO(source.getbuffer())
    # bytes-like
    elif isinstance(source, (bytes, bytearray)):
        excel_obj = io.BytesIO(source)
    # file-like: ensure we're at start if possible
    elif hasattr(source, "read") and not isinstance(source, str):
        with suppress(Exception):
            source.seek(0)
        excel_obj = source

    # Read all sheets raw (no header) to allow header detection
    xls_dict = pd.read_excel(excel_obj, sheet_name=None, header=None, engine="openpyxl")

    found_df: pd.DataFrame | None = None
    _found_cols: dict[str, int] = {}
    _found_sheet: str | None = None
    _header_row: int | None = None

    # Try each sheet until we find Year & Age & (Total/Female/Male)
    for sheet_name, df in xls_dict.items():
        hrow, cmap = _find_header_and_map(df)
        if hrow is None:
            continue

        table = _read_table_with_header(df, hrow, cmap)

        if {"Year", "Age"} - set(table.columns):
            continue
        if not ({"Total", "Female", "Male"} & set(table.columns)):
            continue

        if found_df is None:
            found_df, _found_cols, _found_sheet, _header_row = (
                table,
                cmap,
                sheet_name,
                hrow,
            )

        if (sex in table.columns) and (found_df is not None) and (sex not in found_df.columns):
            found_df, _found_cols, _found_sheet, _header_row = (
                table,
                cmap,
                sheet_name,
                hrow,
            )

    if found_df is None:
        raise ValueError(
            "Could not find a sheet with Year & Age & (Total/Female/Male). "
            f"Scanned sheets: {list(xls_dict.keys())}."
        )

    df = found_df

    # Choose the rate column to use
    rate_col = sex if sex in df.columns else ("Total" if "Total" in df.columns else None)
    if rate_col is None:
        rate_col = "Female" if "Female" in df.columns else "Male"

    # Filter ranges
    if year_min is not None:
        df = df[df["Year"] >= year_min]
    if year_max is not None:
        df = df[df["Year"] <= year_max]
    df = df[(df["Age"] >= age_min) & (df["Age"] <= age_max)]

    if df.empty:
        raise ValueError("No rows left after filtering age/year. Check filters.")

    # Build regular grids
    ages = cast(IntArray, np.arange(df["Age"].min(), df["Age"].max() + 1, dtype=int))
    years = cast(IntArray, np.arange(df["Year"].min(), df["Year"].max() + 1, dtype=int))

    # Pivot to (A, T)
    pivot = df.pivot(index="Age", columns="Year", values=rate_col).reindex(
        index=ages, columns=years
    )
    m = cast(FloatArray, pivot.to_numpy(dtype=float))

    if drop_years is not None:
        mask = ~np.isin(years, np.array(list(drop_years)))
        years = years[mask]
        m = m[:, mask]

    # Simple imputation if gaps exist (ffill/bfill along time then age)
    if np.isnan(m).any():
        m = (
            pd.DataFrame(m, index=ages, columns=years)
            .ffill(axis=1)
            .bfill(axis=1)
            .ffill(axis=0)
            .bfill(axis=0)
            .to_numpy()
        )
        if np.isnan(m).any():
            raise ValueError("Missing values remain after simple imputation.")

    m = cast(FloatArray, np.clip(m, m_floor, None))

    return {"m": (ages, years, m)}
