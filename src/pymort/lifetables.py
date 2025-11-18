from __future__ import annotations

from typing import Dict, Iterable, Literal, Optional, Tuple

import numpy as np
import pandas as pd

Sex = Literal["Total", "Female", "Male"]


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
    sheet_df: pd.DataFrame, max_scan_rows: int = 30
) -> Tuple[Optional[int], Dict[str, int]]:
    """
    Scan top rows to find the header row and a mapping {CanonName -> column index}.
    Returns (header_row_index, mapping). If not found, (None, {}).
    """
    # work on raw values (no header)
    raw = sheet_df.copy()
    raw.columns = [f"col{j}" for j in range(raw.shape[1])]

    for r in range(min(max_scan_rows, len(raw))):
        # row r as candidate header
        row_vals = raw.iloc[r].tolist()
        normalized = [_norm(v) for v in row_vals]

        # try to map required keys
        colmap: Dict[str, int] = {}
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
    sheet_df: pd.DataFrame, header_row: int, colmap: Dict[str, int]
) -> pd.DataFrame:
    """
    Build a tidy DataFrame with canonical columns present in the sheet.
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
    sub["Year"] = pd.to_numeric(sub.get("Year"), errors="coerce")
    sub["Age"] = pd.to_numeric(sub.get("Age"), errors="coerce")
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
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load a mortality table from an Excel file.

    The function automatically detects columns 'Year', 'Age', and one mortality column
    ('Total', 'Male', or 'Female'), regardless of their order in the sheet. It returns
    a clean mortality surface m[age, year] along with the corresponding age and year grids.
    """
    # Read all sheets raw (no header) to allow header detection
    xls_dict = pd.read_excel(path, sheet_name=None, header=None, engine="openpyxl")
    found_df: Optional[pd.DataFrame] = None
    found_cols: Dict[str, int] = {}
    found_sheet: Optional[str] = None
    header_row: Optional[int] = None

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
            found_df, found_cols, found_sheet, header_row = (
                table,
                cmap,
                sheet_name,
                hrow,
            )
        # Prefer sheet where requested sex is available
        if (
            (sex in table.columns)
            and (found_df is not None)
            and (sex not in found_df.columns)
        ):
            found_df, found_cols, found_sheet, header_row = (
                table,
                cmap,
                sheet_name,
                hrow,
            )

    if found_df is None:
        # Build a helpful error message listing columns seen
        seen = {
            name: list(
                map(
                    _norm,
                    xls_dict[name].iloc[0:30].astype(str).fillna("").values.ravel(),
                )
            )
            for name in xls_dict
        }
        raise ValueError(
            "Could not find a sheet with Year & Age & (Total/Female/Male). "
            f"Scanned sheets: {list(xls_dict.keys())}."
        )

    df = found_df

    # Choose the rate column to use
    rate_col = (
        sex if sex in df.columns else ("Total" if "Total" in df.columns else None)
    )
    if rate_col is None:
        # If requested sex not present and no Total, fallback to Female or Male (whichever exists)
        rate_col = "Female" if "Female" in df.columns else "Male"

    if drop_years is not None:
        mask = ~np.isin(years, np.array(list(drop_years)))
        years = years[mask]
        m = m[:, mask]

    # Filter ranges
    if year_min is not None:
        df = df[df["Year"] >= year_min]
    if year_max is not None:
        df = df[df["Year"] <= year_max]
    df = df[(df["Age"] >= age_min) & (df["Age"] <= age_max)]

    if df.empty:
        raise ValueError("No rows left after filtering age/year. Check filters.")

    # Build regular grids
    ages = np.arange(df["Age"].min(), df["Age"].max() + 1, dtype=int)
    years = np.arange(df["Year"].min(), df["Year"].max() + 1, dtype=int)

    # Pivot to (A, T)
    pivot = df.pivot(index="Age", columns="Year", values=rate_col).reindex(
        index=ages, columns=years
    )
    m = pivot.to_numpy(dtype=float)

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
    m = np.clip(m, m_floor, None)

    return {"m": (ages, years, m)}


def m_to_q(m: np.ndarray) -> np.ndarray:
    """
    Convert central death rates m_x,t into one-year death probabilities q_x,t
    using the standard approximation q = m / (1 + 0.5*m). The output is clipped
    to maintain 0 < q < 1.
    """
    q = m / (1.0 + 0.5 * m)
    return np.clip(q, 1e-10, 1 - 1e-10)


def q_to_m(q: np.ndarray) -> np.ndarray:
    """
    Convert one-year death probabilities q_x,t back to central death rates m_x,t
    via m = 2q / (1 - q). The result is clipped to ensure numerical stability.
    """
    q = np.clip(q, 1e-10, 1 - 1e-10)
    return (2.0 * q) / (1.0 - q)


def survival_from_q(q: np.ndarray) -> np.ndarray:
    """
    Compute survival probabilities S_x(t) from one-year death probabilities q_x,t
    by cumulative multiplication of (1 - q). Survival is computed along the time axis.
    """
    return np.cumprod(1.0 - q, axis=1)


def validate_q(q: np.ndarray) -> None:
    """
    Validate that all q_x,t lie strictly within (0,1).
    Raises an AssertionError if invalid values are detected.
    """
    if not (np.all(q > 0) and np.all(q < 1)):
        raise AssertionError("q must be in (0,1).")


def validate_survival_monotonic(S: np.ndarray) -> None:
    """
    Check that survival curves S_x(t) are non-increasing over time.
    Raises an AssertionError if any survival path increases.
    """
    if np.any(np.diff(S, axis=1) > 1e-12):
        raise AssertionError("S_x(t) must be non-increasing in t.")
