from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pymort.lifetables import load_m_from_excel, m_to_q
from pymort.models.cbd_m5 import fit_cbd, reconstruct_logit_q
from pymort.models.lc_m1 import fit_lee_carter, reconstruct_log_m

# -----------------------
# Small helpers
# -----------------------


def _read_stmomo_surface_csv(path: Path) -> pd.DataFrame:
    """Expect a long CSV with columns: Age, Year, value
    or Age, Year, logm_fitted / logitq_fitted.
    """
    df = pd.read_csv(path)

    if not {"Age", "Year"}.issubset(df.columns):
        raise ValueError(f"{path.name}: expected columns Age, Year. Got: {df.columns.tolist()}")

    if "Value" in df.columns:
        val_col = "Value"
    elif "logm_fitted" in df.columns:
        val_col = "logm_fitted"
    elif "logitq_fitted" in df.columns:
        val_col = "logitq_fitted"
    else:
        raise ValueError(
            f"{path.name}: expected a value column among Value/logm_fitted/logitq_fitted. "
            f"Got: {df.columns.tolist()}"
        )

    out = df[["Age", "Year", val_col]].copy()
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    out[val_col] = pd.to_numeric(out[val_col], errors="coerce")
    out = out.dropna(subset=["Age", "Year", val_col])
    out["Age"] = out["Age"].astype(int)
    out["Year"] = out["Year"].astype(int)
    return out.rename(columns={val_col: "Value"})


def _read_stmomo_series_csv(path: Path) -> pd.DataFrame:
    """Read a time series CSV from StMoMo, e.g. lc_kt.csv (Year, kt)"""
    df = pd.read_csv(path)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def _read_stmomo_params_csv(path: Path) -> pd.DataFrame:
    """Read age params CSV from StMoMo, e.g. lc_params.csv (Age, ax, bx)"""
    df = pd.read_csv(path)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def _long_to_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ages = np.sort(df["Age"].unique()).astype(int)
    years = np.sort(df["Year"].unique()).astype(int)

    piv = df.pivot(index="Age", columns="Year", values="Value").reindex(index=ages, columns=years)
    mat = piv.to_numpy(dtype=float)

    if np.isnan(mat).any():
        n = int(np.isnan(mat).sum())
        raise ValueError(
            f"StMoMo surface has {n} missing cells after pivot; export should be rectangular."
        )
    return ages.astype(float), years.astype(int), mat


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size or a.size < 2:
        raise ValueError("corr: arrays must have same size >= 2")
    c = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(c):
        raise ValueError("corr is not finite")
    return c


@dataclass
class SurfaceCheck:
    name: str
    rmse: float
    max_abs: float
    ok: bool


def _check_surface(
    name: str,
    ref: np.ndarray,
    got: np.ndarray,
    rmse_tol: float,
    max_abs_tol: float,
) -> SurfaceCheck:
    if ref.shape != got.shape:
        raise ValueError(f"{name}: shape mismatch ref={ref.shape} got={got.shape}")
    rmse = _rmse(ref, got)
    mx = _max_abs(ref, got)
    ok = (rmse <= rmse_tol) and (mx <= max_abs_tol)
    return SurfaceCheck(name=name, rmse=rmse, max_abs=mx, ok=ok)


def _assert_same_grid(
    ages_ref: np.ndarray,
    years_ref: np.ndarray,
    ages: np.ndarray,
    years: np.ndarray,
    context: str,
) -> None:
    if not np.array_equal(ages_ref, ages) or not np.array_equal(years_ref, years):
        raise ValueError(
            f"{context}: grid mismatch.\n"
            f"StMoMo ages[{ages_ref[0]}..{ages_ref[-1]}] n={len(ages_ref)}, years[{years_ref[0]}..{years_ref[-1]}] n={len(years_ref)}\n"
            f"PYMORT ages[{ages[0]}..{ages[-1]}] n={len(ages)}, years[{years[0]}..{years[-1]}] n={len(years)}\n"
            "=> Ensure your R scripts filtered the exact same age/year window as load_m_from_excel() here."
        )


def _normalize_lc(
    ax: np.ndarray, bx: np.ndarray, kt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize LC to a common identifiability convention:
    - sum(bx) = 1
    - mean(kt) = 0
    """
    bx = np.asarray(bx, dtype=float).copy()
    kt = np.asarray(kt, dtype=float).copy()
    ax = np.asarray(ax, dtype=float).copy()

    s = float(bx.sum())
    if abs(s) < 1e-14:
        raise ValueError("bx.sum() too close to 0; cannot normalize.")
    bx = bx / s
    kt = kt * s

    kbar = float(kt.mean())
    kt = kt - kbar
    ax = ax + bx * kbar

    return ax, bx, kt


def _align_sign(bx_py, kt_py, kt_r):
    """Fix sign indeterminacy: if kt correlation is negative, flip Python (bx, kt)."""
    corr = np.corrcoef(np.asarray(kt_py).ravel(), np.asarray(kt_r).ravel())[0, 1]
    if np.isfinite(corr) and corr < 0:
        bx_py = -bx_py
        kt_py = -kt_py
    return bx_py, kt_py


# -----------------------
# Main validation
# -----------------------


def main() -> None:
    repo = Path()
    xlsx_path = repo / "Data" / "data_france.xlsx"
    out_dir = repo / "validation_against_StMoMo" / "outputs"

    # MUST match the filters you used in the R StMoMo scripts
    ages, years, m = load_m_from_excel(
        str(xlsx_path),
        sex="Total",
        age_min=60,
        age_max=100,
        year_min=1970,
        year_max=2019,
    )["m"]

    # =========================================================
    # LCM1 vs StMoMo (ROBUST)
    # =========================================================

    # 1) Load StMoMo params + kt + fitted surface
    stmomo_params = _read_stmomo_params_csv(out_dir / "lc_params.csv")
    stmomo_kt = _read_stmomo_series_csv(out_dir / "lc_kt.csv")
    stmomo_logm = _read_stmomo_surface_csv(out_dir / "lc_fitted_logm.csv")

    # Align grid on surface (this is your strongest guardrail)
    ages_ref, years_ref, logm_ref = _long_to_matrix(stmomo_logm)
    _assert_same_grid(ages_ref, years_ref, ages, years, "LCM1")

    # Extract ax/bx/kt from StMoMo on same grids
    stmomo_params = stmomo_params.sort_values("Age")
    ax_r = stmomo_params["ax"].to_numpy(dtype=float)
    bx_r = stmomo_params["bx"].to_numpy(dtype=float)

    stmomo_kt = stmomo_kt.sort_values("Year")
    kt_r = stmomo_kt["kt"].to_numpy(dtype=float)

    # 2) Fit PYMORT LC
    params_lc = fit_lee_carter(m=m)
    logm_got = reconstruct_log_m(params_lc)

    ax_py = np.asarray(params_lc.a, dtype=float)
    bx_py = np.asarray(params_lc.b, dtype=float)
    kt_py = np.asarray(params_lc.k, dtype=float)

    # 3) Normalize + align sign
    ax_py_n, bx_py_n, kt_py_n = _normalize_lc(ax_py, bx_py, kt_py)
    ax_r_n, bx_r_n, kt_r_n = _normalize_lc(ax_r, bx_r, kt_r)
    bx_py_n, kt_py_n = _align_sign(bx_py_n, kt_py_n, bx_r_n, kt_r_n)

    # 4) Robust metrics
    corr_kt = _corr(kt_py_n, kt_r_n)
    corr_bx = _corr(bx_py_n, bx_r_n)

    rmse_kt = _rmse(kt_py_n, kt_r_n)
    kt_scale = float(np.std(kt_r_n, ddof=1))
    rmse_kt_rel = rmse_kt / (kt_scale + 1e-12)

    ax_py_c = ax_py_n - ax_py_n.mean()
    ax_r_c = ax_r_n - ax_r_n.mean()
    rmse_ax = _rmse(ax_py_c, ax_r_c)

    # Surface check = informative, but not the primary pass/fail
    lc_surface = _check_surface(
        "LCM1 fitted log(m) surface",
        ref=logm_ref,
        got=logm_got,
        rmse_tol=2e-2,  # realistic band
        max_abs_tol=0.10,  # realistic band
    )

    # Primary LC pass/fail
    lc_ok = (corr_kt >= 0.99) and (corr_bx >= 0.99) and (rmse_kt_rel <= 0.10)

    # =========================================================
    # CBDM5 vs StMoMo (surface)
    # =========================================================
    q = m_to_q(m)

    stmomo_cbd = _read_stmomo_surface_csv(out_dir / "cbd_fitted_logitq.csv")
    ages_ref2, years_ref2, logitq_ref = _long_to_matrix(stmomo_cbd)
    _assert_same_grid(ages_ref2, years_ref2, ages, years, "CBDM5")

    params_cbd = fit_cbd(q=q, ages=ages)
    logitq_got = reconstruct_logit_q(params_cbd)

    cbd_check = _check_surface(
        "CBDM5 fitted logit(q) surface",
        ref=logitq_ref,
        got=logitq_got,
        rmse_tol=0.10,  # CBD tends to differ more across implementations
        max_abs_tol=0.30,
    )

    # =========================================================
    # Report
    # =========================================================
    print("\n=== Validation vs StMoMo (robust) ===")

    print("\n[LCM1] parameter checks (PRIMARY)")
    print(f"- corr(k)     = {corr_kt:.6f} (target >= 0.99)")
    print(f"- corr(b)     = {corr_bx:.6f} (target >= 0.99)")
    print(f"- rmse(k)     = {rmse_kt:.6g}")
    print(f"- rmse(k)/sd  = {rmse_kt_rel:.6f} (target <= 0.10)")
    print(f"- rmse(a) (demeaned) = {rmse_ax:.6g}")

    print("\n[LCM1] surface check (SECONDARY)")
    print(f"- RMSE(logm)   = {lc_surface.rmse:.6g} (tol {2e-2})")
    print(f"- max_abs(logm)= {lc_surface.max_abs:.6g} (tol {0.10})")
    print(f"- surface OK   = {lc_surface.ok}")

    print("\n[CBDM5] surface check")
    print(f"- RMSE(logitq)   = {cbd_check.rmse:.6g} (tol {0.10})")
    print(f"- max_abs(logitq)= {cbd_check.max_abs:.6g} (tol {0.30})")
    print(f"- surface OK     = {cbd_check.ok}")

    if not lc_ok:
        raise SystemExit("❌ LC validation failed: (corr or kt RMSE rel) beyond tolerance.")
    if not cbd_check.ok:
        raise SystemExit("❌ CBD validation failed: surface beyond tolerance.")
    print("\n✅ Validation passed (robust criteria).")


if __name__ == "__main__":
    main()
