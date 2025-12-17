from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from cpsplines.fittings.fit_cpsplines import CPsplines
from cpsplines.utils.rearrange_data import grid_to_scatter


def smooth_mortality_with_cpsplines(
    m: np.ndarray,
    ages: np.ndarray,
    years: np.ndarray,
    deg: Tuple[int, int] = (3, 3),
    ord_d: Tuple[int, int] = (2, 2),
    k: Tuple[int, int] | None = None,
    sp_method: str = "grid_search",
    sp_args: Optional[dict] = None,
    horizon: int = 50,
    verbose: bool = False,
) -> Dict[str, object]:
    """
    Smooth log(m_x,t) with CPsplines on a 2D surface (Age * Year).

    Args
    ----
    m : array (A, T)
        Central death rates m_x,t on an age * year grid.
    ages : array (A,)
    years : array (T,)
    deg : tuple
        Spline degrees (age, year).
    ord_d : tuple
        Penalty derivative orders (age, year).
    k : tuple or None
        Number of B-spline basis functions per dimension (age, year).
        If None, chosen automatically from (A, T).
    sp_method : {"grid_search", "optimizer"}
        Smoothing parameter selection method.
    horizon : int
        Number of years to forecast.
    verbose : bool
        If True, print diagnostics (k chosen, etc.).

    Returns
    -------
    dict with:
        - "m_fitted":   smoothed m on historical years (A, T)
        - "m_forecast": CPsplines forecast for horizon H (A, H)
        - "years_forecast": array of forecast years (H,)
        - "model": CPsplines object
    """

    m = np.asarray(m, dtype=float)
    ages = np.asarray(ages, dtype=float)
    years = np.asarray(years, dtype=int)

    # sécurité
    m = np.clip(m, 1e-12, np.inf)
    if m.shape != (ages.shape[0], years.shape[0]):
        raise ValueError(
            f"Shape mismatch: m has shape {m.shape}, expected ({ages.shape[0]}, {years.shape[0]})."
        )

    A, T = m.shape

    # ========= auto-choice for k if not provided =========
    if k is None:
        k_age = min(max(A // 2, 8), A - 1)
        k_year = min(max(T // 3, 8), T - 1)
        k = (int(k_age), int(k_year))

        if verbose:
            print(f"[CPsplines] auto k = (k_age={k_age}, k_year={k_year})")

    # ========= 1) Prepare data in scatter format =========
    # CPsplines attend un DataFrame de type {x0, x1, y}
    df = grid_to_scatter(
        x=[ages, years],
        y=np.log(m),  # on lisse log(m), plus stable
    )
    df = df.rename(columns={"x0": "age", "x1": "year"})

    # ========= 2) Fit CPsplines surface (Age × Year) =========
    if sp_args is None:
        sp_args = {"top_n": 5, "parallel": True}
    model = CPsplines(
        deg=deg,
        ord_d=ord_d,
        k=k,
        sp_method=sp_method,
        sp_args=sp_args,
        x_range={
            "age": (float(ages.min()), float(ages.max())),
            "year": (float(years.min()) - 2, float(years.max()) + horizon + 2),
        },
    )

    if verbose:
        print("[CPsplines] fitting 2D surface on log m ...")

    model.fit(data=df, y_col="y")

    # ========= 3) Predict historical fitted =========
    df_hist = pd.DataFrame(
        {
            "age": np.repeat(ages, T),
            "year": np.tile(years, A),
        }
    )

    eta_hist = model.predict(df_hist).reshape(A, T)
    m_fitted = np.exp(eta_hist)

    # ========= 4) Predict future years =========
    if horizon <= 0:
        years_forecast = np.asarray([], dtype=int)
        m_forecast = np.zeros((A, 0), dtype=float)
    else:
        years_forecast = np.arange(years[-1] + 1, years[-1] + 1 + horizon)

        df_future = pd.DataFrame(
            {
                "age": np.repeat(ages, horizon),
                "year": np.tile(years_forecast, A),
            }
        )

        eta_future = model.predict(df_future).reshape(A, horizon)
        m_forecast = np.exp(eta_future)

    return {
        "m_fitted": m_fitted,
        "m_forecast": m_forecast,
        "years_forecast": years_forecast,
        "model": model,
    }
