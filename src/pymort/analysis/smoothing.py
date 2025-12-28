from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from cpsplines.fittings.fit_cpsplines import CPsplines


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
    """
    try:
        from cpsplines.fittings.fit_cpsplines import CPsplines
        from cpsplines.utils.rearrange_data import grid_to_scatter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "cpsplines is required for CPsplines smoothing. Install with `pip install .[cpsplines]`."
        ) from exc
    m = np.asarray(m, dtype=float)
    ages = np.asarray(ages, dtype=float)
    years = np.asarray(years, dtype=int)

    # ---- input validation (MUST happen before any clipping) ----
    if m.ndim != 2:
        raise ValueError("m must be a 2D array with shape (A, T).")
    if ages.ndim != 1:
        raise ValueError("ages must be a 1D array.")
    if years.ndim != 1:
        raise ValueError("years must be a 1D array.")

    A, T = m.shape
    if ages.shape[0] != A or years.shape[0] != T:
        raise ValueError(
            f"Shape mismatch: m has shape {m.shape}, expected ({ages.shape[0]}, {years.shape[0]})."
        )

    deg_age = int(deg[0])
    deg_year = int(deg[1])
    ord_age = int(ord_d[0])
    ord_year = int(ord_d[1])

    # degrees must be >= 1 and not exceed data size-1
    deg_age = max(1, min(deg_age, A - 1))
    deg_year = max(1, min(deg_year, T - 1))

    # penalty orders must satisfy 0 <= ord < deg
    ord_age = max(0, min(ord_age, deg_age - 1))
    ord_year = max(0, min(ord_year, deg_year - 1))

    deg = (deg_age, deg_year)
    ord_d = (ord_age, ord_year)

    # OPTIONAL explicit guard (should never trigger now)
    if ord_d[0] >= deg[0] or ord_d[1] >= deg[1]:
        raise ValueError(
            f"CPsplines requires ord_d < deg, got deg={deg}, ord_d={ord_d}"
        )

    if not np.isfinite(m).all():
        raise ValueError("m must contain finite values (no NaN/Inf).")
    if not np.isfinite(ages).all():
        raise ValueError("ages must contain finite values (no NaN/Inf).")
    if not np.isfinite(years.astype(float)).all():
        raise ValueError("years must contain finite values.")

    if (m <= 0.0).any():
        raise ValueError("m must be strictly positive everywhere.")
    if horizon < 0:
        raise ValueError("horizon must be >= 0.")

    # Now safe to clip tiny values for numerical stability of log(m)
    m_safe = np.clip(m, 1e-12, np.inf)

    # ========= auto-choice for k if not provided =========
    if k is None:
        # safest choice for tiny grids used in tests
        k = (max(deg_age + 1, A), max(deg_year + 1, T))
    else:
        k = (int(k[0]), int(k[1]))
        if k[0] <= deg_age or k[1] <= deg_year:
            raise ValueError(f"k must be > deg componentwise, got k={k}, deg={deg}")

    # ========= 1) Prepare data in scatter format =========
    df = grid_to_scatter(
        x=[ages, years],
        y=np.log(m_safe),
    )
    df = df.rename(columns={"x0": "age", "x1": "year"})

    # ========= 2) Fit CPsplines surface =========
    if sp_args is None:
        sp_args = {"top_n": 5, "parallel": False}

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
    df_hist = pd.DataFrame({"age": np.repeat(ages, T), "year": np.tile(years, A)})
    eta_hist = model.predict(df_hist).reshape(A, T)
    m_fitted = np.exp(eta_hist)

    # ========= 4) Predict future years =========
    if horizon == 0:
        years_forecast = np.asarray([], dtype=int)
        m_forecast = np.zeros((A, 0), dtype=float)
    else:
        years_forecast = np.arange(years[-1] + 1, years[-1] + 1 + horizon, dtype=int)
        df_future = pd.DataFrame(
            {"age": np.repeat(ages, horizon), "year": np.tile(years_forecast, A)}
        )
        eta_future = model.predict(df_future).reshape(A, horizon)
        m_forecast = np.exp(eta_future)

    return {
        "m_fitted": m_fitted,
        "m_forecast": m_forecast,
        "years_forecast": years_forecast,
        "model": model,
    }
