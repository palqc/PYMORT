from typing import Optional

import numpy as np

from pymort.analysis import MortalityScenarioSet


def find_nearest_age_index(ages: np.ndarray, target_age: float) -> int:
    """
    Return the index of the age in `ages` closest to `target_age`.
    """
    ages = np.asarray(ages, dtype=float)
    return int(np.argmin(np.abs(ages - float(target_age))))


def build_discount_factors(
    scen_set: MortalityScenarioSet,
    short_rate: Optional[float],
    discount_factors: Optional[np.ndarray],
    H: int,
) -> np.ndarray:
    """
    Determine discount factors for the pricing horizon.

    Priority:
        1) explicit discount_factors (if provided),
        2) scen_set.discount_factors (if present),
        3) flat short_rate (if provided).

    Returns 1D (H,) or 2D (N,H) array when scenario-specific discounting is provided.
    """

    # 1) Explicit discount factors
    if discount_factors is not None:
        df = np.asarray(discount_factors, dtype=float)
        if df.ndim == 1:
            if df.shape[0] < H:
                raise ValueError(
                    f"discount_factors must have length >= {H}, got {df.shape[0]}."
                )
            df = df[:H]
        elif df.ndim == 2:
            if df.shape[1] < H:
                raise ValueError(
                    f"discount_factors must have second dimension >= H; got {df.shape}."
                )
            df = df[:, :H]
        else:
            raise ValueError("discount_factors must be 1D or 2D.")
        if np.any(df <= 0.0) or not np.all(np.isfinite(df)):
            raise ValueError("discount_factors must be positive and finite.")
        return df

    # 2) Scenario-set discount factors
    if scen_set.discount_factors is not None:
        df = np.asarray(scen_set.discount_factors, dtype=float)
        if df.ndim == 1:
            if df.shape[0] < H:
                raise ValueError(
                    f"scen_set.discount_factors must have length >= H "
                    f"({df.shape[0]} vs {H})."
                )
            df = df[:H]
        elif df.ndim == 2:
            if df.shape[1] < H:
                raise ValueError(
                    f"scen_set.discount_factors must have shape (N, >=H); got {df.shape}."
                )
            df = df[:, :H]
        else:
            raise ValueError("scen_set.discount_factors must be 1D or 2D.")
        if np.any(df <= 0.0) or not np.all(np.isfinite(df)):
            raise ValueError("discount_factors must be positive and finite.")
        return df

    # 3) Flat short rate
    if short_rate is None:
        raise ValueError(
            "No discount factors available: provide either "
            "`discount_factors`, `scen_set.discount_factors` or `short_rate`."
        )

    r = float(short_rate)
    if not np.isfinite(r):
        raise ValueError("short_rate must be finite.")

    t = np.arange(1, H + 1, dtype=float)
    df = np.exp(-r * t)
    return df
