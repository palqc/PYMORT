import numpy as np

from pymort.analysis import MortalityScenarioSet


def find_nearest_age_index(ages: np.ndarray, target_age: float) -> int:
    """Return the index of the age in `ages` closest to `target_age`."""
    ages = np.asarray(ages, dtype=float)
    return int(np.argmin(np.abs(ages - float(target_age))))


def build_discount_factors(
    scen_set: MortalityScenarioSet,
    short_rate: float | None,
    discount_factors: np.ndarray | None,
    H: int,
) -> np.ndarray:
    """Determine discount factors for the pricing horizon.

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
                raise ValueError(f"discount_factors must have length >= {H}, got {df.shape[0]}.")
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
                    f"scen_set.discount_factors must have length >= H ({df.shape[0]} vs {H})."
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


def pv_from_cf_paths(cf_paths: np.ndarray, discount_factors: np.ndarray) -> np.ndarray:
    """Compute PV per scenario from cashflow paths and discount factors.

    Parameters
    ----------
    cf_paths : np.ndarray
        Cashflows by scenario and time, shape (N, H).
    discount_factors : np.ndarray
        Discount factors, shape (H,) or (1,H) or (N,H).

    Returns:
    -------
    np.ndarray
        PV per scenario, shape (N,).
    """
    cf = np.asarray(cf_paths, dtype=float)
    if cf.ndim != 2:
        raise ValueError(f"cf_paths must be 2D (N,H); got shape {cf.shape}.")

    N, H = cf.shape
    df = np.asarray(discount_factors, dtype=float)

    if df.ndim == 1:
        if df.shape[0] != H:
            raise ValueError(f"discount_factors length {df.shape[0]} must equal H={H}.")
        return (cf * df[None, :]).sum(axis=1)

    if df.ndim == 2:
        if df.shape[1] != H:
            raise ValueError(f"discount_factors second dim {df.shape[1]} must equal H={H}.")
        if df.shape[0] == 1:
            return (cf * np.repeat(df, N, axis=0)).sum(axis=1)
        if df.shape[0] != N:
            raise ValueError(f"discount_factors first dim must be 1 or N={N}; got {df.shape[0]}.")
        return (cf * df).sum(axis=1)

    raise ValueError("discount_factors must be 1D or 2D.")


def pv_matrix_from_cf_paths(cf_paths: np.ndarray, discount_factors: np.ndarray) -> np.ndarray:
    """Compute PV-by-horizon matrix from cashflow paths.

    pv[n,h] = sum_{t>=h} cf[n,t] * D[n,t]/D[n,h]

    Parameters
    ----------
    cf_paths : (N,T)
    discount_factors : (T,) or (1,T) or (N,T)

    Returns:
    -------
    pv : (N,T)
        PV at each horizon h for each scenario n.
    """
    cf = np.asarray(cf_paths, dtype=float)
    if cf.ndim != 2:
        raise ValueError(f"cf_paths must be 2D (N,T); got {cf.shape}.")
    n, t = cf.shape

    df = np.asarray(discount_factors, dtype=float)
    if df.ndim == 1:
        if df.shape[0] != t:
            raise ValueError(f"discount_factors length {df.shape[0]} must equal T={t}.")
        df = np.repeat(df[None, :], n, axis=0)
    elif df.ndim == 2:
        if df.shape[1] != t:
            raise ValueError(f"discount_factors second dim {df.shape[1]} must equal T={t}.")
        if df.shape[0] == 1:
            df = np.repeat(df, n, axis=0)
        elif df.shape[0] != n:
            raise ValueError(f"discount_factors first dim must be 1 or N={n}; got {df.shape[0]}.")
    else:
        raise ValueError("discount_factors must be 1D or 2D.")

    if not np.all(np.isfinite(df)) or np.any(df <= 0.0):
        raise ValueError("discount_factors must be positive and finite.")

    # discounted cashflows
    cfd = cf * df  # (N,T)

    # tail sums: sum_{t>=h} cf*D
    tail = np.cumsum(cfd[:, ::-1], axis=1)[:, ::-1]  # (N,T)

    # divide by D_h -> PV at horizon h
    pv = tail / df
    return pv


def cohort_survival_full_horizon_from_q(
    q_paths: np.ndarray,
    ages: np.ndarray,
    *,
    age0: float,
    horizon: int,
    age_fit_min: int = 80,
    age_fit_max: int = 95,
    m_floor: float = 1e-12,
) -> np.ndarray:
    q_paths = np.asarray(q_paths, dtype=float)  # (N,A,H)
    ages = np.asarray(ages, dtype=float)
    N, A, H = q_paths.shape
    Ht = min(int(horizon), H)

    a0_idx = int(np.argmin(np.abs(ages - float(age0))))
    age0_snap = float(ages[a0_idx])
    max_age = float(ages.max())

    # --- NEW: robust fit window under slicing ---
    age_fit_max_eff = min(float(age_fit_max), max_age)
    if age_fit_max_eff < float(age_fit_min):
        age_fit_min_eff = max(max_age - 15.0, float(ages.min()))
        age_fit_max_eff = max_age
    else:
        age_fit_min_eff = float(age_fit_min)

    fit_mask = (ages >= age_fit_min_eff) & (ages <= age_fit_max_eff)
    if not np.any(fit_mask):
        raise ValueError("Gompertz fit window not available in age grid.")
    x = ages[fit_mask]
    x_mean = float(x.mean())
    x_var = float(np.mean((x - x_mean) ** 2))
    if x_var <= 0:
        raise ValueError("Degenerate Gompertz fit window.")

    q_diag = np.empty((N, Ht), dtype=float)

    for k in range(Ht):
        age_k = age0_snap + k

        if age_k <= max_age and (a0_idx + k) < A:
            q_diag[:, k] = q_paths[:, a0_idx + k, k]
            continue

        q_fit_raw = q_paths[:, fit_mask, k]
        if not np.isfinite(q_fit_raw).all():
            q_fit_raw = np.nan_to_num(q_fit_raw, nan=0.0, posinf=1.0 - 1e-12, neginf=0.0)
        q_fit = np.clip(q_fit_raw, 0.0, 1.0 - 1e-12)

        m_fit = np.clip(-np.log(1.0 - q_fit), m_floor, None)
        y = np.log(m_fit)

        y_mean = np.mean(y, axis=1, keepdims=True)
        cov = np.mean((x[None, :] - x_mean) * (y - y_mean), axis=1)
        b = cov / x_var
        a = y_mean[:, 0] - b * x_mean

        m_k = np.exp(a + b * float(age_k))
        q_k = 1.0 - np.exp(-m_k)
        q_diag[:, k] = np.clip(q_k, 0.0, 1.0 - 1e-12)

    if not np.isfinite(q_diag).all():
        q_diag = np.nan_to_num(q_diag, nan=0.0, posinf=1.0 - 1e-12, neginf=0.0)
    q_diag = np.clip(q_diag, 0.0, 1.0 - 1e-12)

    S = np.cumprod(1.0 - q_diag, axis=1)
    return np.clip(S, 0.0, 1.0)
