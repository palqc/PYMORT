from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymort.lifetables import validate_q
from pymort.models.utils import _estimate_rw_params


@dataclass
class CBDM5Params:
    """Parameters of the basic two–factor CBD model:

        logit(q_{x,t}) = kappa1_t + kappa2_t * (x - x_bar)

    kappa1_t controls the level, kappa2_t the age–slope.
    """

    kappa1: np.ndarray  # (T,)
    kappa2: np.ndarray  # (T,)
    ages: np.ndarray  # (A,)
    x_bar: float  # mean age used for centering

    # optional RW+drift parameters
    mu1: float | None = None
    sigma1: float | None = None
    mu2: float | None = None
    sigma2: float | None = None


def _logit(p: np.ndarray) -> np.ndarray:
    """Numerically stable logit transform log(p / (1 - p))."""
    p_clipped = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p_clipped / (1.0 - p_clipped))


def _build_cbd_design(ages: np.ndarray) -> tuple[np.ndarray, float]:
    """Build the CBD design matrix X = [1, x - x_bar] for given ages.

    Parameters
    ----------
    ages : np.ndarray
        1D array of ages (A,).

    Returns:
    -------
    X : np.ndarray
        Design matrix of shape (A, 2) with columns [1, x - x_bar].
    x_bar : float
        Mean age used for centering.
    """
    if ages.ndim != 1:
        raise ValueError("ages must be a 1D array.")
    x_bar = float(ages.mean())
    z = ages - x_bar
    X = np.column_stack([np.ones_like(z, dtype=float), z.astype(float)])  # (A, 2)
    return X, x_bar


def fit_cbd(q: np.ndarray, ages: np.ndarray) -> CBDM5Params:
    """Fit the Cairns–Blake–Dowd (CBD) model to a mortality surface q[age, year].
    The model is only appropriate for higher ages (e.g. 60+); users should filter ages before calling this function.

    We assume:
        logit(q_{x,t}) = kappa1_t + kappa2_t * (x - x_bar)

    Steps:
      1) Build design matrix X = [1, (x - x_bar)].
      2) For each year t, regress logit(q_{x,t}) on X to get kappa1_t, kappa2_t.
    """
    q = np.asarray(q, dtype=float)
    ages = np.asarray(ages, dtype=float)
    if q.ndim != 2:
        raise ValueError("q must be a 2D array with shape (A, T).")
    if ages.ndim != 1:
        raise ValueError("ages must be 1D.")
    A, T = q.shape
    if ages.shape[0] != A:
        raise ValueError("ages length must match q.shape[0].")
    if not np.isfinite(q).all():
        raise ValueError("q must contain finite values.")
    if (q <= 0).any() or (q >= 1).any():
        raise ValueError("q must be strictly in (0,1).")
    if not np.isfinite(ages).all():
        raise ValueError("ages must be finite.")
    validate_q(q)

    # Build design matrix once (same ages for all years)
    X, x_bar = _build_cbd_design(ages)  # X: (A, 2)
    XtX = X.T @ X

    # transform q to logits
    y = _logit(q)  # (A, T)
    Xty = X.T @ y
    try:
        beta_hat_all = np.linalg.solve(XtX, Xty)  # (2, T)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("X'X is singular in CBD fit.") from exc

    kappa1 = beta_hat_all[0, :]  # (T,)
    kappa2 = beta_hat_all[1, :]  # (T,)

    return CBDM5Params(
        kappa1=kappa1,
        kappa2=kappa2,
        ages=ages,
        x_bar=x_bar,
    )


def reconstruct_logit_q(params: CBDM5Params) -> np.ndarray:
    """Reconstruct the logit mortality surface logit(q_{x,t}) from CBD parameters.
    Returns an array of shape (A, T).
    """
    k1 = params.kappa1  # (T,)
    k2 = params.kappa2  # (T,)
    ages = params.ages  # (A,)
    z = ages - params.x_bar  # (A,)

    # logit_q[x,t] = kappa1_t + kappa2_t * (x - x_bar)
    # → (A, T) = (1,T) + (A,1)*(1,T)
    return k1[None, :] + z[:, None] * k2[None, :]


def reconstruct_q(params: CBDM5Params) -> np.ndarray:
    """Reconstruct mortality probabilities q_{x,t} from CBD parameters.
    Returns an array with shape (A, T).
    """
    logit_q = reconstruct_logit_q(params)
    return 1.0 / (1.0 + np.exp(-logit_q))


def estimate_rw_params_cbd(params: CBDM5Params) -> CBDM5Params:
    """Estimate RW+drift parameters for (kappa1_t, kappa2_t) and store them
    in the CBDParams object.
    """
    mu1, sigma1 = _estimate_rw_params(params.kappa1)
    mu2, sigma2 = _estimate_rw_params(params.kappa2)
    params.mu1, params.sigma1 = mu1, sigma1
    params.mu2, params.sigma2 = mu2, sigma2
    return params


class CBDM5:
    def __init__(self) -> None:
        self.params: CBDM5Params | None = None

    def fit(self, q: np.ndarray, ages: np.ndarray) -> CBDM5:
        """Fit the CBD model on a mortality surface q[age, year]
        and store the resulting parameters in self.params.
        """
        self.params = fit_cbd(q, ages)
        return self

    def estimate_rw(self) -> tuple[float, float, float, float]:
        """Estimate RW+drift parameters for (kappa1_t, kappa2_t)
        and store them in self.params.
        """
        if self.params is None:
            raise ValueError("Fit the model first.")
        self.params = estimate_rw_params_cbd(self.params)
        return (
            self.params.mu1,
            self.params.sigma1,
            self.params.mu2,
            self.params.sigma2,
        )

    def predict_logit_q(self) -> np.ndarray:
        """Reconstruct the logit-mortality surface from fitted CBD parameters."""
        if self.params is None:
            raise ValueError("Fit the model first.")
        return reconstruct_logit_q(self.params)

    def predict_q(self) -> np.ndarray:
        """Reconstruct mortality probabilities q_{x,t} from fitted CBD parameters."""
        if self.params is None:
            raise ValueError("Fit the model first.")
        return reconstruct_q(self.params)

    def simulate_kappa(
        self,
        kappa_index: str,
        horizon: int,
        n_sims: int = 1000,
        seed: int | None = None,
        include_last: bool = False,
    ) -> np.ndarray:
        """Simulate RW forecasts for kappa1_t or kappa2_t using the fitted drift/vol."""
        if self.params is None:
            raise ValueError("Fit the model first.")

        if kappa_index == "kappa1":
            k_last = float(self.params.kappa1[-1])
            mu = float(self.params.mu1)
            sigma = float(self.params.sigma1)
        elif kappa_index == "kappa2":
            k_last = float(self.params.kappa2[-1])
            mu = float(self.params.mu2)
            sigma = float(self.params.sigma2)
        else:
            raise ValueError("kappa_index must be 'kappa1' or 'kappa2'.")

        horizon = int(horizon)
        n_sims = int(n_sims)
        if horizon <= 0 or n_sims <= 0:
            raise ValueError("horizon and n_sims must be positive integers.")

        rng = np.random.default_rng(seed)

        from pymort.analysis.projections import simulate_random_walk_paths

        paths = simulate_random_walk_paths(
            k_last=k_last,
            mu=mu,
            sigma=sigma,
            horizon=horizon,
            n_sims=n_sims,
            rng=rng,
            include_last=include_last,
        )

        return paths
