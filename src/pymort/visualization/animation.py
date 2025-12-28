from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from pymort.analysis import MortalityScenarioSet


def animate_mortality_surface(
    scen_set: MortalityScenarioSet,
    *,
    value: Literal["q", "S"] = "q",
    statistic: Literal["mean", "median"] = "median",
    interval: int = 200,
    save_path: str | None = None,
    dpi: int = 100,
) -> animation.FuncAnimation:
    """Animate age×time mortality/survival surface over projection horizon.

    Parameters
    ----------
    value : {'q','S'}
        Surface to animate.
    statistic : {'mean','median'}
        Aggregation across scenarios per frame.
    interval : int
        Delay between frames (ms).
    save_path : str, optional
        If provided, save animation to this path (mp4/gif depending on extension).
    """
    if value == "q":
        data = np.asarray(scen_set.q_paths, dtype=float)
    else:
        data = np.asarray(scen_set.S_paths, dtype=float)
    agg = np.median if statistic == "median" else np.mean
    surf = agg(data, axis=0)  # (A, H)

    ages = np.asarray(scen_set.ages, dtype=float)
    years = np.asarray(scen_set.years, dtype=int)
    X, Y = np.meshgrid(years, ages)

    fig, ax = plt.subplots(figsize=(8, 5))
    vmin, vmax = float(surf.min()), float(surf.max())
    pcm = ax.pcolormesh(
        X[:, :1],
        Y[:, :1],
        surf[:, [0]],
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    fig.colorbar(pcm, ax=ax, label=value)
    ax.set_xlabel("Year")
    ax.set_ylabel("Age")

    def update(frame: int):
        pcm.set_array(surf[:, frame].ravel())
        ax.set_title(f"{value} {statistic} | year={years[frame]}")
        return (pcm,)

    anim = animation.FuncAnimation(fig, update, frames=surf.shape[1], interval=interval, blit=True)
    if save_path:
        anim.save(save_path, dpi=dpi)
    return anim


def animate_survival_curves(
    scen_set: MortalityScenarioSet,
    *,
    ages: Iterable[float] | None = None,
    statistic: Literal["mean", "median"] = "median",
    interval: int = 200,
    save_path: str | None = None,
    dpi: int = 100,
) -> animation.FuncAnimation:
    """Animate survival curves over calendar time for selected ages."""
    ages_grid = np.asarray(scen_set.ages, dtype=float)
    years = np.asarray(scen_set.years, dtype=int)
    if ages is None:
        ages_sel = [
            float(ages_grid[0]),
            float(ages_grid[len(ages_grid) // 2]),
            float(ages_grid[-1]),
        ]
    else:
        ages_sel = list(ages)
    idx = [int(np.argmin(np.abs(ages_grid - a))) for a in ages_sel]

    S = np.asarray(scen_set.S_paths, dtype=float)  # (N,A,H)
    agg = np.median if statistic == "median" else np.mean
    S_agg = agg(S, axis=0)  # (A,H)

    fig, ax = plt.subplots(figsize=(8, 5))
    lines = []
    for i in idx:
        (ln,) = ax.plot([], [], label=f"age≈{ages_grid[i]:.1f}")
        lines.append(ln)
    ax.set_xlim(years[0], years[-1])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Year")
    ax.set_ylabel("Survival probability")
    ax.legend()

    def update(frame: int):
        for ln, i in zip(lines, idx):
            ln.set_data(years[: frame + 1], S_agg[i, : frame + 1])
        ax.set_title(f"Survival curves up to year={years[frame]}")
        return lines

    anim = animation.FuncAnimation(fig, update, frames=S_agg.shape[1], interval=interval, blit=True)
    if save_path:
        anim.save(save_path, dpi=dpi)
    return anim
