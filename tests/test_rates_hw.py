from __future__ import annotations

import numpy as np

from pymort.interest_rates.hull_white import build_interest_rate_scenarios


def test_hw_simulation_shapes():
    times = np.arange(1, 6, dtype=float)
    zero = np.full_like(times, 0.02, dtype=float)
    scen = build_interest_rate_scenarios(
        times=times,
        zero_rates=zero,
        a=0.1,
        sigma=0.01,
        n_scenarios=5,
        seed=123,
    )
    assert scen.r_paths.shape == (5, len(times))
    assert scen.discount_factors.shape == (5, len(times))
    assert np.all(scen.discount_factors > 0.0)
