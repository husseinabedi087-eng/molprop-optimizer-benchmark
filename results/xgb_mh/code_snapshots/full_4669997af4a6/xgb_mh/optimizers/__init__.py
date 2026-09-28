"""Optimizers over [0,1]^d. Each run(objective, dim, budget, seed) spends exactly `budget` calls.

Population methods (GWO, WOA, PSO) use pop=10 and budget/pop iterations; the initial population
is iteration 1, so there are (iterations - 1) position updates. Coefficients that decrease
"2 -> 0" do so linearly over those updates: 2 on the first update, exactly 0 on the last one.
"""
from __future__ import annotations

import numpy as np

from ..objective import Objective
from . import gwo, pso, random_search, tpe, woa
from ._common import INITIAL_DESIGN_NOTE, N_INITIAL, initial_design

OPTIMIZERS = {
    "random": random_search.run,
    "tpe": tpe.run,
    "gwo": gwo.run,
    "woa": woa.run,
    "pso": pso.run,
}


def run(name: str, objective: Objective, dim: int, seed: int) -> dict:
    """Run one optimizer and assert that it spent exactly the budget."""
    meta = OPTIMIZERS[name](objective, dim, objective.budget, seed) or {}
    if objective.n_calls != objective.budget:
        raise AssertionError(f"{name} made {objective.n_calls} fitness calls, budget is {objective.budget}")
    first = np.array([c.x for c in objective.calls[:N_INITIAL]])
    if not np.array_equal(first, initial_design(np.random.default_rng(seed), dim)[:objective.budget]):
        raise AssertionError(f"{name} did not start from the common initial design")
    return {"initial_design": INITIAL_DESIGN_NOTE, **meta}
