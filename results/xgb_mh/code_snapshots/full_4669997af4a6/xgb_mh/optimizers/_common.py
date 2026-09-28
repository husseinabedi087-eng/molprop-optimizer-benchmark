"""Initial design and schedules shared by the optimizers."""
from __future__ import annotations

import numpy as np

N_INITIAL = 10
INITIAL_DESIGN_NOTE = f"common initial design ({N_INITIAL} points): first draw of default_rng(seed).random(({N_INITIAL}, d))"


def initial_design(rng: np.random.Generator, dim: int) -> np.ndarray:
    """The common initial design. Every optimizer calls this FIRST on a fresh default_rng(seed),
    so all five evaluate identical points in calls 1-10."""
    return rng.random((N_INITIAL, dim))


def linear_decay(update: int, n_updates: int, start: float = 2.0, end: float = 0.0) -> float:
    """Value for the update-th position update (0-based) of n_updates, from start to end inclusive."""
    if n_updates <= 1:
        return start
    return start + (end - start) * update / (n_updates - 1)


def iterations_for(budget: int, pop: int) -> int:
    if budget % pop:
        raise ValueError(f"budget {budget} must be divisible by population {pop}")
    return budget // pop
