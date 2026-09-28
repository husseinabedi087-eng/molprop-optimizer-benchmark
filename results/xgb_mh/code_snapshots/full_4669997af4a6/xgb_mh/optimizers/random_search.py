"""Random Search: iid uniform samples in [0,1]^d (the first 10 are the common initial design)."""
from __future__ import annotations

import numpy as np

from ._common import initial_design


def run(objective, dim: int, budget: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    design = initial_design(rng, dim)[:budget]
    for x in design:
        objective(x)
    for _ in range(budget - len(design)):
        objective(rng.random(dim))
    return {}
