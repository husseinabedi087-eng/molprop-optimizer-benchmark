"""Particle Swarm Optimization (global best), w = 0.7, c1 = c2 = 1.5.

    v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x),  v clipped to [-0.2, 0.2]
    x = clip(x + v, 0, 1)
Velocities start uniform in [-0.2, 0.2]; r1, r2 are per particle, per dimension.
"""
from __future__ import annotations

import numpy as np

from ._common import N_INITIAL, initial_design, iterations_for

POP = N_INITIAL
W, C1, C2 = 0.7, 1.5, 1.5
V_MAX = 0.2


def run(objective, dim: int, budget: int, seed: int) -> dict:
    iterations = iterations_for(budget, POP)
    rng = np.random.default_rng(seed)
    X = initial_design(rng, dim)
    V = rng.uniform(-V_MAX, V_MAX, (POP, dim))
    pbest, pbest_f = X.copy(), np.full(POP, np.inf)
    for t in range(iterations):
        fit = np.array([objective(x) for x in X])
        improved = fit < pbest_f
        pbest[improved], pbest_f[improved] = X[improved], fit[improved]
        if t == iterations - 1:
            break
        gbest = pbest[np.argmin(pbest_f)]
        r1, r2 = rng.random((POP, dim)), rng.random((POP, dim))
        V = np.clip(W * V + C1 * r1 * (pbest - X) + C2 * r2 * (gbest - X), -V_MAX, V_MAX)
        X = np.clip(X + V, 0.0, 1.0)
    return {"pop": POP, "iterations": iterations, "w": W, "c1": C1, "c2": C2, "v_max": V_MAX}
