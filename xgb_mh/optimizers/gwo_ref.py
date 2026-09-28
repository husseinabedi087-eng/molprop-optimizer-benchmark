"""Grey Wolf Optimizer, reference-faithful schedule (Mirjalili et al., 2014; reference MATLAB code).

Identical to gwo.py (label gwo_a0) except for the coefficient schedule. As in the reference code,
a = 2 - t * (2 / T) is computed BEFORE each position update, t = 0 .. T-1, T = iterations = 10.
With 10 evaluation rounds (initial pack = round 1) there are 9 evaluated updates, t = 0 .. 8:
    a = 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4
The reference loop's final update (t = 9, a = 0.2) is never evaluated, so it is not performed.
Evaluated positions therefore never use a = 0 (which in gwo_a0 collapses the pack to one point).

Otherwise as gwo.py: A = 2a*r1 - a, C = 2*r2 per leader and dimension; X_new = mean of the three
leader-guided moves; alpha/beta/delta = best three positions found so far; clipping to [0,1].
"""
from __future__ import annotations

import numpy as np

from ._common import N_INITIAL, initial_design, iterations_for

POP = N_INITIAL


def reference_a(t: int, iterations: int) -> float:
    """Mirjalili reference schedule: a = 2 - t * (2 / T)."""
    return 2.0 - t * (2.0 / iterations)


def run(objective, dim: int, budget: int, seed: int) -> dict:
    iterations = iterations_for(budget, POP)
    rng = np.random.default_rng(seed)
    X = initial_design(rng, dim)
    leaders = np.zeros((3, dim))
    leader_fit = np.full(3, np.inf)
    a_used = []
    for t in range(iterations):
        fit = np.array([objective(x) for x in X])
        pool_x = np.vstack([leaders, X])
        pool_f = np.concatenate([leader_fit, fit])
        order = np.argsort(pool_f, kind="stable")[:3]
        leaders, leader_fit = pool_x[order].copy(), pool_f[order].copy()
        if t == iterations - 1:
            break
        a = reference_a(t, iterations)
        a_used.append(a)
        r1 = rng.random((POP, 3, dim))
        r2 = rng.random((POP, 3, dim))
        A = 2 * a * r1 - a
        C = 2 * r2
        D = np.abs(C * leaders[None] - X[:, None])
        X = np.clip((leaders[None] - A * D).mean(axis=1), 0.0, 1.0)
    return {"pop": POP, "iterations": iterations, "schedule": "reference: a = 2 - t*(2/T), t = 0..T-2 evaluated",
            "a_used": [round(v, 10) for v in a_used]}
