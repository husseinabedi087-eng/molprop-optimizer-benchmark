"""Whale Optimization Algorithm, reference-faithful schedule (Mirjalili & Lewis, 2016; reference code), b = 1.

Identical to woa.py (label woa_a0) except for the coefficient schedules. As in the reference code,
computed BEFORE each position update, t = 0 .. T-1, T = iterations = 10:
    a  =  2 - t * (2 / T)   -> evaluated updates t = 0..8: 2.0, 1.8, ..., 0.4
    a2 = -1 + t * (-1 / T)  -> evaluated updates t = 0..8: -1.0, -1.1, ..., -1.8
The reference loop's final update (t = 9: a = 0.2, a2 = -1.9) is never evaluated, so it is not performed.
Evaluated positions never use a = 0 (which in woa_a0 sends every p < 0.5 whale exactly onto X*).

Otherwise as woa.py: scalar A, C, l, p per whale; random whale drawn per dimension (reference code);
synchronous update (documented deviation: the reference code updates in place); clipping to [0,1].
"""
from __future__ import annotations

import numpy as np

from ._common import N_INITIAL, initial_design, iterations_for

POP = N_INITIAL
B = 1.0


def reference_schedule(t: int, iterations: int) -> tuple[float, float]:
    return 2.0 - t * (2.0 / iterations), -1.0 + t * (-1.0 / iterations)


def run(objective, dim: int, budget: int, seed: int) -> dict:
    iterations = iterations_for(budget, POP)
    rng = np.random.default_rng(seed)
    X = initial_design(rng, dim)
    best_x, best_f = np.zeros(dim), np.inf
    branches = {"search": 0, "encircle": 0, "spiral": 0}
    a_used, a2_used = [], []
    leader_spiral_stays = 0
    for t in range(iterations):
        for x in X:
            f = objective(x)
            if f < best_f:
                best_x, best_f = x.copy(), f
        if t == iterations - 1:
            break
        a, a2 = reference_schedule(t, iterations)
        a_used.append(a)
        a2_used.append(a2)
        X_old = X.copy()
        for i in range(POP):
            r1, r2, p = rng.random(3)
            A, C = 2 * a * r1 - a, 2 * r2
            l = (a2 - 1) * rng.random() + 1
            if p < 0.5:
                if abs(A) >= 1:
                    x_rand = X_old[rng.integers(POP, size=dim), np.arange(dim)]  # per-dimension draw
                    X[i] = x_rand - A * np.abs(C * x_rand - X_old[i])
                    branches["search"] += 1
                else:
                    X[i] = best_x - A * np.abs(C * best_x - X_old[i])
                    branches["encircle"] += 1
            else:
                X[i] = np.abs(best_x - X_old[i]) * np.exp(B * l) * np.cos(2 * np.pi * l) + best_x
                branches["spiral"] += 1
                # Intrinsic to WOA (also in the reference code): the whale sitting exactly on X* has
                # D = 0, so the spiral leaves it on X* and the next evaluation repeats a known point.
                leader_spiral_stays += bool(np.array_equal(X_old[i], best_x))
        X = np.clip(X, 0.0, 1.0)
    return {"pop": POP, "iterations": iterations, "b": B, "branch_counts": branches,
            "random_whale": "per_dimension", "update": "synchronous (deviation: reference code is in-place)",
            "schedule": "reference: a = 2 - t*(2/T), a2 = -1 + t*(-1/T), t = 0..T-2 evaluated",
            "a_used": [round(v, 10) for v in a_used], "a2_used": [round(v, 10) for v in a2_used],
            "leader_spiral_stays": leader_spiral_stays}
