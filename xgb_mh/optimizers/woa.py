"""Whale Optimization Algorithm (Mirjalili & Lewis, 2016), b = 1.

Per whale per update (scalars r1, r2, p, l as in the reference implementation):
    A = 2a*r1 - a, C = 2*r2, l ~ U[a2, 1] with a2 decreasing -1 -> -2
    p < 0.5 and |A| >= 1:  X_new = X_rand - A * |C * X_rand - X|      (search, random whale)
    p < 0.5 and |A| <  1:  X_new = X* - A * |C * X* - X|              (encircling the best)
    p >= 0.5:              X_new = |X* - X| * e^(b l) * cos(2 pi l) + X*  (spiral)
X* is the best position found so far. As in the reference MATLAB code, the random whale is drawn
independently for each dimension. a decreases linearly 2 -> 0. Positions are clipped to [0,1].

Documented deviation from the reference code: updates are synchronous (X_rand and X come from the
positions at the start of the update), whereas the reference code updates positions in place.
"""
from __future__ import annotations

import numpy as np

from ._common import N_INITIAL, initial_design, iterations_for, linear_decay

POP = N_INITIAL
B = 1.0


def run(objective, dim: int, budget: int, seed: int) -> dict:
    iterations = iterations_for(budget, POP)
    rng = np.random.default_rng(seed)
    X = initial_design(rng, dim)
    best_x, best_f = np.zeros(dim), np.inf
    branches = {"search": 0, "encircle": 0, "spiral": 0}
    for t in range(iterations):
        for x in X:
            f = objective(x)
            if f < best_f:
                best_x, best_f = x.copy(), f
        if t == iterations - 1:
            break
        a = linear_decay(t, iterations - 1)
        a2 = linear_decay(t, iterations - 1, start=-1.0, end=-2.0)
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
        X = np.clip(X, 0.0, 1.0)
    return {"pop": POP, "iterations": iterations, "b": B, "branch_counts": branches,
            "random_whale": "per_dimension", "update": "synchronous (deviation: reference code is in-place)"}
