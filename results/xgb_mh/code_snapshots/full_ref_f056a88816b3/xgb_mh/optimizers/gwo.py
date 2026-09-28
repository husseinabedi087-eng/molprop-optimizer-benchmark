"""Grey Wolf Optimizer (Mirjalili, Mirjalili & Lewis, 2014).

Each iteration: evaluate the whole pack, update alpha/beta/delta (best three positions found
so far), then move every wolf:
    A = 2a*r1 - a, C = 2*r2 (per leader, per dimension)
    D_l = |C * X_l - X|,  X_l' = X_l - A * D_l,  X_new = mean(X_alpha', X_beta', X_delta')
a decreases linearly 2 -> 0 over the position updates. Positions are clipped to [0,1].
"""
from __future__ import annotations

import numpy as np

from ._common import N_INITIAL, initial_design, iterations_for, linear_decay

POP = N_INITIAL


def run(objective, dim: int, budget: int, seed: int) -> dict:
    iterations = iterations_for(budget, POP)
    rng = np.random.default_rng(seed)
    X = initial_design(rng, dim)
    leaders = np.zeros((3, dim))
    leader_fit = np.full(3, np.inf)
    for t in range(iterations):
        fit = np.array([objective(x) for x in X])
        # Best three of {current leaders, pack}; stable sort keeps incumbents on ties.
        pool_x = np.vstack([leaders, X])
        pool_f = np.concatenate([leader_fit, fit])
        order = np.argsort(pool_f, kind="stable")[:3]
        leaders, leader_fit = pool_x[order].copy(), pool_f[order].copy()
        if t == iterations - 1:
            break
        a = linear_decay(t, iterations - 1)
        r1 = rng.random((POP, 3, dim))
        r2 = rng.random((POP, 3, dim))
        A = 2 * a * r1 - a
        C = 2 * r2
        D = np.abs(C * leaders[None] - X[:, None])
        X = np.clip((leaders[None] - A * D).mean(axis=1), 0.0, 1.0)
    return {"pop": POP, "iterations": iterations}
