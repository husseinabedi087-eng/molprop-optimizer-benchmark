"""Budgeted, cached objective shared by all optimizers.

Every call counts toward the budget, including cache hits. Exceeding the budget raises.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable

import numpy as np


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Call:
    call_idx: int
    x: np.ndarray
    fitness: float
    best_so_far: float
    cache_hit: bool
    info: dict
    wall_time: float  # time inside this call (fitness evaluation, ~0 on a cache hit)
    overhead: float   # optimizer/sampler time since the previous call returned (or since creation)


@dataclass
class Objective:
    """fn(x) -> (fitness, info). key(x) -> hashable cache key (default: the raw vector)."""
    fn: Callable[[np.ndarray], tuple[float, dict]]
    budget: int
    key: Callable[[np.ndarray], Hashable] | None = None
    on_call: Callable[[Call], Any] | None = None
    calls: list[Call] = field(default_factory=list)
    _cache: dict = field(default_factory=dict)
    _last_return: float = field(default_factory=time.perf_counter)

    def __call__(self, x: np.ndarray) -> float:
        start = time.perf_counter()
        if len(self.calls) >= self.budget:
            raise BudgetExceeded(f"budget of {self.budget} fitness calls exhausted")
        x = np.array(x, dtype=float)
        k = self.key(x) if self.key else x.tobytes()
        hit = k in self._cache
        if not hit:
            self._cache[k] = self.fn(x)
        fitness, info = self._cache[k]
        best = min(fitness, self.calls[-1].best_so_far) if self.calls else fitness
        call = Call(len(self.calls), x, float(fitness), float(best), hit, info,
                    time.perf_counter() - start, start - self._last_return)
        self.calls.append(call)
        if self.on_call:
            self.on_call(call)
        self._last_return = time.perf_counter()
        return float(fitness)

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def n_unique(self) -> int:
        return len(self._cache)

    def best(self) -> Call:
        return min(self.calls, key=lambda c: c.fitness)  # first occurrence on ties
