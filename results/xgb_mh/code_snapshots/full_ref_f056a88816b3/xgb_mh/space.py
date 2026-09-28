"""Unit-hypercube search space and the ONE shared decode() used by every optimizer.

x[:8] are XGBoost hyperparameters; in Experiment B, x[8:8+F] are feature-selection genes.
Mappings (all exact at x = 0 and x = 1):
  int:    floor(lo + x * (hi - lo + 1)), capped at hi  -> equal-width bins, both ends reachable
  linear: lo * (1 - x) + hi * x
  log:    lo ** (1 - x) * hi ** x                       -> geometric interpolation
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

# name, kind, low, high
HP_SPEC: tuple[tuple[str, str, float, float], ...] = (
    ("n_estimators_max", "int", 100, 1000),
    ("max_depth", "int", 3, 10),
    ("learning_rate", "log", 0.01, 0.3),
    ("subsample", "linear", 0.5, 1.0),
    ("colsample_bytree", "linear", 0.3, 1.0),
    ("min_child_weight", "log", 1.0, 10.0),
    ("reg_lambda", "log", 1e-3, 10.0),
    ("reg_alpha", "log", 1e-3, 10.0),
)
N_HP = len(HP_SPEC)
FS_THRESHOLD = 0.5


def dimension(n_features: int | None) -> int:
    """Search dimension: 8 for Experiment A (n_features=None), 8 + F for Experiment B."""
    return N_HP + (n_features or 0)


def _map(kind: str, lo: float, hi: float, x: float):
    if kind == "int":
        return int(min(hi, math.floor(lo + x * (hi - lo + 1))))
    if kind == "linear":
        return float(lo * (1 - x) + hi * x)
    if kind == "log":
        return float(lo ** (1 - x) * hi ** x)
    raise ValueError(kind)


@dataclass(frozen=True)
class Config:
    params: dict
    selected: tuple[int, ...] | None  # None = all features (Experiment A)

    @property
    def key(self) -> str:
        """Cache key: the full decoded configuration."""
        return json.dumps({"params": self.params, "selected": self.selected}, sort_keys=True)

    def n_selected(self, n_features: int) -> int:
        return n_features if self.selected is None else len(self.selected)


def select_features(genes: np.ndarray) -> tuple[int, ...]:
    """Feature j is selected if gene_j > 0.5; if none, the single argmax gene is selected."""
    chosen = np.flatnonzero(genes > FS_THRESHOLD)
    if chosen.size == 0:
        chosen = np.array([int(np.argmax(genes))])
    return tuple(int(j) for j in chosen)


def mask_to_hex(selected: tuple[int, ...] | None, n_features: int) -> str | None:
    """Compact log form of a feature subset: bit j set <=> feature j selected."""
    if selected is None:
        return None
    return format(sum(1 << j for j in selected), f"0{(n_features + 3) // 4}x")


def hex_to_mask(value: str, n_features: int) -> np.ndarray:
    bits = int(value, 16)
    return np.array([(bits >> j) & 1 for j in range(n_features)], dtype=bool)


def decode(x: np.ndarray, n_features: int | None = None) -> Config:
    x = np.asarray(x, dtype=float)
    if x.shape != (dimension(n_features),) or not np.isfinite(x).all():
        raise ValueError(f"expected a finite vector of length {dimension(n_features)}, got {x.shape}")
    x = np.clip(x, 0.0, 1.0)
    params = {name: _map(kind, lo, hi, float(v)) for (name, kind, lo, hi), v in zip(HP_SPEC, x[:N_HP])}
    selected = select_features(x[N_HP:]) if n_features else None
    return Config(params, selected)
