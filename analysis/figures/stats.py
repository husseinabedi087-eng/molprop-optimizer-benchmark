"""Statistics used by the figures."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata

BOOT_N, BOOT_SEED = 10_000, 20260927


def bootstrap_median_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """Median and percentile bootstrap CI (resampling seeds)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    meds = np.median(x[rng.integers(0, len(x), (BOOT_N, len(x)))], axis=1)
    lo, hi = np.percentile(meds, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(np.median(x)), float(lo), float(hi)


def a12(x: np.ndarray, y: np.ndarray) -> float:
    """Vargha-Delaney A12 on goodness values: P(X better than Y) + 0.5 P(tie)."""
    x, y = np.asarray(x), np.asarray(y)
    return float(((x[:, None] > y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y)))


def nogueira_stability(Z: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """Nogueira, Sechidis & Brown (2018, JMLR) stability of M selected subsets (Z: M x d binary),
    with the asymptotic confidence interval from the same paper (reference implementation)."""
    Z = np.asarray(Z, dtype=float)
    M, d = Z.shape
    p = Z.mean(axis=0)
    k = Z.sum(axis=1)
    kbar = p.sum()
    denom = (kbar / d) * (1 - kbar / d)
    stab = 1 - (M / (M - 1)) * np.mean(p * (1 - p)) / denom
    phi = (1 / denom) * ((Z * p).mean(axis=1) - k * kbar / d ** 2
                         + (stab / 2) * (2 * k * kbar / d ** 2 - k / d - kbar / d + 1))
    var = (4 / M ** 2) * np.sum((phi - phi.mean()) ** 2)
    half = norm.ppf(0.5 + level / 2) * np.sqrt(var)
    return float(stab), float(stab - half), float(stab + half)


# Studentized range q_0.05 / sqrt(2) for the Nemenyi test (Demsar 2006, Table 5a), k = 2..10.
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def average_ranks(goodness: np.ndarray) -> np.ndarray:
    """goodness: N datasets x k methods (higher = better). Rank 1 = best; ties averaged."""
    return np.mean([rankdata(-row) for row in goodness], axis=0)


def nemenyi_cd(k: int, n: int) -> float:
    return NEMENYI_Q05[k] * np.sqrt(k * (k + 1) / (6 * n))
