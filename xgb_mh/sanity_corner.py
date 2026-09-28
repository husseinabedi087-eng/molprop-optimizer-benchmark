"""Corner-bias sphere check: optimum at 0.2 vs 0.8 in every dimension, d in {10, 200}, budget 100, seeds 0-29.

Reproduces results/xgb_mh/sanity/sphere_corner_bias.csv (originally produced by equivalent inline code during
development). A ratio (0.8 / 0.2) far from Random's indicates a bias toward x = 0 (ratio > 1) or x = 1 (< 1).
Usage: python -m xgb_mh.sanity_corner [--out PATH]
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
from joblib import Parallel, delayed

from . import RESULTS_DIR
from .objective import Objective
from .optimizers import OPTIMIZERS, run

LEVELS, DIMS, SEEDS, BUDGET = (0.2, 0.8), (10, 200), range(30), 100


def one(name: str, level: float, dim: int, seed: int) -> tuple:
    c = np.full(dim, level)
    obj = Objective(lambda x: (float(np.sum((x - c) ** 2)), {}), BUDGET)
    run(name, obj, dim, seed)
    return name, level, dim, obj.calls[-1].best_so_far


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RESULTS_DIR / "sanity" / "sphere_corner_bias.csv"))
    out = parser.parse_args().out
    res = Parallel(n_jobs=10)(delayed(one)(n, l, d, s) for n in OPTIMIZERS for l in LEVELS for d in DIMS for s in SEEDS)
    rows = []
    for n in OPTIMIZERS:
        r = {"optimizer": n}
        for d in DIMS:
            lo = np.median([b for m, l, dd, b in res if (m, l, dd) == (n, 0.2, d)])
            hi = np.median([b for m, l, dd, b in res if (m, l, dd) == (n, 0.8, d)])
            r.update({f"d{d}_opt_at_0.2": round(lo, 4), f"d{d}_opt_at_0.8": round(hi, 4),
                      f"d{d}_ratio_0.8_over_0.2": round(hi / lo, 3)})
        rows.append(r)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
