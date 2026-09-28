"""Sphere sanity check: centered vs shifted optimum, d in {10, 200}, budget 100, seeds 0-29.

Center bias: random search is unbiased by construction, so an optimizer's shifted/centered ratio
of median final best fitness is compared with random search's ratio. A relative ratio > 2 is flagged.
Usage: python -m xgb_mh.sanity_sphere
"""
from __future__ import annotations

import csv

import numpy as np
from joblib import Parallel, delayed

from . import RESULTS_DIR
from .objective import Objective
from .optimizers import OPTIMIZERS, run

BUDGET, SEEDS, DIMS = 100, range(30), (10, 200)
SHIFT_SEED = 12345
BIAS_FLAG = 2.0


def optimum(sphere: str, dim: int) -> np.ndarray:
    if sphere == "centered":
        return np.full(dim, 0.5)
    return np.random.default_rng(SHIFT_SEED).uniform(0.1, 0.9, dim)


def one_run(name: str, sphere: str, dim: int, seed: int) -> dict:
    c = optimum(sphere, dim)
    obj = Objective(lambda x: (float(np.sum((x - c) ** 2)), {}), BUDGET)
    run(name, obj, dim, seed)
    return {"optimizer": name, "sphere": sphere, "dim": dim, "seed": seed,
            "best": obj.calls[-1].best_so_far, "unique": obj.n_unique}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizers", default=",".join(OPTIMIZERS),
                        help="rerun only these; the other rows are read back from sphere_runs.csv")
    names = parser.parse_args().optimizers.split(",")
    runs_path = RESULTS_DIR / "sanity" / "sphere_runs.csv"
    rows = []
    if set(names) != set(OPTIMIZERS):
        with runs_path.open() as handle:
            rows = [{"optimizer": r["optimizer"], "sphere": r["sphere"], "dim": int(r["dim"]), "seed": int(r["seed"]),
                     "best": float(r["best"]), "unique": int(r["unique"])}
                    for r in csv.DictReader(handle) if r["optimizer"] not in names]
    jobs = [(n, s, d, k) for d in DIMS for s in ("centered", "shifted") for n in names for k in SEEDS]
    rows += Parallel(n_jobs=10)(delayed(one_run)(*j) for j in jobs)
    med = {}
    for d in DIMS:
        for s in ("centered", "shifted"):
            for n in OPTIMIZERS:
                vals = np.array([r["best"] for r in rows if (r["optimizer"], r["sphere"], r["dim"]) == (n, s, d)])
                med[n, s, d] = (np.median(vals), *np.percentile(vals, [25, 75]))

    out = []
    for n in OPTIMIZERS:
        row = {"optimizer": n}
        for d in DIMS:
            c, s = med[n, "centered", d][0], med[n, "shifted", d][0]
            ratio = s / c
            rel = ratio / (med["random", "shifted", d][0] / med["random", "centered", d][0])
            row.update({f"d{d}_centered_median": c, f"d{d}_centered_iqr": f"{med[n, 'centered', d][1]:.4g}-{med[n, 'centered', d][2]:.4g}",
                        f"d{d}_shifted_median": s, f"d{d}_shifted_iqr": f"{med[n, 'shifted', d][1]:.4g}-{med[n, 'shifted', d][2]:.4g}",
                        f"d{d}_shift_over_center": ratio, f"d{d}_ratio_vs_random": rel})
        row["center_bias_flag"] = any(row[f"d{d}_ratio_vs_random"] > BIAS_FLAG for d in DIMS)
        out.append(row)

    for d in DIMS:  # reference: expected fitness of one uniform random point
        c = optimum("shifted", d)
        out.append({"optimizer": f"reference_uniform_point_d{d}",
                    f"d{d}_centered_median": d / 12, f"d{d}_shifted_median": d / 12 + float(np.sum((c - 0.5) ** 2))})

    path = RESULTS_DIR / "sanity" / "sphere_tests.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(out[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    with runs_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in out:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
