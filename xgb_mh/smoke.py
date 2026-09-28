"""Step 4 smoke test: ESOL + BBBP, seeds 0-1, budget 20, experiments A and B, all optimizers + baseline.

Also: poisoned-test replay (search must be identical when test rows are garbage), static check of the
objective's inputs, a validation-only timing probe on the other three datasets (no refit, no test),
sampler overhead at the full 100-call budget, and a runtime estimate for the full study.
Usage: python -m xgb_mh.smoke
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import DATASETS, RESULTS_DIR
from .experiment import EXPERIMENTS, SearchData, _peak_rss_mb, load, make_fitness, run_baseline, run_search
from .objective import Objective
from .optimizers import OPTIMIZERS, run as run_optimizer
from .space import decode

OUT = RESULTS_DIR / "smoke"
SMOKE_DATASETS, SEEDS, BUDGET = ("ESOL", "BBBP"), (0, 1), 20
PROBE_DATASETS = tuple(d for d in DATASETS if d not in SMOKE_DATASETS)
FULL_SEEDS, FULL_BUDGET, N_JOBS = 30, 100, 10


def probe(dataset: str, experiment: str, seed: int) -> dict:
    """Validation-only timing: random search, no refit, no test access."""
    data, _ = load(dataset)
    fitness, dim, nf = make_fitness(data, experiment, seed)
    obj = Objective(fitness, BUDGET, key=lambda x: decode(x, nf).key)
    run_optimizer("random", obj, dim, seed)
    fits = [c.info["fit_time"] for c in obj.calls if not c.cache_hit]
    return {"dataset": dataset, "experiment": experiment, "seed": seed, "fit_times": fits,
            "n_trees": [c.info["n_trees"] for c in obj.calls], "peak_rss_mb": _peak_rss_mb()}


def overhead(optimizer: str, dim: int, seed: int) -> dict:
    """Optimizer/sampler time per call at the full budget with a free objective."""
    obj = Objective(lambda x: (float(np.sum((x - 0.5) ** 2)), {}), FULL_BUDGET)
    run_optimizer(optimizer, obj, dim, seed)
    return {"optimizer": optimizer, "dim": dim, "seed": seed, "per_call": [c.overhead for c in obj.calls]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # --- static check: the objective can only see SearchData (train/val) ---
    fields = [f.name for f in dataclasses.fields(SearchData)]
    params = list(inspect.signature(make_fitness).parameters)
    assert not any("test" in f for f in fields), fields
    assert params == ["data", "experiment", "seed"], params
    static = {"SearchData_fields": fields, "make_fitness_params": params}

    dims = {d: 8 + load(d)[0].n_features for d in DATASETS}
    jobs = [delayed(run_search)(d, e, o, s, BUDGET, OUT) for d in SMOKE_DATASETS for e in EXPERIMENTS
            for o in OPTIMIZERS for s in SEEDS]
    jobs += [delayed(probe)(d, e, s) for d in PROBE_DATASETS for e in EXPERIMENTS for s in SEEDS]
    jobs += [delayed(overhead)(o, dim, 0) for o in OPTIMIZERS for dim in sorted({8, *dims.values()})]
    t0 = time.perf_counter()
    out = Parallel(n_jobs=N_JOBS)(jobs)
    parallel_wall = time.perf_counter() - t0
    runs = pd.DataFrame([r for r in out if "optimizer" in r and "per_call" not in r])
    probes = [r for r in out if "fit_times" in r]
    overheads = [r for r in out if "per_call" in r]

    # --- baseline (twice: must be deterministic) ---
    base = []
    for d in SMOKE_DATASETS:
        b1, b2 = run_baseline(d), run_baseline(d)
        same = all(b1[k] == b2[k] for k in b1 if k.startswith("test_") or k == "best_val_raw_metric")
        assert same, (b1, b2)
        base.append({**b1, "deterministic": same})
    base = pd.DataFrame(base)

    # --- poisoned-test replay: search must be bit-identical when test rows are garbage ---
    replay = []
    for d, e, o, s in [("ESOL", "B", "gwo", 0), ("ESOL", "A", "tpe", 1), ("BBBP", "B", "woa", 0),
                       ("BBBP", "A", "pso", 1), ("BBBP", "B", "random", 0)]:
        clean = runs[(runs.dataset == d) & (runs.experiment == e) & (runs.optimizer == o) & (runs.seed == s)].iloc[0]
        try:
            poisoned = run_search(d, e, o, s, BUDGET, OUT, poison_test=True, write=False)
            traj_same = poisoned["_fitness_trajectory"] == clean["_fitness_trajectory"]
            test_after = "computed (garbage)"
        except ValueError as err:  # e.g. ROC-AUC on the all -999 poisoned labels: fails only AFTER search
            traj_same, test_after = None, f"refit/test step raised as expected: {err.__class__.__name__}"
            # re-run the search alone to compare the trajectory
            data, _ = load(d, True)
            fitness, dim, nf = make_fitness(data, e, s)
            obj = Objective(fitness, BUDGET, key=lambda x, nf=nf: decode(x, nf).key)
            run_optimizer(o, obj, dim, s)
            traj_same = [c.fitness for c in obj.calls] == clean["_fitness_trajectory"]
        assert traj_same, (d, e, o, s)
        replay.append({"run": f"{d}_{e}_{o}_s{s}", "trajectory_identical_with_poisoned_test": traj_same,
                       "test_step": test_after})
    assert (runs.refit_n_trees == runs.best_iteration + 1).all()

    # --- per-call timing ---
    calls = pd.concat([pd.read_json(OUT / "calls" / f"{r.dataset}_{r.experiment}_{r.optimizer}_s{r.seed}.jsonl", lines=True)
                       for r in runs.itertuples()])
    fresh = calls[~calls.cache_hit]
    fit_ds = fresh.groupby(["dataset", "experiment"]).fit_time.agg(["mean", "median", lambda s: s.quantile(.9), "count"])
    fit_ds.columns = ["fit_mean_s", "fit_median_s", "fit_p90_s", "n_fresh_calls"]
    pool: dict[tuple, list] = {}
    for p in probes:
        pool.setdefault((p["dataset"], p["experiment"]), []).extend(p["fit_times"])
    probe_df = pd.DataFrame([{"dataset": d, "experiment": e, "fit_mean_s": np.mean(t), "fit_median_s": np.median(t),
                              "fit_p90_s": np.quantile(t, .9), "n_fresh_calls": len(t)} for (d, e), t in pool.items()])
    fit_ds = pd.concat([fit_ds, probe_df.set_index(["dataset", "experiment"])])
    fit_ds["source"] = ["smoke" if d in SMOKE_DATASETS else "probe (random search, val only)" for d, _ in fit_ds.index]

    ov = pd.DataFrame([{"optimizer": r["optimizer"], "dim": r["dim"], "total_s": sum(r["per_call"]),
                        "per_call_ms": 1e3 * np.mean(r["per_call"]), "last10_per_call_ms": 1e3 * np.mean(r["per_call"][-10:])}
                       for r in overheads])
    smoke_overhead = calls.groupby(["optimizer", "dataset", "experiment"]).optimizer_overhead.mean().mul(1e3).rename("smoke_overhead_ms")
    unique = runs.assign(frac=runs.n_unique / runs.n_calls).groupby(["optimizer", "experiment"]).frac.mean()
    refit = runs.groupby("dataset").refit_time.mean()

    # --- estimate for the full study ---
    est = []
    for d in DATASETS:
        for e in EXPERIMENTS:
            fit = fit_ds.loc[(d, e)]
            dim = 8 if e == "A" else dims[d]
            for o in OPTIMIZERS:
                u = unique.get((o, e), 1.0)
                over = ov[(ov.optimizer == o) & (ov.dim == dim)].total_s.iloc[0]
                rf = refit.get(d, refit.mean() * fit["fit_mean_s"] / fit_ds["fit_mean_s"].mean())
                for label, t in (("mean", fit["fit_mean_s"]), ("p90", fit["fit_p90_s"])):
                    est.append({"dataset": d, "experiment": e, "optimizer": o, "basis": label,
                                "cpu_s": FULL_SEEDS * (FULL_BUDGET * u * t + over + rf)})
    est = pd.DataFrame(est)
    per_ds = est.pivot_table(index="dataset", columns="basis", values="cpu_s", aggfunc="sum") / N_JOBS / 3600
    per_ds.loc["TOTAL"] = per_ds.sum()

    peak = pd.Series([r["peak_rss_mb"] for r in out if "peak_rss_mb" in r])
    report = {
        "static_check": static, "poisoned_test_replay": replay,
        "refit_trees_equal_best_iteration_plus_1": bool((runs.refit_n_trees == runs.best_iteration + 1).all()),
        "baseline": base.to_dict("records"),
        "parallel_wall_s": parallel_wall, "total_wall_s": time.perf_counter() - t_start,
        "peak_rss_mb_per_worker": {"max": float(peak.max()), "median": float(peak.median())},
    }
    (OUT / "smoke_report.json").write_text(json.dumps(report, indent=2, default=str))
    runs.drop(columns=["_fitness_trajectory"]).to_csv(OUT / "smoke_runs.csv", index=False)
    fit_ds.to_csv(OUT / "fit_time_per_call.csv")
    ov.to_csv(OUT / "optimizer_overhead_budget100.csv", index=False)
    per_ds.to_csv(OUT / "runtime_estimate_hours.csv")
    base.to_csv(OUT / "baseline.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n== static check ==", static)
    print("\n== poisoned-test replay ==\n", pd.DataFrame(replay).to_string(index=False))
    print("\n== fit time per fresh call (s), under", N_JOBS, "concurrent workers ==\n", fit_ds.round(3))
    print("\n== optimizer overhead at budget 100, free objective ==\n", ov.round(3).to_string(index=False))
    print("\n== overhead per call in smoke (ms) ==\n", smoke_overhead.unstack(["dataset", "experiment"]).round(2))
    print("\n== unique fraction (budget 20) ==\n", unique.unstack().round(3))
    print("\n== refit time (s) ==\n", refit.round(3))
    print("\n== estimated wall-clock hours, n_jobs=10 ==\n", per_ds.round(2))
    print("\n== peak RSS per worker (MB) ==", report["peak_rss_mb_per_worker"])
    print("\n== parallel section wall (s) ==", round(parallel_wall, 1))
    cols = ["dataset", "experiment", "optimizer", "seed", "n_unique", "best_val_raw_metric", "n_selected",
            "best_iteration", "refit_n_trees", "test_rmse", "test_mae", "test_r2", "test_roc_auc", "test_pr_auc"]
    print("\n== smoke test metrics (sanity only) ==\n", runs[cols].sort_values(cols[:4]).round(4).to_string(index=False))
    print("\n== baseline ==\n", base.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
