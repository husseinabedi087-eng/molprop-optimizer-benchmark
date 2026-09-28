"""Full study runner, resumable across sessions (shutdown / hibernate / Ctrl+C are all safe).

A run is DONE iff runs/<tag>.json exists (renamed into place last, after its calls/<tag>.jsonl).
On start: stray *.tmp files are deleted, finished runs are skipped, anything else is redone from
scratch. Every run is independently seeded, so results do not depend on how sessions are split.

Usage: python -m xgb_mh.full_run [--out-dir DIR] [--datasets BBBP,...] [--seeds 0-29] [--budget 100] [--n-jobs 10]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import PROJECT_ROOT, RESULTS_DIR
from .experiment import EXPERIMENTS, _atomic_write, load, run_baseline, run_path, run_search
from .optimizers import MAIN_RUN_OPTIMIZERS, OPTIMIZERS

ORDER = ("BBBP", "FreeSolv", "BACE", "ESOL", "Lipophilicity")
CHECKPOINT_EVERY = 50
SMOKE_DIR = RESULTS_DIR / "smoke"
# Modules whose code determines results; a change between sessions aborts the run.
RESULT_CODE = ("__init__.py", "space.py", "objective.py", "experiment.py", "features.py", "data.py", "optimizers/*.py")


def code_hash() -> str:
    pkg = PROJECT_ROOT / "xgb_mh"
    digest = hashlib.sha256()
    for pattern in RESULT_CODE:
        for path in sorted(pkg.glob(pattern)):
            digest.update(path.relative_to(pkg).as_posix().encode() + b"\0" + path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


class Log:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        lo, hi = map(int, text.split("-"))
        return list(range(lo, hi + 1))
    return [int(s) for s in text.split(",")]


def smoke_estimates(budget: int) -> tuple[dict, dict]:
    """Median fit time per call (s) per (dataset, experiment) and estimated seconds per run."""
    fit = pd.read_csv(SMOKE_DIR / "fit_time_per_call.csv").set_index(["dataset", "experiment"]).fit_median_s.to_dict()
    over = pd.read_csv(SMOKE_DIR / "optimizer_overhead_budget100.csv")
    est = {}
    for (d, e), t in fit.items():
        dim = 8 if e == "A" else 8 + load(d)[0].n_features
        for o in OPTIMIZERS:
            ov = over[(over.optimizer == o) & (over.dim == dim)].total_s
            est[d, e, o] = budget * t + (float(ov.iloc[0]) * budget / 100 if len(ov) else 0.0) + 1.2 * t  # + refit
    return fit, est


def task(dataset, experiment, optimizer, seed, budget, out_dir) -> dict:
    t0 = time.perf_counter()
    try:
        row = run_search(dataset, experiment, optimizer, seed, budget, out_dir)
        row.pop("_fitness_trajectory")
        return {"ok": True, "row": row, "duration": time.perf_counter() - t0}
    except Exception:
        return {"ok": False, "key": (dataset, experiment, optimizer, seed), "error": traceback.format_exc()}


def fit_times_from_calls(path: Path) -> list[float]:
    times = []
    with path.open() as handle:
        for line in handle:
            r = json.loads(line)
            if not r["cache_hit"]:
                times.append(r["fit_time"])
    return times


def build_test_summary(out_dir: Path, datasets, seeds) -> Path:
    rows = [json.loads(p.read_text()) for p in sorted((out_dir / "runs").glob("*.json"))]
    for d in datasets:
        base = json.loads((out_dir / "baseline" / f"{d}.json").read_text())
        for e in EXPERIMENTS:
            for s in seeds:  # deterministic baseline: identical row per seed (zero spread, by design)
                rows.append({**base, "experiment": e, "seed": s, "optimizer": "default"})
    df = pd.DataFrame(rows)
    first = ["dataset", "experiment", "optimizer", "seed", "test_rmse", "test_mae", "test_r2", "test_roc_auc", "test_pr_auc",
             "best_val_fitness", "best_val_raw_metric", "n_selected", "n_features_total", "n_calls", "n_unique",
             "best_iteration", "refit_n_trees", "search_time", "refit_time", "fit_time_median"]
    df = df[[c for c in first if c in df] + [c for c in df if c not in first]]
    order = {d: i for i, d in enumerate(datasets)}
    df = df.sort_values(["dataset", "experiment", "optimizer", "seed"], key=lambda c: c.map(order) if c.name == "dataset" else c)
    path = out_dir / "test_summary.csv"
    df.to_csv(path, index=False)
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(RESULTS_DIR / "full"))
    p.add_argument("--datasets", default=",".join(ORDER))
    p.add_argument("--seeds", default="0-29")
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--n-jobs", type=int, default=10)
    p.add_argument("--optimizers", default=",".join(MAIN_RUN_OPTIMIZERS),
                   help=f"comma-separated subset of {', '.join(OPTIMIZERS)}")
    p.add_argument("--allow-code-change", action="store_true", help="resume even if result code changed (not recommended)")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    datasets, seeds = args.datasets.split(","), parse_seeds(args.seeds)
    optimizers = args.optimizers.split(",")
    unknown = set(optimizers) - set(OPTIMIZERS)
    if unknown:
        raise SystemExit(f"unknown optimizers: {sorted(unknown)}")
    log = Log(out / "full_run.log")
    session_start = time.time()

    # --- metadata / code-consistency across sessions ---
    meta_path = out / "run_metadata.json"
    config = {"datasets": datasets, "experiments": list(EXPERIMENTS), "optimizers": optimizers, "seeds": seeds,
              "budget": args.budget, "n_jobs": args.n_jobs}
    chash = code_hash()
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["config"] != config:
            raise SystemExit(f"Config differs from the existing run in {out}:\n{meta['config']}\nvs\n{config}")
        if meta["code_sha256"] != chash and not args.allow_code_change:
            raise SystemExit("Result-determining code changed since this run started; refusing to mix results. "
                             "Use a new --out-dir (or --allow-code-change if you are sure).")
    else:
        import optuna, rdkit, sklearn, xgboost, psutil
        feats = RESULTS_DIR / "features"
        meta = {"created_utc": datetime.now(timezone.utc).isoformat(), "config": config, "code_sha256": chash,
                "initial_design": "common initial design (10 points) from default_rng(seed), shared by every optimizer "
                                  "(also across run folders, e.g. full and full_ref)",
                "packages": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                             "xgboost": xgboost.__version__, "optuna": optuna.__version__, "rdkit": rdkit.__version__,
                             "scikit-learn": sklearn.__version__},
                "hardware": {"cpu": platform.processor(), "logical_cpus": psutil.cpu_count(),
                             "physical_cores": psutil.cpu_count(logical=False), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1)},
                "feature_files_sha256": {d: hashlib.sha256((feats / f"{d}.npz").read_bytes()).hexdigest() for d in datasets},
                "sessions": []}
    meta["sessions"].append({"start_utc": datetime.now(timezone.utc).isoformat()})
    _atomic_write(meta_path, json.dumps(meta, indent=2))
    prev_elapsed = sum(s.get("elapsed_s", 0) for s in meta["sessions"][:-1])

    def close_session(status: str) -> None:
        meta["sessions"][-1].update({"end_utc": datetime.now(timezone.utc).isoformat(),
                                     "elapsed_s": time.time() - session_start, "status": status})
        _atomic_write(meta_path, json.dumps(meta, indent=2))

    # --- clean up and plan ---
    stray = list(out.glob("calls/*.tmp")) + list(out.glob("runs/*.tmp")) + list(out.glob("baseline/*.tmp"))
    for f in stray:
        f.unlink()
    all_tasks = [(d, e, o, s) for d in datasets for e in EXPERIMENTS for o in optimizers for s in seeds]
    done = {t for t in all_tasks if run_path(out, *t).exists()}
    partial = [t for t in all_tasks if t not in done and (out / "calls" / f"{t[0]}_{t[1]}_{t[2]}_s{t[3]}.jsonl").exists()]
    todo = [t for t in all_tasks if t not in done]
    per_ds_total = {d: sum(t[0] == d for t in all_tasks) for d in datasets}
    per_ds_done = {d: sum(t[0] == d for t in done) for d in datasets}
    log(f"=== session {len(meta['sessions'])} start | {out} | code {chash[:12]} | n_jobs={args.n_jobs}")
    log(f"finished runs skipped: {len(done)}/{len(all_tasks)} | partial runs to redo: {len(partial)} | "
        f"stray .tmp files removed: {len(stray)} | to run now: {len(todo)}")
    for d in datasets:
        log(f"  {d}: {per_ds_done[d]}/{per_ds_total[d]} runs done" + (" (complete)" if per_ds_done[d] == per_ds_total[d] else ""))

    # --- baselines (cheap, deterministic) ---
    for d in datasets:
        path = out / "baseline" / f"{d}.json"
        if not path.exists():
            _atomic_write(path, json.dumps(run_baseline(d), indent=1))
            log(f"baseline {d} done")

    smoke_fit, est = smoke_estimates(args.budget)
    fit_times: dict[tuple, list] = {}
    for t in done:
        calls = out / "calls" / f"{t[0]}_{t[1]}_{t[2]}_s{t[3]}.jsonl"
        fit_times.setdefault(t[:2], []).extend(fit_times_from_calls(calls))
    n_done, failures = len(done), []
    actual_sum = est_sum = 0.0
    remaining_est = sum(est[t[:3]] for t in todo)

    def checkpoint() -> None:
        log(f"--- checkpoint at {n_done} runs: median fit time per call, actual vs smoke estimate ---")
        for (d, e), ts in sorted(fit_times.items(), key=lambda kv: (datasets.index(kv[0][0]), kv[0][1])):
            act, ref = float(np.median(ts)), smoke_fit[d, e]
            log(f"    {d:13s} {e}: actual {act:6.2f} s (n={len(ts)}) | smoke {ref:6.2f} s | ratio {act / ref:4.2f}")

    status = "interrupted"
    try:
        jobs = (delayed(task)(*t, args.budget, out) for t in todo)
        for res in Parallel(n_jobs=args.n_jobs, return_as="generator_unordered", pre_dispatch="2*n_jobs")(jobs):
            if not res["ok"]:
                failures.append(res["key"])
                with (out / "errors.log").open("a") as handle:
                    handle.write(f"[{datetime.now()}] {res['key']}\n{res['error']}\n")
                log(f"FAILED {res['key']} (see errors.log; it will be retried next session)")
                continue
            row = res["row"]
            key = (row["dataset"], row["experiment"], row["optimizer"], row["seed"])
            n_done += 1
            per_ds_done[key[0]] += 1
            fit_times.setdefault(key[:2], []).extend(row["_fit_times"])
            actual_sum += res["duration"]
            est_sum += est[key[:3]]
            remaining_est -= est[key[:3]]
            calib = actual_sum / est_sum
            session_elapsed = time.time() - session_start
            eta = calib * remaining_est / args.n_jobs
            log(f"{n_done}/{len(all_tasks)} {key[0]}_{key[1]}_{key[2]}_s{key[3]} ({res['duration']:.0f}s) | "
                f"elapsed session {hms(session_elapsed)}, total {hms(prev_elapsed + session_elapsed)} | "
                f"ETA {hms(eta)} (actual/smoke-median cost ratio {calib:.2f})")
            meta["sessions"][-1].update({"elapsed_s": session_elapsed, "last_update_utc": datetime.now(timezone.utc).isoformat()})
            _atomic_write(meta_path, json.dumps(meta, indent=2))  # survives a hard kill / power loss
            if per_ds_done[key[0]] == per_ds_total[key[0]]:
                log(f"{key[0]} complete: {per_ds_done[key[0]]}/{per_ds_total[key[0]]} runs")
            if n_done % CHECKPOINT_EVERY == 0:
                checkpoint()
        status = "complete" if n_done == len(all_tasks) else "finished_with_failures"
    finally:
        close_session(status)
        log(f"=== session end: {status} | {n_done}/{len(all_tasks)} runs done | failures this session: {len(failures)}")

    if n_done == len(all_tasks):
        checkpoint()
        path = build_test_summary(out, datasets, seeds)
        log(f"ALL RUNS COMPLETE. test summary: {path}")
    elif failures:
        log("Some runs failed; rerun the same command to retry them.")


if __name__ == "__main__":
    main()
