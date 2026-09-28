"""One search run (dataset x experiment x optimizer x seed) and the default baseline.

Test isolation is structural: the objective is built from a SearchData (train + validation
arrays only). Test rows live in a separate HeldOut object that only refit_and_test() receives,
after the search has finished.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score

from . import FEATURES_DIR, task_type
from .objective import Objective
from .optimizers import run as run_optimizer
from .space import Config, decode, dimension, mask_to_hex

EXPERIMENTS = ("A", "B")          # A: HPO only; B: joint feature selection + HPO
EARLY_STOPPING_ROUNDS = 50
FS_PENALTY = 0.01


@dataclass(frozen=True)
class SearchData:
    """Everything the objective may see. Deliberately has no test fields."""
    dataset: str
    task: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


@dataclass(frozen=True)
class HeldOut:
    X_test: np.ndarray
    y_test: np.ndarray


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.flags.writeable = False
    return a


@lru_cache(maxsize=None)
def load(dataset: str, poison_test: bool = False) -> tuple[SearchData, HeldOut]:
    """poison_test=True replaces every test value with garbage (used to prove search never reads it)."""
    z = np.load(FEATURES_DIR / f"{dataset}.npz", allow_pickle=False)
    X, y, task = z["X"], z["y"], task_type(dataset)
    tr, va, te = z["train"], z["validation"], z["test"]
    if set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te):
        raise RuntimeError(f"{dataset}: overlapping split rows")
    if task == "classification":
        y = y.astype(int)
    X_test, y_test = X[te], y[te]
    if poison_test:
        X_test = np.full_like(X_test, np.nan)
        y_test = np.full_like(y_test, -999)
    search = SearchData(dataset, task, *(_readonly(a) for a in (X[tr], y[tr], X[va], y[va])))
    return search, HeldOut(_readonly(X_test), _readonly(y_test))


def make_model(task: str, params: dict, n_estimators: int, seed: int | None, early_stopping: bool):
    from xgboost import XGBClassifier, XGBRegressor
    hp = {k: v for k, v in params.items() if k != "n_estimators_max"}
    kwargs = dict(n_estimators=n_estimators, tree_method="hist", n_jobs=1, random_state=seed,
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS if early_stopping else None, **hp)
    if task == "regression":
        return XGBRegressor(eval_metric="rmse", **kwargs)
    return XGBClassifier(eval_metric="auc", **kwargs)


def _predict(model, task: str, X: np.ndarray, n_trees: int | None = None) -> np.ndarray:
    kw = {"iteration_range": (0, n_trees)} if n_trees else {}
    return model.predict(X, **kw) if task == "regression" else model.predict_proba(X, **kw)[:, 1]


def raw_val_metric(task: str, y: np.ndarray, pred: np.ndarray) -> float:
    """Regression: RMSE. Classification: 1 - ROC-AUC. Lower is better."""
    if task == "regression":
        return float(np.sqrt(np.mean((pred - y) ** 2)))
    return float(1.0 - roc_auc_score(y, pred))


def make_fitness(data: SearchData, experiment: str, seed: int):
    """Fitness on VALIDATION data only. Returns fn(x) -> (fitness, info)."""
    F = data.n_features
    nf = F if experiment == "B" else None

    def fitness(x: np.ndarray) -> tuple[float, dict]:
        cfg = decode(x, nf)
        cols = list(cfg.selected) if cfg.selected is not None else slice(None)
        t0 = time.perf_counter()
        model = make_model(data.task, cfg.params, cfg.params["n_estimators_max"], seed, early_stopping=True)
        model.fit(data.X_train[:, cols], data.y_train, eval_set=[(data.X_val[:, cols], data.y_val)], verbose=False)
        n_trees = int(model.best_iteration) + 1
        raw = raw_val_metric(data.task, data.y_val, _predict(model, data.task, data.X_val[:, cols], n_trees))
        n_sel = cfg.n_selected(F)
        fit = raw + FS_PENALTY * n_sel / F if experiment == "B" else raw
        return fit, {"config": cfg, "raw_metric": raw, "n_features": n_sel, "best_iteration": int(model.best_iteration),
                     "n_trees": n_trees, "fit_time": time.perf_counter() - t0}

    return fitness, dimension(nf), nf


def test_metrics(task: str, y: np.ndarray, pred: np.ndarray) -> dict:
    if task == "regression":
        return {"test_rmse": float(np.sqrt(np.mean((pred - y) ** 2))), "test_mae": float(mean_absolute_error(y, pred)),
                "test_r2": float(r2_score(y, pred))}
    return {"test_roc_auc": float(roc_auc_score(y, pred)), "test_pr_auc": float(average_precision_score(y, pred))}


def refit_and_test(cfg: Config, n_trees: int, data: SearchData, held: HeldOut, seed: int | None,
                   default_params: bool = False) -> dict:
    """Refit on train+val with a FIXED number of trees (no early stopping), evaluate once on test."""
    cols = list(cfg.selected) if cfg.selected is not None else slice(None)
    X = np.vstack([data.X_train, data.X_val])[:, cols]
    y = np.concatenate([data.y_train, data.y_val])
    t0 = time.perf_counter()
    if default_params:
        from xgboost import XGBClassifier, XGBRegressor
        model = (XGBRegressor if data.task == "regression" else XGBClassifier)(n_jobs=1, tree_method="hist")
    else:
        model = make_model(data.task, cfg.params, n_trees, seed, early_stopping=False)
    model.fit(X, y, verbose=False)
    trained = model.get_booster().num_boosted_rounds()
    if not default_params and trained != n_trees:
        raise AssertionError(f"refit trained {trained} trees, expected best_iteration + 1 = {n_trees}")
    out = test_metrics(data.task, held.y_test, _predict(model, data.task, held.X_test[:, cols]))
    out.update({"refit_n_trees": int(trained), "refit_time": time.perf_counter() - t0})
    return out


def _peak_rss_mb() -> float:
    import psutil
    mem = psutil.Process().memory_info()
    return getattr(mem, "peak_wset", mem.rss) / 2**20


def _call_record(call, dataset, experiment, optimizer, seed, F) -> dict:
    info = call.info
    cfg: Config = info["config"]
    return {"dataset": dataset, "experiment": experiment, "optimizer": optimizer, "seed": seed,
            "call_idx": call.call_idx, "config": cfg.params, "selected_mask_hex": mask_to_hex(cfg.selected, F),
            "n_features": info["n_features"], "raw_metric": info["raw_metric"], "fitness": call.fitness,
            "best_so_far": call.best_so_far, "cache_hit": call.cache_hit, "best_iteration": info["best_iteration"],
            "n_trees": info["n_trees"], "fit_time": 0.0 if call.cache_hit else info["fit_time"],
            "wall_time": call.wall_time, "optimizer_overhead": call.overhead}


def run_search(dataset: str, experiment: str, optimizer: str, seed: int, budget: int, out_dir: Path,
               poison_test: bool = False, write: bool = True) -> dict:
    data, held = load(dataset, poison_test)
    F = data.n_features
    fitness, dim, nf = make_fitness(data, experiment, seed)
    tag = f"{dataset}_{experiment}_{optimizer}_s{seed}"
    calls_path = out_dir / "calls" / f"{tag}.jsonl"
    records = []
    objective = Objective(fitness, budget, key=lambda x: decode(x, nf).key,
                          on_call=lambda c: records.append(_call_record(c, dataset, experiment, optimizer, seed, F)))
    t0 = time.perf_counter()
    meta = run_optimizer(optimizer, objective, dim, seed)          # search: SearchData only
    search_time = time.perf_counter() - t0
    best = objective.best()
    result = refit_and_test(best.info["config"], best.info["n_trees"], data, held, seed)  # test used once, here
    row = {"dataset": dataset, "experiment": experiment, "optimizer": optimizer, "seed": seed, "budget": budget,
           "n_calls": objective.n_calls, "n_unique": objective.n_unique, "best_call_idx": best.call_idx,
           "best_val_fitness": best.fitness, "best_val_raw_metric": best.info["raw_metric"],
           "n_selected": best.info["n_features"], "n_features_total": F,
           "best_iteration": best.info["best_iteration"], **result,
           "search_time": search_time, "fit_time_sum": sum(r["fit_time"] for r in records),
           "overhead_sum": sum(r["optimizer_overhead"] for r in records),
           "peak_rss_mb": _peak_rss_mb(), "best_config": json.dumps(best.info["config"].params),
           "best_selected_mask_hex": mask_to_hex(best.info["config"].selected, F),
           "optimizer_meta": json.dumps(meta)}
    fresh = [r["fit_time"] for r in records if not r["cache_hit"]]
    row["fit_time_median"] = float(np.median(fresh)) if fresh else None
    if write:
        # Write-to-temp then rename. runs/<tag>.json is renamed LAST: it is the completion marker.
        _atomic_write(calls_path, "".join(json.dumps(r) + "\n" for r in records))
        _atomic_write(run_path(out_dir, dataset, experiment, optimizer, seed), json.dumps(row, indent=1))
    row["_fitness_trajectory"] = [r["fitness"] for r in records]
    row["_fit_times"] = fresh
    return row


def run_path(out_dir: Path, dataset: str, experiment: str, optimizer: str, seed: int) -> Path:
    return out_dir / "runs" / f"{dataset}_{experiment}_{optimizer}_s{seed}.json"


def _atomic_write(path: Path, text: str) -> None:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def run_baseline(dataset: str) -> dict:
    """XGBoost defaults, all features, no search: fit on train+val, test once. Deterministic."""
    data, held = load(dataset)
    t0 = time.perf_counter()
    from xgboost import XGBClassifier, XGBRegressor
    val_model = (XGBRegressor if data.task == "regression" else XGBClassifier)(n_jobs=1, tree_method="hist")
    val_model.fit(data.X_train, data.y_train, verbose=False)
    val_raw = raw_val_metric(data.task, data.y_val, _predict(val_model, data.task, data.X_val))
    cfg = Config({}, None)
    result = refit_and_test(cfg, 0, data, held, None, default_params=True)
    return {"dataset": dataset, "optimizer": "default", "best_val_raw_metric": val_raw, "n_selected": data.n_features,
            "n_features_total": data.n_features, **result, "search_time": time.perf_counter() - t0,
            "peak_rss_mb": _peak_rss_mb()}
