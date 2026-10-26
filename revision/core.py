"""Shared machinery for the revision experiments.

Generalises xgb_mh.experiment.run_search with three switches that the original runner does not have:
  mirror_mode : "none" | "all" (x -> 1 - x, as in the original mirror check) | "desc" (only descriptor genes mirrored)
  lam, lam_mode: subset-size penalty; "abs" adds lam * |S|/F, "rel" adds lam * m_default_val * |S|/F
  data        : any (SearchData, HeldOut) pair, so alternative scaffold partitions can be searched
The fitness, decoder, optimizers, budget accounting, cache and refit are imported unchanged from xgb_mh.
"""
from __future__ import annotations

import json, os, platform, time
from pathlib import Path

import numpy as np

from xgb_mh.experiment import (HeldOut, SearchData, _predict, load, make_model, raw_val_metric, refit_and_test)
from xgb_mh.objective import Objective
from xgb_mh.optimizers import run as run_optimizer
from xgb_mh.space import N_HP, decode, dimension, mask_to_hex

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("REVISION_OUT", ROOT / "results" / "revision"))


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with tmp.open("w") as h:
        h.write(text); h.flush(); os.fsync(h.fileno())
    os.replace(tmp, path)


def encoder(mode: str):
    if mode == "none":
        return lambda x: np.asarray(x, dtype=float)
    if mode == "all":
        return lambda x: 1.0 - np.asarray(x, dtype=float)
    if mode == "desc":
        def enc(x):
            x = np.array(x, dtype=float)
            x[N_HP:] = 1.0 - x[N_HP:]
            return x
        return enc
    raise ValueError(mode)


def default_val_metric(data: SearchData) -> float:
    from xgboost import XGBClassifier, XGBRegressor
    m = (XGBRegressor if data.task == "regression" else XGBClassifier)(n_jobs=1, tree_method="hist")
    m.fit(data.X_train, data.y_train, verbose=False)
    return raw_val_metric(data.task, data.y_val, _predict(m, data.task, data.X_val))


def make_fitness(data: SearchData, experiment: str, seed: int, penalty: float):
    """Identical to xgb_mh.experiment.make_fitness except that the penalty weight is a parameter."""
    F = data.n_features
    nf = F if experiment == "B" else None

    def fitness(x):
        cfg = decode(x, nf)
        cols = list(cfg.selected) if cfg.selected is not None else slice(None)
        t0 = time.perf_counter()
        model = make_model(data.task, cfg.params, cfg.params["n_estimators_max"], seed, early_stopping=True)
        model.fit(data.X_train[:, cols], data.y_train, eval_set=[(data.X_val[:, cols], data.y_val)], verbose=False)
        n_trees = int(model.best_iteration) + 1
        raw = raw_val_metric(data.task, data.y_val, _predict(model, data.task, data.X_val[:, cols], n_trees))
        n_sel = cfg.n_selected(F)
        fit = raw + penalty * n_sel / F if experiment == "B" else raw
        return fit, {"config": cfg, "raw_metric": raw, "n_features": n_sel, "best_iteration": int(model.best_iteration),
                     "n_trees": n_trees, "fit_time": time.perf_counter() - t0}
    return fitness, dimension(nf), nf


def run_search2(data: SearchData, held: HeldOut, experiment: str, optimizer: str, seed: int, budget: int,
                mirror_mode: str = "none", lam: float = 0.01, lam_mode: str = "abs", default_val: float | None = None,
                calls_path: Path | None = None) -> dict:
    F = data.n_features
    penalty = lam if lam_mode == "abs" else lam * (default_val if default_val is not None else default_val_metric(data))
    fit_fn, dim, nf = make_fitness(data, experiment, seed, penalty)
    enc = encoder(mirror_mode)
    records = []
    def on_call(c):
        i = c.info
        records.append({"i": c.call_idx, "raw": i["raw_metric"], "fit": c.fitness, "best": c.best_so_far,
                        "nf": i["n_features"], "hit": c.cache_hit, "nt": i["n_trees"],
                        "fit_time": 0.0 if c.cache_hit else i["fit_time"], "wall": c.wall_time, "ovh": c.overhead})
    objective = Objective(lambda x: fit_fn(enc(x)), budget, key=lambda x: decode(enc(x), nf).key, on_call=on_call)
    t0 = time.perf_counter()
    meta = run_optimizer(optimizer, objective, dim, seed)
    search_time = time.perf_counter() - t0
    best = objective.best()
    res = refit_and_test(best.info["config"], best.info["n_trees"], data, held, seed)
    if calls_path is not None:
        atomic_write(calls_path, "".join(json.dumps(r) + "\n" for r in records))
    return {"experiment": experiment, "optimizer": optimizer, "seed": seed, "budget": budget,
            "mirror_mode": mirror_mode, "lam": lam, "lam_mode": lam_mode, "penalty_weight": penalty,
            "n_calls": objective.n_calls, "n_unique": objective.n_unique, "best_call_idx": best.call_idx,
            "best_val_fitness": best.fitness, "best_val_raw_metric": best.info["raw_metric"],
            "n_selected": best.info["n_features"], "n_features_total": F, "best_iteration": best.info["best_iteration"],
            **res, "search_time": search_time, "fit_time_sum": float(sum(r["fit_time"] for r in records)),
            "best_config": json.dumps(best.info["config"].params),
            "best_selected_mask_hex": mask_to_hex(best.info["config"].selected, F), "optimizer_meta": json.dumps(meta),
            "platform": platform.platform()}
