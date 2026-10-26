"""Task lists and workers for the revision experiments. Usage: python -m revision.jobs <job> [--workers 2]

Jobs (pre-registered in revision/PREREGISTRATION_REVISION.md before any of them was run):
  J1 reeval   : re-evaluate the 1,500 primary selected configurations (pre-refit diagnostic, model-seed control, predictions)
  J3 mirror   : descriptor-only mirror + same-platform standard / full-mirror references (Exp B, BBBP, FreeSolv)
  J4 penalty  : subset-size penalty sensitivity (lam = 0, 1 % and 5 % of the default validation metric) + references
  J5 budget   : budget 300 (Exp B, BBBP, FreeSolv)
  J2 pool     : 200-configuration pool, single split vs 5-fold inner scaffold CV, untouched outer test (Exp A)
  J6 splits   : five independently constructed scaffold partitions (Exp A; ESOL, FreeSolv, BBBP, BACE)
Every task writes one JSON (and optional npz / jsonl) atomically; finished tasks are skipped on restart.
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np

from xgb_mh import task_type
from xgb_mh.experiment import _predict, load, make_model, raw_val_metric, test_metrics
from xgb_mh.space import decode, hex_to_mask
from .core import OUT, ROOT, atomic_write, default_val_metric, run_search2

RES = ROOT / "results" / "xgb_mh"
OPTS = ("random", "tpe", "gwo_ref", "woa_ref", "pso")
MODEL_SEEDS = (1000, 1001)


def code_hash() -> str:
    files = sorted((ROOT / "revision").glob("*.py")) + sorted((ROOT / "xgb_mh").rglob("*.py"))
    return hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()


# ---------------------------------------------------------------- J1: re-evaluation of selected configurations
def j1_tasks():
    t = []
    for folder, opts in (("full", ("random", "tpe", "pso")), ("full_ref", ("gwo_ref", "woa_ref"))):
        for p in sorted((RES / folder / "runs").glob("*.json")):
            ds, ex, *rest = p.stem.split("_")
            opt = "_".join(rest[:-1])
            if opt in opts:
                t.append(("J1", str(p)))
    for ds in ("ESOL", "FreeSolv", "Lipophilicity", "BBBP", "BACE"):
        t.append(("J1", f"default:{ds}"))
    return t


def j1_run(arg: str) -> tuple[Path, dict, dict]:
    if arg.startswith("default:"):
        ds = arg.split(":")[1]
        data, held = load(ds)
        from xgboost import XGBClassifier, XGBRegressor
        cls = XGBRegressor if data.task == "regression" else XGBClassifier
        m = cls(n_jobs=1, tree_method="hist"); m.fit(data.X_train, data.y_train, verbose=False)
        r = cls(n_jobs=1, tree_method="hist"); r.fit(np.vstack([data.X_train, data.X_val]),
                                                      np.concatenate([data.y_train, data.y_val]), verbose=False)
        preds = {"val_pre": _predict(m, data.task, data.X_val), "test_pre": _predict(m, data.task, held.X_test),
                 "test_refit": _predict(r, data.task, held.X_test)}
        out = OUT / "J1" / f"{ds}_default"
        row = {"dataset": ds, "experiment": "AB", "optimizer": "default", "seed": -1}
    else:
        run = json.loads(Path(arg).read_text())
        ds, ex, opt, seed = run["dataset"], run["experiment"], run["optimizer"], int(run["seed"])
        data, held = load(ds)
        params = json.loads(run["best_config"])
        F = data.n_features
        cols = list(np.flatnonzero(hex_to_mask(run["best_selected_mask_hex"], F))) if run["best_selected_mask_hex"] else slice(None)
        pre = make_model(data.task, params, params["n_estimators_max"], seed, early_stopping=True)
        pre.fit(data.X_train[:, cols], data.y_train, eval_set=[(data.X_val[:, cols], data.y_val)], verbose=False)
        nt_pre = int(pre.best_iteration) + 1
        preds = {"val_pre": _predict(pre, data.task, data.X_val[:, cols], nt_pre),
                 "test_pre": _predict(pre, data.task, held.X_test[:, cols], nt_pre)}
        X = np.vstack([data.X_train, data.X_val])[:, cols]; y = np.concatenate([data.y_train, data.y_val])
        nt = int(run["best_iteration"]) + 1
        for tag, ms in (("test_refit", seed),) + tuple((f"test_refit_m{s}", s) for s in MODEL_SEEDS):
            r = make_model(data.task, params, nt, ms, early_stopping=False); r.fit(X, y, verbose=False)
            preds[tag] = _predict(r, data.task, held.X_test[:, cols])
        out = OUT / "J1" / f"{ds}_{ex}_{opt}_s{seed}"
        row = {"dataset": ds, "experiment": ex, "optimizer": opt, "seed": seed, "n_trees_logged": nt, "n_trees_pre_linux": nt_pre,
               "logged_val_raw": run["best_val_raw_metric"], "logged_test_primary":
               run["test_roc_auc"] if data.task == "classification" else run["test_rmse"]}
    task = data.task
    if "logged_test_primary" in row:
        key = "test_refit__test_roc_auc" if task == "classification" else "test_refit__test_rmse"
        row["_pending_repro_key"] = key
    row["val_pre_raw"] = raw_val_metric(task, data.y_val, preds["val_pre"])
    for k, v in preds.items():
        if k.startswith("test"):
            row.update({f"{k}__{m}": val for m, val in test_metrics(task, held.y_test, v).items()})
    if "_pending_repro_key" in row:
        row["refit_reproduces_primary_exactly"] = bool(row[row.pop("_pending_repro_key")] == row["logged_test_primary"])
        row["val_reproduces_logged_exactly"] = bool(row["val_pre_raw"] == row["logged_val_raw"])
    return out, row, {k: np.asarray(v, dtype=np.float32) for k, v in preds.items()}


# ---------------------------------------------------------------- J3 / J4 / J5: searches on the original split
def search_tasks(job):
    t = []
    DS = ("FreeSolv", "BBBP")
    if job == "J3":
        for ds in DS:
            for s in range(30):
                for opt in OPTS:
                    t.append((job, json.dumps({"ds": ds, "opt": opt, "seed": s, "mirror": "desc", "budget": 100, "lam": 0.01, "mode": "abs"})))
                for opt in ("gwo_ref", "random"):
                    for mm in ("none", "all"):
                        t.append((job, json.dumps({"ds": ds, "opt": opt, "seed": s, "mirror": mm, "budget": 100, "lam": 0.01, "mode": "abs"})))
    if job == "J4":
        for ds in DS:
            for s in range(10):
                for opt in ("tpe", "woa_ref", "pso"):   # gwo_ref / random references come from J3 (seeds 0-9)
                    t.append((job, json.dumps({"ds": ds, "opt": opt, "seed": s, "mirror": "none", "budget": 100, "lam": 0.01, "mode": "abs"})))
                for lam, mode in ((0.0, "abs"), (0.01, "rel"), (0.05, "rel")):
                    for opt in OPTS:
                        t.append((job, json.dumps({"ds": ds, "opt": opt, "seed": s, "mirror": "none", "budget": 100, "lam": lam, "mode": mode})))
                for opt in ("gwo_ref", "random"):
                    t.append((job, json.dumps({"ds": ds, "opt": opt, "seed": s, "mirror": "all", "budget": 100, "lam": 0.0, "mode": "abs"})))
    if job == "J5":
        for ds in DS:
            for s in range(10):
                for opt in OPTS:
                    t.append((job, json.dumps({"ds": ds, "opt": opt, "seed": s, "mirror": "none", "budget": 300, "lam": 0.01, "mode": "abs"})))
    return t


def search_name(c):
    return f"{c['ds']}_B_{c['opt']}_s{c['seed']}_m-{c['mirror']}_b{c['budget']}_lam{c['lam']}{c['mode']}"


def search_run(job, arg):
    c = json.loads(arg)
    data, held = load(c["ds"])
    name = search_name(c)
    row = run_search2(data, held, "B", c["opt"], c["seed"], c["budget"], mirror_mode=c["mirror"], lam=c["lam"],
                      lam_mode=c["mode"], calls_path=OUT / "calls" / f"{name}.jsonl")
    row["dataset"] = c["ds"]
    return OUT / "searches" / name, row, None


# ---------------------------------------------------------------- J2: configuration pool, single split vs inner CV
POOL_N, POOL_SEED, N_FOLDS = 200, 20261025, 5


def scaffold_of(smi):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smi)), isomericSmiles=False)


from functools import lru_cache


@lru_cache(maxsize=None)
def inner_folds(ds):
    """5 scaffold-group folds over train + validation: groups largest first, each to the currently smallest fold."""
    z = np.load(RES / "features" / f"{ds}.npz", allow_pickle=False)
    rows = np.concatenate([z["train"], z["validation"]])
    groups = {}
    for pos, r in enumerate(rows):
        groups.setdefault(scaffold_of(str(z["smiles"][r])), []).append(pos)
    order = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    fold = np.empty(len(rows), dtype=int); sizes = [0] * N_FOLDS
    for g in order:
        k = int(np.argmin(sizes)); fold[g] = k; sizes[k] += len(g)
    return fold


def j2_tasks():
    return [("J2", json.dumps({"ds": ds, "k": k})) for ds in ("FreeSolv", "BACE", "BBBP", "ESOL", "Lipophilicity")
            for k in range(-1, POOL_N)]   # k = -1: default configuration


def j2_run(arg):
    c = json.loads(arg); ds, k = c["ds"], c["k"]
    data, held = load(ds); task = data.task
    Xd = np.vstack([data.X_train, data.X_val]); yd = np.concatenate([data.y_train, data.y_val])
    fold = inner_folds(ds)
    from xgboost import XGBClassifier, XGBRegressor
    if k < 0:
        params = None
        mk = lambda n=None, es=False: (XGBRegressor if task == "regression" else XGBClassifier)(
            n_jobs=1, tree_method="hist", random_state=0, **({"early_stopping_rounds": 50, "n_estimators": 100} if es else {}))
    else:
        x = np.random.default_rng(POOL_SEED).random((POOL_N, 8))[k]
        params = decode(x).params
        mk = lambda n=None, es=False: make_model(task, params, n if n else params["n_estimators_max"], 0, early_stopping=es)
    row = {"dataset": ds, "k": k, "params": params}
    # single split (as in the benchmark): train fit, early stopping on validation
    if k < 0:
        m = mk(); m.fit(data.X_train, data.y_train, verbose=False); nt = 100
    else:
        m = mk(es=True); m.fit(data.X_train, data.y_train, eval_set=[(data.X_val, data.y_val)], verbose=False)
        nt = int(m.best_iteration) + 1
    vp = _predict(m, task, data.X_val, nt if k >= 0 else None); tp = _predict(m, task, held.X_test, nt if k >= 0 else None)
    row.update({"single_val": raw_val_metric(task, data.y_val, vp), "single_nt": nt,
                "test_pre": raw_val_metric(task, held.y_test, tp)})
    r = mk(n=nt) if k >= 0 else mk(); r.fit(Xd, yd, verbose=False)
    row["test_refit_single"] = raw_val_metric(task, held.y_test, _predict(r, task, held.X_test))
    # 5-fold inner scaffold CV on train + validation (test untouched)
    cv, nts = [], []
    for f in range(N_FOLDS):
        tr, va = fold != f, fold == f
        if k < 0:
            mm = mk(); mm.fit(Xd[tr], yd[tr], verbose=False); ntf = 100
            cv.append(raw_val_metric(task, yd[va], _predict(mm, task, Xd[va])))
        else:
            mm = mk(es=True); mm.fit(Xd[tr], yd[tr], eval_set=[(Xd[va], yd[va])], verbose=False)
            ntf = int(mm.best_iteration) + 1
            cv.append(raw_val_metric(task, yd[va], _predict(mm, task, Xd[va], ntf)))
        nts.append(ntf)
    row.update({"cv_folds": cv, "cv_mean": float(np.mean(cv)), "cv_nt_median": int(np.median(nts))})
    if k >= 0:
        rc = mk(n=int(np.median(nts))); rc.fit(Xd, yd, verbose=False)
        row["test_refit_cv"] = raw_val_metric(task, held.y_test, _predict(rc, task, held.X_test))
    else:
        row["test_refit_cv"] = row["test_refit_single"]
    return OUT / "J2" / f"{ds}_k{k}", row, None


# ---------------------------------------------------------------- J6: independent scaffold partitions
J6_DATASETS, J6_PARTITIONS, J6_SEEDS = ("FreeSolv", "BACE", "BBBP", "ESOL"), (1, 2, 3, 4, 5), (0, 1, 2)


def balanced_scaffold_split(smiles, seed, frac=(.8, .1, .1)):
    """Randomised balanced scaffold split (Chemprop-style): scaffold groups larger than half the test size go to
    training first; the remaining groups are shuffled with `seed` and assigned greedily. Returns row index lists."""
    import random
    n = len(smiles); n_tr, n_va = int(frac[0] * n), int(frac[1] * n); n_te = n - n_tr - n_va
    groups = {}
    for i, s in enumerate(smiles):
        groups.setdefault(scaffold_of(s), []).append(i)
    big = [g for g in groups.values() if len(g) > n_te / 2]; small = [g for g in groups.values() if len(g) <= n_te / 2]
    rng = random.Random(seed); rng.shuffle(big); rng.shuffle(small)
    tr, va, te = [], [], []
    for g in big + small:
        if len(tr) + len(g) <= n_tr: tr += g
        elif len(va) + len(g) <= n_va: va += g
        else: te += g
    return sorted(tr), sorted(va), sorted(te)


_PART_CACHE = {}


def partition_data(ds, part):
    key = (ds, part)
    if key in _PART_CACHE:
        return _PART_CACHE[key]
    from xgb_mh.experiment import HeldOut, SearchData
    from xgb_mh.features import compute_descriptors, sanitize, train_filter
    z = np.load(RES / "features" / f"{ds}.npz", allow_pickle=False)
    smiles = [str(s) for s in z["smiles"]]
    cache = OUT / "J6_features" / f"{ds}_full.npy"
    if cache.exists():
        Xall = np.load(cache)
    else:
        Xraw, _ = compute_descriptors(smiles, n_jobs=1)
        Xall, _, _ = sanitize(Xraw); cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.stem + ".tmp.npy"); np.save(tmp, Xall); tmp.replace(cache)
    tr, va, te = balanced_scaffold_split(smiles, part)
    keep, _, _ = train_filter(Xall[tr])
    X = Xall[:, keep].astype(np.float32); y = z["y"].astype(int) if task_type(ds) == "classification" else z["y"]
    sd = SearchData(ds, task_type(ds), X[tr], y[tr], X[va], y[va]); ho = HeldOut(X[te], y[te])
    info = {"n_train": len(tr), "n_val": len(va), "n_test": len(te), "F": int(keep.sum()),
            "split_sha256": hashlib.sha256(json.dumps([tr, va, te]).encode()).hexdigest()}
    _PART_CACHE[key] = (sd, ho, info)
    return _PART_CACHE[key]


def j6_tasks():
    t = []
    for ds in J6_DATASETS:
        for p in J6_PARTITIONS:
            t.append(("J6", json.dumps({"ds": ds, "part": p, "opt": "default", "seed": -1})))
            for s in J6_SEEDS:
                for opt in OPTS:
                    t.append(("J6", json.dumps({"ds": ds, "part": p, "opt": opt, "seed": s})))
    return t


def j6_run(arg):
    c = json.loads(arg)
    sd, ho, info = partition_data(c["ds"], c["part"])
    name = f"{c['ds']}_p{c['part']}_A_{c['opt']}_s{c['seed']}"
    if c["opt"] == "default":
        from xgb_mh.experiment import Config, refit_and_test
        row = refit_and_test(Config({}, None), 0, sd, ho, None, default_params=True)
        row["best_val_raw_metric"] = default_val_metric(sd)
    else:
        row = run_search2(sd, ho, "A", c["opt"], c["seed"], 100, calls_path=OUT / "J6_calls" / f"{name}.jsonl")
    row.update({"dataset": c["ds"], "partition": c["part"], "optimizer": c["opt"], "seed": c["seed"], **info})
    return OUT / "J6" / name, row, None


# ---------------------------------------------------------------- driver
def tasks_for(job):
    return {"J1": j1_tasks, "J2": j2_tasks, "J6": j6_tasks}.get(job, lambda: search_tasks(job))()


def out_name(job, arg):
    if job == "J1":
        return OUT / "J1" / (f"{arg.split(':')[1]}_default" if arg.startswith("default:") else Path(arg).stem)
    if job in ("J3", "J4", "J5"):
        return OUT / "searches" / search_name(json.loads(arg))
    if job == "J2":
        c = json.loads(arg); return OUT / "J2" / f"{c['ds']}_k{c['k']}"
    c = json.loads(arg); return OUT / "J6" / f"{c['ds']}_p{c['part']}_A_{c['opt']}_s{c['seed']}"


def work(job, arg, chash):
    t0 = time.perf_counter()
    fn = {"J1": lambda a: j1_run(a), "J2": j2_run, "J6": j6_run}.get(job, lambda a: search_run(job, a))
    out, row, arrays = fn(arg)
    row.update({"job": job, "elapsed_s": time.perf_counter() - t0, "code_sha256": chash})
    if arrays is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp.npz"); np.savez_compressed(tmp, **arrays); tmp.replace(out.with_name(out.name + ".npz"))
    atomic_write(out.with_name(out.name + ".json"), json.dumps(row, default=float))
    return out.name


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("job"); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    chash = code_hash()
    todo = [t for t in tasks_for(a.job) if not (lambda o: o.with_name(o.name + ".json"))(out_name(*t)).exists()]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[{time.strftime('%H:%M:%S')}] {a.job}: {len(todo)} tasks to do, code {chash[:12]}", flush=True)
    if a.job == "J6" and todo:
        for ds in J6_DATASETS:
            partition_data(ds, J6_PARTITIONS[0])   # builds results/revision/J6_features/<ds>_full.npy once
        print(f"[{time.strftime('%H:%M:%S')}] J6 descriptor caches ready", flush=True)
    from joblib import Parallel, delayed
    done = 0
    for name in Parallel(n_jobs=a.workers, return_as="generator_unordered", batch_size=1)(
            delayed(work)(job, arg, chash) for job, arg in todo):
        done += 1
        if done % 20 == 0 or done == len(todo):
            print(f"[{time.strftime('%H:%M:%S')}] {a.job}: {done}/{len(todo)} ({name})", flush=True)


if __name__ == "__main__":
    # import through the package so that worker functions are pickled by reference (needed on Windows too)
    from revision.jobs import main as _main
    _main()
