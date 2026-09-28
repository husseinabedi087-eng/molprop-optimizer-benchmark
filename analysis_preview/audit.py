"""PRELIMINARY integrity + descriptive audit of completed datasets (read-only).

Reads results/xgb_mh/full/{runs,calls}/ and errors.log; never writes there; does NOT import xgb_mh.
Writes analysis_preview/output/audit_<stamp>/ (AUDIT.md + CSVs).
Usage: python analysis_preview/audit.py [--datasets BBBP,FreeSolv,BACE,ESOL]
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from preview import FULL, LABELS, OPTIMIZERS, OUT, code_hash  # same folder; preview.py never imports xgb_mh

EXPERIMENTS = ("A", "B")
BUDGET, N_INITIAL = 100, 10
PRIMARY = {"ESOL": "test_rmse", "FreeSolv": "test_rmse", "Lipophilicity": "test_rmse",
           "BBBP": "test_roc_auc", "BACE": "test_roc_auc"}
# Mirror of xgb_mh/space.py::HP_SPEC (copied, not imported).
HP_SPEC = (("n_estimators_max", "int", 100, 1000), ("max_depth", "int", 3, 10), ("learning_rate", "log", 0.01, 0.3),
           ("subsample", "linear", 0.5, 1.0), ("colsample_bytree", "linear", 0.3, 1.0),
           ("min_child_weight", "log", 1.0, 10.0), ("reg_lambda", "log", 1e-3, 10.0), ("reg_alpha", "log", 1e-3, 10.0))
BOUND_TOL = 0.01          # "within 1% of the range" (log parameters: 1% of the log range)
FLAG_UNIQUE, FLAG_BOUND, FLAG_CONV, MAD_K = 90, 0.30, 90, 3.0


def position(kind: str, lo: float, hi: float, v: float) -> float:
    """Where v sits in its range, 0..1 (log scale for log parameters)."""
    if kind == "log":
        return math.log(v / lo) / math.log(hi / lo)
    return (v - lo) / (hi - lo)


def uniform_bound_rate(kind: str, lo: float, hi: float) -> tuple[float, float]:
    """Expected share at the lower / upper bound if the decoded x were uniform (floor-bin integer decoding)."""
    if kind == "int":
        n_values = hi - lo + 1
        tol_values = math.floor(BOUND_TOL * (hi - lo)) + 1  # integer values within 1% of each end
        return tol_values / n_values, tol_values / n_values
    return BOUND_TOL, BOUND_TOL


def main() -> None:
    global FULL, OPTIMIZERS  # rebound from the command line; every check below reads these
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="BBBP,FreeSolv,BACE,ESOL")
    parser.add_argument("--run-dir", default=str(FULL), help="run folder to audit (default: the main run)")
    parser.add_argument("--optimizers", default=None, help="comma-separated; default: the main run's five (labelled)")
    parser.add_argument("--design-ref", default=None,
                        help="another run folder whose calls 1-10 (per dataset/experiment/seed) must match this run's")
    args = parser.parse_args()
    datasets = args.datasets.split(",")
    FULL = Path(args.run_dir)
    if args.optimizers:
        OPTIMIZERS = tuple(LABELS.get(o, o) for o in args.optimizers.split(","))
    meta = json.loads((FULL / "run_metadata.json").read_text())
    seeds = meta["config"]["seeds"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT / f"audit_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report, verdict = [], {}

    # ---------- load ----------
    runs, calls, problems = [], [], []
    for d in datasets:
        for path in sorted((FULL / "runs").glob(f"{d}_*.json")):
            r = json.loads(path.read_text())
            tag = f"{r['dataset']}_{r['experiment']}_{r['optimizer']}_s{r['seed']}"
            if tag != path.stem:
                problems.append(f"{path.name}: content says {tag}")
            r["tag"] = path.stem  # file names keep the internal optimizer names
            runs.append(r)
            cpath = FULL / "calls" / f"{path.stem}.jsonl"
            if not cpath.exists():
                problems.append(f"{path.stem}: calls file missing")
                continue
            c = pd.read_json(cpath, lines=True, precise_float=True)  # default parsing rounds the last digits
            c["tag"] = path.stem
            calls.append(c)
    runs = pd.DataFrame(runs)
    runs["optimizer"] = runs.optimizer.replace(LABELS)
    calls = pd.concat(calls, ignore_index=True)

    # ---------- 1. integrity ----------
    expected = {(d, e, o, s) for d in datasets for e in EXPERIMENTS for o in OPTIMIZERS for s in seeds}
    keys = list(zip(runs.dataset, runs.experiment, runs.optimizer, runs.seed))
    dup = len(keys) - len(set(keys))
    missing, extra = expected - set(keys), set(keys) - expected
    per_ds = runs.groupby("dataset").size().reindex(datasets, fill_value=0)
    per_call = calls.groupby("tag").agg(n=("call_idx", "size"), idx_ok=("call_idx", lambda s: list(s) == list(range(BUDGET))))
    bad_calls = per_call[(per_call.n != BUDGET) | ~per_call.idx_ok]
    bad_ncalls = runs[runs.n_calls != BUDGET]
    no_design = runs[~runs.optimizer_meta.str.contains("common initial design", regex=False)]
    # calls 1-10 identical across optimizers: decoded config + feature mask + fitness
    first = calls[calls.call_idx < N_INITIAL].copy()
    first["sig"] = first.config.map(lambda c: json.dumps(c, sort_keys=True)) + "|" + \
        first.selected_mask_hex.fillna("all_features").astype(str) + "|" + \
        first.fitness.map(lambda v: repr(float(v)))  # Exp A mask is null; float() so the repr matches json-parsed values
    first[["dataset", "experiment", "optimizer", "seed"]] = first.tag.str.extract(r"^(\w+?)_([AB])_(\w+)_s(\d+)$").values
    first["optimizer"] = first.optimizer.replace(LABELS)
    shared = first.groupby(["dataset", "experiment", "seed", "call_idx"]).agg(n_opt=("optimizer", "nunique"), n_sig=("sig", "nunique"))
    not_shared = shared[(shared.n_opt != len(OPTIMIZERS)) | (shared.n_sig != 1)]
    errors_log = FULL / "errors.log"
    err_text = errors_log.read_text() if errors_log.exists() else ""
    integ = pd.DataFrame([{"check": k, "value": v} for k, v in {
        **{f"runs_{d}": int(per_ds[d]) for d in datasets}, "duplicates": dup, "missing": len(missing),
        "unexpected": len(extra), "runs_with_n_calls_ne_100": len(bad_ncalls),
        "calls_files_not_exactly_100_ordered_calls": len(bad_calls), "runs_without_initial_design_meta": len(no_design),
        "initial_design_groups_checked": len(shared), "initial_design_groups_not_identical": len(not_shared),
        "filename_content_mismatches": len(problems), "errors_log": "absent" if not errors_log.exists() else f"{len(err_text)} bytes",
        "stray_tmp_files": len(list(FULL.glob("**/*.tmp")))}.items()])
    integ.to_csv(out_dir / "1_integrity.csv", index=False)
    ok1 = (all(per_ds == len(EXPERIMENTS) * len(OPTIMIZERS) * len(seeds)) and dup == 0 and not missing and not extra
           and bad_ncalls.empty and bad_calls.empty and no_design.empty and not_shared.empty and not problems and not err_text)
    verdict[1] = ("PASS" if ok1 else "FLAG",
                  f"{len(runs)} runs ({', '.join(f'{d} {per_ds[d]}' for d in datasets)}); duplicates {dup}, missing {len(missing)}; "
                  f"all runs 100 calls in order: {bad_calls.empty and bad_ncalls.empty}; calls 1-10 identical across the "
                  f"{len(OPTIMIZERS)} optimizers in {len(shared) - len(not_shared)}/{len(shared)} (dataset, experiment, seed, call) groups; "
                  f"errors.log {'absent' if not errors_log.exists() else 'PRESENT'}; stray .tmp files {integ.value.iloc[-1]}"
                  + (f"; problems: {problems[:3]}" if problems else ""))

    # ---------- 1c. calls 1-10 vs another run's shared design ----------
    if args.design_ref:
        ref_dir = Path(args.design_ref)
        mism, checked = [], 0
        for (d, e, s), g in first.groupby(["dataset", "experiment", "seed"]):
            ref_path = next(iter(sorted((ref_dir / "calls").glob(f"{d}_{e}_*_s{s}.jsonl"))), None)
            if ref_path is None:
                mism.append((d, e, s, "no reference run"))
                continue
            with ref_path.open() as handle:
                ref_lines = [json.loads(next(handle)) for _ in range(N_INITIAL)]
            ref_sig = [json.dumps(r["config"], sort_keys=True) + "|" + str(r["selected_mask_hex"] or "all_features") + "|"
                       + repr(r["fitness"]) for r in ref_lines]
            for opt, go in g.groupby("optimizer"):
                checked += 1
                if list(go.sort_values("call_idx").sig) != ref_sig:
                    mism.append((d, e, s, opt))
        verdict["1c"] = ("PASS" if not mism else "FLAG",
                         f"calls 1-10 identical to {ref_dir.name}'s shared design in {checked - len(mism)}/{checked} "
                         f"(dataset, experiment, seed, optimizer) runs" + (f"; mismatches: {mism[:5]}" if mism else ""))

    # ---------- 1b. test_summary.csv (whole study) ----------
    ts_path = FULL / "test_summary.csv"
    if ts_path.exists():
        ts = pd.read_csv(ts_path)
        all_ds = meta["config"]["datasets"]
        opt_rows = ts[ts.optimizer != "default"]
        base_rows = ts[ts.optimizer == "default"]
        n_opt_expected = len(all_ds) * len(EXPERIMENTS) * len(OPTIMIZERS) * len(seeds)
        per_ds_opt = opt_rows.groupby("dataset").size().reindex(all_ds, fill_value=0)
        per_ds_base = base_rows.groupby("dataset").size().reindex(all_ds, fill_value=0)
        dup_ts = int(ts.duplicated(["dataset", "experiment", "optimizer", "seed"]).sum())
        base_const = bool(base_rows.groupby("dataset")[[c for c in ts if c.startswith("test_")]].nunique(dropna=True).max().max() <= 1)
        pm_missing = int(sum(ts[ts.dataset == d][PRIMARY[d]].isna().sum() for d in all_ds))
        ok1b = (len(opt_rows) == n_opt_expected and (per_ds_base == len(EXPERIMENTS) * len(seeds)).all()
                and dup_ts == 0 and base_const and pm_missing == 0)
        verdict["1b"] = ("PASS" if ok1b else "FLAG",
                         f"test_summary.csv: {len(opt_rows)} optimizer rows (expected {n_opt_expected}; per dataset "
                         + ", ".join(f"{d} {per_ds_opt[d]}" for d in all_ds) + f") + {len(base_rows)} default rows ("
                         + ", ".join(f"{d} {per_ds_base[d]}" for d in all_ds) + f" = 2 experiments x 30 seeds); duplicates {dup_ts}; "
                         f"default rows constant within dataset: {base_const}; missing primary metric: {pm_missing}")
    else:
        verdict["1b"] = ("FLAG", "test_summary.csv not found")

    # ---------- 2. unique evaluations ----------
    fresh = calls[~calls.cache_hit].groupby("tag").size()
    runs["n_unique_from_calls"] = runs.tag.map(fresh)
    uniq = runs.groupby(["optimizer", "experiment"]).n_unique.agg(["median", "min", "max"]).reindex(
        pd.MultiIndex.from_product([OPTIMIZERS, EXPERIMENTS], names=["optimizer", "experiment"]))
    uniq["flag_median_lt_90"] = uniq["median"] < FLAG_UNIQUE
    # Where do the cache hits happen? (1-based call numbers; last iteration = calls 91-100)
    hits = calls[calls.cache_hit].copy()
    hits[["optimizer", "experiment"]] = hits.tag.str.extract(r"^\w+?_([AB])_(\w+)_s\d+$")[[1, 0]].values
    hits["optimizer"] = hits.optimizer.replace(LABELS)
    n_runs = runs.groupby(["optimizer", "experiment"]).size()
    uniq["cache_hits_per_run"] = hits.groupby(["optimizer", "experiment"]).size().reindex(uniq.index, fill_value=0) / n_runs.reindex(uniq.index)
    last = hits[hits.call_idx >= BUDGET - N_INITIAL].groupby(["optimizer", "experiment"]).size().reindex(uniq.index, fill_value=0)
    uniq["pct_hits_in_last_iteration"] = 100 * last / hits.groupby(["optimizer", "experiment"]).size().reindex(uniq.index)
    uniq.to_csv(out_dir / "2_unique_evaluations.csv")
    runs.groupby(["dataset", "experiment", "optimizer"]).n_unique.agg(["median", "min"]).to_csv(out_dir / "2_unique_by_dataset.csv")
    consistent = bool((runs.n_unique == runs.n_unique_from_calls).all())
    flagged = uniq[uniq.flag_median_lt_90]
    verdict[2] = ("PASS" if flagged.empty and consistent else "FLAG",
                  "median unique per run: " + "; ".join(
                      f"{o} A {uniq.loc[(o, 'A'), 'median']:.0f} (min {uniq.loc[(o, 'A'), 'min']:.0f}) / "
                      f"B {uniq.loc[(o, 'B'), 'median']:.0f} (min {uniq.loc[(o, 'B'), 'min']:.0f})" for o in OPTIMIZERS)
                  + ("" if consistent else "; n_unique disagrees with the calls log")
                  + (f"; below 90: {', '.join(f'{o} {e}' for o, e in flagged.index)}" if len(flagged) else ""))

    # ---------- 2b. cache-hit causes for the reference-faithful variants ----------
    ref_opts = [o for o in OPTIMIZERS if o in ("gwo_ref", "woa_ref")]
    if ref_opts:
        rr = runs[runs.optimizer.isin(ref_opts)].copy()
        rr["hits"] = rr.n_calls - rr.n_unique
        rr["leader_stays"] = rr.optimizer_meta.map(lambda m: json.loads(m).get("leader_spiral_stays", 0))
        # Clipped-corner repeats: every hyperparameter at a bound of its range (the move overshot and was
        # clipped to [0,1] in all 8 coordinates), so distinct x decode to one configuration.
        def at_corner(cfg: dict) -> bool:
            return all(min(abs(cfg[k] - lo), abs(cfg[k] - hi)) < 1e-12 for k, _, lo, hi in HP_SPEC)
        hit_rows = calls[calls.cache_hit]
        corner = hit_rows[hit_rows.config.map(at_corner)].groupby("tag").size()
        rr["corner_hits"] = rr.tag.map(corner).fillna(0).astype(int)
        # Upper bound on schedule-caused hits: non-corner hits not accounted for by leader-stays.
        rr["unexplained"] = (rr.hits - rr.corner_hits - rr.leader_stays).clip(lower=0)
        rr["a_min"] = rr.optimizer_meta.map(lambda m: min(json.loads(m)["a_used"]))
        rr[["tag", "optimizer", "experiment", "hits", "leader_stays", "corner_hits", "unexplained", "a_min"]].to_csv(
            out_dir / "2b_cache_hit_causes.csv", index=False)
        parts, ok = [], True
        for o in ref_opts:
            g = rr[rr.optimizer == o]
            parts.append(f"{o}: hits/run median {g.hits.median():.0f} (max {g.hits.max()}); leader-stays median "
                         f"{g.leader_stays.median():.0f}; clipped-corner repeats total {g.corner_hits.sum()} in "
                         f"{(g.corner_hits > 0).sum()} runs (max {g.corner_hits.max()}/run); unexplained (upper bound on "
                         f"schedule-caused) max {g.unexplained.max()} (runs with any: {(g.unexplained > 0).sum()}); "
                         f"smallest a used {g.a_min.min():.2f}")
            ok &= g.a_min.min() > 0 and g.unexplained.max() <= 2 and (o != "gwo_ref" or g.hits.max() == 0)
        verdict["2b"] = ("PASS" if ok else "FLAG", "; ".join(parts))

    # ---------- 3. boundary hits of SELECTED configs ----------
    cfg = pd.DataFrame(runs.best_config.map(json.loads).tolist())
    rows = []
    for name, kind, lo, hi in HP_SPEC:
        pos = cfg[name].map(lambda v: position(kind, lo, hi, v))
        exp_lo, exp_hi = uniform_bound_rate(kind, lo, hi)
        for o in OPTIMIZERS:
            for scope, mask in (("all", runs.optimizer == o), ("A", (runs.optimizer == o) & (runs.experiment == "A")),
                                ("B", (runs.optimizer == o) & (runs.experiment == "B"))):
                p = pos[mask]
                lo_rate, hi_rate = float((p <= BOUND_TOL + 1e-12).mean()), float((p >= 1 - BOUND_TOL - 1e-12).mean())
                rows.append({"hyperparameter": name, "optimizer": o, "experiments": scope, "n": int(mask.sum()),
                             "at_lower_pct": 100 * lo_rate, "at_upper_pct": 100 * hi_rate,
                             "uniform_expected_lower_pct": 100 * exp_lo, "uniform_expected_upper_pct": 100 * exp_hi,
                             "flag_gt_30pct": max(lo_rate, hi_rate) > FLAG_BOUND})
    bounds = pd.DataFrame(rows)
    bounds.to_csv(out_dir / "3_boundary_hits.csv", index=False)
    bflag = bounds[(bounds.experiments == "all") & bounds.flag_gt_30pct]
    verdict[3] = ("PASS" if bflag.empty else "FLAG",
                  "no hyperparameter has > 30% of selected configs at a bound for any optimizer" if bflag.empty else
                  "; ".join(f"{r.optimizer} {r.hyperparameter}: lower {r.at_lower_pct:.0f}% / upper {r.at_upper_pct:.0f}% "
                            f"(uniform would give {r.uniform_expected_lower_pct:.0f}% / {r.uniform_expected_upper_pct:.0f}%)"
                            for r in bflag.itertuples()))

    # ---------- 4. early stopping never triggered ----------
    runs["n_estimators_max"] = cfg["n_estimators_max"].values
    runs["hit_cap"] = runs.best_iteration + 1 == runs.n_estimators_max
    es = runs.groupby(["optimizer", "experiment"]).hit_cap.mean().mul(100).unstack().reindex(OPTIMIZERS)
    es.to_csv(out_dir / "4_early_stopping_cap.csv")
    runs.groupby(["dataset", "experiment", "optimizer"]).hit_cap.mean().mul(100).to_csv(out_dir / "4_early_stopping_cap_by_dataset.csv")
    by_ds = runs.groupby("dataset").hit_cap.mean().mul(100).reindex(datasets)
    verdict[4] = ("INFO", f"selected configs whose trees hit n_estimators_max (early stopping did not trigger): overall "
                  f"{100 * runs.hit_cap.mean():.1f}%; by dataset " + ", ".join(f"{d} {v:.0f}%" for d, v in by_ds.items()))

    # ---------- 5. outliers (> 3 raw MAD from the method's median) ----------
    rows = []
    for (d, e, o), g in runs.groupby(["dataset", "experiment", "optimizer"]):
        m = PRIMARY[d]
        med = g[m].median()
        mad = (g[m] - med).abs().median()
        if mad == 0:
            continue
        for r in g[(g[m] - med).abs() > MAD_K * mad].itertuples():
            rows.append({"dataset": d, "experiment": e, "optimizer": o, "seed": r.seed, "metric": m.replace("test_", ""),
                         "value": getattr(r, m), "method_median": med, "MAD": mad, "distance_in_MAD": abs(getattr(r, m) - med) / mad,
                         "worse_than_median": (getattr(r, m) > med) == (m == "test_rmse")})
    outl = pd.DataFrame(rows, columns=["dataset", "experiment", "optimizer", "seed", "metric", "value", "method_median", "MAD",
                                       "distance_in_MAD", "worse_than_median"])
    outl = outl.sort_values("distance_in_MAD", ascending=False)
    outl.to_csv(out_dir / "5_outliers.csv", index=False)
    verdict[5] = ("PASS" if outl.empty else "FLAG",
                  f"{len(outl)} of {len(runs)} runs are > 3 MAD (unscaled) from their method's median"
                  + ("" if outl.empty else f" ({int(outl.worse_than_median.sum())} worse, {int((~outl.worse_than_median).sum())} better)"))

    # ---------- 6. convergence ----------
    runs["best_call"] = runs.best_call_idx + 1  # 1-based call number where the final best was first reached
    conv = runs.groupby(["optimizer", "experiment"]).best_call.agg(
        median="median", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75),
        pct_in_last_10=lambda s: 100 * (s > BUDGET - 10).mean(), pct_in_first_10=lambda s: 100 * (s <= N_INITIAL).mean())
    conv = conv.reindex(pd.MultiIndex.from_product([OPTIMIZERS, EXPERIMENTS], names=["optimizer", "experiment"]))
    conv["flag_median_gt_90"] = conv["median"] > FLAG_CONV
    conv.to_csv(out_dir / "6_convergence.csv")
    runs.groupby(["dataset", "experiment", "optimizer"]).best_call.median().to_csv(out_dir / "6_convergence_by_dataset.csv")
    cflag = conv[conv.flag_median_gt_90]
    verdict[6] = ("PASS" if cflag.empty else "FLAG",
                  "median call of final best: " + "; ".join(
                      f"{o} A {conv.loc[(o, 'A'), 'median']:.0f} / B {conv.loc[(o, 'B'), 'median']:.0f}" for o in OPTIMIZERS))

    # ---------- report ----------
    chash = code_hash()
    names = {1: "Integrity", "1b": "Final test_summary.csv", "1c": "Calls 1-10 vs reference design",
             2: "Unique evaluations", "2b": "Cache-hit causes (reference variants)", 3: "Boundary hits (selected configs)",
             4: "Early stopping hit the cap", 5: "Outliers (> 3 MAD)", 6: "Convergence (call of final best)"}
    a0_present = any(o in OPTIMIZERS for o in ("gwo_a0", "woa_a0"))
    report += [f"# PRELIMINARY audit - {FULL.name}: {', '.join(datasets)} - optimizers {', '.join(OPTIMIZERS)}", ""]
    if a0_present:
        report += ["gwo_a0 / woa_a0 = GWO / WOA with the schedule reaching a = 0 (sensitivity analysis only; "
                   "reference-faithful variants are gwo_ref / woa_ref).", ""]
    report += [f"Generated {datetime.now():%Y-%m-%d %H:%M}. Read-only over {FULL}. Supporting CSVs in this folder.", ""]
    report += [f"- **{i}. {names[i]}: {verdict[i][0]}** - {verdict[i][1]}" for i in verdict]  # insertion order
    report += ["", "## 2. Unique evaluations per run (median / min) and where cache hits occur", "",
               "| optimizer | A median | A min | B median | B min | cache hits/run (A / B) | % of hits in calls 91-100 (A / B) |",
               "| --- | --- | --- | --- | --- | --- | --- |"]
    pct = lambda o, e: "-" if pd.isna(uniq.loc[(o, e), "pct_hits_in_last_iteration"]) else f"{uniq.loc[(o, e), 'pct_hits_in_last_iteration']:.0f}%"
    report += [f"| {o} | {uniq.loc[(o, 'A'), 'median']:.0f} | {uniq.loc[(o, 'A'), 'min']:.0f} | "
               f"{uniq.loc[(o, 'B'), 'median']:.0f} | {uniq.loc[(o, 'B'), 'min']:.0f} | "
               f"{uniq.loc[(o, 'A'), 'cache_hits_per_run']:.1f} / {uniq.loc[(o, 'B'), 'cache_hits_per_run']:.1f} | "
               f"{pct(o, 'A')} / {pct(o, 'B')} |" for o in OPTIMIZERS]
    if a0_present:
        report += ["", "gwo_a0/woa_a0: a reaches exactly 0 on the final position update, so A = 0 there. gwo_a0: every wolf "
                   "moves to mean(alpha, beta, delta), i.e. one point -> calls 92-100 are cache hits (effective budget 91). "
                   "woa_a0: whales in the p < 0.5 branch move exactly onto the best-so-far position (a cache hit). In the "
                   "reference GWO code the last evaluated update uses a = 2/T > 0, so this collapse is specific to our "
                   "'a = 0 on the last update' schedule."]
    if ref_opts:
        report += ["", "gwo_ref/woa_ref: reference schedule a = 2 - t*(2/T), evaluated updates use a = 2.0 ... 0.4. woa_ref's "
                   "remaining hits are WOA's built-in leader-stay (the whale on X* takes the spiral with D = 0 and "
                   "re-evaluates X*); see 2b_cache_hit_causes.csv."]
    piv = bounds[bounds.experiments == "all"].assign(
        cell=lambda t: t.at_lower_pct.round(0).astype(int).astype(str) + " / " + t.at_upper_pct.round(0).astype(int).astype(str))
    piv = piv.pivot(index="hyperparameter", columns="optimizer", values="cell").reindex([h[0] for h in HP_SPEC])[list(OPTIMIZERS)]
    expct = bounds[(bounds.experiments == "all") & (bounds.optimizer == OPTIMIZERS[0])].set_index("hyperparameter")
    report += ["", "## 3. % of selected configs at lower / upper bound (A and B pooled; within 1% of range)", "",
               "| hyperparameter | " + " | ".join(OPTIMIZERS) + " | uniform expectation |", "| --- |" + " --- |" * (len(OPTIMIZERS) + 1)]
    report += [f"| {h} | " + " | ".join(piv.loc[h]) + f" | {expct.loc[h, 'uniform_expected_lower_pct']:.0f} / "
               f"{expct.loc[h, 'uniform_expected_upper_pct']:.0f} |" for h in piv.index]
    report += ["", "## 4. % of selected configs where best_iteration + 1 == n_estimators_max", "",
               "| optimizer | A | B |", "| --- | --- | --- |"]
    report += [f"| {o} | {es.loc[o, 'A']:.1f} | {es.loc[o, 'B']:.1f} |" for o in OPTIMIZERS]
    report += ["", "## 5. Outliers (> 3 unscaled MAD from the method's median)", ""]
    if outl.empty:
        report += ["None."]
    else:
        report += ["| dataset | exp | optimizer | seed | metric | value | method median | distance (MAD) | direction |",
                   "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        report += [f"| {r.dataset} | {r.experiment} | {r.optimizer} | {r.seed} | {r.metric} | {r.value:.4f} | "
                   f"{r.method_median:.4f} | {r.distance_in_MAD:.1f} | {'worse' if r.worse_than_median else 'better'} |"
                   for r in outl.itertuples()]
    report += ["", "## 6. Call number (1-100) at which the final best validation fitness was first reached", "",
               "| optimizer | A median [IQR] | A % in calls 91-100 | B median [IQR] | B % in calls 91-100 |",
               "| --- | --- | --- | --- | --- |"]
    report += [f"| {o} | {conv.loc[(o, 'A'), 'median']:.0f} [{conv.loc[(o, 'A'), 'q25']:.0f}, {conv.loc[(o, 'A'), 'q75']:.0f}] | "
               f"{conv.loc[(o, 'A'), 'pct_in_last_10']:.0f}% | {conv.loc[(o, 'B'), 'median']:.0f} "
               f"[{conv.loc[(o, 'B'), 'q25']:.0f}, {conv.loc[(o, 'B'), 'q75']:.0f}] | {conv.loc[(o, 'B'), 'pct_in_last_10']:.0f}% |"
               for o in OPTIMIZERS]
    report += ["", "---", f"Runner code hash now `{chash}`; recorded `{meta['code_sha256']}`; unchanged: **{chash == meta['code_sha256']}**"]
    text = "\n".join(report)
    (out_dir / "AUDIT.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[written to {out_dir}]")


if __name__ == "__main__":
    main()
