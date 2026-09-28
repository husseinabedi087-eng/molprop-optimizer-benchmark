"""PRELIMINARY read-only preview of finished xgb_mh runs.

Pre-specified analyses per dataset x experiment:
  1. Does tuning help?  each optimizer vs the default baseline (median test primary metric,
     fraction of the 30 seeds where the optimizer is strictly better).
  2. Which optimizer is best?  Friedman over the 5 optimizers only (default excluded), mean ranks,
     pairwise Wilcoxon signed-rank (zero_method="pratt"), Holm over the 10 pairs, Vargha-Delaney A12.
  3. Validation-test gap: median best validation score vs median test score, Spearman(val, test) over seeds.
Plus: Experiment B median selected features; secondary metric (PR-AUC / R2) medians.

Reads results/xgb_mh/full/{runs,baseline}/*.json and never writes there. Does NOT import xgb_mh
(importing would create xgb_mh/__pycache__). A dataset is included only when all of its runs exist.
Writes to analysis_preview/output/ only.
Usage: python analysis_preview/preview.py [--datasets BBBP,FreeSolv,BACE,ESOL]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "results" / "xgb_mh" / "full"
PKG = ROOT / "xgb_mh"
OUT = Path(__file__).resolve().parent / "output"
# GWO/WOA in this run use a schedule reaching a = 0 on the last update (sensitivity analysis only);
# reference-faithful variants will be run separately and will be the primary GWO/WOA results.
LABELS = {"gwo": "gwo_a0", "woa": "woa_a0"}
OPTIMIZERS = ("random", "tpe", "gwo_a0", "woa_a0", "pso")
METHODS = ("default",) + OPTIMIZERS
FEATURE_SUMMARY = ROOT / "results" / "xgb_mh" / "features" / "summary.json"
EXPERIMENTS = ("A", "B")
TASK = {"ESOL": "regression", "FreeSolv": "regression", "Lipophilicity": "regression",
        "BBBP": "classification", "BACE": "classification"}
PRIMARY = {"regression": "test_rmse", "classification": "test_roc_auc"}
SECONDARY = {"regression": "test_r2", "classification": "test_pr_auc"}
HIGHER_BETTER = {"test_rmse": False, "test_roc_auc": True}
# Must mirror xgb_mh/full_run.py::RESULT_CODE and code_hash() exactly.
RESULT_CODE = ("__init__.py", "space.py", "objective.py", "experiment.py", "features.py", "data.py", "optimizers/*.py")
BANNER = ("PRELIMINARY - gwo_a0/woa_a0 = GWO/WOA with a reaching 0 (sensitivity analysis only; reference-faithful "
          "variants pending); exploratory, not for conclusions")


def code_hash() -> str:
    digest = hashlib.sha256()
    for pattern in RESULT_CODE:
        for path in sorted(PKG.glob(pattern)):
            digest.update(path.relative_to(PKG).as_posix().encode() + b"\0" + path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def load(dataset: str, seeds: list[int]) -> pd.DataFrame | None:
    """All runs + deterministic baseline replicated per seed; None if the dataset is incomplete."""
    rows = [json.loads(p.read_text()) for p in (FULL / "runs").glob(f"{dataset}_*.json")]
    expected = len(EXPERIMENTS) * len(OPTIMIZERS) * len(seeds)
    if len(rows) != expected:
        print(f"{dataset}: {len(rows)}/{expected} runs finished -> skipped (incomplete)")
        return None
    base = json.loads((FULL / "baseline" / f"{dataset}.json").read_text())
    rows += [{**base, "optimizer": "default", "experiment": e, "seed": s} for e in EXPERIMENTS for s in seeds]
    df = pd.DataFrame(rows)
    df["optimizer"] = df.optimizer.replace(LABELS)
    # Validation score on the same scale as the test metric (raw metric, i.e. without the FS penalty).
    raw = df.best_val_raw_metric
    df["val_score"] = 1.0 - raw if TASK[dataset] == "classification" else raw
    return df


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted, running, m = [0.0] * len(pvalues), 0.0, len(pvalues)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def a12(x: np.ndarray, y: np.ndarray) -> float:
    """Vargha-Delaney A12 on 'goodness' values: P(X better than Y) + 0.5 P(tie). > 0.5 favours X."""
    gt = (x[:, None] > y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(x) * len(y)))


def analyse(df: pd.DataFrame, dataset: str, experiment: str) -> dict:
    task = TASK[dataset]
    pm, sm = PRIMARY[task], SECONDARY[task]
    sign = 1.0 if HIGHER_BETTER[pm] else -1.0  # goodness = sign * metric
    sub = df[df.experiment == experiment]
    wide = sub.pivot(index="seed", columns="optimizer", values=pm)[list(METHODS)]
    good = sign * wide

    # 2. Friedman on the 5 optimizers only; rank 1 = best.
    g = good[list(OPTIMIZERS)].values
    ranks = np.apply_along_axis(lambda r: rankdata(-r), 1, g)
    chi2, p_friedman = friedmanchisquare(*g.T)
    mean_rank = dict(zip(OPTIMIZERS, ranks.mean(axis=0)))

    pairs = []
    for a, b in itertools.combinations(OPTIMIZERS, 2):
        d = good[a].values - good[b].values
        p = 1.0 if np.all(d == 0) else float(wilcoxon(good[a], good[b], zero_method="pratt").pvalue)
        # d is on the goodness scale: > 0 favours `a` (for RMSE this is rmse_b - rmse_a).
        pairs.append({"pair": f"{a} vs {b}", "median_diff": float(np.median(d)), "p_raw": p,
                      "A12": a12(good[a].values, good[b].values)})
    for row, adj in zip(pairs, holm([r["p_raw"] for r in pairs])):
        row["p_holm"] = adj

    # 1 + 3 + extras, per method.
    default = good["default"].values
    methods = []
    for m in METHODS:
        ms = sub[sub.optimizer == m].sort_values("seed")
        row = {"method": m, "test_median": float(ms[pm].median()),
               "beats_default": None if m == "default" else float(np.mean(good[m].values > default)),
               "ties_default": None if m == "default" else float(np.mean(good[m].values == default)),
               "val_median": float(ms.val_score.median()),
               "gap_test_minus_val": float(ms[pm].median() - ms.val_score.median()),
               "spearman_rho": None, "spearman_p": None,
               "secondary_median": float(ms[sm].median()),
               "mean_rank_5opt": mean_rank.get(m),
               "n_selected_median": float(ms.n_selected.median()) if experiment == "B" else None}
        if ms[pm].nunique() > 1 and ms.val_score.nunique() > 1:
            rho, p = spearmanr(ms.val_score, ms[pm])
            row["spearman_rho"], row["spearman_p"] = float(rho), float(p)
        methods.append(row)
    return {"dataset": dataset, "experiment": experiment, "primary": pm, "secondary": sm,
            "n_features_total": int(sub.n_features_total.iloc[0]), "n_seeds": len(wide),
            "friedman_chi2": float(chi2), "friedman_p": float(p_friedman), "methods": methods, "pairs": pairs}


def f(v, spec=".4f"):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)


def fp(p):
    return "-" if p is None else ("<0.001" if p < 1e-3 else f"{p:.3f}")


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = [header, ["---"] * len(header)] + rows
    return "\n".join("| " + " | ".join(r) + " |" for r in lines)


def render_compact(results: list[dict], chash: str, recorded: str) -> str:
    """Short paste-back version: one table per pre-specified analysis."""
    cell = lambda r: f"{r['dataset']} {r['experiment']}"
    by = lambda r: {m["method"]: m for m in r["methods"]}
    out = [f"# {BANNER}", "",
           "Primary: ROC-AUC (BBBP, BACE; higher better) / RMSE (FreeSolv, ESOL, Lipophilicity; lower better). n = 30 seeds. "
           "Default = XGBoost defaults (deterministic).", ""]

    out += ["## PRELIMINARY 1. Does tuning help? test median (% of seeds beating default)", ""]
    rows = [[cell(r), r["primary"].replace("test_", ""), f(by(r)["default"]["test_median"])] +
            [f"{f(by(r)[m]['test_median'])} ({by(r)[m]['beats_default']:.0%})" for m in OPTIMIZERS] for r in results]
    out += [table(["cell", "metric", "default"] + list(OPTIMIZERS), rows), ""]

    out += ["## PRELIMINARY 2. Which optimizer is best? Friedman (5 optimizers) + mean rank (1 = best)", ""]
    rows = []
    for r in results:
        best_pair = min(r["pairs"], key=lambda p: p["p_holm"])
        n_sig = sum(p["p_holm"] < 0.05 for p in r["pairs"])
        rows.append([cell(r), f"{r['friedman_chi2']:.2f}", fp(r["friedman_p"])] +
                    [f(by(r)[m]["mean_rank_5opt"], ".2f") for m in OPTIMIZERS] +
                    [f"{n_sig}/10", f"{best_pair['pair']}: {fp(best_pair['p_holm'])} (A12 {best_pair['A12']:.2f})"])
    out += [table(["cell", "chi2", "p"] + list(OPTIMIZERS) + ["Holm p<.05", "smallest Holm p (pair)"], rows), ""]

    out += ["## PRELIMINARY 3. Validation-test gap: test - val (median) / Spearman rho(val, test) across seeds", ""]
    rows = []
    for r in results:
        b = by(r)
        rho = lambda m: "-" if b[m]["spearman_rho"] is None else f"{b[m]['spearman_rho']:+.2f}" + (
            "*" if b[m]["spearman_p"] < .05 else "")
        rows.append([cell(r), f(b["default"]["gap_test_minus_val"], "+.3f")] +
                    [f"{f(b[m]['gap_test_minus_val'], '+.3f')} / {rho(m)}" for m in OPTIMIZERS])
    out += [table(["cell", "default gap"] + list(OPTIMIZERS), rows),
            "", "Gap sign: ROC-AUC gap < 0 = test below val; RMSE gap > 0 = test error above val. * = Spearman p < 0.05 (unadjusted).", ""]

    out += ["## PRELIMINARY Experiment B: median selected features", ""]
    rows = [[r["dataset"], str(r["n_features_total"])] + [f(by(r)[m]["n_selected_median"], ".0f") for m in OPTIMIZERS]
            for r in results if r["experiment"] == "B"]
    out += [table(["dataset", "F"] + list(OPTIMIZERS), rows), ""]

    out += ["## PRELIMINARY Secondary metric medians (PR-AUC for BBBP/BACE, R2 for FreeSolv/ESOL/Lipophilicity)", ""]
    rows = [[cell(r), r["secondary"].replace("test_", "")] + [f(by(r)[m]["secondary_median"], ".3f") for m in METHODS]
            for r in results]
    out += [table(["cell", "metric"] + list(METHODS), rows), "",
            f"Runner code hash unchanged: **{chash == recorded}** (`{chash[:16]}...`)", ""]
    return "\n".join(out)


PREDICTION = ("Lipophilicity, with the largest validation set (420), should show a stronger validation-test "
              "correlation and a clearer tuning benefit than the other datasets.")
# Decision rule, fixed in code before the Lipophilicity analysis was run:
#   supported        = Lipophilicity ranks 1st of 5 on BOTH median Spearman rho and median fraction beating default
#   partly supported = 1st on exactly one of the two
#   not supported    = 1st on neither
# Medians are over the 10 optimizer x experiment values per dataset (5 optimizers x A, B).


def render_prediction(results: list[dict]) -> str:
    sizes = json.loads(FEATURE_SUMMARY.read_text())
    rows, stats = [], {}
    for d in dict.fromkeys(r["dataset"] for r in results):
        cells = [m for r in results if r["dataset"] == d for m in r["methods"] if m["method"] != "default"]
        rho = [m["spearman_rho"] for m in cells if m["spearman_rho"] is not None]
        beat = [m["beats_default"] for m in cells]
        per_exp = {e: float(np.median([m["spearman_rho"] for r in results if r["dataset"] == d and r["experiment"] == e
                                       for m in r["methods"] if m["method"] != "default" and m["spearman_rho"] is not None]))
                   for e in EXPERIMENTS}
        stats[d] = (float(np.median(rho)), float(np.median(beat)))
        rows.append([d, str(sizes[d]["final_split_sizes"]["validation"]), f"{stats[d][0]:+.2f}",
                     f"{per_exp['A']:+.2f} / {per_exp['B']:+.2f}", f"{min(rho):+.2f} to {max(rho):+.2f}",
                     f"{stats[d][1]:.0%}", f"{min(beat):.0%} to {max(beat):.0%}"])
    out = ["## PRELIMINARY Prediction test", "", f"Prediction (recorded before Lipophilicity results): \"{PREDICTION}\"", "",
           "Decision rule (fixed before running this analysis): supported = Lipophilicity ranks 1st of 5 datasets on BOTH "
           "the median Spearman rho(val, test) and the median fraction of seeds beating the default; partly supported = 1st "
           "on one; not supported = neither. Medians over the 10 optimizer x experiment cells per dataset.", "",
           table(["dataset", "val size", "median rho", "median rho A / B", "rho range", "median beats default",
                  "beats-default range"], rows), ""]
    if "Lipophilicity" in stats:
        # Competition ranking: rank = 1 + number of datasets strictly higher; ties are reported, not broken.
        lip = stats["Lipophilicity"]
        info = []
        n_strict = n_tied = 0
        for k, name in ((0, "median rho"), (1, "median beats-default")):
            higher = [d for d in stats if stats[d][k] > lip[k] + 1e-12]
            tied = [d for d in stats if d != "Lipophilicity" and abs(stats[d][k] - lip[k]) <= 1e-12]
            rank = 1 + len(higher)
            n_strict += rank == 1 and not tied
            n_tied += rank == 1 and bool(tied)
            info.append(f"{name}: Lipophilicity {lip[k]:.4f}, rank {rank}"
                        + (f" (tied with {', '.join(tied)})" if tied else "")
                        + (f"; higher: {', '.join(f'{d} {stats[d][k]:.4f}' for d in higher)}" if higher else ""))
        word = {2: "SUPPORTED", 1: "PARTLY SUPPORTED", 0: "NOT SUPPORTED"}
        out += ["Exact values: " + " | ".join(info), "",
                f"**Verdict, ties NOT counted as 1st (Lipophilicity must exceed every other dataset, as the prediction's "
                f"wording 'than the other datasets' implies): {word[n_strict]}.**",
                f"Verdict if ties counted as joint 1st (the rule did not specify ties): {word[n_strict + n_tied]}.", ""]
    return "\n".join(out)


def render(results: list[dict], chash: str, recorded: str) -> str:
    out = [f"# {BANNER}", "", f"Generated {datetime.now():%Y-%m-%d %H:%M}. Source: results/xgb_mh/full (read-only). "
           "30 tuning seeds per method; default = XGBoost defaults, deterministic (zero spread by design).",
           "Primary: ROC-AUC (BBBP, BACE; higher better), RMSE (FreeSolv, ESOL, Lipophilicity; lower better). "
           "Val score = best validation raw metric (Exp B: without the feature penalty).", ""]

    out += ["## PRELIMINARY overview", ""]
    rows = []
    for r in results:
        by = {m["method"]: m for m in r["methods"]}
        best = min(OPTIMIZERS, key=lambda m: by[m]["mean_rank_5opt"])
        n_sig = sum(p["p_holm"] < 0.05 for p in r["pairs"])
        better_med = sum((by[m]["test_median"] > by["default"]["test_median"]) == HIGHER_BETTER[r["primary"]]
                         and by[m]["test_median"] != by["default"]["test_median"] for m in OPTIMIZERS)
        rows.append([r["dataset"], r["experiment"], r["primary"].replace("test_", ""), f(by["default"]["test_median"]),
                     f"{better_med}/5", f"{best} ({by[best]['mean_rank_5opt']:.2f})", fp(r["friedman_p"]), f"{n_sig}/10"])
    out += [table(["dataset", "exp", "primary", "default", "optimizers beating default (median)",
                   "best optimizer (mean rank)", "Friedman p (5 opt)", "Holm-sig. pairs"], rows), ""]

    for r in results:
        pm = r["primary"].replace("test_", "")
        sm = r["secondary"].replace("test_", "")
        label = "HPO only" if r["experiment"] == "A" else "joint FS + HPO"
        out += [f"## PRELIMINARY - {r['dataset']}, Experiment {r['experiment']} ({label})", "",
                f"Friedman on 5 optimizers (default excluded), n = {r['n_seeds']} seeds: "
                f"chi2 = {r['friedman_chi2']:.2f}, p = {fp(r['friedman_p'])}", ""]
        header = ["method", f"test {pm} (median)", "beats default", "mean rank", f"val {pm} (median)",
                  "test - val", "Spearman val~test (p)", f"{sm} (median)"]
        if r["experiment"] == "B":
            header.append(f"n selected (of {r['n_features_total']})")
        rows = []
        for m in sorted(r["methods"], key=lambda m: (m["mean_rank_5opt"] is None, m["mean_rank_5opt"] or 0)):
            beats = "-" if m["beats_default"] is None else f"{m['beats_default']:.0%}" + (
                f" (ties {m['ties_default']:.0%})" if m["ties_default"] else "")
            rho = "-" if m["spearman_rho"] is None else f"{m['spearman_rho']:+.2f} ({fp(m['spearman_p'])})"
            row = [m["method"], f(m["test_median"]), beats, f(m["mean_rank_5opt"], ".2f"), f(m["val_median"]),
                   f(m["gap_test_minus_val"], "+.4f"), rho, f(m["secondary_median"])]
            if r["experiment"] == "B":
                row.append(f(m["n_selected_median"], ".0f"))
            rows.append(row)
        out += [table(header, rows), ""]
        rows = [[p["pair"], f(p["median_diff"], "+.4f"), fp(p["p_raw"]), fp(p["p_holm"]) + (" *" if p["p_holm"] < .05 else ""),
                 f(p["A12"], ".2f")] for p in r["pairs"]]
        out += [f"Pairwise Wilcoxon (pratt), Holm over 10 pairs. median_diff and A12 oriented so that "
                f"> 0 / > 0.5 favours the FIRST method.", "",
                table(["pair", "median diff", "p raw", "p Holm", "A12"], rows), ""]

    out += ["---", f"Runner code hash now: `{chash}`", f"Recorded at run start: `{recorded}`",
            f"Unchanged: **{chash == recorded}**", "", f"_{BANNER}_"]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="BBBP,FreeSolv,BACE,ESOL,Lipophilicity")
    args = parser.parse_args()
    meta = json.loads((FULL / "run_metadata.json").read_text())
    seeds = meta["config"]["seeds"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT.mkdir(exist_ok=True)

    results = []
    for d in args.datasets.split(","):
        df = load(d, seeds)
        if df is not None:
            results += [analyse(df, d, e) for e in EXPERIMENTS]

    chash, recorded = code_hash(), meta["code_sha256"]
    report = render(results, chash, recorded)
    compact = render_compact(results, chash, recorded) + "\n" + render_prediction(results)
    (OUT / f"PRELIMINARY_preview_{stamp}.md").write_text(report, encoding="utf-8")
    (OUT / f"PRELIMINARY_compact_{stamp}.md").write_text(compact, encoding="utf-8")
    pd.DataFrame([{"dataset": r["dataset"], "experiment": r["experiment"], **m} for r in results for m in r["methods"]]) \
        .to_csv(OUT / f"PRELIMINARY_methods_{stamp}.csv", index=False)
    pd.DataFrame([{"dataset": r["dataset"], "experiment": r["experiment"], **p} for r in results for p in r["pairs"]]) \
        .to_csv(OUT / f"PRELIMINARY_pairwise_{stamp}.csv", index=False)
    pd.DataFrame([{k: r[k] for k in ("dataset", "experiment", "primary", "n_seeds", "friedman_chi2", "friedman_p")}
                  for r in results]).to_csv(OUT / f"PRELIMINARY_friedman_{stamp}.csv", index=False)
    print(compact)
    print(f"\n[full report + compact + CSVs written to {OUT}\\PRELIMINARY_*_{stamp}.*]")


if __name__ == "__main__":
    main()
