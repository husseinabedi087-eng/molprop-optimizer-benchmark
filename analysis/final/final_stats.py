"""Step 5: final statistics. Read-only over results/; writes analysis/final/ only.

Primary methods: Random, TPE, PSO (results/xgb_mh/full), GWO = gwo_ref, WOA = woa_ref (results/xgb_mh/full_ref),
Default. gwo_a0 / woa_a0 (results/xgb_mh/full: gwo, woa) are used only in the sensitivity analysis (6).
Usage (Idle priority): python analysis/final/final_stats.py
"""
from __future__ import annotations

import hashlib
import itertools
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = ROOT / "analysis" / "figures"
sys.path.insert(0, str(FIG))
from data import feature_mask  # noqa: E402
from stats import BOOT_N, BOOT_SEED, a12, average_ranks, bootstrap_median_ci, nemenyi_cd, nogueira_stability  # noqa: E402

RES = ROOT / "results" / "xgb_mh"
FULL, FULL_REF, MIRROR = RES / "full", RES / "full_ref", RES / "mirror"
MIRROR_EVAL_DIR = ROOT / "analysis_preview" / "output" / "mirror_20260928_111045"
OUT = HERE
ANALYSIS_FILES = [HERE / "final_stats.py", FIG / "stats.py", FIG / "data.py", FIG / "style.py", FIG / "figures.py"]

DATASETS = ("ESOL", "FreeSolv", "Lipophilicity", "BBBP", "BACE")
EXPERIMENTS = ("A", "B")
TASK = {"ESOL": "regression", "FreeSolv": "regression", "Lipophilicity": "regression",
        "BBBP": "classification", "BACE": "classification"}
# (column, label, higher_is_better, role)
METRICS = {"regression": [("test_rmse", "RMSE", False, "primary"), ("test_mae", "MAE", False, "secondary"),
                          ("test_r2", "R2", True, "secondary")],
           "classification": [("test_roc_auc", "ROC-AUC", True, "primary"), ("test_pr_auc", "PR-AUC", True, "secondary")]}
OPTS = ("Random", "TPE", "GWO", "WOA", "PSO")
METHODS = ("Default",) + OPTS
MAIN = {"default": "Default", "random": "Random", "tpe": "TPE", "pso": "PSO"}
REF = {"gwo_ref": "GWO", "woa_ref": "WOA"}
A0 = {"gwo": "GWO", "woa": "WOA"}
CALL_CAP_S = 600  # calls longer than this include machine sleep and are dropped from compute statistics


# ------------------------------------------------------------------------------------------------ helpers
def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def holm(p) -> list[float]:
    p = list(p)
    order, adj, running, m = np.argsort(p), [0.0] * len(p), 0.0, len(p)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def wilcox(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=float)
    return 1.0 if np.all(d == 0) else float(wilcoxon(d, zero_method="pratt").pvalue)


def spearman_boot_ci(x: np.ndarray, y: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    rho = float(spearmanr(x, y)[0])
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(x), (BOOT_N, len(x)))
    rx, ry = rankdata(x[idx], axis=1), rankdata(y[idx], axis=1)
    rx -= rx.mean(1, keepdims=True)
    ry -= ry.mean(1, keepdims=True)
    den = np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))
    r = np.where(den > 0, (rx * ry).sum(1) / np.where(den > 0, den, 1), np.nan)
    lo, hi = np.nanpercentile(r, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return rho, float(lo), float(hi)


def fp(p) -> str:
    return "-" if p is None or (isinstance(p, float) and np.isnan(p)) else ("<0.001" if p < 1e-3 else f"{p:.3f}")


def md_table(header, rows) -> str:
    return "\n".join("| " + " | ".join(map(str, r)) + " |" for r in [header, ["---"] * len(header)] + rows)


def load(variant: str = "ref") -> pd.DataFrame:
    full = pd.read_csv(FULL / "test_summary.csv")
    parts = [full[full.optimizer.isin(MAIN)].assign(method=lambda d: d.optimizer.map(MAIN), source="full")]
    if variant == "ref":
        ref = pd.read_csv(FULL_REF / "test_summary.csv")
        parts.append(ref[ref.optimizer.isin(REF)].assign(method=lambda d: d.optimizer.map(REF), source="full_ref"))
    else:
        parts.append(full[full.optimizer.isin(A0)].assign(method=lambda d: d.optimizer.map(A0), source="full (a0)"))
    df = pd.concat(parts, ignore_index=True)
    df["task"] = df.dataset.map(TASK)
    df["val_primary"] = np.where(df.task == "classification", 1 - df.best_val_raw_metric, df.best_val_raw_metric)
    df["test_primary"] = np.where(df.task == "classification", df.test_roc_auc, df.test_rmse)
    df["seed"] = df.seed.astype(int)
    return df


def cell(df, ds, exp, method, col) -> np.ndarray:
    s = df[(df.dataset == ds) & (df.experiment == exp) & (df.method == method)].sort_values("seed")
    return s[col].to_numpy(dtype=float)


# ------------------------------------------------------------------------------------------------ analyses
def tuning_vs_default(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for exp in EXPERIMENTS:
            for col, label, hb, role in METRICS[TASK[ds]]:
                d0 = float(cell(df, ds, exp, "Default", col)[0])
                fam = []
                for m in OPTS:
                    x = cell(df, ds, exp, m, col)
                    imp = (x - d0) if hb else (d0 - x)  # improvement over default: > 0 = better
                    med, lo, hi = bootstrap_median_ci(x)
                    dmed, dlo, dhi = bootstrap_median_ci(imp)
                    fam.append({"dataset": ds, "experiment": exp, "metric": label, "role": role, "method": m,
                                "default": d0, "median": med, "ci_low": lo, "ci_high": hi,
                                "median_improvement": dmed, "impr_ci_low": dlo, "impr_ci_high": dhi,
                                "pct_seeds_beating_default": 100 * float(np.mean(imp > 0)),
                                "pct_seeds_tied": 100 * float(np.mean(imp == 0)), "p_raw": wilcox(imp)})
                for r, adj in zip(fam, holm([r["p_raw"] for r in fam])):
                    r["p_holm"] = adj
                    r["conclusion"] = ("better" if r["median_improvement"] > 0 else "worse") if adj < 0.05 else "n.s."
                rows += fam
    return pd.DataFrame(rows)


def optimizer_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fr, pw = [], []
    for ds in DATASETS:
        for exp in EXPERIMENTS:
            for col, label, hb, role in METRICS[TASK[ds]]:
                g = np.column_stack([cell(df, ds, exp, m, col) * (1 if hb else -1) for m in OPTS])  # goodness
                chi2, p = friedmanchisquare(*g.T)
                ranks = np.apply_along_axis(lambda r: rankdata(-r), 1, g).mean(0)
                fr.append({"dataset": ds, "experiment": exp, "metric": label, "role": role, "friedman_chi2": float(chi2),
                           "friedman_p": float(p), **{f"mean_rank_{m}": float(r) for m, r in zip(OPTS, ranks)}})
                fam = []
                for (i, a), (j, b) in itertools.combinations(enumerate(OPTS), 2):
                    d = g[:, i] - g[:, j]
                    fam.append({"dataset": ds, "experiment": exp, "metric": label, "role": role, "pair": f"{a} vs {b}",
                                "median_diff_goodness": float(np.median(d)), "p_raw": wilcox(d),
                                "A12": a12(g[:, i], g[:, j])})
                for r, adj in zip(fam, holm([r["p_raw"] for r in fam])):
                    r["p_holm"] = adj
                pw += fam
    return pd.DataFrame(fr), pd.DataFrame(pw)


def validation_reliability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for exp in EXPERIMENTS:
            for m in METHODS:
                v, t = cell(df, ds, exp, m, "val_primary"), cell(df, ds, exp, m, "test_primary")
                row = {"dataset": ds, "experiment": exp, "method": m, "metric": "ROC-AUC" if TASK[ds] == "classification" else "RMSE",
                       "median_val": float(np.median(v)), "median_test": float(np.median(t)),
                       "median_gap_test_minus_val": float(np.median(t - v)), "rho": np.nan, "rho_ci_low": np.nan,
                       "rho_ci_high": np.nan}
                if m != "Default":
                    row["rho"], row["rho_ci_low"], row["rho_ci_high"] = spearman_boot_ci(v, t)
                rows.append(row)
    return pd.DataFrame(rows)


def feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for m in OPTS:
            s = df[(df.dataset == ds) & (df.experiment == "B") & (df.method == m)].sort_values("seed")
            F = int(s.n_features_total.iloc[0])
            frac = s.n_selected.to_numpy() / F
            med, lo, hi = bootstrap_median_ci(frac)
            stab, slo, shi = nogueira_stability(np.vstack([feature_mask(h, F) for h in s.best_selected_mask_hex]))
            rows.append({"dataset": ds, "method": m, "F": F, "fraction_median": med, "fraction_ci_low": lo,
                         "fraction_ci_high": hi, "fraction_q25": float(np.quantile(frac, .25)),
                         "fraction_q75": float(np.quantile(frac, .75)), "nogueira": stab, "nogueira_ci_low": slo,
                         "nogueira_ci_high": shi})
    return pd.DataFrame(rows)


def sensitivity(ref: pd.DataFrame, a0: pd.DataFrame, tvd_ref: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for m in ("GWO", "WOA"):
        fam = []
        for ds in DATASETS:
            for exp in EXPERIMENTS:
                hb = TASK[ds] == "classification"
                r, z = cell(ref, ds, exp, m, "test_primary"), cell(a0, ds, exp, m, "test_primary")
                d = (r - z) if hb else (z - r)  # > 0: ref better than a0
                fam.append({"method": m, "dataset": ds, "experiment": exp, "metric": "ROC-AUC" if hb else "RMSE",
                            "median_ref": float(np.median(r)), "median_a0": float(np.median(z)),
                            "median_paired_diff_ref_better": float(np.median(d)), "p_raw": wilcox(d),
                            "unique_ref_median": float(np.median(cell(ref, ds, exp, m, "n_unique"))),
                            "unique_a0_median": float(np.median(cell(a0, ds, exp, m, "n_unique")))})
        for row, adj in zip(fam, holm([f["p_raw"] for f in fam])):  # Holm over the 10 cells, per method
            row["p_holm"] = adj
        rows += fam
    paired = pd.DataFrame(rows)
    tvd_a0 = tuning_vs_default(a0)
    key = ["dataset", "experiment", "metric", "method"]
    cmp = tvd_ref[tvd_ref.role == "primary"][key + ["conclusion", "p_holm", "median_improvement"]].merge(
        tvd_a0[tvd_a0.role == "primary"][key + ["conclusion", "p_holm", "median_improvement"]], on=key, suffixes=("_ref", "_a0"))
    cmp["changed"] = cmp.conclusion_ref != cmp.conclusion_a0
    return paired, cmp


def cd_optimizers(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Path]]:
    import style
    from figures import _draw_cd
    import matplotlib.pyplot as plt
    style.apply()
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(style.SINGLE_COL, 3.6))
    for idx, exp in enumerate(EXPERIMENTS):
        good = np.array([[np.median(cell(df, ds, exp, m, "test_primary")) * (1 if TASK[ds] == "classification" else -1)
                          for m in OPTS] for ds in DATASETS])
        ranks = average_ranks(good)
        cd = nemenyi_cd(len(OPTS), len(DATASETS))
        p = float(friedmanchisquare(*good.T).pvalue)
        _draw_cd(axes[idx], list(OPTS), ranks, cd,
                 f"Experiment {exp}: optimizers, mean rank over N = {len(DATASETS)} datasets\n"
                 f"LOW-POWERED; Nemenyi CD at α = 0.05; Friedman p = {p:.2f}")
        style.panel_label(axes[idx], idx, x=0.0, y=0.93)
        rows += [{"experiment": exp, "method": m, "mean_rank": float(r), "CD": float(cd), "friedman_p_across_datasets": p,
                  "N_datasets": len(DATASETS)} for m, r in zip(OPTS, ranks)]
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.02, hspace=0.15)
    return pd.DataFrame(rows), style.save(fig, "figS_cd_optimizers", OUT)


def headroom(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for ds in DATASETS:
        hb = TASK[ds] == "classification"
        for exp in EXPERIMENTS:
            dval = float(cell(df, ds, exp, "Default", "val_primary")[0])
            dtest = float(cell(df, ds, exp, "Default", "test_primary")[0])
            opt = df[(df.dataset == ds) & (df.experiment == exp) & (df.method.isin(OPTS))]
            v, t = float(opt.val_primary.median()), float(opt.test_primary.median())
            head = (v - dval) if hb else (dval - v)       # validation improvement available/realised over default
            gain = (t - dtest) if hb else (dtest - t)     # median test improvement over default
            rows.append({"dataset": ds, "experiment": exp, "default_val": dval, "median_best_val_optimizers": v,
                         "val_headroom": head, "val_headroom_pct": 100 * head / abs(dval), "default_test": dtest,
                         "median_test_optimizers": t, "test_improvement": gain, "test_improvement_pct": 100 * gain / abs(dtest)})
    h = pd.DataFrame(rows)
    rho_pct, p_pct = spearmanr(h.val_headroom_pct, h.test_improvement_pct)
    rho_raw, p_raw = spearmanr(h.val_headroom, h.test_improvement)
    return h, {"spearman_rho_pct": float(rho_pct), "p_pct": float(p_pct), "spearman_rho_raw_units": float(rho_raw),
               "p_raw_units": float(p_raw), "n_cells": len(h)}


def compute_stats() -> tuple[pd.DataFrame, dict]:
    sources = [(FULL, o, MAIN[o]) for o in ("random", "tpe", "pso")] + [(FULL_REF, o, REF[o]) for o in REF]
    rows, dropped = [], []
    for src, opt, name in sources:
        for p in sorted((src / "calls").glob(f"*_{opt}_s*.jsonl")):
            run = json.loads((src / "runs" / f"{p.stem}.json").read_text())
            with p.open() as handle:
                calls = [json.loads(line) for line in handle]
            fresh = [c for c in calls if not c["cache_hit"]]
            slow = [c for c in calls if c["wall_time"] > CALL_CAP_S]
            dropped += [{"run": p.stem, "call": c["call_idx"] + 1, "wall_time_s": c["wall_time"]} for c in slow]
            kept = [c for c in calls if c["wall_time"] <= CALL_CAP_S]
            rows.append({"dataset": run["dataset"], "experiment": run["experiment"], "method": name,
                         "median_fresh_call_s": float(np.median([c["wall_time"] for c in fresh if c["wall_time"] <= CALL_CAP_S])),
                         "n_unique": run["n_unique"], "n_dropped_calls": len(slow),
                         "run_time_s_clean": sum(c["wall_time"] + c["optimizer_overhead"] for c in kept) + run["refit_time"],
                         "run_time_s_raw": run["search_time"] + run["refit_time"]})
    per_run = pd.DataFrame(rows)
    per_run.to_csv(OUT / "9_compute_per_run.csv", index=False)
    pd.DataFrame(dropped).to_csv(OUT / "9_dropped_calls_over_600s.csv", index=False)
    agg = per_run.groupby(["dataset", "method"]).agg(
        median_s_per_call=("median_fresh_call_s", "median"), median_unique=("n_unique", "median"),
        min_unique=("n_unique", "min"), total_cpu_h_clean=("run_time_s_clean", lambda s: s.sum() / 3600),
        total_cpu_h_raw=("run_time_s_raw", lambda s: s.sum() / 3600), dropped_calls=("n_dropped_calls", "sum")).reset_index()
    return agg, {"calls_dropped": len(dropped), "runs_affected": int((per_run.n_dropped_calls > 0).sum()),
                 "total_cpu_h_clean": float(per_run.run_time_s_clean.sum() / 3600),
                 "total_cpu_h_raw": float(per_run.run_time_s_raw.sum() / 3600)}


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    t0 = datetime.now()
    ref, a0 = load("ref"), load("a0")
    assert len(ref[ref.method != "Default"]) == 1500 and len(ref[ref.method == "Default"]) == 300
    prevalence = {d: json.loads((RES / "features" / "summary.json").read_text())[d]["positive_rate"]["test"]
                  for d in ("BBBP", "BACE")}

    tvd = tuning_vs_default(ref)
    tvd.to_csv(OUT / "1_tuning_vs_default.csv", index=False)
    fr, pw = optimizer_comparison(ref)
    fr.to_csv(OUT / "2_friedman.csv", index=False)
    pw.to_csv(OUT / "2_pairwise.csv", index=False)
    vr = validation_reliability(ref)
    vr.to_csv(OUT / "3_validation_reliability.csv", index=False)
    fs = feature_selection(ref)
    fs.to_csv(OUT / "4_feature_selection.csv", index=False)
    shutil.copyfile(MIRROR_EVAL_DIR / "prespecified_tests.csv", OUT / "5_mirror_prespecified_tests.csv")
    mirror_md = (MIRROR_EVAL_DIR / "MIRROR_EVAL.md").read_text(encoding="utf-8")
    sens, rq1 = sensitivity(ref, a0, tvd)
    sens.to_csv(OUT / "6_sensitivity_a0_vs_ref.csv", index=False)
    rq1.to_csv(OUT / "6_sensitivity_rq1_conclusions.csv", index=False)
    cd, cd_files = cd_optimizers(ref)
    cd.to_csv(OUT / "7_cd_ranks.csv", index=False)
    hr, hr_stats = headroom(ref)
    hr.to_csv(OUT / "8_headroom_EXPLORATORY.csv", index=False)
    comp, comp_tot = compute_stats()
    comp.to_csv(OUT / "9_compute.csv", index=False)

    # ---------- provenance ----------
    analysis_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in ANALYSIS_FILES}
    analysis_hash = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(analysis_hashes.items())).encode()).hexdigest()
    provenance = {
        "generated": t0.isoformat(timespec="seconds"), "analysis_code_sha256": analysis_hash, "analysis_files": analysis_hashes,
        "bootstrap": {"resamples": BOOT_N, "seed": BOOT_SEED, "type": "percentile, resampling tuning seeds"},
        "inputs": {name: {"test_summary_sha256": sha(d / "test_summary.csv"),
                          "run_code_sha256": json.loads((d / "run_metadata.json").read_text())["code_sha256"]}
                   for name, d in (("full", FULL), ("full_ref", FULL_REF), ("mirror", MIRROR))},
        "mirror_eval": {"source": str(MIRROR_EVAL_DIR.relative_to(ROOT)).replace("\\", "/"),
                        "prediction_md_sha256": sha(MIRROR / "prediction.md")},
        "methods": {"Random/TPE/PSO/Default": "results/xgb_mh/full", "GWO/WOA": "results/xgb_mh/full_ref (gwo_ref, woa_ref)",
                    "sensitivity only": "gwo_a0/woa_a0 = results/xgb_mh/full (gwo, woa)"},
        "pr_auc_test_prevalence": prevalence,
    }
    rec = lambda d: json.loads(d.to_json(orient="records"))
    (OUT / "final_stats.json").write_text(json.dumps({
        "provenance": provenance, "tuning_vs_default": rec(tvd), "friedman": rec(fr), "pairwise": rec(pw),
        "validation_reliability": rec(vr), "feature_selection": rec(fs), "sensitivity_paired": rec(sens),
        "sensitivity_rq1": rec(rq1), "cd_ranks": rec(cd), "headroom_EXPLORATORY": {"cells": rec(hr), **hr_stats},
        "compute": {"per_dataset_method": rec(comp), **comp_tot}}, indent=1), encoding="utf-8")

    # ---------- RESULTS.md ----------
    L = ["# Final statistics - xgb_mh", "",
         f"Generated {t0:%Y-%m-%d %H:%M}. Analysis code sha256 `{analysis_hash}` (files in final_stats.json). "
         f"Inputs: full `{provenance['inputs']['full']['run_code_sha256'][:12]}`, full_ref "
         f"`{provenance['inputs']['full_ref']['run_code_sha256'][:12]}`, mirror `{provenance['inputs']['mirror']['run_code_sha256'][:12]}`.",
         "GWO / WOA = reference-faithful gwo_ref / woa_ref. n = 30 tuning seeds per method. Bootstrap: 10,000 percentile "
         f"resamples of seeds, seed {BOOT_SEED}. Wilcoxon: zero_method = pratt. Improvement over default is oriented so > 0 = better.", ""]

    def tvd_row(ds, exp, metric):
        g = tvd[(tvd.dataset == ds) & (tvd.experiment == exp) & (tvd.metric == metric)].set_index("method")
        return [ds, exp, f"{g.default.iloc[0]:.3f}"] + [
            f"{g.loc[m, 'median_improvement']:+.3f} [{g.loc[m, 'impr_ci_low']:+.3f}, {g.loc[m, 'impr_ci_high']:+.3f}] "
            f"{g.loc[m, 'pct_seeds_beating_default']:.0f}%{' *' if g.loc[m, 'p_holm'] < .05 else ''}" for m in OPTS]
    L += ["## 1. Tuning vs default (primary metric)", "",
          "Cell: median improvement over default [bootstrap 95% CI], % of seeds beating default; * = one-sample Wilcoxon "
          "vs the default value, Holm over the 5 optimizers, p < 0.05.", "",
          md_table(["dataset", "exp", "default"] + list(OPTS),
                   [tvd_row(ds, e, METRICS[TASK[ds]][0][1]) for ds in DATASETS for e in EXPERIMENTS]), ""]
    sec = tvd[tvd.role == "secondary"].groupby(["metric", "conclusion"]).size().unstack(fill_value=0)
    L += ["Secondary metrics (own Holm families; full table in 1_tuning_vs_default.csv). Count of optimizer x dataset x "
          f"experiment cells by conclusion. PR-AUC test prevalence (chance level): BBBP {prevalence['BBBP']:.3f}, "
          f"BACE {prevalence['BACE']:.3f}.", "",
          md_table(["metric"] + list(sec.columns), [[i] + list(r) for i, r in sec.iterrows()]), ""]

    L += ["## 2. Optimizers (default excluded; primary metric)", "",
          md_table(["dataset", "exp", "Friedman chi2", "p"] + [f"rank {m}" for m in OPTS] + ["Holm p<.05 pairs", "smallest Holm p"],
                   [[r.dataset, r.experiment, f"{r.friedman_chi2:.2f}", fp(r.friedman_p)] +
                    [f"{getattr(r, f'mean_rank_{m}'):.2f}" for m in OPTS] +
                    [f"{int((g := pw[(pw.dataset == r.dataset) & (pw.experiment == r.experiment) & (pw.metric == r.metric)]).p_holm.lt(.05).sum())}/10",
                     f"{g.loc[g.p_holm.idxmin(), 'pair']}: {fp(g.p_holm.min())} (A12 {g.loc[g.p_holm.idxmin(), 'A12']:.2f})"]
                    for r in fr[fr.role == "primary"].itertuples()]), ""]
    secf = fr[fr.role == "secondary"]
    L += [f"Secondary metrics: Friedman p < 0.05 in {int((secf.friedman_p < .05).sum())}/{len(secf)} cells; Holm-significant "
          f"pairs {int((pw[pw.role == 'secondary'].p_holm < .05).sum())}/{int((pw.role == 'secondary').sum())} "
          "(2_friedman.csv, 2_pairwise.csv).", ""]

    L += ["## 3. Validation reliability (primary metric)", "",
          "Cell: median (test - val) / Spearman rho(val, test) over 30 seeds [bootstrap 95% CI].", "",
          md_table(["dataset", "exp", "Default gap"] + list(OPTS),
                   [[ds, e, f"{vr[(vr.dataset == ds) & (vr.experiment == e) & (vr.method == 'Default')].median_gap_test_minus_val.iloc[0]:+.3f}"] +
                    [(lambda r: f"{r.median_gap_test_minus_val:+.3f} / {r.rho:+.2f} [{r.rho_ci_low:+.2f}, {r.rho_ci_high:+.2f}]")(
                        vr[(vr.dataset == ds) & (vr.experiment == e) & (vr.method == m)].iloc[0]) for m in OPTS]
                    for ds in DATASETS for e in EXPERIMENTS]), ""]

    L += ["## 4. Experiment B feature selection", "",
          "Cell: median selected fraction [bootstrap 95% CI] / Nogueira stability [95% CI].", "",
          md_table(["dataset", "F"] + list(OPTS),
                   [[ds, int(fs[fs.dataset == ds].F.iloc[0])] +
                    [(lambda r: f"{r.fraction_median:.2f} [{r.fraction_ci_low:.2f}, {r.fraction_ci_high:.2f}] / "
                                f"{r.nogueira:.3f} [{r.nogueira_ci_low:.3f}, {r.nogueira_ci_high:.3f}]")(
                        fs[(fs.dataset == ds) & (fs.method == m)].iloc[0]) for m in OPTS] for ds in DATASETS]), ""]

    L += ["## 5. Mirror check (included as-is from MIRROR_EVAL.md)", "", "```text", mirror_md.strip(), "```", ""]

    L += ["## 6. Sensitivity: a0 schedule vs reference schedule (primary metric, paired by seed)", "",
          "median paired diff > 0 = reference variant better; Holm over the 10 dataset x experiment cells, per method.", "",
          md_table(["method", "dataset", "exp", "median ref", "median a0", "paired diff", "p Holm", "unique ref / a0"],
                   [[r.method, r.dataset, r.experiment, f"{r.median_ref:.4f}", f"{r.median_a0:.4f}",
                     f"{r.median_paired_diff_ref_better:+.4f}", fp(r.p_holm), f"{r.unique_ref_median:.0f} / {r.unique_a0_median:.0f}"]
                    for r in sens.itertuples()]), ""]
    ch = rq1[rq1.changed]
    L += [f"RQ1 (tuning vs default) conclusions with a0 replacing ref: {len(ch)} of {len(rq1)} optimizer x cell conclusions change"
          + ("." if ch.empty else ":"), ""]
    if not ch.empty:
        L += [md_table(["dataset", "exp", "method", "ref", "a0"], [[r.dataset, r.experiment, r.method, r.conclusion_ref,
                                                                   r.conclusion_a0] for r in ch.itertuples()]), ""]

    L += ["## 7. Supplement: CD diagrams across datasets (optimizers only; LOW-POWERED, N = 5)", "",
          md_table(["exp"] + list(OPTS) + ["CD", "Friedman p"],
                   [[e] + [f"{cd[(cd.experiment == e) & (cd.method == m)].mean_rank.iloc[0]:.2f}" for m in OPTS] +
                    [f"{cd.CD.iloc[0]:.2f}", fp(cd[cd.experiment == e].friedman_p_across_datasets.iloc[0])] for e in EXPERIMENTS]),
          "", f"Figure: {', '.join(p.name for p in cd_files)}", ""]

    L += ["## 8. EXPLORATORY (not pre-specified): validation headroom vs test gain", "",
          "Headroom = median best validation score of the 5 optimizers relative to the default's validation score, oriented "
          "so > 0 = optimizers better on validation (for RMSE: default - optimizers; for ROC-AUC: optimizers - default). "
          "Test gain = median test improvement over default (same orientation). % = relative to the default's value.", "",
          md_table(["dataset", "exp", "val headroom", "val headroom %", "test gain", "test gain %"],
                   [[r.dataset, r.experiment, f"{r.val_headroom:+.4f}", f"{r.val_headroom_pct:+.1f}", f"{r.test_improvement:+.4f}",
                     f"{r.test_improvement_pct:+.1f}"] for r in hr.itertuples()]),
          "", f"Spearman over the {hr_stats['n_cells']} cells: relative (%) rho = {hr_stats['spearman_rho_pct']:+.2f} "
          f"(p = {fp(hr_stats['p_pct'])}); raw units rho = {hr_stats['spearman_rho_raw_units']:+.2f} (p = {fp(hr_stats['p_raw_units'])}). "
          "EXPLORATORY; 10 cells; no correction.", ""]

    L += ["## 9. Compute", "",
          f"Calls with wall time > {CALL_CAP_S} s (machine sleep) dropped: {comp_tot['calls_dropped']} calls in "
          f"{comp_tot['runs_affected']} runs. Total run time, all primary runs: {comp_tot['total_cpu_h_clean']:.1f} worker-hours "
          f"after dropping ({comp_tot['total_cpu_h_raw']:.1f} h raw). Time per call measured under 10 concurrent workers.", "",
          md_table(["dataset"] + [f"{m} s/call" for m in OPTS] + [f"{m} unique" for m in OPTS] + ["worker-h (all methods)"],
                   [[ds] + [f"{comp[(comp.dataset == ds) & (comp.method == m)].median_s_per_call.iloc[0]:.2f}" for m in OPTS] +
                    [f"{comp[(comp.dataset == ds) & (comp.method == m)].median_unique.iloc[0]:.0f}" for m in OPTS] +
                    [f"{comp[comp.dataset == ds].total_cpu_h_clean.sum():.1f}"] for ds in DATASETS]), ""]
    (OUT / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nanalysis code sha256: {analysis_hash}\ndone in {(datetime.now() - t0).total_seconds():.0f} s")


if __name__ == "__main__":
    main()
