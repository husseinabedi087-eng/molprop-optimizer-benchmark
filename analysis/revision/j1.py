"""J1 analyses: reproduction, pre-refit diagnostic (R1.3), model-seed control (R1.2), parent structures (R2.4),
prediction-level stability (R2.6). Writes analysis/revision/out/J1_*.csv"""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, friedmanchisquare, wilcoxon, pearsonr
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "final")); sys.path.insert(0, str(ROOT / "analysis" / "figures"))
import final_stats as fs
OUT = ROOT / "analysis" / "revision" / "out"; OUT.mkdir(parents=True, exist_ok=True)
J1 = ROOT / "results" / "revision" / "J1"; RES = ROOT / "results" / "xgb_mh"
NAME = {"random": "Random", "tpe": "TPE", "gwo_ref": "GWO", "woa_ref": "WOA", "pso": "PSO"}
TASK = fs.TASK

rows = []
for f in J1.glob("*.json"):
    d = json.loads(f.read_text())
    if d["optimizer"] == "default": continue
    cls = TASK[d["dataset"]] == "classification"
    k = "test_roc_auc" if cls else "test_rmse"
    rows.append({"dataset": d["dataset"], "experiment": d["experiment"], "method": NAME[d["optimizer"]], "seed": d["seed"],
                 "val": (1 - d["val_pre_raw"]) if cls else d["val_pre_raw"], "test_pre": d[f"test_pre__{k}"],
                 "test_refit": d[f"test_refit__{k}"], "test_m1000": d[f"test_refit_m1000__{k}"], "test_m1001": d[f"test_refit_m1001__{k}"],
                 "repro": d["refit_reproduces_primary_exactly"] and d["val_reproduces_logged_exactly"], "stem": f.stem})
D = pd.DataFrame(rows)
print("reproduced exactly:", int(D.repro.sum()), "/", len(D))
defaults = {}
for ds in fs.DATASETS:
    d = json.loads((J1 / f"{ds}_default.json").read_text()); cls = TASK[ds] == "classification"; k = "test_roc_auc" if cls else "test_rmse"
    defaults[ds] = {"val": (1 - d["val_pre_raw"]) if cls else d["val_pre_raw"], "test_pre": d[f"test_pre__{k}"], "test_refit": d[f"test_refit__{k}"]}

def sp_ci(x, y, B=10000):
    rho = spearmanr(x, y)[0]; rng = np.random.default_rng(20261025); n = len(x); r = []
    idx = rng.integers(0, n, (B, n))
    for i in idx:
        if len(set(x[i])) > 1 and len(set(y[i])) > 1: r.append(spearmanr(x[i], y[i])[0])
    return rho, *np.percentile(r, [2.5, 97.5])

# ---- R1.3 pre-refit diagnostic
rr = []
for (ds, ex, m), g in D.groupby(["dataset", "experiment", "method"]):
    g = g.sort_values("seed"); v, tp, tr = g.val.values, g.test_pre.values, g.test_refit.values
    a = sp_ci(v, tp); b = sp_ci(v, tr)
    cls = TASK[ds] == "classification"; dflt = defaults[ds]
    gap_pre = np.median(tp - v); gap_ref = np.median(tr - v)
    rr.append({"dataset": ds, "experiment": ex, "method": m, "rho_pre": a[0], "rho_pre_lo": a[1], "rho_pre_hi": a[2],
               "rho_refit": b[0], "rho_refit_lo": b[1], "rho_refit_hi": b[2], "delta_rho_refit_minus_pre": b[0] - a[0],
               "median_gap_pre": gap_pre, "median_gap_refit": gap_ref, "default_gap_pre": dflt["test_pre"] - dflt["val"],
               "default_gap_refit": dflt["test_refit"] - dflt["val"],
               "optimism_pre": gap_pre - (dflt["test_pre"] - dflt["val"]),
               "median_refit_minus_pre_test": np.median(tr - tp)})
R = pd.DataFrame(rr); R.to_csv(OUT / "J1_prerefit_diagnostic.csv", index=False)
print("pre-refit rho CI includes 0:", int(((R.rho_pre_lo <= 0) & (R.rho_pre_hi >= 0)).sum()), "/ 50;",
      "refit:", int(((R.rho_refit_lo <= 0) & (R.rho_refit_hi >= 0)).sum()), "/ 50")
print("median rho pre / refit:", R.rho_pre.median().round(3), R.rho_refit.median().round(3), "| median delta:", R.delta_rho_refit_minus_pre.median().round(3),
      "| IQR delta:", R.delta_rho_refit_minus_pre.quantile([.25, .75]).round(3).tolist())
print(R.groupby("dataset")[["rho_pre", "rho_refit"]].median().round(3))
print("sig positive pre:", R[(R.rho_pre_lo > 0)][["dataset","experiment","method","rho_pre"]].to_string())
print("sig negative pre:", R[(R.rho_pre_hi < 0)][["dataset","experiment","method","rho_pre"]].to_string())
gp = R.groupby(["dataset", "experiment"]).agg(gap_pre=("median_gap_pre", "median"), gap_refit=("median_gap_refit", "median"),
        default_gap_pre=("default_gap_pre", "first"), default_gap_refit=("default_gap_refit", "first"), optimism=("optimism_pre", "median")).reset_index()
gp.to_csv(OUT / "J1_gaps.csv", index=False); print(gp.round(4).to_string())

# ---- R1.2 model-seed control
ms = []
for (ds, ex, m), g in D.groupby(["dataset", "experiment", "method"]):
    T = g[["test_refit", "test_m1000", "test_m1001"]].values
    within = T.var(axis=1, ddof=1).mean(); total = T.flatten().var(ddof=1)
    between_tuning = g.test_refit.var(ddof=1)
    ms.append({"dataset": ds, "experiment": ex, "method": m, "sd_within_config_model_seed": np.sqrt(within),
               "sd_across_tuning_seeds": np.sqrt(between_tuning), "share_model_seed": within / total,
               "deterministic_configs": int((T.std(axis=1) == 0).sum())})
M = pd.DataFrame(ms); M.to_csv(OUT / "J1_model_seed.csv", index=False)
print("model-seed share of test variance: median", M.share_model_seed.median().round(3), "range", M.share_model_seed.min().round(3), M.share_model_seed.max().round(3))
print(M.groupby("dataset")[["sd_within_config_model_seed", "sd_across_tuning_seeds", "share_model_seed"]].median().round(4))
# does the optimizer comparison survive averaging over model seeds?
alt = []
for ds in fs.DATASETS:
    for ex in fs.EXPERIMENTS:
        sub = D[(D.dataset == ds) & (D.experiment == ex)].copy(); cls = TASK[ds] == "classification"
        sub["avg3"] = sub[["test_refit", "test_m1000", "test_m1001"]].mean(1)
        for col in ("test_refit", "avg3"):
            g = np.column_stack([sub[sub.method == m].sort_values("seed")[col].values * (1 if cls else -1) for m in fs.OPTS])
            p = friedmanchisquare(*g.T).pvalue
            ps = [fs.wilcox(g[:, i] - g[:, j]) for i, j in itertools.combinations(range(5), 2)]
            alt.append({"dataset": ds, "experiment": ex, "outcome": col, "friedman_p": p, "n_holm_sig": int(sum(np.array(fs.holm(ps)) < .05))})
A = pd.DataFrame(alt); A.to_csv(OUT / "J1_model_seed_avg_comparison.csv", index=False)
print(A.pivot_table(index=["dataset", "experiment"], columns="outcome", values=["friedman_p", "n_holm_sig"]).round(3))

# ---- R2.6 prediction-level stability
ps = []
for (ds, ex, m), g in D.groupby(["dataset", "experiment", "method"]):
    P = np.vstack([np.load(J1 / f"{s}.npz")["test_refit"] for s in g.sort_values("seed").stem])
    C = np.corrcoef(P); iu = np.triu_indices(len(P), 1)
    ps.append({"dataset": ds, "experiment": ex, "method": m, "mean_pairwise_r": C[iu].mean(), "min_pairwise_r": C[iu].min()})
P = pd.DataFrame(ps); P.to_csv(OUT / "J1_prediction_stability.csv", index=False)
print(P.groupby(["dataset", "experiment"]).mean_pairwise_r.agg(["min", "median", "max"]).round(3))
