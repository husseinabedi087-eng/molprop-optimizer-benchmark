"""Rebuild the per-run table from the exact runs/*.json files (test_summary.csv rounds the 16th digit, which breaks
exact ROC-AUC ties). Writes analysis/revision/out/exact_runs.csv, read by the revision figures."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "final")); sys.path.insert(0, str(ROOT / "analysis" / "figures"))
import final_stats as fs
SRC = {"Random": ("full", "random"), "TPE": ("full", "tpe"), "PSO": ("full", "pso"), "GWO": ("full_ref", "gwo_ref"), "WOA": ("full_ref", "woa_ref")}
COLS = ["test_rmse", "test_mae", "test_r2", "test_roc_auc", "test_pr_auc", "best_val_raw_metric"]
ex = fs.load("ref")
for i, r in ex.iterrows():
    src = (ROOT / "results/xgb_mh/full/baseline" / f"{r.dataset}.json") if r.method == "Default" else \
          (ROOT / "results/xgb_mh" / SRC[r.method][0] / "runs" / f"{r.dataset}_{r.experiment}_{SRC[r.method][1]}_s{r.seed}.json")
    run = json.loads(src.read_text())
    for c in COLS:
        if c in run and not pd.isna(r[c]): ex.at[i, c] = run[c]
ex["val_primary"] = np.where(ex.task == "classification", 1 - ex.best_val_raw_metric, ex.best_val_raw_metric)
ex["test_primary"] = np.where(ex.task == "classification", ex.test_roc_auc, ex.test_rmse)
out = ROOT / "analysis/revision/out"; out.mkdir(parents=True, exist_ok=True)
ex.to_csv(out / "exact_runs.csv", index=False, float_format="%.17g")
fr, pw = fs.optimizer_comparison(ex)
fr.to_csv(out / "exact_friedman.csv", index=False); pw.to_csv(out / "exact_pairwise.csv", index=False)
b = fr[(fr.dataset == "BBBP") & (fr.experiment == "B") & (fr.role == "primary")].iloc[0]
print(f"BBBP Exp. B Friedman from exact values: chi2 = {b.friedman_chi2:.2f}, p = {b.friedman_p:.3f}")
