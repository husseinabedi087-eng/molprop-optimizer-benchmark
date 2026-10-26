"""Log-based revision analyses: R2.2 ranges of searched and selected validation scores, R2.3 unique configurations and elapsed compute."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "analysis" / "final")); sys.path.insert(0, str(ROOT / "analysis" / "figures"))
import final_stats as fs
from stats import nogueira_stability
from data import feature_mask
OUT = ROOT / "analysis" / "revision" / "out"; RES = ROOT / "results" / "xgb_mh"
SRC = {"Random": ("full", "random"), "TPE": ("full", "tpe"), "PSO": ("full", "pso"), "GWO": ("full_ref", "gwo_ref"), "WOA": ("full_ref", "woa_ref")}

rows, curves = [], []
for ds in fs.DATASETS:
    cls = fs.TASK[ds] == "classification"
    for ex in fs.EXPERIMENTS:
        for m, (f, o) in SRC.items():
            for s in range(30):
                c = [json.loads(l) for l in open(RES / f / "calls" / f"{ds}_{ex}_{o}_s{s}.jsonl")]
                raw = np.array([x["raw_metric"] for x in c]); fit = np.array([x["fitness"] for x in c])
                prim = 1 - raw if cls else raw
                ft = np.cumsum([x["fit_time"] if x["wall_time"] <= 600 else 0 for x in c]); best = np.minimum.accumulate(fit)
                rows.append({"dataset": ds, "experiment": ex, "method": m, "seed": s, "unique": int(sum(not x["cache_hit"] for x in c)),
                             "val_q05": np.quantile(prim, .05), "val_q50": np.quantile(prim, .5), "val_q95": np.quantile(prim, .95),
                             "best_raw": prim[np.argmin(fit)], "fit_time_total_s": ft[-1],
                             "improve_last_quarter": best[74] - best[-1], "improve_last_tenth": best[89] - best[-1]})
                for frac in (0.25, 0.5, 0.75, 1.0):
                    k = min(int(np.searchsorted(ft, frac * ft[-1])), 99)
                    curves.append({"dataset": ds, "experiment": ex, "method": m, "seed": s, "time_frac": frac, "calls_used": k + 1, "best_fitness": best[k]})
R = pd.DataFrame(rows); R.to_csv(OUT / "R2_logs_per_run.csv", index=False)
C = pd.DataFrame(curves); C.to_csv(OUT / "R2_3_time_curves.csv", index=False)
g = R.groupby(["dataset", "experiment"]).agg(searched_q05=("val_q05", "median"), searched_q50=("val_q50", "median"),
        searched_q95=("val_q95", "median"), selected_min=("best_raw", "min"), selected_max=("best_raw", "max"), selected_sd=("best_raw", "std")).reset_index()
g["selected_range"] = g.selected_max - g.selected_min; g["searched_q05_q95"] = (g.searched_q95 - g.searched_q05).abs()
g["ratio_selected_to_searched"] = g.selected_range / g.searched_q05_q95
g.to_csv(OUT / "R2_2_val_ranges.csv", index=False); print(g.round(4).to_string())
u = R.groupby(["experiment", "method"]).unique.describe()[["min", "25%", "50%", "max"]]; print(u); u.to_csv(OUT / "R2_3_unique.csv")
imp = R.groupby(["dataset", "experiment"]).agg(med_improve_last_quarter=("improve_last_quarter", "median"),
        share_runs_improving_last_tenth=("improve_last_tenth", lambda v: float(np.mean(v > 0)))).reset_index()
imp.to_csv(OUT / "R2_3_late_improvement.csv", index=False); print(imp.round(4).to_string())
tc = C.groupby(["dataset", "experiment", "method", "time_frac"]).calls_used.median().unstack(); tc.to_csv(OUT / "R2_3_calls_at_time_fraction.csv"); print(tc.groupby(level=[1]).median())

