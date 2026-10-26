"""J5: budget 300 vs 100 (Exp B, BBBP, FreeSolv, seeds 0-9)."""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "final")); sys.path.insert(0, str(ROOT / "analysis" / "figures"))
import final_stats as fs
from stats import nogueira_stability
from data import feature_mask
OUT = ROOT / "analysis" / "revision" / "out"; S = ROOT / "results" / "revision" / "searches"
NAME = {"random": "Random", "tpe": "TPE", "gwo_ref": "GWO", "woa_ref": "WOA", "pso": "PSO"}
rows = []
for b in (100, 300):
    for ds in ("BBBP", "FreeSolv"):
        for opt, m in NAME.items():
            for s in range(10):
                d = json.loads((S / f"{ds}_B_{opt}_s{s}_m-none_b{b}_lam0.01abs.json").read_text()); cls = ds == "BBBP"
                rows.append({"budget": b, "dataset": ds, "method": m, "seed": s, "val_fit": d["best_val_fitness"],
                             "val": (1 - d["best_val_raw_metric"]) if cls else d["best_val_raw_metric"],
                             "test": d["test_roc_auc"] if cls else d["test_rmse"], "frac": d["n_selected"] / d["n_features_total"],
                             "unique": d["n_unique"], "time_h": (d["search_time"]) / 3600, "mask": d["best_selected_mask_hex"], "F": d["n_features_total"]})
R = pd.DataFrame(rows)
def ci(d):
    d = np.asarray(d); m = d.mean(); h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)); return m, m - h, m + h
out = []
for ds in ("BBBP", "FreeSolv"):
    cls = ds == "BBBP"; d0 = float(fs.cell(fs.load("ref"), ds, "A", "Default", "test_primary")[0])
    for m in fs.OPTS:
        a = R[(R.budget == 100) & (R.dataset == ds) & (R.method == m)].sort_values("seed"); b = R[(R.budget == 300) & (R.dataset == ds) & (R.method == m)].sort_values("seed")
        dv = a.val_fit.values - b.val_fit.values          # > 0: lower (better) fitness at 300
        dt = (b.test.values - a.test.values) if cls else (a.test.values - b.test.values)   # > 0: better test at 300
        st100 = nogueira_stability(np.vstack([feature_mask(h, a.F.iloc[0]) for h in a["mask"]]).astype(float))[0]
        st300 = nogueira_stability(np.vstack([feature_mask(h, b.F.iloc[0]) for h in b["mask"]]).astype(float))[0]
        out.append({"dataset": ds, "method": m, "val_gain_mean": ci(dv)[0], "val_gain_lo": ci(dv)[1], "val_gain_hi": ci(dv)[2], "p_val": fs.wilcox(dv),
                    "test_gain_mean": ci(dt)[0], "test_gain_lo": ci(dt)[1], "test_gain_hi": ci(dt)[2], "p_test": fs.wilcox(dt),
                    "median_test_100": a.test.median(), "median_test_300": b.test.median(), "default": d0,
                    "frac_100": a.frac.median(), "frac_300": b.frac.median(), "stab_100": st100, "stab_300": st300,
                    "unique_300_median": b.unique.median(), "unique_300_min": b.unique.min(), "time_ratio": b.time_h.sum() / a.time_h.sum()})
    for key in ("p_val", "p_test"):
        fam = [r for r in out if r["dataset"] == ds]
        for r, h in zip(fam, fs.holm([r[key] for r in fam])): r[key + "_holm"] = h
O = pd.DataFrame(out); O.to_csv(OUT / "J5_budget.csv", index=False)
pd.set_option("display.width", 250); print(O.round(4).to_string())
for ds in ("BBBP", "FreeSolv"):
    cls = ds == "BBBP"
    for b in (100, 300):
        g = np.column_stack([R[(R.budget == b) & (R.dataset == ds) & (R.method == m)].sort_values("seed").test.values * (1 if cls else -1) for m in fs.OPTS])
        pw = [fs.wilcox(g[:, i] - g[:, j]) for i, j in itertools.combinations(range(5), 2)]
        print(ds, b, "Friedman p", round(stats.friedmanchisquare(*g.T).pvalue, 3), "Holm pairs", int(sum(np.array(fs.holm(pw)) < .05)))
