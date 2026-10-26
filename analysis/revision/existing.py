"""Revision analyses that use only the existing (original-platform) results. Writes analysis/revision/out/."""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "analysis" / "final")); sys.path.insert(0, str(ROOT / "analysis" / "figures"))
import final_stats as fs
OUT = ROOT / "analysis" / "revision" / "out"; OUT.mkdir(parents=True, exist_ok=True)
RES = ROOT / "results" / "xgb_mh"
rng_seed = 20261025

def holm(p): return fs.holm(p)

def tost(d, delta):
    d = np.asarray(d, float); n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        p = 0.0 if abs(m) < delta else 1.0; lo90 = hi90 = lo95 = hi95 = m
    else:
        p1 = stats.t.sf((m + delta) / se, n - 1); p2 = stats.t.cdf((m - delta) / se, n - 1); p = max(p1, p2)
        lo90, hi90 = m + np.array([-1, 1]) * stats.t.ppf(.95, n - 1) * se
        lo95, hi95 = m + np.array([-1, 1]) * stats.t.ppf(.975, n - 1) * se
    rng = np.random.default_rng(rng_seed); bm = np.median(d[rng.integers(0, n, (10000, n))], axis=1)
    return {"mean": m, "ci95_low": lo95, "ci95_high": hi95, "ci90_low": lo90, "ci90_high": hi90,
            "median": float(np.median(d)), "median_ci95_low": float(np.percentile(bm, 2.5)), "median_ci95_high": float(np.percentile(bm, 97.5)),
            "tost_p": float(p), "min_equiv_margin": float(max(abs(lo90), abs(hi90)))}

df = fs.load("ref")
# ---------- R1.1 dimensions from the logs
dims = []
for ds in fs.DATASETS:
    rec = json.loads(next(open(RES / "full" / "calls" / f"{ds}_B_random_s0.jsonl")))
    F_log = len(rec["selected_mask_hex"]) * 4
    F = int(df[(df.dataset == ds) & (df.experiment == "B")].n_features_total.iloc[0])
    z = np.load(RES / "features" / f"{ds}.npz"); dims.append({"dataset": ds, "F_npz": z["X"].shape[1], "F_runs": F, "d_B": 8 + F, "mask_hex_bits": F_log})
pd.DataFrame(dims).to_csv(OUT / "R1_1_dimensions.csv", index=False); print(pd.DataFrame(dims))

# ---------- R1.4 paired optimizer differences + TOST
rows = []
for ds in fs.DATASETS:
    cls = fs.TASK[ds] == "classification"
    d0 = float(fs.cell(df, ds, "A", "Default", "test_primary")[0]); delta = 0.01 if cls else 0.02 * d0
    for exp in fs.EXPERIMENTS:
        g = {m: fs.cell(df, ds, exp, m, "test_primary") * (1 if cls else -1) for m in fs.OPTS}
        fam = []
        for a, b in itertools.combinations(fs.OPTS, 2):
            r = {"dataset": ds, "experiment": exp, "pair": f"{a} vs {b}", "delta": delta, **tost(g[a] - g[b], delta)}
            fam.append(r)
        for r, ph in zip(fam, holm([r["tost_p"] for r in fam])): r["tost_p_holm"] = ph; r["equivalent"] = ph < .05
        for r in fam: r["equivalent_half_margin"] = tost(g[r["pair"].split(" vs ")[0]] - g[r["pair"].split(" vs ")[1]], delta / 2)["tost_p"] < .05
        rows += fam
eq = pd.DataFrame(rows); eq.to_csv(OUT / "R1_4_pairwise_equivalence.csv", index=False)
cellsum = eq.groupby(["dataset", "experiment"]).agg(n_equiv=("equivalent", "sum"), max_min_margin=("min_equiv_margin", "max"), delta=("delta", "first")).reset_index()
cellsum.to_csv(OUT / "R1_4_cell_summary.csv", index=False)
print("R1.4 equivalent pairs (Holm TOST):", int(eq.equivalent.sum()), "/", len(eq)); print(cellsum.to_string())

# tuned vs default equivalence (one-sample vs the deterministic default)
rows = []
for ds in fs.DATASETS:
    cls = fs.TASK[ds] == "classification"; d0 = float(fs.cell(df, ds, "A", "Default", "test_primary")[0]); delta = 0.01 if cls else 0.02 * d0
    for exp in fs.EXPERIMENTS:
        fam = [{"dataset": ds, "experiment": exp, "method": m, "delta": delta,
                **tost((fs.cell(df, ds, exp, m, "test_primary") - d0) * (1 if cls else -1), delta)} for m in fs.OPTS]
        for r, ph in zip(fam, holm([r["tost_p"] for r in fam])): r["tost_p_holm"] = ph; r["equivalent"] = ph < .05
        rows += fam
pd.DataFrame(rows).to_csv(OUT / "R1_4_vs_default_equivalence.csv", index=False)

# ---------- R1.5 paired mirror changes, original platform (S vs M), with CIs
mir = pd.read_csv(RES / "mirror" / "test_summary.csv")
rows = []
name = {"random": "Random", "tpe": "TPE", "gwo_ref": "GWO", "woa_ref": "WOA", "pso": "PSO"}
for ds in ("BBBP", "FreeSolv"):
    cls = ds == "BBBP"; d0 = float(fs.cell(df, ds, "A", "Default", "test_primary")[0]); delta = 0.01 if cls else 0.02 * d0
    for opt, m in name.items():
        s = df[(df.dataset == ds) & (df.experiment == "B") & (df.method == m)].sort_values("seed")
        mm = mir[(mir.dataset == ds) & (mir.optimizer == opt)].sort_values("seed")
        assert (s.seed.values == mm.seed.values).all()
        F = s.n_features_total.iloc[0]
        fr = mm.n_selected.values / F - s.n_selected.values / F
        tp = (mm.test_roc_auc.values - s.test_roc_auc.values) if cls else (s.test_rmse.values - mm.test_rmse.values)
        rows.append({"dataset": ds, "method": m, "outcome": "selected fraction (M - S)", "delta": 0.03, **tost(fr, 0.03)})
        rows.append({"dataset": ds, "method": m, "outcome": "test goodness (M - S)", "delta": delta, **tost(tp, delta)})
mr = pd.DataFrame(rows); mr.to_csv(OUT / "R1_5_mirror_paired_changes.csv", index=False)
print(mr[["dataset", "method", "outcome", "mean", "ci95_low", "ci95_high", "tost_p", "min_equiv_margin"]].round(4).to_string())
