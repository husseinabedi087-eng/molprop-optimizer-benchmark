"""J3 (descriptor-only mirror, R1.5) and J4 (penalty sensitivity, R1.6) analyses."""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "final")); sys.path.insert(0, str(ROOT / "analysis" / "figures")); sys.path.insert(0, str(ROOT / "analysis" / "revision"))
import final_stats as fs
OUT = ROOT / "analysis" / "revision" / "out"; RES = ROOT / "results" / "xgb_mh"; S = ROOT / "results" / "revision" / "searches"
NAME = {"random": "Random", "tpe": "TPE", "gwo_ref": "GWO", "woa_ref": "WOA", "pso": "PSO"}
FOLDER = {"random": "full", "tpe": "full", "pso": "full", "gwo_ref": "full_ref", "woa_ref": "full_ref"}

def tost(d, delta):
    d = np.asarray(d, float); n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    if se == 0: return {"mean": m, "ci95_low": m, "ci95_high": m, "tost_p": 0.0 if abs(m) < delta else 1.0}
    lo, hi = m + np.array([-1, 1]) * stats.t.ppf(.975, n - 1) * se
    return {"mean": m, "ci95_low": lo, "ci95_high": hi,
            "tost_p": max(stats.t.sf((m + delta) / se, n - 1), stats.t.cdf((m - delta) / se, n - 1))}

rows = []
for f in S.glob("*.json"):
    d = json.loads(f.read_text()); cls = d["dataset"] == "BBBP"
    rows.append({"dataset": d["dataset"], "opt": d["optimizer"], "method": NAME[d["optimizer"]], "seed": d["seed"], "mirror": d["mirror_mode"],
                 "budget": d["budget"], "lam": d["lam"], "mode": d["lam_mode"], "frac": d["n_selected"] / d["n_features_total"],
                 "test": d["test_roc_auc"] if cls else d["test_rmse"], "val": d["best_val_raw_metric"], "n_unique": d["n_unique"],
                 "search_time": d["search_time"], "fit_time_sum": d["fit_time_sum"], "file": f.stem})
R = pd.DataFrame(rows)
R["cond"] = np.where(R["mode"] == "rel", "rel" + (R.lam * 100).round().astype(int).astype(str) + "%", "abs" + R.lam.astype(str))

# ---------- reproduction of the original S (full/full_ref) and M (mirror) runs on the original platform
rep = []
for _, r in R[(R.budget == 100) & (R.lam == 0.01) & (R["mode"] == "abs") & R.mirror.isin(["none", "all"])].iterrows():
    folder = "mirror" if r.mirror == "all" else FOLDER[r.opt]
    o = json.loads((RES / folder / "runs" / f"{r.dataset}_B_{r.opt}_s{r.seed}.json").read_text())
    k = "test_roc_auc" if r.dataset == "BBBP" else "test_rmse"
    rep.append(r.test == o[k] and r.frac == o["n_selected"] / o["n_features_total"])
print("J3/J4 reference runs identical to the published runs:", sum(rep), "/", len(rep))

# ---------- J3: S (original), M (original mirror), D (new). Use the published S and M for all five optimizers (identical where rerun)
orig = []
for opt, m in NAME.items():
    for ds in ("BBBP", "FreeSolv"):
        for s in range(30):
            k = "test_roc_auc" if ds == "BBBP" else "test_rmse"
            for mir, folder in (("none", FOLDER[opt]), ("all", "mirror")):
                o = json.loads((RES / folder / "runs" / f"{ds}_B_{opt}_s{s}.json").read_text())
                orig.append({"dataset": ds, "method": m, "seed": s, "mirror": mir, "frac": o["n_selected"] / o["n_features_total"], "test": o[k]})
O = pd.DataFrame(orig)
Dm = R[(R.mirror == "desc") & (R.budget == 100)][["dataset", "method", "seed", "frac", "test"]].assign(mirror="desc")
J3 = pd.concat([O, Dm], ignore_index=True)
res = []
for ds in ("BBBP", "FreeSolv"):
    cls = ds == "BBBP"; d0 = float(fs.cell(fs.load("ref"), ds, "A", "Default", "test_primary")[0]); delta = 0.01 if cls else 0.02 * d0
    fam = []
    for m in fs.OPTS:
        g = {mm: J3[(J3.dataset == ds) & (J3.method == m) & (J3.mirror == mm)].sort_values("seed") for mm in ("none", "all", "desc")}
        fS, fM, fD = (g[k].frac.values for k in ("none", "all", "desc"))
        tS, tM, tD = (g[k].test.values for k in ("none", "all", "desc"))
        good = lambda a, b: (a - b) if cls else (b - a)
        r = {"dataset": ds, "method": m, "med_frac_S": np.median(fS), "med_frac_M": np.median(fM), "med_frac_D": np.median(fD),
             "n_D_gt_S": int((fD > fS).sum()), "p_frac_D_vs_S": fs.wilcox(fD - fS)}
        r.update({f"DminusS_frac_{k}": v for k, v in tost(fD - fS, 0.03).items()})
        r.update({f"DminusM_frac_{k}": v for k, v in tost(fD - fM, 0.03).items()})
        r.update({f"MminusD_frac_{k}": v for k, v in tost(fM - fD, 0.03).items() if k != "tost_p"})
        r.update({f"DminusS_test_{k}": v for k, v in tost(good(tD, tS), delta).items()})
        r["p_test_D_vs_S"] = fs.wilcox(good(tD, tS)); r["delta_test"] = delta
        fam.append(r)
    for key in ("p_frac_D_vs_S", "p_test_D_vs_S", "DminusM_frac_tost_p", "DminusS_test_tost_p"):
        for r, h in zip(fam, fs.holm([r[key] for r in fam])): r[key + "_holm"] = h
    res += fam
J3R = pd.DataFrame(res); J3R.to_csv(OUT / "J3_descriptor_mirror.csv", index=False)
pd.set_option("display.width", 250)
print(J3R[["dataset", "method", "med_frac_S", "med_frac_M", "med_frac_D", "n_D_gt_S", "p_frac_D_vs_S_holm", "DminusM_frac_mean",
           "DminusM_frac_ci95_low", "DminusM_frac_ci95_high", "DminusM_frac_tost_p_holm", "MminusD_frac_mean", "DminusS_test_mean",
           "DminusS_test_ci95_low", "DminusS_test_ci95_high", "p_test_D_vs_S_holm", "DminusS_test_tost_p_holm"]].round(4).to_string())

# ---------- J4: penalty conditions, seeds 0-9
J4 = R[(R.budget == 100) & (R.mirror == "none") & (R.seed < 10)]
rk, fr = [], []
for ds in ("BBBP", "FreeSolv"):
    cls = ds == "BBBP"
    for c in ("abs0.01", "abs0.0", "rel1%", "rel5%"):
        sub = J4[(J4.dataset == ds) & (J4.cond == c)]
        g = np.column_stack([sub[sub.method == m].sort_values("seed").test.values * (1 if cls else -1) for m in fs.OPTS])
        assert g.shape == (10, 5), (ds, c, g.shape)
        chi, p = stats.friedmanchisquare(*g.T); ranks = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, g).mean(0)
        pw = [fs.wilcox(g[:, i] - g[:, j]) for i, j in itertools.combinations(range(5), 2)]
        pen = sub.iloc[0]
        fr.append({"dataset": ds, "condition": c, "friedman_chi2": chi, "friedman_p": p, "n_holm_pairs": int(sum(np.array(fs.holm(pw)) < .05)),
                   **{f"rank_{m}": v for m, v in zip(fs.OPTS, ranks)},
                   **{f"frac_{m}": sub[sub.method == m].frac.median() for m in fs.OPTS},
                   **{f"test_{m}": sub[sub.method == m].test.median() for m in fs.OPTS}})
F4 = pd.DataFrame(fr)
for ds in ("BBBP", "FreeSolv"):
    ref = F4[(F4.dataset == ds) & (F4.condition == "abs0.01")][[f"rank_{m}" for m in fs.OPTS]].values[0]
    for i in F4.index[F4.dataset == ds]:
        F4.loc[i, "kendall_tau_vs_ref"] = stats.kendalltau(ref, F4.loc[i, [f"rank_{m}" for m in fs.OPTS]].values.astype(float))[0]
F4.to_csv(OUT / "J4_penalty.csv", index=False)
print(F4[["dataset", "condition", "friedman_p", "n_holm_pairs", "kendall_tau_vs_ref"] + [f"rank_{m}" for m in fs.OPTS] + [f"frac_{m}" for m in fs.OPTS]].round(3).to_string())
# penalty weights actually applied
pw_ = R[(R.budget == 100) & (R["mode"] == "rel")].groupby(["dataset", "cond"]).apply(lambda g: json.loads((S / f"{g.file.iloc[0]}.json").read_text())["penalty_weight"])
print(pw_)
# mirrored subset-size effect without penalty (GWO, Random; seeds 0-9)
mm = []
for ds in ("BBBP", "FreeSolv"):
    for m in ("GWO", "Random"):
        a = R[(R.dataset == ds) & (R.method == m) & (R.budget == 100) & (R.lam == 0.0) & (R.mirror == "none")].sort_values("seed")
        b = R[(R.dataset == ds) & (R.method == m) & (R.budget == 100) & (R.lam == 0.0) & (R.mirror == "all")].sort_values("seed")
        mm.append({"dataset": ds, "method": m, "frac_S_lam0": a.frac.median(), "frac_M_lam0": b.frac.median(),
                   "n_M_gt_S": int((b.frac.values > a.frac.values).sum()), "p": fs.wilcox(b.frac.values - a.frac.values)})
MM = pd.DataFrame(mm); MM.to_csv(OUT / "J4_mirror_lam0.csv", index=False); print(MM.round(4).to_string())
