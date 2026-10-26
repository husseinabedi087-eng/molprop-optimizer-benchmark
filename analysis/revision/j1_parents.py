"""R2.4: standardized parent structures grouped across splits; re-score the fixed selected configurations."""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.stats import friedmanchisquare, spearmanr
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "final")); sys.path.insert(0, str(ROOT / "analysis" / "figures"))
import final_stats as fs
RDLogger.DisableLog("rdApp.*")
OUT = ROOT / "analysis" / "revision" / "out"; J1 = ROOT / "results" / "revision" / "J1"; RES = ROOT / "results" / "xgb_mh"
NAME = {"random": "Random", "tpe": "TPE", "gwo_ref": "GWO", "woa_ref": "WOA", "pso": "PSO"}
te = rdMolStandardize.TautomerEnumerator(); un = rdMolStandardize.Uncharger(); fc = rdMolStandardize.LargestFragmentChooser()

def parent_key(smi):
    m = Chem.MolFromSmiles(smi)
    m = rdMolStandardize.Cleanup(m); m = fc.choose(m); m = un.uncharge(m)
    try: m = te.Canonicalize(m)
    except Exception: pass
    Chem.RemoveStereochemistry(m)
    return Chem.MolToInchiKey(m).split("-")[0]

summ, rows = [], []
for ds in fs.DATASETS:
    cls = fs.TASK[ds] == "classification"
    z = np.load(RES / "features" / f"{ds}.npz"); y = z["y"]; smi = [str(s) for s in z["smiles"]]
    part = np.empty(len(y), dtype=object)
    for p in ("train", "validation", "test"): part[z[p]] = p
    keys = np.array([parent_key(s) for s in smi])
    df = pd.DataFrame({"key": keys, "part": part, "y": y})
    conflict = set()
    for k, g in df.groupby("key"):
        if len(g) > 1 and ((g.y.nunique() > 1) if cls else (g.y.max() - g.y.min() > 0.5)): conflict.add(k)
    tr_keys = set(df[df.part == "train"].key); trva_keys = tr_keys | set(df[df.part == "validation"].key)
    ex_val = (df.part == "validation") & (df.key.isin(tr_keys) | df.key.isin(conflict))
    ex_te = (df.part == "test") & (df.key.isin(trva_keys) | df.key.isin(conflict))
    groups_multi = int((df.groupby("key").size() > 1).sum())
    cross = int(sum(df[df.key == k].part.nunique() > 1 for k, g in df.groupby("key") if len(g) > 1))
    summ.append({"dataset": ds, "parent_groups_multi": groups_multi, "groups_crossing_splits": cross,
                 "label_conflict_groups": len(conflict), "train_in_conflict": int(((df.part == "train") & df.key.isin(conflict)).sum()),
                 "val_excluded": int(ex_val.sum()), "n_val": int((df.part == "validation").sum()),
                 "test_excluded": int(ex_te.sum()), "n_test": int((df.part == "test").sum())})
    vmask = ~ex_val[z["validation"]].values; tmask = ~ex_te[z["test"]].values
    yv, yt = y[z["validation"]], y[z["test"]]
    def score(pred, yy, mask):
        if cls: return roc_auc_score(yy[mask], pred[mask])
        return float(np.sqrt(np.mean((pred[mask] - yy[mask]) ** 2)))
    full_v = np.ones(len(yv), bool); full_t = np.ones(len(yt), bool)
    for f in J1.glob(f"{ds}_*.npz"):
        P = np.load(f); meta = json.loads(f.with_suffix(".json").read_text())
        m = "Default" if meta["optimizer"] == "default" else NAME[meta["optimizer"]]
        exps = ("A", "B") if m == "Default" else (meta["experiment"],)
        for e in exps:
            rows.append({"dataset": ds, "experiment": e, "method": m, "seed": meta["seed"],
                         "test_orig": score(P["test_refit"], yt, full_t), "test_clean": score(P["test_refit"], yt, tmask),
                         "val_orig": score(P["val_pre"], yv, full_v), "val_clean": score(P["val_pre"], yv, vmask)})
S = pd.DataFrame(summ); S.to_csv(OUT / "J1_parents_summary.csv", index=False); print(S.to_string())
R = pd.DataFrame(rows); R.to_csv(OUT / "J1_parents_scores.csv", index=False)

out = []
for ds in fs.DATASETS:
    cls = fs.TASK[ds] == "classification"; sgn = 1 if cls else -1
    for e in fs.EXPERIMENTS:
        sub = R[(R.dataset == ds) & (R.experiment == e)]
        d0 = sub[sub.method == "Default"].iloc[0]
        for ver in ("orig", "clean"):
            g = np.column_stack([sub[sub.method == m].sort_values("seed")[f"test_{ver}"].values for m in fs.OPTS])
            fp = friedmanchisquare(*(g * sgn).T).pvalue
            ps = [fs.wilcox((g[:, i] - g[:, j]) * sgn) for i, j in itertools.combinations(range(5), 2)]
            nsig = int(sum(np.array(fs.holm(ps)) < .05))
            tv = []
            for k in range(5):
                imp = (g[:, k] - d0[f"test_{ver}"]) * sgn
                tv.append(fs.wilcox(imp))
            tvh = fs.holm(tv); verdict = "".join("+" if (h < .05 and np.median((g[:, k] - d0[f'test_{ver}']) * sgn) > 0) else ("-" if h < .05 else "0") for k, h in enumerate(tvh))
            rhos = [spearmanr(sub[sub.method == m].sort_values("seed")[f"val_{ver}"], sub[sub.method == m].sort_values("seed")[f"test_{ver}"])[0] for m in fs.OPTS]
            out.append({"dataset": ds, "experiment": e, "version": ver, "friedman_p": fp, "n_holm_pairs": nsig,
                        "tuned_vs_default (R,T,G,W,P)": verdict, "median_rho_val_test_prerefit_val": float(np.median(rhos))})
O = pd.DataFrame(out); O.to_csv(OUT / "J1_parents_conclusions.csv", index=False); print(O.round(3).to_string())
