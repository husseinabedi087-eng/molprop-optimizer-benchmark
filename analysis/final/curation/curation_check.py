"""Read-only duplicate / leakage check on the exact molecules and scaffold split used in the study."""
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
PROJECT = Path(__file__).resolve().parents[3]  # analysis/final/curation/ -> project root
FEAT = PROJECT / "results" / "xgb_mh" / "features"
OUT = Path(sys.argv[1])
DATASETS = {"ESOL": "reg", "FreeSolv": "reg", "Lipophilicity": "reg", "BBBP": "clf", "BACE": "clf"}
SPLITS = ("train", "validation", "test")
detail = []


def main():
    for ds, task in DATASETS.items():
        z = np.load(FEAT / f"{ds}.npz", allow_pickle=False)
        smi, y = z["smiles"], z["y"]
        rows = []
        for split in SPLITS:
            for pos in z[split]:
                mol = Chem.MolFromSmiles(str(smi[pos]))
                key = Chem.MolToInchiKey(mol) if mol is not None else ""
                rows.append({"split": split, "pos": int(pos), "pyg_index": int(z["pyg_index"][pos]), "smiles": str(smi[pos]),
                             "canonical": Chem.MolToSmiles(mol) if mol is not None else "", "inchikey": key,
                             "block1": key.split("-")[0] if key else "", "y": float(y[pos])})
        df = pd.DataFrame(rows)
        n_fail = int((df.inchikey == "").sum())
        df = df[df.inchikey != ""]
        line = {"dataset": ds, "n": len(df), "inchikey_failures": n_fail}
        for split in SPLITS:
            s = df[df.split == split]
            g = s.groupby("inchikey").size()
            line[f"{split}_dup_groups"] = int((g > 1).sum())
            line[f"{split}_extra_copies"] = int((g[g > 1] - 1).sum())
            # label spread inside within-split duplicate groups
            spread = s[s.inchikey.isin(g[g > 1].index)].groupby("inchikey").y.agg(lambda v: v.max() - v.min())
            line[f"{split}_dup_label_conflicts"] = int((spread > 0).sum()) if task == "clf" else round(float(spread.max()), 3) if len(spread) else 0
        for a, b in itertools.combinations(SPLITS, 2):
            A, B = df[df.split == a], df[df.split == b]
            shared = sorted(set(A.inchikey) & set(B.inchikey))
            tag = f"{a[:5]}-{b[:5]}"
            line[f"{tag}_shared"] = len(shared)
            agree, diffs = 0, []
            for k in shared:
                ya, yb = A[A.inchikey == k].y.to_numpy(), B[B.inchikey == k].y.to_numpy()
                for u, v in itertools.product(ya, yb):
                    d = abs(u - v)
                    diffs.append(d)
                    agree += d == 0 if task == "clf" else 0
                    detail.append({"dataset": ds, "pair": tag, "type": "exact (full InChIKey)", "inchikey": k,
                                   "smiles_a": A[A.inchikey == k].smiles.iloc[0], "smiles_b": B[B.inchikey == k].smiles.iloc[0],
                                   "y_a": u, "y_b": v, "abs_diff": d})
            if task == "clf":
                line[f"{tag}_label"] = f"{agree}/{len(diffs)} same class" if diffs else "-"
            else:
                line[f"{tag}_label"] = (f"|dy| median {np.median(diffs):.3f}, max {max(diffs):.3f}" if diffs else "-")
            # stereo-only: same connectivity block, different full key, across the split pair
            stereo = 0
            for blk in sorted(set(A.block1) & set(B.block1)):  # sorted: set order varies between runs
                ka, kb = set(A[A.block1 == blk].inchikey), set(B[B.block1 == blk].inchikey)
                pairs = [(u, v) for u in sorted(ka) for v in sorted(kb) if u != v]
                if pairs:
                    stereo += 1
                    for u, v in pairs:
                        ra, rb = A[A.inchikey == u].iloc[0], B[B.inchikey == v].iloc[0]
                        detail.append({"dataset": ds, "pair": tag, "type": "stereo-only (same block 1)", "inchikey": f"{u} | {v}",
                                       "smiles_a": ra.smiles, "smiles_b": rb.smiles, "y_a": ra.y, "y_b": rb.y,
                                       "abs_diff": abs(ra.y - rb.y)})
            line[f"{tag}_stereo_only"] = stereo
        print(pd.Series(line).to_string())
        print()
    pd.DataFrame(detail).to_csv(OUT, index=False)
    print("detail rows:", len(detail), "->", OUT)


if __name__ == "__main__":
    main()
