"""Step 2: build scaffold splits and RDKit descriptor matrices for every dataset.

Usage: python -m xgb_mh.prepare [--datasets ESOL,BBBP]
Outputs under results/xgb_mh/: splits/<ds>.json, features/<ds>.npz,
features/<ds>_descriptor_log.csv, features/<ds>_dropped_molecules.csv, features/summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, RDLogger

from . import DATASETS, FEATURES_DIR, task_type
from .data import cached_split, load_molecules
from .features import compute_descriptors, descriptor_names, sanitize, train_filter


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(name: str) -> dict:
    RDLogger.DisableLog("rdApp.*")
    smiles, y, source = load_molecules(name)
    dropped: list[dict] = []

    # PyG MoleculeNet silently skips raw rows RDKit cannot parse; reconcile against the raw CSV.
    raw_path = Path(source["raw_files"][0]["path"])
    with raw_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        column = next(c for c in reader.fieldnames if c.lower() in ("smiles", "mol"))
        raw_smiles = [row[column] for row in reader]
    loaded = set(smiles)
    for row_number, smi in enumerate(raw_smiles):
        if smi not in loaded:
            reason = "rdkit_parse_failed" if Chem.MolFromSmiles(smi) is None else "removed_by_pyg_loader"
            dropped.append({"pyg_index": "", "smiles": smi, "stage": f"pyg_loader(raw_row={row_number})", "reason": reason})
    source["n_raw_rows"] = len(raw_smiles)

    valid_ids = []
    for i, smi in enumerate(smiles):
        if Chem.MolFromSmiles(smi) is None:
            dropped.append({"pyg_index": i, "smiles": smi, "stage": "before_split", "reason": "rdkit_parse_failed"})
        elif not np.isfinite(y[i]):
            dropped.append({"pyg_index": i, "smiles": smi, "stage": "before_split", "reason": "missing_label"})
        else:
            valid_ids.append(i)
    split, split_info = cached_split(name, [smiles[i] for i in valid_ids], valid_ids)

    names = descriptor_names()
    X_raw, failures = compute_descriptors([smiles[i] for i in valid_ids])
    position = {pyg: row for row, pyg in enumerate(valid_ids)}
    # A molecule is dropped only if descriptor computation failed for every descriptor.
    total_failure = {valid_ids[r] for r, f in enumerate(failures) if len(f) == len(names)}
    partition_of = {i: part for part, ids in split.items() for i in ids}
    for i in sorted(total_failure):
        dropped.append({"pyg_index": i, "smiles": smiles[i], "stage": partition_of[i], "reason": "descriptor_computation_failed"})
    kept_split = {part: [i for i in ids if i not in total_failure] for part, ids in split.items()}

    ids = np.array(sorted(i for ids_ in kept_split.values() for i in ids_))
    X_raw = X_raw[[position[i] for i in ids]]
    failures = [failures[position[i]] for i in ids]
    row_of = {pyg: row for row, pyg in enumerate(ids)}
    rows = {part: np.array([row_of[i] for i in part_ids], dtype=int) for part, part_ids in kept_split.items()}
    tr = rows["train"]

    X, inf_mask, overflow_mask = sanitize(X_raw)
    exception_mask = np.zeros_like(inf_mask)
    for r, failed in enumerate(failures):
        for n in failed:
            exception_mask[r, names.index(n)] = True
    keep, constant, too_many_nan = train_filter(X[tr])

    log_rows = []
    nan_frac = np.isnan(X[tr]).mean(axis=0)
    for j, n in enumerate(names):
        reason = "" if keep[j] else ";".join(r for r, flag in (("constant_on_train", constant[j]), ("nan_fraction_gt_0.2", too_many_nan[j])) if flag)
        log_rows.append({
            "descriptor": n, "kept": bool(keep[j]), "drop_reason": reason,
            "train_inf_to_nan": int(inf_mask[tr, j].sum()),
            "train_float32_overflow_to_nan": int(overflow_mask[tr, j].sum()),
            "train_converted_to_nan_total": int(inf_mask[tr, j].sum() + overflow_mask[tr, j].sum()),
            "train_rdkit_exception_nan": int(exception_mask[tr, j].sum()),
            "train_nan_fraction_after_conversion": round(float(nan_frac[j]), 6),
            "all_converted_to_nan_total": int(inf_mask[:, j].sum() + overflow_mask[:, j].sum()),
        })
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(FEATURES_DIR / f"{name}_descriptor_log.csv", log_rows, list(log_rows[0]))
    _write_csv(FEATURES_DIR / f"{name}_dropped_molecules.csv", dropped, ["pyg_index", "smiles", "stage", "reason"])

    feature_names = np.array(names)[keep]
    X_kept = X[:, keep].astype(np.float32)
    assert np.isfinite(X_kept[~np.isnan(X_kept)]).all()
    np.savez_compressed(FEATURES_DIR / f"{name}.npz", X=X_kept, y=y[ids], pyg_index=ids,
                        smiles=np.array([smiles[i] for i in ids]), feature_names=feature_names,
                        train=rows["train"], validation=rows["validation"], test=rows["test"])

    info = {
        "dataset": name, "task": task_type(name), "source": source, "split": split_info,
        "n_raw_rows": len(raw_smiles), "n_molecules_pyg": len(smiles), "n_dropped": len(dropped),
        "dropped_by_reason": {r: sum(d["reason"] == r for d in dropped) for r in sorted({d["reason"] for d in dropped})},
        "final_split_sizes": {p: int(len(v)) for p, v in rows.items()},
        "n_descriptors_total": len(names), "n_features_kept": int(keep.sum()),
        "n_dropped_constant": int((constant & ~too_many_nan).sum()),
        "n_dropped_nan_gt_0.2": int((too_many_nan & ~constant).sum()),
        "n_dropped_both": int((constant & too_many_nan).sum()),
        "n_descriptors_with_train_conversions": int(sum(r["train_converted_to_nan_total"] > 0 for r in log_rows)),
        "train_nan_fraction_kept_features": float(np.isnan(X_kept[tr]).mean()),
        "X_shape": list(X_kept.shape),
    }
    if info["task"] == "classification":
        info["positive_rate"] = {p: float(y[ids][v].mean()) for p, v in rows.items()}
    else:
        info["target_mean_sd"] = {p: [float(y[ids][v].mean()), float(y[ids][v].std())] for p, v in rows.items()}
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DATASETS))
    args = parser.parse_args()
    summary_path = FEATURES_DIR / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary["_environment"] = {"rdkit": rdkit.__version__, "numpy": np.__version__, "python": platform.python_version()}
    for name in args.datasets.split(","):
        info = prepare(name)
        summary[name] = info
        print(f"{name:14s} X={tuple(info['X_shape'])} kept={info['n_features_kept']}/{info['n_descriptors_total']} "
              f"split={info['final_split_sizes']} dropped={info['dropped_by_reason']} "
              f"legacy_match={info['split'].get('matches_legacy', 'n/a')}", flush=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
