"""Read-only data access for the figures.

Random / TPE / PSO / Default come from results/xgb_mh/full. GWO / WOA come from results/xgb_mh/full_ref
(gwo_ref, woa_ref) once that run is complete; until then from results/xgb_mh/full (gwo, woa = the a0
variants) and every figure is marked DRAFT. Never writes to results/ and never imports xgb_mh.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "results" / "xgb_mh" / "full"
FULL_REF = ROOT / "results" / "xgb_mh" / "full_ref"

DATASETS = ("ESOL", "FreeSolv", "Lipophilicity", "BBBP", "BACE")  # grouped by task
EXPERIMENTS = ("A", "B")
EXPERIMENT_LABEL = {"A": "Experiment A: HPO only", "B": "Experiment B: feature selection + HPO"}
TASK = {"ESOL": "regression", "FreeSolv": "regression", "Lipophilicity": "regression",
        "BBBP": "classification", "BACE": "classification"}
METRIC = {"regression": "test_rmse", "classification": "test_roc_auc"}
METRIC_NAME = {"regression": "RMSE", "classification": "ROC-AUC"}
AXIS_LABEL = {"regression": "RMSE (lower is better)", "classification": "ROC-AUC (higher is better)"}
HIGHER_BETTER = {"regression": False, "classification": True}
MAIN_NAMES = {"default": "Default", "random": "Random", "tpe": "TPE", "pso": "PSO"}
REF_NAMES = {"gwo_ref": "GWO", "woa_ref": "WOA"}
A0_NAMES = {"gwo": "GWO", "woa": "WOA"}


@dataclass
class Study:
    runs: pd.DataFrame          # one row per dataset x experiment x method x seed
    draft: bool                 # True while GWO/WOA are the a0 variants
    gwo_woa_source: str


def ref_complete() -> bool:
    meta_path, summary = FULL_REF / "run_metadata.json", FULL_REF / "test_summary.csv"
    if not (meta_path.exists() and summary.exists()):
        return False
    meta = json.loads(meta_path.read_text())
    cfg = meta["config"]
    expected = len(cfg["datasets"]) * len(cfg["experiments"]) * len(cfg["optimizers"]) * len(cfg["seeds"])
    return len(list((FULL_REF / "runs").glob("*.json"))) == expected and meta["sessions"][-1].get("status") == "complete"


def load_study() -> Study:
    main = pd.read_csv(FULL / "test_summary.csv")
    parts = [main[main.optimizer.isin(MAIN_NAMES)].assign(run_dir=str(FULL))]
    if ref_complete():
        ref = pd.read_csv(FULL_REF / "test_summary.csv")
        parts.append(ref[ref.optimizer.isin(REF_NAMES)].assign(run_dir=str(FULL_REF)))
        names, draft, source = {**MAIN_NAMES, **REF_NAMES}, False, "full_ref (gwo_ref, woa_ref)"
    else:
        parts.append(main[main.optimizer.isin(A0_NAMES)].assign(run_dir=str(FULL)))
        names, draft, source = {**MAIN_NAMES, **A0_NAMES}, True, "full (gwo, woa = a0 variants) - DRAFT"
    df = pd.concat(parts, ignore_index=True)
    df["method"] = df.optimizer.map(names)
    df["task"] = df.dataset.map(TASK)
    df["test_primary"] = np.where(df.task == "regression", df.test_rmse, df.test_roc_auc)
    raw = df.best_val_raw_metric
    df["val_primary"] = np.where(df.task == "classification", 1.0 - raw, raw)  # same scale as test metric
    df["seed"] = df.seed.astype(int)
    return Study(df, draft, source)


def feature_mask(hex_value: str, n_features: int) -> np.ndarray:
    bits = int(hex_value, 16)
    return np.array([(bits >> j) & 1 for j in range(n_features)], dtype=bool)


def load_curves(study: Study) -> dict[tuple, np.ndarray]:
    """(dataset, experiment, method) -> array (n_seeds, budget) of best-so-far validation fitness."""
    curves: dict[tuple, list] = {}
    runs = study.runs[study.runs.method != "Default"].sort_values("seed")
    for r in runs.itertuples():
        path = Path(r.run_dir) / "calls" / f"{r.dataset}_{r.experiment}_{r.optimizer}_s{r.seed}.jsonl"
        with path.open() as handle:
            best = [json.loads(line)["best_so_far"] for line in handle]
        curves.setdefault((r.dataset, r.experiment, r.method), []).append(best)
    return {k: np.array(v) for k, v in curves.items()}
