"""Metaheuristic HPO + feature selection for XGBoost on MoleculeNet descriptors."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "xgb_mh"
FEATURES_DIR = RESULTS_DIR / "features"
SPLITS_DIR = RESULTS_DIR / "splits"
LEGACY_SPLITS = {
    "ESOL": PROJECT_ROOT / "priority_primary_20260924" / "primary_ESOL" / "split_indices.json",
    "BBBP": PROJECT_ROOT / "priority_primary_20260924" / "primary_BBBP" / "split_indices.json",
}

DATASETS = ("ESOL", "FreeSolv", "Lipophilicity", "BBBP", "BACE")
REGRESSION = ("ESOL", "FreeSolv", "Lipophilicity")
CLASSIFICATION = ("BBBP", "BACE")
SPLIT_SEED = 42


def task_type(name: str) -> str:
    if name in REGRESSION:
        return "regression"
    if name in CLASSIFICATION:
        return "classification"
    raise ValueError(f"Unsupported dataset: {name}")
