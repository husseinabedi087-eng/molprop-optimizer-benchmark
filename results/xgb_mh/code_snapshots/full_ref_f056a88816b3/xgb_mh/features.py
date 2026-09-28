"""RDKit 2D descriptor features with train-only filtering."""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

FLOAT32_MAX = float(np.finfo(np.float32).max)
MAX_NAN_FRACTION = 0.20


def descriptor_names() -> list[str]:
    return [name for name, _ in Descriptors.descList]


def _descriptor_rows(smiles_chunk: list[str]) -> list[tuple[np.ndarray, list[str]]]:
    """Return (values, names of descriptors that raised) per molecule; values NaN on failure."""
    RDLogger.DisableLog("rdApp.*")
    rows = []
    for smi in smiles_chunk:
        mol = Chem.MolFromSmiles(smi)
        values = np.full(len(Descriptors.descList), np.nan)
        failed = []
        for j, (name, fn) in enumerate(Descriptors.descList):
            try:
                values[j] = float(fn(mol))
            except Exception:
                failed.append(name)
        rows.append((values, failed))
    return rows


def compute_descriptors(smiles: list[str], n_jobs: int = 10) -> tuple[np.ndarray, list[list[str]]]:
    chunks = [smiles[i:i + 100] for i in range(0, len(smiles), 100)]
    parts = Parallel(n_jobs=n_jobs)(delayed(_descriptor_rows)(c) for c in chunks)
    rows = [row for part in parts for row in part]
    return np.vstack([r[0] for r in rows]), [r[1] for r in rows]


def sanitize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Set inf and |x| > float32 max to NaN. Returns X and the per-entry inf / overflow masks."""
    X = X.copy()
    inf_mask = np.isinf(X)
    overflow_mask = np.isfinite(X) & (np.abs(X) > FLOAT32_MAX)
    X[inf_mask | overflow_mask] = np.nan
    return X, inf_mask, overflow_mask


def train_filter(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep mask using TRAIN rows only: drop constant features and features with >20% NaN."""
    nan_fraction = np.isnan(X_train).mean(axis=0)
    n_unique = np.array([np.unique(col[~np.isnan(col)]).size for col in X_train.T])
    constant = n_unique <= 1
    too_many_nan = nan_fraction > MAX_NAN_FRACTION
    return ~(constant | too_many_nan), constant, too_many_nan
