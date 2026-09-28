"""Dataset loading and Bemis-Murcko scaffold split.

`scaffold_split` is the logic of molecular_gnn/data.py::scaffold_split, operating on
a SMILES list so that workers never import torch. Indices refer to PyG MoleculeNet order.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from . import DATA_DIR, LEGACY_SPLITS, SPLIT_SEED, SPLITS_DIR

# Our dataset name -> PyG MoleculeNet name.
PYG_NAMES = {"ESOL": "ESOL", "FreeSolv": "FreeSolv", "Lipophilicity": "Lipo", "BBBP": "BBBP", "BACE": "BACE"}


def load_molecules(name: str, data_dir: Path = DATA_DIR) -> tuple[list[str], np.ndarray, dict]:
    """Return SMILES and labels in PyG MoleculeNet order, plus source info."""
    from torch_geometric.datasets import MoleculeNet  # heavy import, only needed here
    dataset = MoleculeNet(root=str(Path(data_dir) / name), name=PYG_NAMES[name])
    smiles = [str(item.smiles) for item in dataset]
    y = np.asarray([float(item.y.view(-1)[0]) for item in dataset], dtype=float)
    raw_files = [Path(dataset.raw_dir) / f for f in dataset.raw_file_names] if isinstance(dataset.raw_file_names, list) \
        else [Path(dataset.raw_dir) / dataset.raw_file_names]
    source = {"loader": "torch_geometric.datasets.MoleculeNet", "pyg_name": PYG_NAMES[name],
              "raw_files": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in raw_files],
              "n_items": len(smiles)}
    return smiles, y, source


def scaffold_split(smiles: Sequence[str], seed: int, train_fraction: float = .8,
                   validation_fraction: float = .1, ids: Sequence[int] | None = None) -> dict[str, list[int]]:
    """Deterministic Bemis-Murcko scaffold split (identical to the GNN study).

    `ids` gives the index reported for each SMILES (default: its position), so the split can
    run on the parseable subset while still reporting original PyG indices.
    """
    ids = list(range(len(smiles))) if ids is None else list(ids)
    groups: dict[str, list[int]] = {}
    for index, smi in zip(ids, smiles):
        molecule = Chem.MolFromSmiles(smi)
        if molecule is None:
            raise ValueError(f"Invalid SMILES at index {index}: {smi!r}")
        scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(molecule), isomericSmiles=False)
        groups.setdefault(scaffold, []).append(index)

    rng = random.Random(seed)
    ordered: list[list[int]] = []
    for size in sorted({len(group) for group in groups.values()}, reverse=True):
        tied = [group for group in groups.values() if len(group) == size]
        rng.shuffle(tied)
        ordered.extend(tied)
    n = len(smiles)
    train_cut, validation_cut = int(n * train_fraction), int(n * (train_fraction + validation_fraction))
    train_ids: list[int] = []
    validation_ids: list[int] = []
    test_ids: list[int] = []
    for group in ordered:
        if len(train_ids) + len(group) <= train_cut:
            train_ids.extend(group)
        elif len(train_ids) + len(validation_ids) + len(group) <= validation_cut:
            validation_ids.extend(group)
        else:
            test_ids.extend(group)
    return {"train": train_ids, "validation": validation_ids, "test": test_ids}


def cached_split(name: str, smiles: Sequence[str], ids: Sequence[int]) -> tuple[dict[str, list[int]], dict]:
    """Compute the split, check it against the legacy GNN cache if one exists, and cache it."""
    split = scaffold_split(smiles, SPLIT_SEED, ids=ids)
    info = {"split_seed": SPLIT_SEED, "sizes": {k: len(v) for k, v in split.items()},
            "sha256": hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()}
    legacy = LEGACY_SPLITS.get(name)
    if legacy is not None and legacy.exists():
        legacy_split = json.loads(legacy.read_text())
        info["legacy_file"] = str(legacy)
        info["matches_legacy"] = legacy_split == split
        if not info["matches_legacy"]:
            raise RuntimeError(f"{name}: recomputed scaffold split differs from {legacy}")
    path = SPLITS_DIR / f"{name}.json"
    if path.exists() and json.loads(path.read_text()) != split:
        raise RuntimeError(f"{name}: recomputed split differs from cache {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split))
    return split, info
