"""Mirrored-encoding tests. Run: python -m xgb_mh.tests.test_mirror (or pytest).

The run-level tests fit real XGBoost models on FreeSolv (Experiment B, budget 20; ~2 min in total) and
read results/xgb_mh/full_ref/calls/ read-only for a regression check of the standard encoding.
"""
import json

import numpy as np

from xgb_mh import RESULTS_DIR
from xgb_mh.experiment import MIRROR_NOTE, load, make_fitness, run_search
from xgb_mh.optimizers._common import initial_design
from xgb_mh.space import decode, dimension, mirror

DS, EXP, OPT, SEED, BUDGET = "FreeSolv", "B", "gwo_ref", 3, 20


def test_mirror_maps_to_one_minus_x():
    rng = np.random.default_rng(0)
    for nf in (None, 50):
        for _ in range(200):
            x = rng.random(dimension(nf))
            assert decode(mirror(x), nf).key == decode(1 - x, nf).key
            assert np.array_equal(mirror(mirror(x)), x)
    # Feature j is selected under the mirror iff x_j < 0.5 (strictly), i.e. the selection is reversed.
    x = np.concatenate([np.full(8, 0.3), [0.2, 0.7, 0.5, 0.49, 0.9]])
    assert decode(mirror(x), 5).selected == (0, 3)
    assert decode(x, 5).selected == (1, 4)
    # Bounds map onto bounds: every hyperparameter end swaps.
    lo, hi = decode(mirror(np.ones(8))).params, decode(np.zeros(8)).params
    assert lo == hi


def _run(mirror_flag: bool) -> dict:
    return run_search(DS, EXP, OPT, SEED, BUDGET, RESULTS_DIR / "unused", write=False, mirror=mirror_flag)


def test_mirrored_run_budget_determinism_and_initial_design():
    a, b = _run(True), _run(True)
    assert a["n_calls"] == BUDGET and b["n_calls"] == BUDGET
    assert a["_fitness_trajectory"] == b["_fitness_trajectory"]          # same seed -> identical trajectory
    assert a["encoding"] == "mirrored" and json.loads(a["optimizer_meta"])["encoding"] == MIRROR_NOTE
    # Calls 1-10 are the common design, mirrored: fitness(decode(1 - design)).
    data, _ = load(DS)
    fitness, dim, nf = make_fitness(data, EXP, SEED)
    design = initial_design(np.random.default_rng(SEED), dim)
    expected = [fitness(1.0 - x)[0] for x in design]
    assert a["_fitness_trajectory"][:10] == expected


def test_standard_encoding_unchanged():
    std = _run(False)
    assert std["encoding"] == "standard"
    path = RESULTS_DIR / "full_ref" / "calls" / f"{DS}_{EXP}_{OPT}_s{SEED}.jsonl"
    with path.open() as handle:
        full_ref_first10 = [json.loads(next(handle))["fitness"] for _ in range(10)]
    assert std["_fitness_trajectory"][:10] == full_ref_first10              # identical to the finished full_ref run
    assert _run(True)["_fitness_trajectory"][:10] != full_ref_first10       # and the mirror really changes them


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
