"""Optuna TPE: one suggest_float in [0,1] per dimension.

The common initial design is injected with enqueue_trial. Enqueued trials are ordinary COMPLETE
trials, so they count toward n_startup_trials = 10: trial 11 is the first TPE-model-based suggestion
(verified in tests/test_optimizers.py). The sampler is seeded with the tuning seed.
"""
from __future__ import annotations

import numpy as np

from ._common import N_INITIAL, initial_design

N_STARTUP_TRIALS = N_INITIAL


def run(objective, dim: int, budget: int, seed: int, n_startup_trials: int = N_STARTUP_TRIALS) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    for x in initial_design(np.random.default_rng(seed), dim)[:budget]:
        study.enqueue_trial({f"x{i}": float(v) for i, v in enumerate(x)})
    study.optimize(lambda trial: objective(np.array([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(dim)])),
                   n_trials=budget)
    # Effective (resolved) Parzen estimator settings; Optuna 5 defaults to multivariate TPE.
    parzen = sampler._parzen_estimator_parameters
    return {"optuna_version": optuna.__version__, "n_startup_trials": n_startup_trials,
            "enqueued_initial_design": N_INITIAL,
            "multivariate": bool(parzen.multivariate), "consider_endpoints": bool(parzen.consider_endpoints),
            "constant_liar": bool(sampler._constant_liar), "n_ei_candidates": sampler._n_ei_candidates,
            # constant_liar only affects RUNNING trials; optimize() here is sequential (n_jobs=1),
            # so it has no effect (verified in tests/test_optimizers.py).
            "constant_liar_effective": False, "study_n_jobs": 1}
