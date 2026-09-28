"""Optimizer contract tests. Run: python -m xgb_mh.tests.test_optimizers (or pytest)."""
import numpy as np

from xgb_mh.objective import BudgetExceeded, Objective
from xgb_mh.optimizers import OPTIMIZERS, run
from xgb_mh.optimizers._common import linear_decay
from xgb_mh.space import decode

BUDGET = 100


def sphere(x):
    return float(np.sum((x - 0.5) ** 2)), {}


def _run(name, dim, seed, key=None, fn=sphere):
    obj = Objective(fn, BUDGET, key=key)
    meta = run(name, obj, dim, seed)
    return obj, meta


def test_exact_budget_and_bounds():
    for name in OPTIMIZERS:
        for dim in (8, 200):
            obj, _ = _run(name, dim, 0)
            assert obj.n_calls == BUDGET, (name, dim)
            xs = np.array([c.x for c in obj.calls])
            assert xs.shape == (BUDGET, dim) and xs.min() >= 0 and xs.max() <= 1, name


def test_same_seed_identical_trajectory():
    for name in OPTIMIZERS:
        for dim in (8, 200):
            a, _ = _run(name, dim, 7)
            b, _ = _run(name, dim, 7)
            c, _ = _run(name, dim, 8)
            xa, xb, xc = (np.array([k.x for k in o.calls]) for o in (a, b, c))
            assert np.array_equal(xa, xb), (name, dim)
            assert [k.fitness for k in a.calls] == [k.fitness for k in b.calls]
            assert not np.array_equal(xa, xc), (name, dim)


def test_cache_hits_count_toward_budget():
    coarse = lambda x: tuple(np.round(x, 0))  # very coarse key -> many hits
    for name in OPTIMIZERS:
        obj, _ = _run(name, 8, 1, key=coarse)
        assert obj.n_calls == BUDGET and obj.n_unique < BUDGET, (name, obj.n_unique)
        assert sum(c.cache_hit for c in obj.calls) == BUDGET - obj.n_unique
    # Decoded-config key, Experiment B shape: still exactly 100 calls.
    for name in OPTIMIZERS:
        obj, _ = _run(name, 8 + 50, 2, key=lambda x: decode(x, 50).key)
        assert obj.n_calls == BUDGET


def test_budget_overrun_raises():
    obj = Objective(sphere, 3)
    for _ in range(3):
        obj(np.zeros(2))
    try:
        obj(np.zeros(2))
    except BudgetExceeded:
        return
    raise AssertionError("fourth call allowed")


def test_best_so_far_monotone():
    for name in OPTIMIZERS:
        obj, _ = _run(name, 10, 3)
        best = [c.best_so_far for c in obj.calls]
        assert all(b2 <= b1 for b1, b2 in zip(best, best[1:]))
        assert best[-1] == min(c.fitness for c in obj.calls)


def test_schedules():
    a = [linear_decay(k, 9) for k in range(9)]
    assert a[0] == 2.0 and a[-1] == 0.0 and np.allclose(np.diff(a), -0.25)
    a2 = [linear_decay(k, 9, -1.0, -2.0) for k in range(9)]
    assert a2[0] == -1.0 and a2[-1] == -2.0


def test_pso_velocity_clamp():
    obj, _ = _run("pso", 20, 4)
    xs = np.array([c.x for c in obj.calls]).reshape(10, 10, 20)  # iteration, particle, dim
    assert np.abs(np.diff(xs, axis=0)).max() <= 0.2 + 1e-12


def test_woa_uses_all_branches():
    for name in ("woa", "woa_ref"):
        totals = {"search": 0, "encircle": 0, "spiral": 0}
        for seed in range(10):
            _, meta = _run(name, 10, seed)
            assert sum(meta["branch_counts"].values()) == 90
            for k, v in meta["branch_counts"].items():
                totals[k] += v
        assert all(v > 0 for v in totals.values()), (name, totals)


def test_reference_schedules():
    # 10 evaluation rounds -> 9 evaluated updates with t = 0..8 of T = 10; a never reaches 0.
    expected_a = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4]
    expected_a2 = [-1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8]
    _, gwo_meta = _run("gwo_ref", 10, 0)
    _, woa_meta = _run("woa_ref", 10, 0)
    assert np.allclose(gwo_meta["a_used"], expected_a) and np.allclose(woa_meta["a_used"], expected_a)
    assert np.allclose(woa_meta["a2_used"], expected_a2)
    assert min(gwo_meta["a_used"]) > 0 and min(woa_meta["a_used"]) > 0


SCHEDULE_HIT_LIMIT = 2


def schedule_cache_hits(name: str, dim: int, seeds=range(30)) -> tuple[list[int], list[int]]:
    """Per run on the sphere (key = exact position vector): (total cache hits, hits not explained by WOA's
    intrinsic leader-stay) - the second number is an upper bound on hits caused by the schedule.

    woa_ref reports `leader_spiral_stays`: the whale sitting exactly on X* takes the spiral branch (D = 0)
    and re-evaluates X*. That is intrinsic to WOA (present in the reference code too), not to the schedule.
    The remaining 0-2 hits seen at d = 8-10 are whales clipped into the same corner of [0,1]^d (every
    coordinate at 0 or 1) after an overshooting move; none occur at d = 200.
    """
    total, schedule = [], []
    for seed in seeds:
        obj, meta = _run(name, dim, seed)
        hits = BUDGET - obj.n_unique
        total.append(hits)
        schedule.append(hits - meta.get("leader_spiral_stays", 0))
    return total, schedule


def test_reference_variants_have_no_schedule_collapse():
    for name in ("gwo_ref", "woa_ref"):
        for dim in (8, 10, 200):
            _, schedule = schedule_cache_hits(name, dim)
            assert min(schedule) >= 0 and max(schedule) <= SCHEDULE_HIT_LIMIT, (name, dim, schedule)


def test_population_structure():
    for name in ("gwo", "woa", "pso", "gwo_ref", "woa_ref"):
        _, meta = _run(name, 10, 0)
        assert meta["pop"] == 10 and meta["iterations"] == 10
    _, meta = _run("tpe", 10, 0)
    assert meta["n_startup_trials"] == 10


def _tpe_trajectory(dim, seed, **sampler_kwargs):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed, **sampler_kwargs))
    study.optimize(lambda t: sphere(np.array([t.suggest_float(f"x{i}", 0, 1) for i in range(dim)]))[0], n_trials=BUDGET)
    return np.array([[t.params[f"x{i}"] for i in range(dim)] for t in study.trials])


def test_tpe_constant_liar_has_no_effect_sequentially():
    # constant_liar only imputes values for RUNNING trials; study.optimize(n_jobs=1) never has any.
    for dim in (8, 200):
        assert np.array_equal(_tpe_trajectory(dim, 3, n_startup_trials=10, constant_liar=True),
                              _tpe_trajectory(dim, 3, n_startup_trials=10, constant_liar=False)), dim


def test_tpe_enqueued_design_counts_toward_startup():
    # Both runs enqueue the same 10-point design. With n_startup_trials=10 the enqueued trials fill the
    # startup phase, so trial 11 is TPE-model-based; with n_startup_trials=100 trial 11 is a random draw.
    # If enqueued trials did NOT count, both trial-11 suggestions would be the same random draw.
    from xgb_mh.optimizers import tpe
    from xgb_mh.optimizers._common import initial_design
    for dim in (8, 200):
        design = initial_design(np.random.default_rng(5), dim)
        runs = {}
        for n_startup in (10, BUDGET):
            obj = Objective(sphere, BUDGET)
            meta = tpe.run(obj, dim, BUDGET, 5, n_startup_trials=n_startup)
            runs[n_startup] = np.array([c.x for c in obj.calls])
        assert np.array_equal(runs[10][:10], design) and np.array_equal(runs[BUDGET][:10], design)
        assert not np.allclose(runs[10][10], runs[BUDGET][10]), dim
    obj, meta = _run("tpe", 8, 5)
    assert meta["n_startup_trials"] == 10 and meta["enqueued_initial_design"] == 10


def test_common_initial_design():
    from xgb_mh.optimizers._common import initial_design
    for dim in (8, 200):
        for seed in (0, 1, 2):
            design = initial_design(np.random.default_rng(seed), dim)
            for name in OPTIMIZERS:
                obj, meta = _run(name, dim, seed)
                assert np.array_equal(np.array([c.x for c in obj.calls[:10]]), design), (name, dim, seed)
                assert meta["initial_design"].startswith("common initial design (10 points)")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
