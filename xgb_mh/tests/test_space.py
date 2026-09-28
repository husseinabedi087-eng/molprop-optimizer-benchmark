"""decode() tests. Run: python -m xgb_mh.tests.test_space (or pytest)."""
import math

import numpy as np

from xgb_mh.space import HP_SPEC, N_HP, decode, dimension, select_features


def _hp(x_value: float, name: str):
    x = np.full(N_HP, 0.5)
    x[[n for n, *_ in HP_SPEC].index(name)] = x_value
    return decode(x).params[name]


def test_ends_are_exact():
    lo, hi = decode(np.zeros(N_HP)).params, decode(np.ones(N_HP)).params
    for name, kind, low, high in HP_SPEC:
        assert lo[name] == low and hi[name] == high, (name, lo[name], hi[name])
        if kind == "int":
            assert isinstance(lo[name], int) and isinstance(hi[name], int)


def test_integer_bounds_reachable_with_equal_bins():
    grid = np.linspace(0, 1, 200_001)
    for name, lo, hi in (("max_depth", 3, 10), ("n_estimators_max", 100, 1000)):
        values = np.array([_hp(v, name) for v in grid])
        assert set(values.tolist()) == set(range(lo, hi + 1)), name
        counts = np.bincount(values - lo)
        # Equal-width bins: the end bins are not half-sized (unlike rounding).
        assert counts.min() / counts.max() > 0.95, (name, counts.min(), counts.max())
    assert _hp(0.999, "max_depth") == 10 and _hp(0.874, "max_depth") == 9 and _hp(0.125, "max_depth") == 4


def test_log_scale():
    for name, kind, lo, hi in HP_SPEC:
        if kind != "log":
            continue
        assert _hp(0.0, name) == lo and _hp(1.0, name) == hi
        assert math.isclose(_hp(0.5, name), math.sqrt(lo * hi), rel_tol=1e-12), name
    assert math.isclose(_hp(0.5, "learning_rate"), math.sqrt(0.003), rel_tol=1e-12)
    assert math.isclose(_hp(0.5, "min_child_weight"), math.sqrt(10), rel_tol=1e-12)
    assert math.isclose(_hp(0.5, "reg_lambda"), 0.1, rel_tol=1e-12)
    assert math.isclose(_hp(0.5, "reg_alpha"), 0.1, rel_tol=1e-12)


def test_linear_midpoint():
    assert math.isclose(_hp(0.5, "subsample"), 0.75) and math.isclose(_hp(0.5, "colsample_bytree"), 0.65)


def test_feature_selection_rule():
    assert select_features(np.array([0.2, 0.6, 0.5, 0.9])) == (1, 3)  # 0.5 itself is not selected
    assert select_features(np.array([0.1, 0.45, 0.3])) == (1,)       # none > 0.5 -> argmax
    assert select_features(np.array([0.5, 0.5, 0.2])) == (0,)        # ties -> first argmax
    assert select_features(np.ones(5)) == (0, 1, 2, 3, 4)


def test_experiment_dimensions():
    assert dimension(None) == 8 and dimension(188) == 196
    assert decode(np.full(8, 0.3)).selected is None
    x = np.concatenate([np.full(8, 0.3), [0.7, 0.1, 0.9]])
    cfg = decode(x, n_features=3)
    assert cfg.selected == (0, 2) and cfg.n_selected(3) == 2
    for bad in (np.zeros(9), np.full(8, np.nan)):
        try:
            decode(bad)
        except ValueError:
            continue
        raise AssertionError("invalid vector accepted")


def test_clipping_and_cache_key():
    assert decode(np.full(8, 1.7)).params == decode(np.ones(8)).params
    assert decode(np.full(8, -3.0)).key == decode(np.zeros(8)).key
    assert decode(np.full(8, 0.2)).key != decode(np.full(8, 0.21)).key


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
