import numpy as np

from bubble_bi.eval.metrics import daily_rank_ic, rank_ic, rank_icir


def _mask_all(shape):
    return np.ones(shape, dtype=bool)


def test_perfect_ranking_gives_rank_ic_one():
    target = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    pred = target.copy()
    assert rank_ic(pred, target, _mask_all(target.shape)) == 1.0


def test_reversed_ranking_gives_rank_ic_minus_one():
    target = np.array([[1.0, 2.0, 3.0, 4.0]])
    pred = -target
    assert rank_ic(pred, target, _mask_all(target.shape)) == -1.0


def test_masked_names_are_ignored():
    target = np.array([[1.0, 2.0, 3.0, np.nan]])
    pred = np.array([[1.0, 2.0, 3.0, 999.0]])
    mask = np.array([[True, True, True, False]])
    assert rank_ic(pred, target, mask) == 1.0


def test_day_with_one_valid_name_is_nan():
    target = np.array([[1.0, np.nan, np.nan]])
    pred = np.array([[1.0, 2.0, 3.0]])
    mask = np.array([[True, False, False]])
    assert np.isnan(daily_rank_ic(pred, target, mask)[0])


def test_rank_icir_is_finite_for_varied_days():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(50, 10))
    pred = target + rng.normal(scale=0.5, size=(50, 10))
    val = rank_icir(pred, target, _mask_all(target.shape))
    assert np.isfinite(val)
