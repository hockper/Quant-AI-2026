import numpy as np
import pandas as pd

from bubble_bi.data.features.fracdiff import frac_diff, weights


def test_weights_follow_the_binomial_recurrence():
    d = 0.45
    w = weights(d)
    assert w[0] == 1.0
    assert np.isclose(w[1], -d)
    for k in range(2, len(w)):
        assert np.isclose(w[k], -w[k - 1] * (d - k + 1) / k)


def test_d_of_one_is_just_an_ordinary_difference():
    x = pd.Series(np.arange(30, dtype=float) ** 1.3)
    got = frac_diff(x, d=1.0)
    want = x.diff()
    both = got.notna() & want.notna()
    assert np.allclose(got[both], want[both])


def test_d_of_zero_changes_nothing():
    x = pd.Series(np.arange(30, dtype=float) ** 1.3)
    got = frac_diff(x, d=0.0).dropna()
    assert np.allclose(got.to_numpy(), x.loc[got.index].to_numpy())


def test_it_never_looks_at_the_future():
    x = pd.Series(np.cumsum(np.random.default_rng(0).normal(size=400)))
    full = frac_diff(x)
    truncated = frac_diff(x.iloc[:250])          # delete the future, recompute
    a, b = full.iloc[:250].to_numpy(), truncated.to_numpy()
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.allclose(a[~both_nan], b[~both_nan], atol=1e-10)


def test_the_weights_do_not_sum_to_zero():
    # This is why a raw running total must never be fed to frac_diff: a fraction of
    # its LEVEL survives, and a level that differs between companies is a company ID.
    # (Truncating the weights is what breaks the sum; the untruncated series sums to 0.)
    total = weights().sum()
    assert total > 0.05
    assert np.isclose(total, 0.108, atol=0.01)
