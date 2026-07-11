import numpy as np
import pandas as pd

from bubble_bi.data.features.fracdiff import frac_diff, weights


def test_weights_follow_the_binomial_recurrence():
    d = 0.45
    w = weights(d, detrend=False)              # the raw, textbook weights
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
    got = frac_diff(x, d=0.0, detrend=False).dropna()   # d=0 is the identity...
    assert np.allclose(got.to_numpy(), x.loc[got.index].to_numpy())


def test_it_never_looks_at_the_future():
    x = pd.Series(np.cumsum(np.random.default_rng(0).normal(size=400)))
    full = frac_diff(x)
    truncated = frac_diff(x.iloc[:250])          # delete the future, recompute
    a, b = full.iloc[:250].to_numpy(), truncated.to_numpy()
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.allclose(a[~both_nan], b[~both_nan], atol=1e-10)


def test_truncating_the_weights_would_leave_a_trend_behind():
    # The bug the detrend correction exists to fix. In theory the weights sum to zero,
    # which is what removes a trend. Truncated at 49 lags they sum to 0.108 -- so 11%
    # of a sixteen-year climb in the log price would survive into the feature.
    assert np.isclose(weights(detrend=False).sum(), 0.108, atol=0.01)


def test_the_corrected_weights_sum_to_zero_so_a_trend_cannot_survive():
    w = weights()                              # detrend=True by default
    assert abs(w.sum()) < 1e-12


def test_the_correction_touches_only_today_and_leaves_the_memory_alone():
    # The long tail IS the memory. The fix must not disturb it -- only today's weight.
    raw, fixed = weights(detrend=False), weights()
    assert np.array_equal(raw[1:], fixed[1:])          # every past day: untouched
    assert not np.isclose(raw[0], fixed[0])            # today: nudged
    assert np.isclose(fixed[0], raw[0] - raw.sum())


def test_a_trend_in_the_input_does_not_survive_into_the_output():
    steady_climb = pd.Series(np.linspace(0, 10, 500))   # like a log price, over years

    leaky = frac_diff(steady_climb, detrend=False).dropna()
    fixed = frac_diff(steady_climb).dropna()

    # The leaky version keeps climbing with the input; the corrected one is dead flat.
    assert leaky.iloc[-1] - leaky.iloc[0] > 0.5
    assert abs(fixed.iloc[-1] - fixed.iloc[0]) < 1e-9
