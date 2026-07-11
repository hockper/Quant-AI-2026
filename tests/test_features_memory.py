import numpy as np
import pandas as pd
import pytest

from bubble_bi.data.features_memory import entropy, hurst


def test_hurst_raises_for_window_too_small_for_two_scales():
    # window=15 -> candidate scales {1,3,7,15}, only 15 survives the >=8 filter,
    # so fewer than 2 scales are usable. Silently returning all-NaN here would
    # zero out the panel's entire valid mask; it must raise instead.
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 200))))
    with pytest.raises(ValueError):
        hurst(close, 15)


def test_hurst_near_half_for_random_walk():
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600))))
    h = hurst(close, window=200).dropna()
    assert abs(h.mean() - 0.5) < 0.15          # random walk -> H ~ 0.5


def _prices_from_ar1(rho, n=1200, seed=0, sigma=0.01):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = rho * r[i - 1] + eps[i]
    return pd.Series(100 * np.exp(np.cumsum(r)))


def test_hurst_higher_for_persistent_than_random_walk():
    h_persistent = hurst(_prices_from_ar1(0.4, seed=1), window=200).dropna().mean()
    h_rw = hurst(_prices_from_ar1(0.0, seed=1), window=200).dropna().mean()
    assert h_persistent > h_rw


def test_hurst_lower_for_antipersistent_than_random_walk():
    h_antipersistent = hurst(_prices_from_ar1(-0.4, seed=1), window=200).dropna().mean()
    h_rw = hurst(_prices_from_ar1(0.0, seed=1), window=200).dropna().mean()
    assert h_antipersistent < h_rw


def test_entropy_zero_for_constant_returns():
    close = pd.Series(100 * np.exp(np.arange(200) * 0.001))   # constant log-return
    e = entropy(close, window=60, bins=10).dropna()
    assert np.allclose(e.to_numpy(), 0.0, atol=1e-9)


def test_entropy_positive_for_noisy_returns():
    rng = np.random.default_rng(2)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))))
    e = entropy(close, window=60, bins=10).dropna()
    assert (e > 0.5).mean() > 0.9


def test_memory_features_are_causal():
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))))
    for fn in (lambda s: hurst(s, 100), lambda s: entropy(s, 60, 10)):
        full = fn(close)
        trunc = fn(close.iloc[:251])
        a, b = full.iloc[:251].to_numpy(), trunc.to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        assert np.allclose(a[~both_nan], b[~both_nan], atol=1e-10)
