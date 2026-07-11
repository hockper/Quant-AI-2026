import numpy as np
import pandas as pd

from bubble_bi.data.features_memory import entropy, hurst


def test_hurst_near_half_for_random_walk():
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600))))
    h = hurst(close, window=200).dropna()
    assert abs(h.mean() - 0.5) < 0.15          # random walk -> H ~ 0.5


def test_hurst_above_half_for_strong_trend():
    n = 600
    rng = np.random.default_rng(1)
    drift = np.linspace(0, 1.2, n)                       # persistent trend
    close = pd.Series(100 * np.exp(drift + 0.001 * rng.normal(size=n)))
    h = hurst(close, window=200).dropna()
    assert h.mean() > 0.6


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
