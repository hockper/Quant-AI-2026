"""Price feature tests. The one that matters most is causality: a feature
that quietly uses tomorrow's price will look great on paper and lose money
for real, so we prove no past value moves when the future changes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bubble_bi.data.features.price import build

FEATURE_NAMES = {
    "log_return", "sma_ratio_5", "sma_ratio_10", "sma_ratio_20",
    "rsi", "macd", "macd_signal", "macd_hist",
}


def _ohlcv(n: int, seed: int = 0) -> pd.DataFrame:
    """A plausible-looking price series: nothing fancy, just enough wiggle for
    every rolling/ewm window to have real data to chew on."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n)
    steps = rng.normal(loc=0.0002, scale=0.02, size=n)
    close = 100.0 * np.exp(np.cumsum(steps))
    high = close * (1.0 + rng.uniform(0.0, 0.01, size=n))
    low = close * (1.0 - rng.uniform(0.0, 0.01, size=n))
    open_ = close * (1.0 + rng.uniform(-0.005, 0.005, size=n))
    volume = rng.integers(1_000, 1_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def _rising(n: int) -> pd.DataFrame:
    # +2% almost every day, with a tiny -0.1% dip every 25 days. A purely
    # monotonic series (no losses at all, ever) makes Wilder's avg-loss exactly
    # zero forever and RSI comes out NaN by this formula's own convention --
    # a handful of small dips keeps it a genuine (near-)division, not that
    # degenerate case, while the series is still overwhelmingly rising.
    steps = np.full(n, 0.02)
    steps[::25] = -0.001
    close = 100.0 * np.exp(np.cumsum(steps))
    index = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": np.full(n, 1_000.0)},
        index=index,
    )


def _falling(n: int) -> pd.DataFrame:
    # Mirror image of _rising: -2% almost every day, tiny +0.1% bounce every 25.
    steps = np.full(n, -0.02)
    steps[::25] = 0.001
    close = 100.0 * np.exp(np.cumsum(steps))
    index = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": np.full(n, 1_000.0)},
        index=index,
    )


def test_every_feature_is_present_named_exactly_and_shares_the_index():
    df = _ohlcv(120)
    out = build(df, {})
    assert set(out) == FEATURE_NAMES
    for name, series in out.items():
        assert isinstance(series, pd.Series), name
        assert series.index.equals(df.index), name


def test_causality_truncating_the_future_does_not_change_the_past():
    df = _ohlcv(300)
    keep = 240
    full = build(df, {})
    truncated = build(df.iloc[:keep], {})

    for name in FEATURE_NAMES:
        past_full = full[name].iloc[:keep].to_numpy()
        past_trunc = truncated[name].to_numpy()
        both_nan = np.isnan(past_full) & np.isnan(past_trunc)
        assert np.allclose(
            np.where(both_nan, 0.0, past_full),
            np.where(both_nan, 0.0, past_trunc),
            equal_nan=False,
        ), f"{name} changed when future rows were deleted"
        # deleting the future must not turn a known value into NaN, or vice versa
        assert np.array_equal(np.isnan(past_full), np.isnan(past_trunc)), name


def test_rsi_on_a_strictly_rising_series_sits_near_100():
    df = _rising(200)
    rsi = build(df, {})["rsi"]
    tail = rsi.iloc[40:]  # past warm-up and the first small dip
    assert not tail.isna().any()
    assert (tail > 95).all()


def test_rsi_on_a_strictly_falling_series_sits_near_0():
    df = _falling(200)
    rsi = build(df, {})["rsi"]
    tail = rsi.iloc[40:]
    assert not tail.isna().any()
    assert (tail < 5).all()
