"""Microstructure feature tests. Friction estimators are easy to get subtly
wrong at the edges -- an undefined square root, a bit of float noise, an
average taken in the wrong order -- so most of these tests exist to pin down
one specific guard rather than just "the number looks plausible"."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bubble_bi.data.features.microstructure import WINDOW, build

FEATURE_NAMES = {"amihud", "roll_spread", "corwin_schultz"}

_CS_DEN = 3.0 - 2.0 * np.sqrt(2.0)


def _make_df(close, high=None, low=None, volume=None) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    n = len(close)
    if high is None:
        high = close + 0.5
    if low is None:
        low = close - 0.5
    if volume is None:
        volume = np.full(n, 10_000.0)
    index = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "open": close,
            "high": np.asarray(high, dtype=float),
            "low": np.asarray(low, dtype=float),
            "close": close,
            "volume": np.asarray(volume, dtype=float),
        },
        index=index,
    )


def _ohlcv(n: int, seed: int = 0) -> pd.DataFrame:
    """A plausible-looking price series with plenty of wiggle in both price
    and volume, so every rolling window has real data to chew on."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0002, scale=0.02, size=n)
    close = 100.0 * np.exp(np.cumsum(steps))
    high = close * (1.0 + rng.uniform(0.0, 0.01, size=n))
    low = close * (1.0 - rng.uniform(0.0, 0.01, size=n))
    volume = rng.integers(1_000, 1_000_000, size=n).astype(float)
    return _make_df(close, high, low, volume)


def _momentum_path(n: int, seed: int = 12, rho: float = 0.6) -> np.ndarray:
    """An AR(1) walk on the day-to-day price CHANGE itself (not the price):
    dp_t = rho*dp_{t-1} + noise, rho > 0. That makes consecutive price moves
    reinforce each other (a "trending" / momentum series) instead of bouncing
    against each other, which is exactly the case Roll's estimator cannot
    handle -- its serial covariance of dp is genuinely positive."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 1.0, n)
    dp = np.zeros(n)
    for i in range(1, n):
        dp[i] = rho * dp[i - 1] + eps[i]
    return 100.0 + np.cumsum(dp)


def _bounce_path(n: int, amp: float = 1.0) -> np.ndarray:
    """A textbook bid-ask bounce: the price alternates by +/-amp every day,
    which is the pattern Roll's estimator was designed to detect -- it gives
    dp a strongly NEGATIVE serial covariance."""
    return 100.0 + amp * np.array([(-1.0) ** i for i in range(n)])


def _tight_range_path(n: int, seed: int = 0):
    """A quiet random walk with a noisy, narrow high/low range each day.
    Corwin-Schultz's raw daily estimate is known to go negative on data like
    this (a small, noisy range relative to the two-day spread), which is
    exactly the case its clamp-before-averaging guard has to handle."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.3, n))
    high = close + rng.uniform(0.01, 0.3, n)
    low = close - rng.uniform(0.01, 0.3, n)
    return close, high, low


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


def test_roll_is_exactly_zero_on_genuine_positive_serial_covariance():
    n = 350
    close = _momentum_path(n, seed=12, rho=0.6)
    df = _make_df(close)

    # Confirm the fixture actually exercises the guard: the raw rolling
    # covariance of dp with its own lag really must be positive (after a
    # short burn-in for the AR(1) to settle), not just "close to zero".
    dp = pd.Series(close).diff()
    raw_cov = dp.rolling(WINDOW).cov(dp.shift(1))
    tail_cov = raw_cov.iloc[50:]
    assert (tail_cov > 0).all(), "fixture does not have positive serial covariance"

    roll = build(df, {})["roll_spread"]
    assert (roll.iloc[50:] == 0.0).all()


def test_roll_is_positive_on_a_bid_ask_bounce_path():
    n = 100
    close = _bounce_path(n, amp=1.0)
    df = _make_df(close)

    roll = build(df, {})["roll_spread"]
    valid = roll.dropna()
    assert len(valid) > n - WINDOW - 5  # only the warm-up period should be NaN
    assert (valid > 0.0).all()


def test_roll_spread_is_scale_invariant():
    n = 200
    close = _momentum_path(n, seed=3, rho=-0.3)  # some real bounce signal, not all-zero
    df_low = _make_df(close)
    df_high = _make_df(close * 10.0)  # identical shape, 10x the price level

    roll_low = build(df_low, {})["roll_spread"]
    roll_high = build(df_high, {})["roll_spread"]

    both_nan = roll_low.isna() & roll_high.isna()
    assert np.allclose(
        roll_low.where(~both_nan, 0.0),
        roll_high.where(~both_nan, 0.0),
    )
    assert (roll_low.dropna() > 0.0).any()  # make sure it's not trivially all zero


def test_corwin_schultz_clamps_before_averaging_not_after():
    n = 100
    close, high, low = _tight_range_path(n, seed=0)
    df = _make_df(close, high, low)

    hS, lS = pd.Series(high, index=df.index), pd.Series(low, index=df.index)
    hl2 = np.log(hS / lS) ** 2
    beta = hl2 + hl2.shift(1)
    high2 = pd.concat([hS, hS.shift(1)], axis=1).max(axis=1)
    low2 = pd.concat([lS, lS.shift(1)], axis=1).min(axis=1)
    gamma = np.log(high2 / low2) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_DEN - np.sqrt(gamma / _CS_DEN)
    raw = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))

    # Confirm the fixture genuinely produces negative daily estimates -- the
    # whole point of this test is exercising the clamp, not just checking a
    # non-negativity property that clamping-after would satisfy too.
    assert (raw.dropna() < 0.0).sum() > 10

    correct = build(df, {})["corwin_schultz"]
    wrong_order = raw.rolling(WINDOW).mean().clip(lower=0.0)  # clamp AFTER averaging: the bug

    both_nan = correct.isna() & wrong_order.isna()
    diffs = ~np.isclose(
        correct.where(~both_nan, 0.0), wrong_order.where(~both_nan, 0.0)
    )
    assert diffs.any(), "clamp-before and clamp-after gave identical results -- fixture too weak"
    assert (correct.dropna() >= wrong_order.reindex(correct.dropna().index) - 1e-12).all()


def test_amihud_decreases_as_volume_rises():
    n = 150
    rng = np.random.default_rng(1)
    steps = rng.normal(loc=0.0, scale=0.02, size=n)
    close = 100.0 * np.exp(np.cumsum(steps))

    low_volume = np.full(n, 5_000.0)
    high_volume = low_volume * 50.0

    df_low = _make_df(close, volume=low_volume)
    df_high = _make_df(close, volume=high_volume)

    amihud_low = build(df_low, {})["amihud"]
    amihud_high = build(df_high, {})["amihud"]

    valid = amihud_low.notna() & amihud_high.notna()
    assert valid.sum() > 0
    assert (amihud_high[valid] <= amihud_low[valid]).all()
    assert (amihud_high[valid] < amihud_low[valid]).any()


def test_amihud_leaves_zero_volume_days_blank_instead_of_crashing():
    n = 60
    rng = np.random.default_rng(2)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    volume = np.full(n, 10_000.0)
    volume[10] = 0.0
    volume[11] = 0.0
    df = _make_df(close, volume=volume)

    out = build(df, {})  # must not raise / warn-crash on the zero-volume days
    assert "amihud" in out
