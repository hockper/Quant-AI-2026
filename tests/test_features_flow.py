"""Flow feature tests.

Two things matter beyond the basics: causality (same reasoning as the price
features — no feature may see tomorrow), and the volume-normalisation fix in
obv_frac. Before that fix, obv_frac's cumulative sum ran at whatever scale a
company's raw share volume happened to be, so it doubled as a company ID
rather than a genuine order-flow signal. The test below is the direct proof
that normalising volume first removes that leak: two companies with identical
prices but volumes a factor of 10,000 apart must produce the same obv_frac.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bubble_bi.data.features.flow import build

FEATURE_NAMES = {"volume_z", "volume_frac", "obv_frac"}


def _ohlcv(n: int, seed: int = 0, volume_scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n)
    steps = rng.normal(loc=0.0002, scale=0.02, size=n)
    close = 100.0 * np.exp(np.cumsum(steps))
    high = close * (1.0 + rng.uniform(0.0, 0.01, size=n))
    low = close * (1.0 - rng.uniform(0.0, 0.01, size=n))
    open_ = close * (1.0 + rng.uniform(-0.005, 0.005, size=n))
    # Base share counts in a realistic thinly-traded range; volume_scale then
    # stretches the whole series by orders of magnitude, as if it were a
    # completely different company's typical turnover.
    volume = rng.integers(1_000, 50_000, size=n).astype(float) * volume_scale
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
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
        assert np.array_equal(np.isnan(past_full), np.isnan(past_trunc)), name


def test_obv_frac_is_unchanged_by_a_company_wide_volume_scale():
    # Same price path (and hence the same sign(close.diff()) buy/sell pattern),
    # but one company trades ~1e5 shares a day and the other ~1e9 -- a factor
    # of 10,000 apart, deliberately bigger than any real-world pair, to make
    # the point unmissable.
    small = _ohlcv(300, seed=1, volume_scale=1.0)
    big = small.copy()
    big["volume"] = small["volume"] * 1e4

    obv_small = build(small, {})["obv_frac"].to_numpy()
    obv_big = build(big, {})["obv_frac"].to_numpy()

    both_nan = np.isnan(obv_small) & np.isnan(obv_big)
    assert np.array_equal(np.isnan(obv_small), np.isnan(obv_big))
    assert np.allclose(
        np.where(both_nan, 0.0, obv_small),
        np.where(both_nan, 0.0, obv_big),
    ), "obv_frac depends on the company's absolute volume scale, not just its flow"
