import numpy as np
import pandas as pd
import pytest

from bubble_bi.data.features import volatility

FEATURE_NAMES = {"realized_vol", "parkinson", "garman_klass", "yang_zhang", "atr_frac"}
# The four variance-based estimators that share WINDOW=20 (atr_frac is separate: it can
# go negative after fractional differencing, so it is not part of the "always >= 0" claim).
NONNEGATIVE_ESTIMATORS = {"realized_vol", "parkinson", "garman_klass", "yang_zhang"}


def _make_ohlc(n: int = 150, seed: int = 0, start_price: float = 100.0) -> pd.DataFrame:
    """A realistic-looking, strictly causal random OHLC path: high/low always bound
    open/close, everything stays positive."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)

    log_ret = rng.normal(0.0, 0.01, n)
    close = start_price * np.exp(np.cumsum(log_ret))

    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * np.exp(rng.normal(0.0, 0.002, n - 1))  # small overnight gap

    wick_up = np.abs(rng.normal(0.0, 0.005, n))
    wick_down = np.abs(rng.normal(0.0, 0.005, n))
    high = np.maximum(open_, close) * (1 + wick_up)
    low = np.minimum(open_, close) * (1 - wick_down)

    volume = rng.integers(1_000, 10_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def test_all_five_features_present_and_share_the_index():
    df = _make_ohlc()
    out = volatility.build(df, {})
    assert set(out) == FEATURE_NAMES
    for name, series in out.items():
        assert isinstance(series, pd.Series), name
        assert series.index.equals(df.index), name


def test_causality_truncating_the_future_never_changes_the_past():
    """The headline test: nothing computed for day t may depend on any day after t."""
    df = _make_ohlc(n=150, seed=1)
    truncated = df.iloc[:-30].copy()  # delete the future

    full_out = volatility.build(df, {})
    truncated_out = volatility.build(truncated, {})

    for name in FEATURE_NAMES:
        full_vals = full_out[name].iloc[: len(truncated)].to_numpy()
        trunc_vals = truncated_out[name].to_numpy()

        # NaN warm-up pattern must be identical -- truncating the future can't
        # somehow "unblock" or "reblock" a value that only ever looked backwards.
        assert np.array_equal(np.isnan(full_vals), np.isnan(trunc_vals)), name

        mask = ~np.isnan(full_vals)
        assert np.allclose(full_vals[mask], trunc_vals[mask]), name


def test_the_four_variance_estimators_are_nonnegative_and_finite_after_warmup():
    df = _make_ohlc(n=150, seed=2)
    out = volatility.build(df, {})
    warmup = 2 * volatility.WINDOW  # generous margin past the rolling window
    for name in NONNEGATIVE_ESTIMATORS:
        tail = out[name].iloc[warmup:]
        assert np.isfinite(tail).all(), name
        assert (tail >= 0).all(), name


def test_parkinson_matches_its_closed_form_for_a_constant_high_low_ratio():
    """If H/L is the same constant ratio r every day, parkinson has a closed form:
    sqrt(mean[(ln r)^2 / (4 ln 2)]) = |ln r| / (2 sqrt(ln 2))."""
    n = 60
    r = 1.05
    dates = pd.bdate_range("2020-01-01", periods=n)
    low = pd.Series(100.0, index=dates)
    high = low * r
    # open/close don't matter for parkinson; keep them valid and boring.
    df = pd.DataFrame({"open": low, "high": high, "low": low, "close": low, "volume": 1000.0})
    df.index = dates

    out = volatility.build(df, {})
    expected = abs(np.log(r)) / (2 * np.sqrt(np.log(2)))

    tail = out["parkinson"].iloc[volatility.WINDOW - 1 :]
    assert np.allclose(tail, expected)


def test_atr_frac_is_scale_invariant_across_price_levels():
    """Two companies with identical price *shapes* but different price *levels*
    must produce the same atr_frac -- that is the whole point of dividing by close
    before fractionally differencing (a raw-dollar ATR would not match)."""
    base = _make_ohlc(n=120, seed=3, start_price=50.0)
    scaled = base.copy()
    for col in ("open", "high", "low", "close"):
        scaled[col] = base[col] * 10.0  # same shape, 10x the price level

    out_base = volatility.build(base, {})
    out_scaled = volatility.build(scaled, {})

    a = out_base["atr_frac"].to_numpy()
    b = out_scaled["atr_frac"].to_numpy()
    mask = ~(np.isnan(a) | np.isnan(b))
    assert mask.any()
    assert np.array_equal(np.isnan(a), np.isnan(b))
    assert np.allclose(a[mask], b[mask])


def test_garman_klass_clips_negative_variance_to_zero_instead_of_nan():
    """Engineer a candle where the -(2ln2-1)*ln(C/O)^2 term dominates the
    0.5*ln(H/L)^2 term: a near-zero range with a huge open-to-close move. The raw
    mean variance goes negative; the result must be clipped to 0, not NaN."""
    n = 30
    dates = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame(
        {
            "open": 50.0,
            "high": 100.0,   # tiny range (high == low) ...
            "low": 100.0,
            "close": 150.0,  # ... versus a huge open->close swing
            "volume": 1000.0,
        },
        index=dates,
    )

    out = volatility.build(df, {})
    tail = out["garman_klass"].iloc[volatility.WINDOW - 1 :]
    assert not tail.isna().any()
    assert np.allclose(tail, 0.0)


def test_yang_zhang_clips_negative_variance_to_zero_instead_of_nan():
    """Same idea as garman_klass: a degenerate, engineered candle can push the
    Yang-Zhang variance sum negative before clipping (real OHLC data is well-behaved
    here, but the clip guards against the pathological case)."""
    n = 30
    dates = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame(
        {
            "open": 50.0,
            "high": 100.0,
            "low": 100.0,
            "close": 150.0,
            "volume": 1000.0,
        },
        index=dates,
    )

    out = volatility.build(df, {})
    # +1: the overnight term needs close.shift(1), which adds one extra NaN at the start.
    tail = out["yang_zhang"].iloc[volatility.WINDOW :]
    assert not tail.isna().any()
    assert np.allclose(tail, 0.0)
