import numpy as np
import pandas as pd

from bubble_bi.data.features_micro import amihud, corwin_schultz, obv, roll_spread


def _df(close, high=None, low=None, volume=None):
    n = len(close)
    close = pd.Series(close, dtype=float)
    high = pd.Series(high if high is not None else close * 1.01, dtype=float)
    low = pd.Series(low if low is not None else close * 0.99, dtype=float)
    volume = pd.Series(volume if volume is not None else np.full(n, 1e6), dtype=float)
    return pd.DataFrame({"open": close, "high": high, "low": low,
                         "close": close, "volume": volume})


def test_obv_accumulates_sign_times_normalized_volume():
    df = _df([10, 11, 10, 12], volume=[100, 200, 300, 400])
    window = 2
    sign = np.sign(df["close"].diff()).fillna(0.0)
    norm = df["volume"] / df["volume"].rolling(window).mean()
    expected = (sign * norm).cumsum()
    assert np.allclose(obv(df, window).to_numpy(), expected.to_numpy(), equal_nan=True)


def test_obv_is_causal():
    df = _df([10, 11, 10, 12, 13, 11, 12], volume=[100, 200, 300, 400, 250, 150, 500])
    full = obv(df, 3)
    trunc = obv(df.iloc[:5], 3)
    a, b = full.iloc[:5].to_numpy(), trunc.to_numpy()
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.allclose(a[~both_nan], b[~both_nan], atol=1e-10)


def test_obv_is_scale_free_across_tickers_with_proportional_volume():
    # This is the whole point of the fix: raw share volume differs by orders
    # of magnitude across tickers, so a raw cumsum of sign*volume would carry
    # a per-ticker level term. Normalizing by trailing mean volume removes it
    # -- two tickers whose volume differs by a constant multiplicative factor
    # (e.g. a big-cap vs. a small-cap trading the same relative pattern) must
    # produce IDENTICAL obv.
    close = [10, 11, 10, 12, 13, 11, 12, 14, 13, 15]
    base_volume = np.array([100, 200, 300, 400, 250, 150, 500, 600, 300, 800], dtype=float)
    window = 3
    small_cap = obv(_df(close, volume=base_volume), window)
    big_cap = obv(_df(close, volume=base_volume * 1000.0), window)
    assert np.allclose(small_cap.to_numpy(), big_cap.to_numpy(), equal_nan=True)


def test_roll_spread_is_zero_for_flat_zero_serial_covariance():
    # constant dP -> the TRUE serial covariance is exactly 0 (not positive);
    # this only exercises the float-noise tolerance path, not the cov > 0 guard.
    df = _df(np.linspace(100, 140, 120))
    s = roll_spread(df, 21).dropna()
    assert (s == 0).all()


def test_roll_spread_is_zero_when_serial_cov_is_genuinely_positive():
    # momentum/trend-following path: AR(1) on the price increments with
    # rho=+0.6 makes consecutive dP genuinely, substantially positively
    # autocorrelated (verified below) -> Roll's estimator is undefined and
    # must return 0. This is the test that actually exercises the cov > 0
    # guard, unlike the flat-price fixture above.
    rng = np.random.default_rng(0)
    n = 200
    dp = np.zeros(n)
    eps = rng.normal(0, 1.0, n)
    for i in range(1, n):
        dp[i] = 0.6 * dp[i - 1] + eps[i]
    close = 100.0 + np.cumsum(dp)
    df = _df(close)

    raw_dp = df["close"].diff()
    cov = raw_dp.rolling(21).cov(raw_dp.shift(1)).dropna()
    assert (cov > 0).mean() > 0.9  # confirm the fixture is genuinely positive-cov

    s = roll_spread(df, 21).dropna()
    assert (s == 0).all()


def test_roll_spread_positive_for_bid_ask_bounce():
    # alternating price -> negative serial covariance -> a real spread estimate
    base = np.full(120, 100.0)
    base[1::2] += 0.5                                   # bounce
    s = roll_spread(_df(base), 21).dropna()
    assert (s > 0).mean() > 0.9


def test_corwin_schultz_never_negative():
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    df = _df(c, high=c * (1 + rng.uniform(0, 0.02, 200)),
             low=c * (1 - rng.uniform(0, 0.02, 200)))
    s = corwin_schultz(df, 21).dropna()
    assert (s >= 0).all()


def test_corwin_schultz_clamps_before_rolling_not_after():
    # Mostly flat, narrow-band days (small positive raw CS) with a single-day
    # spike to a far-away level every 5th day. That spike day's two-day range
    # (gamma) dwarfs the single-day ranges (beta) on both sides of it, driving
    # the raw CS estimate sharply negative for those two adjacent day-pairs
    # while leaving the other days in the same rolling window positive --
    # exactly the mix needed to distinguish clamp-before from clamp-after.
    n = 60
    levels = np.full(n, 100.0)
    levels[np.arange(4, n, 5)] = 250.0
    high = levels * 1.002
    low = levels * 0.998
    df = _df(levels, high=high, low=low)

    h, l = df["high"], df["low"]
    _DEN = 3.0 - 2.0 * np.sqrt(2.0)
    hl2 = np.log(h / l) ** 2
    beta = hl2 + hl2.shift(1)
    h2 = pd.concat([h, h.shift(1)], axis=1).max(axis=1)
    l2 = pd.concat([l, l.shift(1)], axis=1).min(axis=1)
    gamma = np.log(h2 / l2) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _DEN - np.sqrt(gamma / _DEN)
    raw = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))

    # Sanity: the fixture must actually exercise the negative-raw-value case,
    # otherwise the ordering distinction below is untested.
    assert (raw < 0).any()

    window = 5
    wrong = raw.rolling(window).mean().clip(lower=0.0)  # clamp AFTER averaging
    right = corwin_schultz(df, window)                   # clamp BEFORE averaging (correct)

    idx = right.dropna().index.intersection(wrong.dropna().index)
    r, w = right.loc[idx], wrong.loc[idx]

    # The two orderings must produce different numbers here...
    assert not np.allclose(r, w)
    # ...and clamping negatives away per-day before averaging can only raise
    # (or leave unchanged) the resulting rolling mean.
    assert (r >= w - 1e-12).all()


def test_amihud_decreases_with_volume():
    c = np.array([100.0, 101.0, 100.0, 101.0] * 30)
    low_vol = amihud(_df(c, volume=np.full(len(c), 1e5)), 21).dropna()
    high_vol = amihud(_df(c, volume=np.full(len(c), 1e8)), 21).dropna()
    assert low_vol.iloc[-1] > high_vol.iloc[-1]


def test_micro_features_are_causal():
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    df = _df(c)
    for fn in (lambda d: amihud(d, 21), lambda d: roll_spread(d, 21),
               lambda d: corwin_schultz(d, 21), lambda d: obv(d, 21)):
        full = fn(df)
        trunc = fn(df.iloc[:201])
        a, b = full.iloc[:201].to_numpy(), trunc.to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        assert np.allclose(a[~both_nan], b[~both_nan], atol=1e-10)
