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


def test_obv_accumulates_signed_volume():
    df = _df([10, 11, 10, 12], volume=[100, 200, 300, 400])
    # signs: nan->0, +, -, +   => 0, +200, -300, +400 cumulated
    assert obv(df).tolist() == [0.0, 200.0, -100.0, 300.0]


def test_roll_spread_is_zero_when_serial_cov_is_positive():
    # a strong trend has POSITIVE serial covariance -> Roll is undefined -> 0
    df = _df(np.linspace(100, 140, 120))
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
               lambda d: corwin_schultz(d, 21), obv):
        full = fn(df)
        trunc = fn(df.iloc[:201])
        a, b = full.iloc[:201].to_numpy(), trunc.to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        assert np.allclose(a[~both_nan], b[~both_nan], atol=1e-10)
