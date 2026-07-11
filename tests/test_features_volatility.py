import numpy as np
import pandas as pd

from bubble_bi.data.features_volatility import (atr, garman_klass, parkinson,
                                                rogers_satchell, yang_zhang)


def _ohlcv(n=200, seed=0):
    rng = np.random.default_rng(seed)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    return pd.DataFrame({
        "open": c.shift(1).fillna(c.iloc[0]),
        "high": c * 1.01,
        "low": c * 0.99,
        "close": c,
        "volume": rng.integers(1e6, 5e6, n).astype(float),
    })


def test_estimators_are_nonnegative_and_finite_after_warmup():
    df = _ohlcv()
    for f in (parkinson(df, 20), garman_klass(df, 20), yang_zhang(df, 20), atr(df, 14)):
        tail = f.iloc[50:]
        assert np.isfinite(tail).all()
        assert (tail >= 0).all()


def test_parkinson_matches_closed_form_for_constant_range():
    # high/low ratio is constant -> parkinson = |ln r| / (2*sqrt(ln 2))
    n = 60
    c = pd.Series(np.full(n, 100.0))
    r = 1.02
    df = pd.DataFrame({"open": c, "high": c * r, "low": c, "close": c,
                       "volume": np.ones(n)})
    expected = abs(np.log(r)) / (2 * np.sqrt(np.log(2)))
    assert np.isclose(parkinson(df, 20).iloc[-1], expected)


def test_rogers_satchell_is_zero_when_open_equals_close_equals_high_equals_low():
    c = pd.Series(np.full(10, 50.0))
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": np.ones(10)})
    assert np.allclose(rogers_satchell(df).to_numpy(), 0.0)


def test_rogers_satchell_matches_hand_computed_value_for_nondegenerate_bar():
    # O/H/L/C chosen so the two cross terms differ (0.00365 vs 0.00183) -- a
    # swapped pairing (H/O with L/C, etc.) or a sign flip would both change
    # the result, unlike the O=H=L=C degenerate test where every term is 0.
    o, h, l, c = 100.0, 108.0, 97.0, 103.0
    df = pd.DataFrame({"open": [o], "high": [h], "low": [l], "close": [c], "volume": [1.0]})
    expected = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    assert np.isclose(rogers_satchell(df).iloc[0], expected)
    assert np.isclose(expected, 0.005476226668581852)


def test_atr_is_causal():
    df = _ohlcv(n=200)
    full = atr(df, 14)
    trunc = atr(df.iloc[:121], 14)
    a, b = full.iloc[:121].to_numpy(), trunc.to_numpy()
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.allclose(a[~both_nan], b[~both_nan], atol=1e-10)
