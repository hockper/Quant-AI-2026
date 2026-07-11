import numpy as np
import pandas as pd

from bubble_bi.data.features.candle import build, rebuild_candles


def _day(o, h, l, c, prev):
    idx = pd.date_range("2020-01-01", periods=2, freq="B")
    return pd.DataFrame({"open": [prev, o], "high": [prev, h], "low": [prev, l],
                         "close": [prev, c], "volume": [1e6, 1e6]}, index=idx)


def test_the_four_numbers_describe_the_candle_we_expect():
    df = _day(o=102, h=106, l=101, c=104, prev=100)
    got = build(df, {})
    assert np.isclose(got["gap"].iloc[1], np.log(102 / 100))        # opened up 2%
    assert np.isclose(got["body"].iloc[1], np.log(104 / 102))       # then rose to 104
    assert np.isclose(got["upper_wick"].iloc[1], np.log(106 / 104)) # poked to 106
    assert np.isclose(got["lower_wick"].iloc[1], np.log(102 / 101)) # dipped to 101


def test_wicks_are_never_negative():
    rng = np.random.default_rng(0)
    n = 300
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = close * (1 + rng.normal(0, 0.01, n))
    top = np.maximum(open_, close)
    bottom = np.minimum(open_, close)
    df = pd.DataFrame(
        {"open": open_, "high": top * (1 + rng.uniform(0, 0.02, n)),
         "low": bottom * (1 - rng.uniform(0, 0.02, n)), "close": close,
         "volume": np.full(n, 1e6)},
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    got = build(df, {})
    assert (got["upper_wick"].dropna() >= -1e-12).all()
    assert (got["lower_wick"].dropna() >= -1e-12).all()


def test_the_candle_can_be_rebuilt_exactly_from_the_four_numbers():
    """This is what makes the reconstruction plot a real reconstruction."""
    rng = np.random.default_rng(1)
    n = 60
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = close * (1 + rng.normal(0, 0.01, n))
    top, bottom = np.maximum(open_, close), np.minimum(open_, close)
    real = pd.DataFrame(
        {"open": open_, "high": top * (1 + rng.uniform(0, 0.03, n)),
         "low": bottom * (1 - rng.uniform(0, 0.03, n)), "close": close,
         "volume": np.full(n, 1e6)},
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    shape = pd.DataFrame(build(real, {})).iloc[1:]        # day 0 has no yesterday
    back = rebuild_candles(shape, first_close=real["close"].iloc[0])

    for column in ("open", "high", "low", "close"):
        assert np.allclose(back[column], real[column].iloc[1:], rtol=1e-9)


def test_a_scale_free_candle_means_the_same_thing_at_any_price():
    # A $20 stock and a $500 stock with the SAME shaped day give the SAME numbers.
    cheap = _day(o=20.4, h=21.2, l=20.2, c=20.8, prev=20)
    dear = _day(o=510, h=530, l=505, c=520, prev=500)
    a, b = build(cheap, {}), build(dear, {})
    for name in ("gap", "body", "upper_wick", "lower_wick"):
        assert np.isclose(a[name].iloc[1], b[name].iloc[1], atol=1e-12)


def test_it_never_looks_at_the_future():
    rng = np.random.default_rng(2)
    n = 200
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    df = pd.DataFrame(
        {"open": close * 0.99, "high": close * 1.02, "low": close * 0.98,
         "close": close, "volume": np.full(n, 1e6)},
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    full = pd.DataFrame(build(df, {}))
    cut = pd.DataFrame(build(df.iloc[:120], {}))
    a, b = full.iloc[:120].to_numpy(), cut.to_numpy()
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.allclose(a[~both_nan], b[~both_nan], atol=1e-12)
