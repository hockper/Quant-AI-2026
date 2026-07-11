import numpy as np
import pandas as pd
import pytest

import bubble_bi as bb
from bubble_bi.data import prices as price_source


@pytest.fixture
def fake_market(tmp_path, monkeypatch):
    """A fake yfinance, so no test ever touches the network."""
    def _fetch(ticker, start, end):
        days = pd.date_range("2010-01-04", periods=500, freq="B", name="date")
        rng = np.random.default_rng(abs(hash(ticker)) % 1000)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
        df = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": rng.integers(1e6, 5e6, len(days)).astype(float),
            },
            index=days,
        )
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        return df

    monkeypatch.setattr(price_source, "_fetch", _fetch)
    return str(tmp_path)


def test_one_row_per_company_per_day(fake_market):
    s = bb.check({"tickers": ["AAA", "BBB"], "data_dir": fake_market})
    table = bb.data.download(s)
    assert table.index.names == ["date", "ticker"]
    assert set(table.columns) == {"open", "high", "low", "close", "volume", "target"}
    assert table.index.get_level_values("ticker").nunique() == 2
    assert not table.index.duplicated().any()


def test_target_is_tomorrows_return_not_todays(fake_market):
    s = bb.check({"tickers": ["AAA"], "data_dir": fake_market})
    table = bb.data.download(s).droplevel("ticker")
    close = table["close"]
    expected = close.shift(-1) / close - 1.0          # tomorrow / today - 1
    assert np.allclose(table["target"].dropna(), expected.dropna())
    # the last day has no tomorrow, so it must be blank -- not silently zero
    assert np.isnan(table["target"].iloc[-1])


def test_narrowing_the_start_date_actually_narrows_the_data(fake_market):
    # The cache can hold MORE history than you asked for. Asking for less must give
    # you less -- otherwise a user who narrows the window silently gets the old one.
    wide = bb.data.download(bb.check({"tickers": ["AAA"], "data_dir": fake_market}))
    narrow = bb.data.download(
        bb.check({"tickers": ["AAA"], "start": "2011-01-01", "data_dir": fake_market})
    )
    assert wide.index.get_level_values("date").min() < pd.Timestamp("2011-01-01")
    assert narrow.index.get_level_values("date").min() >= pd.Timestamp("2011-01-01")
    assert len(narrow) < len(wide)


def test_rerunning_is_cached_and_gives_the_same_table(fake_market):
    s = bb.check({"tickers": ["AAA"], "data_dir": fake_market})
    first = bb.data.download(s)
    second = bb.data.download(s)          # now served from the parquet cache
    pd.testing.assert_frame_equal(first, second)


def test_the_leak_detector_actually_catches_a_leak(fake_market, monkeypatch):
    """A check that cannot fail is worse than no check. Plant a leak; it must fire."""
    from bubble_bi.data.features import price

    s = bb.check({"tickers": ["AAA", "BBB"], "data_dir": fake_market})
    table = bb.data.download(s)
    assert bb.data.find_leaks(table, s) == []          # honest features: clean

    real = price.build

    def cheating(df, settings):
        out = real(df, settings)
        out["rsi"] = out["rsi"].shift(-1)              # tomorrow's RSI, today
        return out

    monkeypatch.setattr(price, "build", cheating)
    assert bb.data.find_leaks(table, s) == ["rsi"]     # caught, and names the culprit
