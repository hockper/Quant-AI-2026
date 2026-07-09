import numpy as np
import pandas as pd

from bubble_bi.data.ingest import ingest


class FakeSource:
    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def fetch(self, ticker, start, end):
        self.calls.append((ticker, start, end))
        df = self.frames[ticker]
        if start is not None:
            df = df[df.index > pd.Timestamp(start)]
        return df


def _frame(n=10, seed=0):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    c = pd.Series(100 + np.cumsum(rng.normal(size=n)), index=dates)
    return pd.DataFrame(
        {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1e6},
        index=dates,
    )


def test_ingest_writes_parquet(tmp_path):
    src = FakeSource({"AAPL": _frame()})
    paths = ingest(["AAPL"], src, str(tmp_path))
    got = pd.read_parquet(paths["AAPL"])
    assert list(got.columns) == ["open", "high", "low", "close", "volume"]
    assert len(got) == 10


def test_ingest_is_incremental(tmp_path):
    full = _frame(n=15)
    src = FakeSource({"AAPL": full.iloc[:10]})
    ingest(["AAPL"], src, str(tmp_path))
    # add later data and re-run; should only fetch the tail
    src.frames["AAPL"] = full
    ingest(["AAPL"], src, str(tmp_path))
    got = pd.read_parquet(next(iter((tmp_path).glob("AAPL.parquet"))))
    assert len(got) == 15
    # second call requested a start date (incremental), not a full refetch
    assert src.calls[-1][1] is not None
