from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


class PriceSource(Protocol):
    def fetch(self, ticker: str, start: str | None, end: str | None) -> pd.DataFrame: ...


class YFinanceSource:
    def fetch(self, ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
        import yfinance as yf

        df = yf.download(
            ticker, start=start, end=end, auto_adjust=True, progress=False
        )
        if df.empty:
            return pd.DataFrame(columns=COLUMNS)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df


def ingest(
    tickers: list[str],
    source: PriceSource,
    raw_dir: str,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, str]:
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for ticker in tickers:
        path = Path(raw_dir) / f"{ticker}.parquet"
        existing = pd.read_parquet(path) if path.exists() else None
        fetch_start = start
        if existing is not None and len(existing):
            fetch_start = str(existing.index.max().date())
        new = source.fetch(ticker, fetch_start, end)
        if existing is not None and len(existing):
            combined = pd.concat([existing, new])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = new.sort_index()
        combined[COLUMNS].to_parquet(path)
        paths[ticker] = str(path)
    return paths
