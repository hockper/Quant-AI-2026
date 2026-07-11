"""Getting the raw prices.

One table, indexed by (date, ticker), with the six numbers a trading day leaves
behind: open, high, low, close, volume — plus the answer we are trying to
predict.

Downloads are cached to disk per company, so re-running is cheap and works
offline. Re-running also refreshes the most recent days, in case the exchange
restated a price.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]


def _fetch(ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
    """Ask yfinance for one company's daily bars."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[OHLCV]
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def _cached(ticker: str, folder: Path, start: str | None, end: str | None) -> pd.DataFrame:
    """One company's history, from disk if we have it, from the web if not."""
    path = folder / f"{ticker}.parquet"
    have = pd.read_parquet(path) if path.exists() else None

    if have is not None and len(have):
        # Refetch from the last day we hold, not the day after: the ranges overlap
        # by one day, the de-duplication below absorbs it, and a restated price
        # overwrites the stale one.
        fresh = _fetch(ticker, str(have.index.max().date()), end)
        both = pd.concat([have, fresh])
        both = both[~both.index.duplicated(keep="last")].sort_index()
    else:
        both = _fetch(ticker, start, end).sort_index()

    both[OHLCV].to_parquet(path)

    # The cache may hold MORE history than was asked for (an earlier run wanted more).
    # Trim to the window actually requested, or narrowing `start` would silently do
    # nothing.
    if start:
        both = both[both.index >= pd.Timestamp(start)]
    if end:
        both = both[both.index <= pd.Timestamp(end)]
    return both[OHLCV]


def download(settings: dict, on_progress=None) -> pd.DataFrame:
    """Every company's daily prices, in one table.

    Returns a DataFrame indexed by (date, ticker) with columns:
        open, high, low, close, volume   -- what happened that day
        target                           -- tomorrow's return

    `target` is the answer we are trying to predict. It is the ONLY column that
    looks forward, and it is never fed to a model as an input.
    """
    folder = Path(settings["data_dir"]) / "raw"
    folder.mkdir(parents=True, exist_ok=True)

    frames = []
    tickers = settings["tickers"]
    for i, ticker in enumerate(tickers, 1):
        df = _cached(ticker, folder, settings["start"], settings["end"])
        if df.empty:
            continue
        df = df.copy()
        # Tomorrow's return. shift(-1) reaches into the future ON PURPOSE -- this is
        # the answer, not a clue. Nothing else in the project is allowed to do this.
        df["target"] = df["close"].shift(-1) / df["close"] - 1.0
        df["ticker"] = ticker
        frames.append(df.reset_index())
        if on_progress:
            on_progress(i, len(tickers), ticker)

    if not frames:
        raise RuntimeError(
            "Nothing downloaded. Check the ticker symbols and your internet connection."
        )

    out = pd.concat(frames, ignore_index=True)
    out = out.set_index(["date", "ticker"]).sort_index()
    return out[OHLCV + ["target"]]
