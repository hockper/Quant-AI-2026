"""The 22 things we measure about a company each day.

Raw prices alone are a poor description of a trading day. These features re-describe
each day from five angles, so the tokenizer has something rich to compress:

    price          where it is going          (returns, moving averages, RSI, MACD)
    volatility     how violently it moves     (four estimators, from different parts
                                               of the candle)
    memory         whether the past echoes    (Hurst, fractional differencing, entropy)
    microstructure what it costs to trade     (illiquidity, bid-ask spread)
    flow           who is pushing, how hard   (volume dynamics, on-balance volume)

⚠️ Every one of them is BACKWARD-LOOKING. A feature on day t may use only days up to
and including t. The only column allowed to look forward is `target` — the answer —
and it is never fed to a model as an input.
"""

from __future__ import annotations

import pandas as pd

from bubble_bi.data.features import (
    flow,
    memory,
    microstructure,
    price,
    volatility,
)

# The five families, in the order their columns appear.
FAMILIES = {
    "price": price,
    "volatility": volatility,
    "memory": memory,
    "microstructure": microstructure,
    "flow": flow,
}


def _for_one_company(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Every feature, for a single company's history."""
    columns: dict[str, pd.Series] = {}
    for family in FAMILIES.values():
        columns.update(family.build(df, settings))
    return pd.DataFrame(columns, index=df.index)


def names() -> list[str]:
    """The feature column names, in order. Cheap — runs on a throwaway series."""
    import numpy as np

    n = 400
    fake = pd.DataFrame(
        {
            "open": np.linspace(100, 110, n),
            "high": np.linspace(101, 111, n),
            "low": np.linspace(99, 109, n),
            "close": np.linspace(100, 110, n),
            "volume": np.full(n, 1e6),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    return list(_for_one_company(fake, {}).columns)


def add_features(prices: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Add every feature to the raw price table.

    Takes the table from `download()` — indexed by (date, ticker) — and returns it
    with the feature columns alongside. Features are computed one company at a time:
    a moving average must never bleed from one company into another.
    """
    out = []
    for ticker, one in prices.groupby(level="ticker", sort=False):
        one = one.droplevel("ticker").sort_index()          # a single company, by date
        feats = _for_one_company(one[["open", "high", "low", "close", "volume"]], settings)
        joined = pd.concat([one, feats], axis=1)
        joined["ticker"] = ticker
        out.append(joined.reset_index())

    table = pd.concat(out, ignore_index=True).set_index(["date", "ticker"]).sort_index()

    # target last: it is the answer, not a feature.
    ordered = [c for c in table.columns if c != "target"] + ["target"]
    return table[ordered]
