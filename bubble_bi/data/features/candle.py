"""The shape of the day itself.

Every other feature in this project describes a *stretch* of days: a 20-day
volatility, a 100-day memory, a 21-day spread. Remarkably, none of them describe
**today**. Today's open is seen by nothing at all; today's high and low appear only
buried inside long averages. The model was being asked to recognise the market's
mood while blind to the shape of the very day it was describing — unable to tell a
violent reversal (huge wicks, tiny body) from a calm grind (barely any range).

These four numbers fix that. They are the raw candle, expressed as *ratios*:

    gap          where it opened, against yesterday's close
    body         what it actually did, open to close
    upper_wick   how far up it reached and gave back
    lower_wick   how far down

Why not simply feed the model the raw open/high/low/close? Because prices trend —
one company went from $6 to $200 over this history — and prices differ wildly
between companies. Handing those numbers to the model puts a drifting,
company-identifying level straight back into it, which is the single most expensive
bug in this project's history. Ratios have no level. They mean the same thing for a
$20 stock and a $500 stock, in 2010 and in 2026.

And they are **exactly invertible**: given yesterday's close, these four numbers
rebuild open, high, low and close to the cent. That is what lets us draw the
candle a token remembered, rather than a vague impression of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build(df: pd.DataFrame, settings: dict) -> dict[str, pd.Series]:
    """df: ONE company, indexed by date, columns open/high/low/close/volume.
    Returns {feature_name: Series}. Every Series must share df's index."""
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    yesterday = close.shift(1)          # the only backward reference we need
    top = np.maximum(open_, close)      # the body's upper edge
    bottom = np.minimum(open_, close)   # ...and its lower edge

    return {
        # Overnight: the market moved while nobody could trade.
        "gap": np.log(open_ / yesterday),
        # The body: what the session actually did. Positive = closed up on its open.
        "body": np.log(close / open_),
        # The wicks: ground taken and then given back. A long wick is a rejection --
        # buyers pushed up and were beaten back, or sellers pushed down and failed.
        # Always >= 0 by construction, since high >= max(open, close) >= min >= low.
        "upper_wick": np.log(high / top),
        "lower_wick": np.log(bottom / low),
    }


def rebuild_candles(shape: pd.DataFrame, first_close: float) -> pd.DataFrame:
    """Turn the four shape numbers back into open / high / low / close.

    The inverse of `build`. Starting from one known close, walk forward:

        open  = yesterday's close x exp(gap)
        close = open              x exp(body)
        high  = max(open, close)  x exp(upper_wick)
        low   = min(open, close)  / exp(lower_wick)

    This is how we draw the candle a single token remembered.
    """
    rows = []
    yesterday = float(first_close)
    for _, day in shape.iterrows():
        open_ = yesterday * np.exp(day["gap"])
        close = open_ * np.exp(day["body"])
        top, bottom = max(open_, close), min(open_, close)
        rows.append({
            "open": open_,
            "high": top * np.exp(day["upper_wick"]),
            "low": bottom / np.exp(day["lower_wick"]),
            "close": close,
        })
        yesterday = close
    return pd.DataFrame(rows, index=shape.index)
