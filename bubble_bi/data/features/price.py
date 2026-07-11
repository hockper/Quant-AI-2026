"""Price features — has this stock been going up, and is it overheating?

Eight numbers computed from the close price alone: the day's return, three
"how far above/below its recent average is it" ratios, one overbought/oversold
gauge (RSI), and a trend-vs-momentum triple (MACD). These are the bread-and-
butter technical-analysis signals — nothing exotic, but they are the cheapest
way to tell a model "this stock has been rising" versus "this stock has been
falling", which is most of what price history has to say.

Every number here only ever looks backwards from day t. That is not a style
preference: a trading model that peeks at tomorrow's close will look perfect
in backtests and lose money live, because tomorrow's close is exactly what it
is being asked to predict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# How many days back each moving-average ratio looks.
SMA_WINDOWS = (5, 10, 20)
# Wilder's original RSI window — the industry-standard choice, not tuned here.
RSI_WINDOW = 14
# MACD's three spans: fast EMA, slow EMA, and the signal line smoothing the gap.
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def _rsi(close: pd.Series, window: int) -> pd.Series:
    """Wilder's Relative Strength Index: 100 when every recent move was up, 0 when
    every recent move was down, ~50 when gains and losses have been balanced."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder's smoothing is an EWM with alpha = 1/window; adjust=False keeps it a
    # simple running update (today's average = a blend of yesterday's average and
    # today's move), which is what makes it causal.
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


def build(df: pd.DataFrame, settings: dict) -> dict[str, pd.Series]:
    """df: ONE company, indexed by date, columns open/high/low/close/volume.
    Returns {feature_name: Series}. Every Series must share df's index."""
    close = df["close"].astype(float)

    out: dict[str, pd.Series] = {}
    out["log_return"] = np.log(close).diff()

    for w in SMA_WINDOWS:
        # How far today's close sits above (positive) or below (negative) its own
        # trailing average, as a fraction — 0.05 means "5% above its W-day mean".
        out[f"sma_ratio_{w}"] = close / close.rolling(w).mean() - 1.0

    out["rsi"] = _rsi(close, RSI_WINDOW)

    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal  # the gap driving MACD crossover signals

    return out
