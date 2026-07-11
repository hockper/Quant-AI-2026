"""Volatility features — how much is this stock actually moving?

There is more than one way to answer that, because a daily candle carries more
information than just its closing price. Each estimator below uses a
different slice of the candle (close-to-close, high/low, open/close) and is
progressively more "efficient" — it needs fewer days to pin down the true
volatility, because it throws away less of what happened during the day:

    realized_vol    close-to-close only          the textbook estimator
    parkinson       high/low range               ~5x more efficient, ignores drift
    garman_klass    high/low + open/close        adds the overnight/intraday split
    yang_zhang      + the previous close         also handles overnight jumps
    atr_frac        Wilder's average true range  a smoothed, trend-following measure

Every number here only ever looks backwards from day t. That is not a style
preference: a trading model that peeks at tomorrow's high or low will look
perfect in backtests and lose money live, because tomorrow's candle is
exactly what it is being asked to predict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bubble_bi.data.features.fracdiff import frac_diff

# Rolling window shared by the four variance-based estimators below.
WINDOW = 20
# Wilder's original ATR window — the industry-standard choice, not tuned here.
ATR_WINDOW = 14

_LN2 = np.log(2.0)


def _rogers_satchell(high: pd.Series, low: pd.Series, close: pd.Series, open_: pd.Series) -> pd.Series:
    """The within-day term shared by Yang-Zhang: captures range *and* drift,
    unlike Parkinson's range-only term."""
    return np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(low / open_)


def build(df: pd.DataFrame, settings: dict) -> dict[str, pd.Series]:
    """df: ONE company, indexed by date, columns open/high/low/close/volume.
    Returns {feature_name: Series}. Every Series must share df's index."""
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)  # yesterday's close — never today's or later

    out: dict[str, pd.Series] = {}

    # --- realized_vol: the plain close-to-close estimator -------------------
    log_ret = np.log(close).diff()
    out["realized_vol"] = log_ret.rolling(WINDOW).std()

    # --- parkinson: uses only the day's high/low range ----------------------
    hl2 = np.log(high / low) ** 2
    out["parkinson"] = np.sqrt(hl2.rolling(WINDOW).mean() / (4.0 * _LN2))

    # --- garman_klass: adds the open/close split on top of the range --------
    co2 = np.log(close / open_) ** 2
    gk_var = (0.5 * hl2 - (2.0 * _LN2 - 1.0) * co2).rolling(WINDOW).mean()
    # the -(2ln2-1) term can dominate and push the mean negative; clip before
    # sqrt or a run of quiet days silently turns into NaN instead of ~0.
    out["garman_klass"] = np.sqrt(gk_var.clip(lower=0.0))

    # --- yang_zhang: overnight jump + intraday move + Rogers-Satchell -------
    log_overnight = np.log(open_ / prev_close)          # yesterday's close -> today's open
    log_open_close = np.log(close / open_)               # today's open -> today's close
    sigma_overnight = log_overnight.rolling(WINDOW).var()
    sigma_open_close = log_open_close.rolling(WINDOW).var()
    sigma_rs = _rogers_satchell(high, low, close, open_).rolling(WINDOW).mean()
    k = 0.34 / (1.34 + (WINDOW + 1) / (WINDOW - 1))
    yz_var = sigma_overnight + k * sigma_open_close + (1.0 - k) * sigma_rs
    out["yang_zhang"] = np.sqrt(yz_var.clip(lower=0.0))  # same negative-variance risk as garman_klass

    # --- atr_frac: Wilder's ATR, made scale-free, then fractionally differenced
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / ATR_WINDOW, adjust=False, min_periods=ATR_WINDOW).mean()
    # Divide by close BEFORE frac-differencing: a raw ATR is in dollars, so a
    # $20 stock and a $500 stock get different numbers for the same *relative*
    # volatility, and the column drifts upward as prices grow over time. As a
    # fraction of price it is comparable across companies and across history.
    out["atr_frac"] = frac_diff(atr / close)

    return out
