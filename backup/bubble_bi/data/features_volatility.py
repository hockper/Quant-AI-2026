from __future__ import annotations

import numpy as np
import pandas as pd

_LN2 = np.log(2.0)


def parkinson(df: pd.DataFrame, window: int) -> pd.Series:
    hl2 = np.log(df["high"] / df["low"]) ** 2
    var = hl2.rolling(window).mean() / (4.0 * _LN2)
    return np.sqrt(var.clip(lower=0.0))


def garman_klass(df: pd.DataFrame, window: int) -> pd.Series:
    hl2 = np.log(df["high"] / df["low"]) ** 2
    co2 = np.log(df["close"] / df["open"]) ** 2
    var = (0.5 * hl2 - (2.0 * _LN2 - 1.0) * co2).rolling(window).mean()
    return np.sqrt(var.clip(lower=0.0))          # the -(2ln2-1) term can go negative


def rogers_satchell(df: pd.DataFrame) -> pd.Series:
    h, l, c, o = df["high"], df["low"], df["close"], df["open"]
    return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)


def yang_zhang(df: pd.DataFrame, window: int) -> pd.Series:
    o, c = df["open"], df["close"]
    log_overnight = np.log(o / c.shift(1))       # close -> next open (the gap)
    log_open_close = np.log(c / o)               # intraday
    sigma_o = log_overnight.rolling(window).var()
    sigma_c = log_open_close.rolling(window).var()
    sigma_rs = rogers_satchell(df).rolling(window).mean()
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    var = sigma_o + k * sigma_c + (1.0 - k) * sigma_rs
    return np.sqrt(var.clip(lower=0.0))


def atr(df: pd.DataFrame, window: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
