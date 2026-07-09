from __future__ import annotations

import numpy as np
import pandas as pd

from bubble_bi.config import FeatureConfig


def FEATURE_NAMES(cfg: FeatureConfig) -> list[str]:
    names = ["log_return"]
    names += [f"sma_ratio_{w}" for w in cfg.ma_windows]
    names += ["rsi", "macd", "macd_signal", "macd_hist", "realized_vol", "volume_z"]
    return names


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    log_ret = np.log(close).diff()

    out = pd.DataFrame(index=df.index)
    out["log_return"] = log_ret
    for w in cfg.ma_windows:
        out[f"sma_ratio_{w}"] = close / close.rolling(w).mean() - 1.0

    out["rsi"] = _rsi(close, cfg.rsi_window)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal
    out["realized_vol"] = log_ret.rolling(cfg.vol_window).std()
    vmean = volume.rolling(cfg.volume_window).mean()
    vstd = volume.rolling(cfg.volume_window).std()
    out["volume_z"] = (volume - vmean) / vstd

    return out[FEATURE_NAMES(cfg)]
