from __future__ import annotations

import numpy as np
import pandas as pd

from bubble_bi.config import FeatureConfig
from bubble_bi.data.fracdiff import frac_diff
from bubble_bi.data.features_memory import entropy, hurst
from bubble_bi.data.features_micro import amihud, corwin_schultz, obv, roll_spread
from bubble_bi.data.features_volatility import atr, garman_klass, parkinson, yang_zhang


def FEATURE_NAMES(cfg: FeatureConfig) -> list[str]:
    names = ["log_return"]
    names += [f"sma_ratio_{w}" for w in cfg.ma_windows]
    names += ["rsi", "macd", "macd_signal", "macd_hist", "realized_vol", "volume_z"]
    names += ["parkinson", "garman_klass", "yang_zhang", "atr_frac"]          # volatility
    names += ["hurst", "close_frac", "entropy"]                               # memory
    names += ["amihud", "roll_spread", "corwin_schultz"]                      # microstructure
    names += ["volume_frac", "obv_frac"]                                      # flow
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

    fd = dict(d=cfg.frac_d, thresh=cfg.frac_thresh, max_lags=cfg.frac_max_lags)

    # --- volatility --------------------------------------------------------
    out["parkinson"] = parkinson(df, cfg.vol_window)
    out["garman_klass"] = garman_klass(df, cfg.vol_window)
    out["yang_zhang"] = yang_zhang(df, cfg.vol_window)
    out["atr_frac"] = frac_diff(atr(df, cfg.atr_window), **fd)

    # --- memory ------------------------------------------------------------
    out["hurst"] = hurst(close, cfg.hurst_window)
    out["close_frac"] = frac_diff(np.log(close), **fd)
    out["entropy"] = entropy(close, cfg.entropy_window, cfg.entropy_bins)

    # --- microstructure ----------------------------------------------------
    out["amihud"] = amihud(df, cfg.amihud_window)
    out["roll_spread"] = roll_spread(df, cfg.roll_window)
    out["corwin_schultz"] = corwin_schultz(df, cfg.cs_window)

    # --- flow --------------------------------------------------------------
    out["volume_frac"] = frac_diff(np.log1p(volume), **fd)
    out["obv_frac"] = frac_diff(obv(df), **fd)

    return out[FEATURE_NAMES(cfg)]
