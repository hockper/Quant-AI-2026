from __future__ import annotations

import numpy as np
import pandas as pd

_DEN = 3.0 - 2.0 * np.sqrt(2.0)      # Corwin-Schultz denominator


def obv(df: pd.DataFrame, window: int) -> pd.Series:
    """On-Balance Volume over VOLUME-NORMALISED flow.

    Raw share volume differs by orders of magnitude across tickers, and the
    truncated FFD weights do not sum to zero -- so a raw cumsum would leave a
    per-ticker level term that survives into obv_frac and acts as a ticker id.
    Dividing by the trailing mean volume makes the accumulated flow comparable
    across tickers.
    """
    sign = np.sign(df["close"].diff()).fillna(0.0)
    vol = df["volume"].astype(float)
    norm = vol / vol.rolling(window).mean()
    return (sign * norm).cumsum()


def amihud(df: pd.DataFrame, window: int) -> pd.Series:
    """Amihud illiquidity: |return| per dollar traded. log1p-compressed (extreme skew)."""
    r = np.log(df["close"]).diff().abs()
    dollar = (df["close"] * df["volume"]).replace(0.0, np.nan)
    illiq = (r / dollar).rolling(window).mean()
    return np.log1p(1e6 * illiq)


def roll_spread(df: pd.DataFrame, window: int) -> pd.Series:
    """Roll (1984) implied spread, relative (fraction-of-price): 2*sqrt(-Cov(dP_t, dP_{t-1})) / P.

    The estimator is UNDEFINED when the serial covariance is positive (common on
    trending daily data); we return 0 there, which is the standard convention.

    A near-linear price path (constant dP) has a true covariance of exactly 0,
    but float64 rolling covariance can land on a tiny negative value purely from
    cancellation error. `tol` is the expected magnitude of that rounding noise
    (machine epsilon scaled by the rolling second moment of dP and window size);
    covariances within it of 0 are treated as non-negative, i.e. spread 0.

    The covariance guard operates on the raw dollar price changes; only the
    final spread is scaled by price, to keep the estimator comparable across
    tickers/price levels.
    """
    dp = df["close"].diff()
    cov = dp.rolling(window).cov(dp.shift(1))
    neg_cov = (-cov).clip(lower=0.0)          # NaN-preserving; >=0 real cov -> 0
    tol = np.finfo(float).eps * window * dp.pow(2).rolling(window).mean()
    noise = (neg_cov > 0.0) & (neg_cov <= tol)
    neg_cov = neg_cov.mask(noise, 0.0)
    return 2.0 * np.sqrt(neg_cov) / df["close"]


def corwin_schultz(df: pd.DataFrame, window: int) -> pd.Series:
    """Corwin-Schultz (2012) bid-ask spread from high/low only. Negatives clamped to 0."""
    h, l = df["high"], df["low"]
    hl2 = np.log(h / l) ** 2
    beta = hl2 + hl2.shift(1)
    h2 = pd.concat([h, h.shift(1)], axis=1).max(axis=1)
    l2 = pd.concat([l, l.shift(1)], axis=1).min(axis=1)
    gamma = np.log(h2 / l2) ** 2

    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _DEN - np.sqrt(gamma / _DEN)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return spread.clip(lower=0.0).rolling(window).mean()
