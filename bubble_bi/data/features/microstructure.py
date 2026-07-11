"""Microstructure features — the hidden friction of trading a stock.

The price on your screen is not the price you pay. Between deciding to trade
and actually owning the shares there is a toll: the bid-ask spread (buyers and
sellers quote slightly different prices, and you cross that gap in both
directions), and market impact (a big order moves the price against you before
it is fully filled). None of that shows up in the closing price — it only
shows up in how *expensive* it was to act on it. These three estimators try to
recover that hidden cost from data that is otherwise perfectly ordinary
open/high/low/close/volume history.

    amihud          |return| per dollar traded         a proxy for market impact
    roll_spread     bid-ask bounce in the price itself   an implied spread, in %
    corwin_schultz  the day's high/low range             another implied spread, in %

Every number here only ever looks backwards from day t. That is not a style
preference: a trading model that peeks at tomorrow's high or low will look
perfect in backtests and lose money live, because tomorrow's candle is
exactly what it is being asked to predict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Shared rolling window for all three estimators.
WINDOW = 21

# Corwin-Schultz denominator, 3 - 2*sqrt(2), used twice in the alpha formula.
_CS_DEN = 3.0 - 2.0 * np.sqrt(2.0)


def build(df: pd.DataFrame, settings: dict) -> dict[str, pd.Series]:
    """df: ONE company, indexed by date, columns open/high/low/close/volume.
    Returns {feature_name: Series}. Every Series must share df's index."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    out: dict[str, pd.Series] = {}

    # --- amihud: illiquidity, |return| absorbed per dollar traded ------------
    log_ret_abs = np.log(close).diff().abs()
    dollar_volume = (close * volume).replace(0.0, np.nan)  # zero-volume days -> NaN, not a crash
    illiq = (log_ret_abs / dollar_volume).rolling(WINDOW).mean()
    out["amihud"] = np.log1p(1e6 * illiq)  # log1p tames the extreme right skew

    # --- roll_spread: Roll (1984) implied spread from the bid-ask bounce ----
    # A dealer quotes a bid below and an ask above the "true" price, so a trade
    # that alternately hits the bid then the ask makes consecutive price
    # changes bounce against each other -- a NEGATIVE serial covariance. Roll's
    # estimator turns that covariance back into an implied spread.
    dp = close.diff()
    cov = dp.rolling(WINDOW).cov(dp.shift(1))
    neg_cov = (-cov).clip(lower=0.0)  # NaN-preserving; cov >= 0 (no bounce signal) -> spread 0

    # Guard 1: the estimator is UNDEFINED when the serial covariance is >= 0
    # (any trending stock has one on daily data) -- the clip above already
    # sends that case to exactly 0, which is the standard convention.
    #
    # Guard 2: a near-linear price path has a TRUE covariance of exactly 0, but
    # float64 cancellation in rolling().cov() can still land on something like
    # -1e-17, which would otherwise produce a spurious tiny spread. `tol` is
    # the expected size of that rounding noise -- machine epsilon scaled by the
    # window and by the typical size of dp^2 -- so it is RELATIVE to the data
    # and can never mask a real spread, only float noise.
    tol = np.finfo(float).eps * WINDOW * dp.pow(2).rolling(WINDOW).mean()
    float_noise = (neg_cov > 0.0) & (neg_cov <= tol)
    neg_cov = neg_cov.mask(float_noise, 0.0)

    # Divide by close: the raw estimator is in DOLLARS, so a $20 stock and a
    # $500 stock get different numbers for the same *relative* trading cost,
    # and the column drifts upward as prices simply grow over time. Dividing
    # by price turns it into a fraction of price, comparable across companies
    # and across history.
    out["roll_spread"] = 2.0 * np.sqrt(neg_cov) / close

    # --- corwin_schultz: implied spread from the high/low range alone -------
    hl2 = np.log(high / low) ** 2
    beta = hl2 + hl2.shift(1)
    high2 = pd.concat([high, high.shift(1)], axis=1).max(axis=1)
    low2 = pd.concat([low, low.shift(1)], axis=1).min(axis=1)
    gamma = np.log(high2 / low2) ** 2

    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_DEN - np.sqrt(gamma / _CS_DEN)
    daily_spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    # Guard 3: clamp negative daily estimates to 0 BEFORE averaging, not after.
    # A negative "spread" is meaningless on its own, but averaging the raw
    # (signed) values first and clamping the mean gives a different, wrong
    # number -- the negative days should not get to drag the average down at
    # all, since they carry no real information.
    daily_spread = daily_spread.clip(lower=0.0)
    out["corwin_schultz"] = daily_spread.rolling(WINDOW).mean()

    return out
