"""Flow features — is money moving into this stock, or out of it?

Volume tells a different story than price: a rally on huge volume is a
different animal from the same rally on a quiet day. These three features
summarise that story: `volume_z` says whether today's volume is unusual,
`volume_frac` gives volume's own trend without over-differencing it away, and
`obv_frac` tracks the running tally of "was volume flowing in on up days or
down days" (On-Balance Volume), also de-trended.

Same causality rule as everywhere else: only `.rolling()`, `.diff()`, `.cumsum()`
and friends looking backwards from day t. No `shift(-1)`, no centred windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bubble_bi.data.features.fracdiff import frac_diff

# Trailing window used to judge "is today's volume unusual" and to normalise
# OBV's running total (see obv_frac below).
VOLUME_WINDOW = 20


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume, over volume NORMALISED by its own trailing mean.

    Classic OBV accumulates raw share volume: +volume on an up day, -volume on
    a down day, running total. That is a trap here. Share counts differ by
    orders of magnitude between companies (a thinly traded stock might turn
    over 1e5 shares a day, a mega-cap 1e9), so the raw cumulative sum lands at
    wildly different scales too — and frac_diff's weights do NOT sum to zero,
    so a chunk of that per-company *level* survives into the "differenced"
    output instead of cancelling out. In an earlier version of this project,
    that leak meant 56% of obv_frac's variance was just which ticker it was,
    not order flow. Dividing volume by its own trailing mean first puts every
    company on the same "how many times its usual volume" scale, so what
    survives the frac-diff is actual buy/sell pressure, not a company ID.
    """
    sign = np.sign(close.diff()).fillna(0.0)
    normalised = volume / volume.rolling(VOLUME_WINDOW).mean()
    return (sign * normalised).cumsum()


def build(df: pd.DataFrame, settings: dict) -> dict[str, pd.Series]:
    """df: ONE company, indexed by date, columns open/high/low/close/volume.
    Returns {feature_name: Series}. Every Series must share df's index."""
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    out: dict[str, pd.Series] = {}

    vmean = volume.rolling(VOLUME_WINDOW).mean()
    vstd = volume.rolling(VOLUME_WINDOW).std()
    out["volume_z"] = (volume - vmean) / vstd

    out["volume_frac"] = frac_diff(np.log1p(volume))
    out["obv_frac"] = frac_diff(_obv(close, volume))

    return out
