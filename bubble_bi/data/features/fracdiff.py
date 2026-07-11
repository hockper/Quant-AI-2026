"""Fractional differencing — making a series stationary without erasing its memory.

A price series trends, which most models cannot cope with. The usual fix is to
take differences (today minus yesterday), but that throws away *all* memory of
where the price came from.

Fractional differencing takes a *partial* difference — 0.45 of one, say — which
removes the trend while keeping a long tail of memory. (López de Prado.)

Every value uses only past days. Nothing here looks forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# How much memory to keep (0 = keep everything and stay trending, 1 = plain difference).
D = 0.45
# Ignore weights smaller than this. Smaller = longer memory, but a longer warm-up.
THRESHOLD = 1e-3
# Never look back further than this, whatever the threshold says.
MAX_LAGS = 200


def weights(d: float = D, threshold: float = THRESHOLD, max_lags: int = MAX_LAGS) -> np.ndarray:
    """The weight given to each past day: w[0] is today, w[1] yesterday, and so on.

    They come from the binomial expansion of (1 - B)^d and decay slowly — which is
    exactly the point: a slow decay is a long memory.
    """
    w = [1.0]
    for k in range(1, max_lags):
        nxt = -w[-1] * (d - k + 1) / k
        if abs(nxt) < threshold:
            break
        w.append(nxt)
    return np.asarray(w, dtype=float)


def frac_diff(series: pd.Series, d: float = D, threshold: float = THRESHOLD,
              max_lags: int = MAX_LAGS) -> pd.Series:
    """Fractionally difference a series: y[t] = sum_k w[k] * x[t-k].

    The first few values are blank (NaN) — there is not enough history yet.

    ⚠️ The weights do NOT sum to zero, so a *level* survives this. Feed it something
    scale-free (a log price, a ratio), never a raw running total whose size differs
    wildly between companies — otherwise the output secretly encodes *which* company
    it is.
    """
    w = weights(d, threshold, max_lags)
    n = len(w)
    x = series.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        # window i is x[i .. i+n-1]; its result belongs to the LAST day in it,
        # and pairing it with reversed weights gives w[0]*today + w[1]*yesterday + ...
        out[n - 1:] = sliding_window_view(x, n) @ w[::-1]
    return pd.Series(out, index=series.index)
