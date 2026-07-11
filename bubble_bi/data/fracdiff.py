from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def fracdiff_weights(d: float, thresh: float = 1e-3, max_lags: int = 200) -> np.ndarray:
    """Binomial weights for fractional differentiation (Lopez de Prado FFD).

    w_0 = 1 ;  w_k = -w_{k-1} * (d - k + 1) / k
    Truncated once |w_k| < thresh, and hard-capped at max_lags.
    """
    w = [1.0]
    for k in range(1, max_lags):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
    return np.asarray(w, dtype=float)


def frac_diff(series: pd.Series, d: float, thresh: float = 1e-3,
              max_lags: int = 200) -> pd.Series:
    """Causal fixed-width fractional differentiation: y_t = sum_k w_k * x_{t-k}."""
    w = fracdiff_weights(d, thresh, max_lags)
    L = len(w)
    x = series.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= L:
        # windows[i] = x[i : i+L]; the value at t = i+L-1 needs x[t], x[t-1], ..., x[t-L+1]
        windows = sliding_window_view(x, L)
        out[L - 1:] = windows @ w[::-1]
    return pd.Series(out, index=series.index)
