from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _rs_for_scale(views: np.ndarray, s: int) -> np.ndarray:
    """Mean rescaled-range (R/S) at sub-scale s, for every window. -> [n_windows]

    Classical R/S: cumulative sum of each chunk's deviation from its own mean.
    """
    n_win, W = views.shape
    n_chunks = W // s
    seg = views[:, :n_chunks * s].reshape(n_win, n_chunks, s)
    mean = seg.mean(axis=2, keepdims=True)
    dev = np.cumsum(seg - mean, axis=2)
    R = dev.max(axis=2) - dev.min(axis=2)          # [n_win, n_chunks]
    S = seg.std(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(S > 0, R / S, np.nan)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(rs, axis=1)              # [n_win]


def hurst(close: pd.Series, window: int) -> pd.Series:
    """Rolling Hurst exponent via rescaled-range: slope of log(R/S) vs log(scale)."""
    r = np.log(close.to_numpy(dtype=float))
    r = np.diff(r, prepend=np.nan)                 # log-returns, NaN at index 0
    out = np.full(len(r), np.nan)
    scales = sorted({s for s in (window // 8, window // 4, window // 2, window) if s >= 8})
    if len(r) < window or len(scales) < 2:
        return pd.Series(out, index=close.index)

    views = sliding_window_view(r, window)         # [n_win, window]
    rs = np.vstack([_rs_for_scale(views, s) for s in scales])       # [n_scales, n_win]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rs = np.log(rs)
    log_n = np.log(np.asarray(scales, dtype=float))[:, None]        # [n_scales, 1]

    # least-squares slope per window, vectorised
    valid = np.isfinite(log_rs).all(axis=0)
    ln_c = log_n - log_n.mean()
    lr_c = log_rs - log_rs.mean(axis=0, keepdims=True)
    with np.errstate(invalid="ignore"):
        slope = (ln_c * lr_c).sum(axis=0) / (ln_c ** 2).sum()
    slope = np.where(valid, slope, np.nan)
    out[window - 1:] = slope
    return pd.Series(out, index=close.index)


def entropy(close: pd.Series, window: int, bins: int) -> pd.Series:
    """Rolling Shannon entropy of log-returns (histogram over the window's own range)."""
    r = np.log(close.to_numpy(dtype=float))
    r = np.diff(r, prepend=np.nan)
    out = np.full(len(r), np.nan)
    if len(r) < window:
        return pd.Series(out, index=close.index)

    views = sliding_window_view(r, window)
    for i in range(views.shape[0]):
        seg = views[i]
        if not np.isfinite(seg).all():
            continue
        lo, hi = seg.min(), seg.max()
        # epsilon (not exact equality) absorbs exp/log round-trip jitter (~1e-16);
        # real market log-returns are orders of magnitude larger than this.
        if hi - lo < 1e-12:                         # constant returns -> no surprise
            out[i + window - 1] = 0.0
            continue
        counts, _ = np.histogram(seg, bins=bins, range=(lo, hi))
        p = counts / counts.sum()
        p = p[p > 0]
        out[i + window - 1] = float(-(p * np.log(p)).sum())
    return pd.Series(out, index=close.index)
