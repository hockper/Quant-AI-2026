"""Memory features — does this stock's price remember its own past?

Most price series look random day to day, but the *character* of that
randomness varies. Some stocks trend: a run of up-days tends to keep running
(the price "remembers" its direction). Others mean-revert: an up-day tends to
be followed by a down-day, as if the price is elastic and snaps back. And some
are genuinely closer to a coin flip, where yesterday tells you nothing about
today. "Memory", here, is a name for that character — how much the past shapes
what comes next, independent of whether the price is going up or down overall.

Three ways of looking at memory:

- `hurst` asks, at many different time-scales inside a rolling window, "how
  much does this series wander compared to a pure random walk?" A Hurst
  exponent above 0.5 means trends tend to persist; below 0.5 means moves tend
  to reverse; right at 0.5 is the coin-flip case.
- `close_frac` is a fractionally-differenced log price: a version of the price
  that has had its trend largely removed (so models can treat it as
  stationary) while deliberately keeping a long, fading tail of memory of
  where the price has been. See `fracdiff.py` for the mechanics.
- `entropy` asks a more day-to-day question: over the last couple of months of
  returns, how disordered / surprising has the mix of moves been? A market
  bouncing between a small number of repeated moves is "orderly" (low
  entropy); one throwing out a wide, unpredictable spread of moves is "high
  entropy".

Every number here only ever looks backwards from day t — see the causality
test in tests/test_features_memory.py.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from bubble_bi.data.features.fracdiff import frac_diff

# Rolling window for the Hurst exponent. Inside it we look at several smaller
# sub-scales (see _rs_for_scale) to estimate how R/S grows with scale.
HURST_WINDOW = 100
# Rolling window for Shannon entropy of log-returns.
ENTROPY_WINDOW = 60
# Number of histogram bins entropy sorts each window's returns into.
ENTROPY_BINS = 10


def _rs_for_scale(views: np.ndarray, s: int) -> np.ndarray:
    """Mean rescaled-range (R/S) at sub-scale s, for every window at once. -> [n_windows]

    This is the classical R/S statistic: split each window into chunks of
    length s, and within EACH chunk subtract that chunk's own mean before
    accumulating (cumsum). That mean-subtraction is not optional — R/S is
    supposed to measure how a series wanders around its own local average,
    not how far a trend has carried it. Skip the mean-subtraction and R/S
    stops measuring memory and starts measuring drift: a plain deterministic
    uptrend (which has zero long-range dependence by construction) would
    score close to 1.0 instead of the ~0.5 a driftless walk gets.
    """
    n_win, w = views.shape
    n_chunks = w // s
    seg = views[:, : n_chunks * s].reshape(n_win, n_chunks, s)
    mean = seg.mean(axis=2, keepdims=True)
    dev = np.cumsum(seg - mean, axis=2)
    R = dev.max(axis=2) - dev.min(axis=2)  # [n_win, n_chunks]
    S = seg.std(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(S > 0, R / S, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(rs, axis=1)  # [n_win]


def _hurst(close: pd.Series, window: int) -> pd.Series:
    """Rolling Hurst exponent: the slope of log(R/S) against log(scale).

    Vectorised across every rolling window at once via sliding_window_view —
    `rolling().apply()` would take minutes over 30 companies x ~4,150 days.
    """
    scales = sorted({s for s in (window // 8, window // 4, window // 2, window) if s >= 8})
    if len(scales) < 2:
        raise ValueError(
            f"HURST_WINDOW={window} yields fewer than 2 usable R/S scales "
            f"(need window >= 16 so at least two of window//8, //4, //2, and "
            f"the full window are >= 8). A single all-NaN feature column "
            f"would silently invalidate the entire dataset."
        )

    log_price = np.log(close.to_numpy(dtype=float))
    r = np.diff(log_price, prepend=np.nan)  # log-returns; NaN at index 0
    out = np.full(len(r), np.nan)
    if len(r) < window:
        return pd.Series(out, index=close.index)

    # Each window of returns r[i .. i+window-1] produces one Hurst value that
    # belongs to the LAST day in the window — day i+window-1 — never earlier.
    views = sliding_window_view(r, window)  # [n_windows, window]
    rs = np.vstack([_rs_for_scale(views, s) for s in scales])  # [n_scales, n_windows]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rs = np.log(rs)
    log_scale = np.log(np.asarray(scales, dtype=float))[:, None]  # [n_scales, 1]

    # Least-squares slope of log(R/S) vs log(scale), done for every window at
    # once: slope = cov(x, y) / var(x) with x = log_scale (same for every
    # window) and y = log_rs (one column per window).
    # A plain (not nan-aware) mean here: every column we actually keep (see
    # `valid` below) is fully finite across scales, so an ordinary mean is
    # exact for it; a column with any NaN just becomes NaN throughout and is
    # masked out below, without np.nanmean's "empty slice" warning on the way.
    valid = np.isfinite(log_rs).all(axis=0)
    x_centered = log_scale - log_scale.mean()
    y_centered = log_rs - log_rs.mean(axis=0, keepdims=True)
    with np.errstate(invalid="ignore"):
        slope = (x_centered * y_centered).sum(axis=0) / (x_centered**2).sum()
    slope = np.where(valid, slope, np.nan)

    out[window - 1 :] = slope
    return pd.Series(out, index=close.index)


def _entropy(close: pd.Series, window: int, bins: int) -> pd.Series:
    """Rolling Shannon entropy of log-returns, histogrammed over each window's
    own min-to-max range. A simple Python loop over windows — slow compared to
    the vectorised Hurst above, but entropy only runs once per day, not once
    per (day, scale), so it stays fast enough."""
    log_price = np.log(close.to_numpy(dtype=float))
    r = np.diff(log_price, prepend=np.nan)
    out = np.full(len(r), np.nan)
    if len(r) < window:
        return pd.Series(out, index=close.index)

    views = sliding_window_view(r, window)
    for i in range(views.shape[0]):
        seg = views[i]
        if not np.isfinite(seg).all():
            continue
        lo, hi = seg.min(), seg.max()
        # A small epsilon, not exact equality, absorbs the ~1e-16 jitter that
        # a log-then-exp round trip leaves behind; real market log-returns
        # are many orders of magnitude larger than that jitter.
        if hi - lo < 1e-12:  # every return in the window is (numerically) the same
            out[i + window - 1] = 0.0  # no disorder -> no surprise -> zero entropy
            continue
        counts, _ = np.histogram(seg, bins=bins, range=(lo, hi))
        p = counts / counts.sum()
        p = p[p > 0]  # 0 * log(0) is defined as 0 in entropy; just drop empty bins
        out[i + window - 1] = float(-(p * np.log(p)).sum())
    return pd.Series(out, index=close.index)


def build(df: pd.DataFrame, settings: dict) -> dict[str, pd.Series]:
    """df: ONE company, indexed by date, columns open/high/low/close/volume.
    Returns {feature_name: Series}. Every Series must share df's index."""
    close = df["close"].astype(float)

    out: dict[str, pd.Series] = {}
    out["hurst"] = _hurst(close, HURST_WINDOW)
    out["close_frac"] = frac_diff(np.log(close))
    out["entropy"] = _entropy(close, ENTROPY_WINDOW, ENTROPY_BINS)
    return out
