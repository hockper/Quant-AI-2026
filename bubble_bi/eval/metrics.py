from __future__ import annotations

import numpy as np


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = x.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def daily_rank_ic(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    T = pred.shape[0]
    out = np.full(T, np.nan)
    for t in range(T):
        m = mask[t] & np.isfinite(pred[t]) & np.isfinite(target[t])
        if m.sum() < 2:
            continue
        out[t] = _corr(_rankdata(pred[t][m]), _rankdata(target[t][m]))
    return out


def rank_ic(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    daily = daily_rank_ic(pred, target, mask)
    return float(np.nanmean(daily))


def rank_icir(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    daily = daily_rank_ic(pred, target, mask)
    sd = np.nanstd(daily)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return float(np.nanmean(daily) / sd)


def information_coefficient(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    T = pred.shape[0]
    vals = np.full(T, np.nan)
    for t in range(T):
        m = mask[t] & np.isfinite(pred[t]) & np.isfinite(target[t])
        if m.sum() < 2:
            continue
        vals[t] = _corr(pred[t][m], target[t][m])
    return float(np.nanmean(vals))
