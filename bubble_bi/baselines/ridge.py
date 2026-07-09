from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from bubble_bi.data.panel import Panel
from bubble_bi.data.splits import WalkForwardSplit
from bubble_bi.eval.metrics import rank_ic, rank_icir


def _flatten(panel: Panel, lo: int, hi: int):
    T, N, D = panel.features.shape
    X = panel.features[lo:hi].reshape(-1, D)
    y = panel.target[lo:hi].reshape(-1)
    m = panel.mask[lo:hi].reshape(-1) & np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X, y, m


def predict_test_ridge(panel: Panel, split: WalkForwardSplit, alpha: float = 1.0) -> np.ndarray:
    lo_tr, hi_tr = split.train
    Xtr, ytr, mtr = _flatten(panel, lo_tr, hi_tr)
    scaler = StandardScaler().fit(Xtr[mtr])
    model = Ridge(alpha=alpha).fit(scaler.transform(Xtr[mtr]), ytr[mtr])

    lo_te, hi_te = split.test
    N = panel.features.shape[1]
    D = panel.features.shape[2]
    Xte = panel.features[lo_te:hi_te].reshape(-1, D)
    valid = np.isfinite(Xte).all(axis=1)
    preds = np.full(Xte.shape[0], np.nan)
    if valid.any():
        preds[valid] = model.predict(scaler.transform(Xte[valid]))
    return preds.reshape(hi_te - lo_te, N)


def evaluate_baseline(panel: Panel, splits: list[WalkForwardSplit], alpha: float = 1.0) -> dict:
    preds_all, target_all, mask_all = [], [], []
    for split in splits:
        lo_te, hi_te = split.test
        preds_all.append(predict_test_ridge(panel, split, alpha))
        target_all.append(panel.target[lo_te:hi_te])
        mask_all.append(panel.mask[lo_te:hi_te])
    if not splits:
        return {"rank_ic": float("nan"), "rank_icir": float("nan"), "n_splits": 0}
    pred = np.concatenate(preds_all, axis=0)
    target = np.concatenate(target_all, axis=0)
    mask = np.concatenate(mask_all, axis=0)
    return {
        "rank_ic": rank_ic(pred, target, mask),
        "rank_icir": rank_icir(pred, target, mask),
        "n_splits": len(splits),
    }
