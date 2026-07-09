from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bubble_bi.config import DataConfig, FeatureConfig
from bubble_bi.data.features import FEATURE_NAMES, compute_features


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    tickers: list[str]
    features: np.ndarray  # [T, N, D]
    target: np.ndarray  # [T, N]
    mask: np.ndarray  # [T, N] bool
    feature_names: list[str]


def build_panel(
    per_ticker: dict[str, pd.DataFrame],
    data_cfg: DataConfig,
    feat_cfg: FeatureConfig,
) -> Panel:
    feature_names = FEATURE_NAMES(feat_cfg)
    kept = {t: df for t, df in per_ticker.items() if len(df) >= data_cfg.min_history}
    tickers = sorted(kept)
    if not tickers:
        raise ValueError("no ticker meets min_history")

    all_dates = sorted(set().union(*[df.index for df in kept.values()]))
    dates = pd.DatetimeIndex(all_dates)
    T, N, D = len(dates), len(tickers), len(feature_names)

    features = np.full((T, N, D), np.nan, dtype=np.float32)
    target = np.full((T, N), np.nan, dtype=np.float32)

    for j, t in enumerate(tickers):
        df = kept[t].reindex(dates)
        feats = compute_features(df, feat_cfg)
        features[:, j, :] = feats.to_numpy(dtype=np.float32)
        close = df["close"].astype(float)
        target[:, j] = (close.shift(-1) / close - 1.0).to_numpy(dtype=np.float32)

    mask = np.isfinite(target) & np.isfinite(features).all(axis=2)
    return Panel(dates, tickers, features, target, mask, feature_names)


def save_panel(panel: Panel, path: str) -> None:
    np.savez_compressed(
        path,
        dates=panel.dates.astype("datetime64[ns]").to_numpy(),
        tickers=np.array(panel.tickers, dtype=object),
        features=panel.features,
        target=panel.target,
        mask=panel.mask,
        feature_names=np.array(panel.feature_names, dtype=object),
    )


def load_panel(path: str) -> Panel:
    z = np.load(path, allow_pickle=True)
    return Panel(
        dates=pd.DatetimeIndex(z["dates"]),
        tickers=list(z["tickers"]),
        features=z["features"],
        target=z["target"],
        mask=z["mask"],
        feature_names=list(z["feature_names"]),
    )
