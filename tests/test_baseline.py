import numpy as np
import pandas as pd

from bubble_bi.config import DataConfig, FeatureConfig, SplitConfig
from bubble_bi.data.panel import build_panel
from bubble_bi.data.splits import walk_forward_splits
from bubble_bi.baselines.ridge import predict_test_ridge, evaluate_baseline


def _panel_with_signal(n=400, N=12, seed=0):
    rng = np.random.default_rng(seed)
    per = {}
    for k in range(N):
        dates = pd.bdate_range("2015-01-01", periods=n)
        c = pd.Series(100 + np.cumsum(rng.normal(size=n)), index=dates)
        v = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
        per[f"T{k:02d}"] = pd.DataFrame(
            {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": v},
            index=dates,
        )
    return build_panel(per, DataConfig(tickers=list(per), min_history=50), FeatureConfig())


def test_prediction_shape_matches_test_window():
    panel = _panel_with_signal()
    cfg = SplitConfig(train_days=200, val_days=40, test_days=40, step_days=40)
    split = walk_forward_splits(len(panel.dates), cfg)[0]
    preds = predict_test_ridge(panel, split, alpha=1.0)
    assert preds.shape == (40, len(panel.tickers))


def test_evaluate_baseline_returns_finite_metrics():
    panel = _panel_with_signal()
    cfg = SplitConfig(train_days=200, val_days=40, test_days=40, step_days=40)
    splits = walk_forward_splits(len(panel.dates), cfg)
    result = evaluate_baseline(panel, splits, alpha=1.0)
    assert result["n_splits"] == len(splits)
    assert np.isfinite(result["rank_ic"])
