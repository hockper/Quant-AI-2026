import numpy as np
import torch

from bubble_bi.data.windows import DayDataset, build_day_loaders


def test_block_contents_and_mask():
    T, N, D, L = 12, 3, 2, 5
    feats = np.arange(T * N * D, dtype=np.float32).reshape(T, N, D)
    mask = np.ones((T, N), dtype=bool)
    mask[6, 1] = False
    ds = DayDataset(feats, mask, window_len=L, day_range=(0, T), min_valid=2)
    # first valid day is t = L-1 = 4
    item = ds[0]                                  # t = 4
    assert item["block"].shape == (N, L, D)
    assert item["valid"].tolist() == [True, True, True]
    # block for stock 0 at t=4 is rows 0..4
    assert torch.allclose(item["block"][0], torch.tensor(feats[0:5, 0, :]))


def test_days_with_too_few_valid_skipped():
    T, N, D, L = 8, 3, 1, 4
    feats = np.zeros((T, N, D), dtype=np.float32)
    mask = np.ones((T, N), dtype=bool)
    mask[:, 1] = False
    mask[:, 2] = False
    ds = DayDataset(feats, mask, window_len=L, day_range=(0, T), min_valid=2)
    assert len(ds) == 0


def test_build_day_loaders_window_len():
    from bubble_bi.config import Config, DataConfig, FeatureConfig, ModelConfig
    from bubble_bi.data.panel import build_panel
    import pandas as pd

    per = {}
    rng = np.random.default_rng(0)
    for k in range(5):
        dates = pd.bdate_range("2015-01-01", periods=300)
        c = pd.Series(100 + np.cumsum(rng.normal(size=300)), index=dates)
        v = rng.integers(1_000_000, 5_000_000, size=300).astype(float)
        per[f"T{k}"] = pd.DataFrame(
            {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": v}, index=dates
        )
    panel = build_panel(per, DataConfig(tickers=list(per), min_history=50), FeatureConfig())
    cfg = Config(data=DataConfig(tickers=list(per), min_history=50), model=ModelConfig())
    cfg.train.batch_size = 8
    loaders, std = build_day_loaders(panel, cfg, window_len=7)
    batch = next(iter(loaders["train"]))
    assert batch["block"].shape[1:] == (5, 7, panel.features.shape[2])
    assert std.mean is not None
