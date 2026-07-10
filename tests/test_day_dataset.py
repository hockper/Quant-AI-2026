import numpy as np
import torch

from bubble_bi.data.windows import DayDataset, build_day_loaders


def test_day_windows_and_mask_and_cs_slice():
    T, N, D, p = 12, 3, 2, 4
    feats = np.arange(T * N * D, dtype=np.float32).reshape(T, N, D)
    mask = np.ones((T, N), dtype=bool)
    mask[6, 1] = False                       # stock 1 invalid on day 6
    ds = DayDataset(feats, mask, p=p, day_range=(0, T), min_valid=2)
    # days included are t=3..11; ds[3] -> t=6
    item = ds[3]
    assert item["windows"].shape == (N, p, D)
    assert item["valid"].tolist() == [True, False, True]
    assert torch.all(item["windows"][1] == 0)           # invalid stock zero-filled
    cs = item["windows"][:, -1, :]                       # CS input = last day of window
    assert torch.allclose(cs[0], torch.tensor(feats[6, 0, :]))
    assert torch.allclose(cs[2], torch.tensor(feats[6, 2, :]))


def test_days_with_too_few_valid_are_skipped():
    T, N, D, p = 8, 3, 1, 4
    feats = np.zeros((T, N, D), dtype=np.float32)
    mask = np.ones((T, N), dtype=bool)
    mask[:, 1] = False
    mask[:, 2] = False                       # only stock 0 valid -> <2 valid
    ds = DayDataset(feats, mask, p=p, day_range=(0, T), min_valid=2)
    assert len(ds) == 0


def test_build_day_loaders_batches_days():
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
    cfg = Config(data=DataConfig(tickers=list(per), min_history=50), model=ModelConfig(p=4))
    cfg.train.batch_size = 8
    loaders, std = build_day_loaders(panel, cfg)
    batch = next(iter(loaders["train"]))
    assert batch["windows"].shape[1:] == (5, 4, panel.features.shape[2])
    assert batch["valid"].shape[1] == 5
    assert std.mean is not None
