import numpy as np
import torch

from bubble_bi.data.windows import WindowDataset, build_loaders


def test_window_contents_match_slice_and_skip_invalid():
    T, N, D, p = 10, 1, 2, 4
    feats = np.arange(T * N * D, dtype=np.float32).reshape(T, N, D)
    mask = np.ones((T, N), dtype=bool)
    mask[5, 0] = False                       # invalidate day 5
    ds = WindowDataset(feats, mask, p=p, day_range=(0, T))
    # last-day t needs days [t-3..t] all valid. t=5,6,7,8 touch day 5 -> excluded;
    # valid last days are t=3, t=4, t=9 -> 3 windows.
    assert len(ds) == 3
    sample = ds[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (p, D)
    # first sample is the window ending at t=3 = rows 0..3
    assert torch.allclose(sample, torch.tensor(feats[0:4, 0, :]))


def test_build_loaders_produces_three_splits():
    from bubble_bi.config import Config, DataConfig, FeatureConfig, ModelConfig
    from bubble_bi.data.panel import build_panel
    import pandas as pd

    per = {}
    rng = np.random.default_rng(0)
    for k in range(4):
        dates = pd.bdate_range("2015-01-01", periods=300)
        c = pd.Series(100 + np.cumsum(rng.normal(size=300)), index=dates)
        v = rng.integers(1_000_000, 5_000_000, size=300).astype(float)
        per[f"T{k}"] = pd.DataFrame(
            {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": v}, index=dates
        )
    panel = build_panel(per, DataConfig(tickers=list(per), min_history=50), FeatureConfig())
    cfg = Config(data=DataConfig(tickers=list(per), min_history=50), model=ModelConfig(p=4))
    loaders, std = build_loaders(panel, cfg)
    assert set(loaders) == {"train", "val", "test"}
    xb = next(iter(loaders["train"]))
    assert xb.shape[1:] == (4, panel.features.shape[2])
    assert std.mean is not None
