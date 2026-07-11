import numpy as np
import pandas as pd

from bubble_bi.config import Config, DataConfig, FeatureConfig, SplitConfig
from bubble_bi.cli import build_panel_from_raw, run_baseline


def _write_raw(raw_dir, n=400, N=12, seed=0):
    rng = np.random.default_rng(seed)
    for k in range(N):
        dates = pd.bdate_range("2015-01-01", periods=n)
        c = pd.Series(100 + np.cumsum(rng.normal(size=n)), index=dates)
        v = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
        df = pd.DataFrame(
            {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": v},
            index=dates,
        )
        df.index.name = "date"
        df.to_parquet(f"{raw_dir}/T{k:02d}.parquet")


def test_end_to_end_pipeline_produces_finite_rank_ic(tmp_path):
    raw = tmp_path / "raw"
    cache = tmp_path / "cache"
    raw.mkdir()
    cache.mkdir()
    _write_raw(str(raw))
    cfg = Config(
        data=DataConfig(
            tickers=[f"T{k:02d}" for k in range(12)],
            raw_dir=str(raw),
            cache_dir=str(cache),
            min_history=50,
        ),
        features=FeatureConfig(),
        splits=SplitConfig(train_days=200, val_days=40, test_days=40, step_days=40),
    )
    panel = build_panel_from_raw(cfg)
    assert (cache / "panel.npz").exists()
    assert panel.features.shape[1] == 12
    result = run_baseline(cfg)
    assert result["n_splits"] >= 1
    assert np.isfinite(result["rank_ic"])
