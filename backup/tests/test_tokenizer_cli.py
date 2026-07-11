import numpy as np
import pandas as pd

from bubble_bi.config import Config, DataConfig, FeatureConfig, ModelConfig, TrainConfig
from bubble_bi.cli import train_tokenizer, eval_tokenizer


def _write_raw(raw, n=320, N=6):
    rng = np.random.default_rng(0)
    for k in range(N):
        dates = pd.bdate_range("2015-01-01", periods=n)
        c = pd.Series(100 + np.cumsum(rng.normal(size=n)), index=dates)
        v = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
        df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": v}, index=dates)
        df.index.name = "date"
        df.to_parquet(f"{raw}/T{k}.parquet")


def _cfg(tmp_path):
    return Config(
        data=DataConfig(tickers=[f"T{k}" for k in range(6)], raw_dir=str(tmp_path / "raw"),
                        cache_dir=str(tmp_path / "cache"), min_history=50),
        features=FeatureConfig(),
        model=ModelConfig(p=4, d_model=16, codebook_size=16, enc_layers=1,
                          dec_layers=1, heads=2, ff=32, dropout=0.0),
        train=TrainConfig(max_steps=10, batch_size=32, val_every=10, ckpt_every=10,
                          device="cpu", amp=False),
    )


def test_train_then_eval_tokenizer(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "cache").mkdir()
    _write_raw(tmp_path / "raw")
    cfg = _cfg(tmp_path)
    train_metrics = train_tokenizer(cfg)
    assert train_metrics["step"] == 10
    assert (tmp_path / "cache" / "checkpoints" / "last.pt").exists()
    ev = eval_tokenizer(cfg)
    assert np.isfinite(ev["recon_mse"])
    assert 0.0 <= ev["codes_used_frac"] <= 1.0
    assert "mean_baseline_mse" in ev
