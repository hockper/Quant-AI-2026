import json

import numpy as np
import pandas as pd

from bubble_bi.config import Config, DataConfig, FeatureConfig, ModelConfig, TrainConfig
from bubble_bi.cli import train_tokenizer


def _write_raw(raw, n=320, N=6):
    rng = np.random.default_rng(0)
    for k in range(N):
        dates = pd.bdate_range("2015-01-01", periods=n)
        c = pd.Series(100 + np.cumsum(rng.normal(size=n)), index=dates)
        v = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
        df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": v}, index=dates)
        df.index.name = "date"
        df.to_parquet(f"{raw}/T{k}.parquet")


def _cfg(tmp_path, max_steps):
    return Config(
        data=DataConfig(tickers=[f"T{k}" for k in range(6)], raw_dir=str(tmp_path / "raw"),
                        cache_dir=str(tmp_path / "cache"), min_history=50),
        features=FeatureConfig(),
        model=ModelConfig(p=4, d_model=16, codebook_size=16, enc_layers=1,
                          dec_layers=1, heads=2, ff=32, dropout=0.0),
        train=TrainConfig(max_steps=max_steps, batch_size=8, val_every=100,
                          ckpt_every=1, log_every=1, device="cpu", amp=False),
    )


def _train_steps_logged(run_dir):
    lines = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    return [r["step"] for r in lines if r["phase"] == "train"]


def test_resume_continues_instead_of_restarting(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "cache").mkdir()
    _write_raw(tmp_path / "raw")
    run = tmp_path / "cache" / "runs" / "a"

    m1 = train_tokenizer(_cfg(tmp_path, 4), run_name="a")
    assert m1["step"] == 4
    assert _train_steps_logged(run) == [1, 2, 3, 4]

    # resume with a bigger budget: must do ONLY steps 5 and 6 (not restart at 1)
    m2 = train_tokenizer(_cfg(tmp_path, 6), run_name="a", resume=True)
    assert m2["step"] == 6
    assert _train_steps_logged(run) == [1, 2, 3, 4, 5, 6]   # a restart would append 1..6 again
