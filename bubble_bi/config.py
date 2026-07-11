from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class DataConfig:
    tickers: list[str]
    start: str | None = None
    end: str | None = None
    raw_dir: str = "artifacts/raw"
    cache_dir: str = "artifacts/cache"
    min_history: int = 252
    train_frac: float = 0.7
    val_frac: float = 0.15


@dataclass
class FeatureConfig:
    ma_windows: list[int] = field(default_factory=lambda: [5, 10, 20])
    rsi_window: int = 14
    vol_window: int = 20
    volume_window: int = 20
    frac_d: float = 0.45
    frac_thresh: float = 1e-3
    frac_max_lags: int = 200
    atr_window: int = 14
    hurst_window: int = 100
    entropy_window: int = 60
    entropy_bins: int = 10
    amihud_window: int = 21
    roll_window: int = 21
    cs_window: int = 21


@dataclass
class SplitConfig:
    train_days: int = 756
    val_days: int = 126
    test_days: int = 126
    step_days: int = 126


@dataclass
class ModelConfig:
    p: int = 4
    d_model: int = 128
    codebook_size: int = 512
    enc_layers: int = 3
    dec_layers: int = 2
    heads: int = 4
    ff: int = 256
    dropout: float = 0.1
    beta_commit: float = 0.25
    lambda_div: float = 0.1
    lambda_ortho: float = 0.1
    ema_decay: float = 0.99
    dead_code_reinit_every: int = 250
    cs_codebook_size: int = 512
    cs_p: int = 5
    fusion_codebook_size: int = 512
    fusion_layers: int = 2
    use_fusion: bool = True
    pred_window: int = 64
    pred_layers: int = 4
    n_kv_heads: int = 0
    rope_theta: float = 10000.0


@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    batch_size: int = 256
    max_steps: int = 2000
    val_every: int = 200
    ckpt_every: int = 200
    log_every: int = 10
    device: str = "auto"
    amp: bool = True
    num_workers: int = 0


@dataclass
class Config:
    data: DataConfig
    features: FeatureConfig = field(default_factory=FeatureConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 42


def load_config(path: str) -> Config:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    data = DataConfig(**raw.get("data", {"tickers": []}))
    features = FeatureConfig(**raw.get("features", {}))
    splits = SplitConfig(**raw.get("splits", {}))
    model = ModelConfig(**raw.get("model", {}))
    train = TrainConfig(**raw.get("train", {}))
    return Config(
        data=data, features=features, splits=splits,
        model=model, train=train, seed=raw.get("seed", 42),
    )
