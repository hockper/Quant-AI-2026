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


@dataclass
class FeatureConfig:
    ma_windows: list[int] = field(default_factory=lambda: [5, 10, 20])
    rsi_window: int = 14
    vol_window: int = 20
    volume_window: int = 20


@dataclass
class SplitConfig:
    train_days: int = 756
    val_days: int = 126
    test_days: int = 126
    step_days: int = 126


@dataclass
class Config:
    data: DataConfig
    features: FeatureConfig = field(default_factory=FeatureConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    seed: int = 42


def load_config(path: str) -> Config:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    data = DataConfig(**raw.get("data", {"tickers": []}))
    features = FeatureConfig(**raw.get("features", {}))
    splits = SplitConfig(**raw.get("splits", {}))
    return Config(data=data, features=features, splits=splits, seed=raw.get("seed", 42))
