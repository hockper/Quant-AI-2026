from __future__ import annotations

from pathlib import Path

from bubble_bi.config import Config, load_config


def load_colab_config(config_path: str, drive_root: str, **overrides) -> Config:
    """Load a config, point its paths at Google Drive, and apply overrides.

    Keeps a single set of YAML configs — the notebook overrides in memory rather
    than maintaining parallel `colab_*.yaml` files that drift out of sync.
    """
    cfg = load_config(config_path)
    root = Path(drive_root)
    cfg.data.raw_dir = str(root / "artifacts" / "raw")
    cfg.data.cache_dir = str(root / "artifacts" / "cache")
    cfg.train.device = "auto"

    for key, value in overrides.items():
        for section in (cfg.train, cfg.model, cfg.data, cfg.features, cfg.splits):
            if hasattr(section, key):
                setattr(section, key, value)
                break
        else:
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                raise AttributeError(f"unknown config field: {key!r}")
    return cfg
