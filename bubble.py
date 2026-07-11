"""Bubble Bi — the friendly layer the notebook talks to.

Everything here is a thin wrapper over the tested engine in `backup/bubble_bi`.
The notebook should never need to import from the engine directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The engine lives in backup/. Make it importable without the notebook knowing.
_ENGINE = Path(__file__).resolve().parent / "backup"
if str(_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ENGINE))

from bubble_bi.config import (  # noqa: E402
    Config,
    DataConfig,
    FeatureConfig,
    ModelConfig,
    SplitConfig,
    TrainConfig,
)
from bubble_bi.runtime import detect_runtime  # noqa: E402


def make_config(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    *,
    # the tokenizer — how each stock-day becomes one "word"
    vocabulary: int = 512,
    days_per_token: int = 4,
    market_days: int = 5,
    model_size: int = 128,
    # the predictor — the GPT that reads the sentences
    sentence_length: int = 64,
    predictor_layers: int = 4,
    # training
    steps: int = 2000,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    seed: int = 42,
    device: str = "auto",
    data_dir: str = "artifacts",
) -> Config:
    """Build the project's configuration from plain notebook values.

    Friendly names in, the engine's names out. Anything not listed here keeps a
    sensible default — you should not have to care about it.
    """
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        raise ValueError("`tickers` is empty — list at least one company, e.g. ['AAPL'].")
    if days_per_token < 1:
        raise ValueError(f"`days_per_token` must be at least 1, got {days_per_token}.")
    if vocabulary < 2:
        raise ValueError(f"`vocabulary` must be at least 2, got {vocabulary}.")

    return Config(
        data=DataConfig(
            tickers=tickers,
            start=start,
            end=end,
            raw_dir=f"{data_dir}/raw",
            cache_dir=f"{data_dir}/cache",
        ),
        features=FeatureConfig(),
        splits=SplitConfig(),
        model=ModelConfig(
            p=days_per_token,
            cs_p=market_days,
            d_model=model_size,
            codebook_size=vocabulary,
            cs_codebook_size=vocabulary,
            fusion_codebook_size=vocabulary,
            pred_window=sentence_length,
            pred_layers=predictor_layers,
        ),
        train=TrainConfig(
            lr=learning_rate,
            batch_size=batch_size,
            max_steps=steps,
            device=device,
        ),
        seed=seed,
    )


def describe(cfg: Config) -> str:
    """A one-glance summary of what is about to run."""
    d, m, t = cfg.data, cfg.model, cfg.train
    where = detect_runtime() if t.device == "auto" else t.device
    period = f"from {d.start}" if d.start else "all available history"
    return (
        f"✅ Ready to run on {where.upper()}\n"
        f"   {len(d.tickers)} companies, {period}\n"
        f"   Vocabulary: {m.codebook_size} market 'words', "
        f"each summarising {m.p} days\n"
        f"   Predictor reads sentences of {m.pred_window} tokens\n"
        f"   Data folder: {d.raw_dir.rsplit('/', 1)[0]}/"
    )
