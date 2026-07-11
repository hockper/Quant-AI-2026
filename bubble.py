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

_TS_KEYS = {"days", "vocabulary", "encoder_depth", "decoder_depth"}
_CS_KEYS = {"days", "vocabulary", "encoder_depth", "decoder_depth"}
_FUSION_KEYS = {"vocabulary", "depth"}
_PREDICTOR_KEYS = {"sentence_length", "depth"}


def _check(block: dict, allowed: set[str], name: str) -> dict:
    unknown = set(block) - allowed
    if unknown:
        raise ValueError(
            f"`{name}` got unknown setting(s) {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}."
        )
    return block


def make_config(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    *,
    ts: dict | None = None,
    cs: dict | None = None,
    fusion: dict | None = None,
    predictor: dict | None = None,
    model_size: int = 128,
    steps: int = 2000,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    seed: int = 42,
    device: str = "auto",
    data_dir: str = "artifacts",
) -> Config:
    """Build the project's configuration from plain notebook values.

    The tokenizer has two entries, each with its own settings:

      ts     -- what THIS stock has been doing (one stock, over time)
      cs     -- what the WHOLE MARKET was doing (all stocks, on a day)
      fusion -- where the two entries merge into the single token we keep

    `model_size` is deliberately shared: the two entries meet in a cross-attention
    layer, which requires both sides to have the same width.
    """
    ts = _check(dict(ts or {}), _TS_KEYS, "ts")
    cs = _check(dict(cs or {}), _CS_KEYS, "cs")
    fusion = _check(dict(fusion or {}), _FUSION_KEYS, "fusion")
    predictor = _check(dict(predictor or {}), _PREDICTOR_KEYS, "predictor")

    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        raise ValueError("`tickers` is empty — list at least one company, e.g. ['AAPL'].")
    for entry, block in (("ts", ts), ("cs", cs)):
        if block.get("days", 1) < 1:
            raise ValueError(f"`{entry}['days']` must be at least 1, got {block['days']}.")
        if block.get("vocabulary", 2) < 2:
            raise ValueError(
                f"`{entry}['vocabulary']` must be at least 2, got {block['vocabulary']}."
            )

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
            d_model=model_size,                                   # shared (see above)
            # entry 1 — TS
            p=ts.get("days", 4),
            codebook_size=ts.get("vocabulary", 512),
            enc_layers=ts.get("encoder_depth", 3),
            dec_layers=ts.get("decoder_depth", 2),
            # entry 2 — CS
            cs_p=cs.get("days", 5),
            cs_codebook_size=cs.get("vocabulary", 512),
            cs_enc_layers=cs.get("encoder_depth", 3),
            cs_dec_layers=cs.get("decoder_depth", 2),
            # where the two entries merge
            fusion_codebook_size=fusion.get("vocabulary", 512),
            fusion_layers=fusion.get("depth", 2),
            # the predictor
            pred_window=predictor.get("sentence_length", 64),
            pred_layers=predictor.get("depth", 4),
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

    ts = f"     TS  this stock, {m.p} days back".ljust(40)
    cs = f"     CS  whole market, {m.cs_p} days back".ljust(40)
    return "\n".join([
        f"✅ Ready to run on {where.upper()}",
        f"   {len(d.tickers)} companies, {period}",
        "",
        "   The tokenizer's two entries:",
        f"{ts}→ {m.codebook_size} words, depth {m.enc_layers}",
        f"{cs}→ {m.cs_codebook_size} words, depth {m.cs_enc_layers}",
        f"     ⤷ merged into ONE token out of {m.fusion_codebook_size}",
        "",
        f"   Predictor reads sentences of {m.pred_window} tokens (depth {m.pred_layers})",
        f"   Shared brain width: {m.d_model}",
    ])
