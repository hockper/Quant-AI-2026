# Bubble Bi — STORM-inspired Trading Model (Spec 1: Data Pipeline + World Model)

> Canonical design spec for the project (moved into the repo to keep it beside the
> implementation plans in `docs/superpowers/plans/`).

## Context

We are starting a deep-learning trading project inspired by the STORM paper
(*"STORM: A Spatio-Temporal Factor Model Based on Dual Vector Quantized
Variational Autoencoders for Financial Trading"*, Zhao et al., WSDM '26 — the
`STORM.pdf` in the project root).

The paper builds a Dual VQ-VAE (a **Time-Series module** patching one stock over
time + a **Cross-Sectional module** patching all stocks on one day), fuses their
latents in a Factor module, and predicts cross-sectional returns; downstream it
feeds a PPO agent (buy/hold/sell) and a portfolio strategy.

**Our adaptation deliberately differs from the paper.** We keep the paper's Dual
VQ-VAE but use it purely as a **tokenizer**. Its discrete tokens feed a **separate
Transformer** that predicts the **next candle** (world-model style). Later
(Spec 2) we drop the Transformer's prediction heads and feed its hidden state to
an RL agent that takes long / short / flat + stop actions.

**Goal:** build *toward a real trading tool* — so from day one we enforce
no-lookahead data handling, walk-forward evaluation, and (in Spec 2) an honest
backtest with costs/slippage. This first spec delivers the data pipeline and the
world model, built **incrementally** so there is always a working, tested system.

## Project roadmap (each is its own spec → plan → build cycle)

1. **This spec — Data pipeline + STORM world model** (Part 1).
2. **Spec 2 — RL trading agent** (Part 2): consumes the frozen Transformer
   world-model artifact from M4; long/short/flat + stops; honest backtest env
   with transaction costs + slippage; walk-forward evaluation.

## Scope & non-goals (this spec)

**In scope:** yfinance data pipeline; Dual VQ-VAE tokenizer; Transformer
predictor with hybrid heads; training/eval/checkpointing; walk-forward metrics.

**Out of scope (Spec 2):** RL agent, trade simulator with costs/slippage/stops,
portfolio management, live/paper trading.

**Non-goals:** matching the paper's exact numbers; full S&P 500 scale on the
first pass; the paper's prior-posterior / linear return head (intentionally
replaced by the Transformer). Note: the Factor module's *cross-attention feature
fusion* IS included (added in M2); only its prediction head is replaced.

## Architecture

```
Stage 1 — Dual VQ-VAE (TOKENIZER)      loss: recon + commitment + diversity + orthogonality
   TS module: per-stock daily state → enc → quantize(codebook_TS) → tokens_TS   (+decoder for recon)
   CS module: per-day cross-section  → enc → quantize(codebook_CS) → tokens_CS   (+decoder for recon)
        │   FREEZE after training
        ▼   discrete token sequences over time
Stage 2 — Transformer (PREDICTOR, causal/GPT-style, spatio-temporal)   loss: CE(next-token) + MSE(next-candle)
   input per step t: stock's TS token history + shared market CS tokens
   ├─ head A: categorical next-token → decode via VQ-VAE → generative candle  (imagination rollouts)
   └─ head B: regression → next candle OHLC / return                          (sharp point forecast, RankIC)
        │   Part 2 (Spec 2): drop both heads
        ▼
   Transformer hidden state → RL agent (long / short / flat + stops)
```

**Training scheme:** two-stage — train the tokenizer to reconstruct, freeze it,
then train the Transformer on the frozen tokens. Optional joint fine-tune at M4.

## Data pipeline & flow

```
yfinance (daily OHLCV, auto_adjust)
  → raw/*.parquet             (per-ticker cache, incremental, idempotent)
  → align → panel [dates × tickers × OHLCV]   (mask/forward-fill; drop thin history)
  → features (causal TA)       → feature panel [T × N × D]
  → target y[t] = ret(close, t→t+1)  (strictly shifted forward — no leakage)
  → walk-forward splitter      → list of (train, val, test) date windows
  → Dataset → windows of W days over N stocks → batch [B, W, N, D]
        ├─ TS view: per-stock over time
        └─ CS view: per-day over stocks
```

- **Universe:** fixed ~30 large-cap US tickers (DJ30-like) in config; expandable.
  Survivorship-bias caveat noted; point-in-time membership is a later enhancement.
- **Features:** ~30 *causal* technical indicators (returns, log-returns, MAs, RSI,
  MACD, realized vol, volume z-scores…), cross-sectionally standardized per day;
  expandable toward the paper's Alpha158.
- **Caching:** raw parquet, feature parquet, built tensors — all keyed by a config
  hash so re-runs are cheap and reproducible.

## Model components

- **VQ codebook (shared impl):** encoder features → nearest-neighbor quantize
  against K vectors of dim H → straight-through estimator. Losses: commitment /
  codebook (or EMA update), **diversity** (paper Eq. 4, entropy of mean soft
  assignment) and **orthogonality** (paper Eq. 5) on the codebook embeddings.
- **TS module:** per-stock daily state → Transformer encoder → quantize
  (codebook_TS) → Transformer decoder reconstruction.
- **CS module:** per-day cross-section of all stocks → Transformer encoder →
  quantize (codebook_CS) → Transformer decoder reconstruction.
- **Transformer predictor:** causal decoder-only, spatio-temporal (per-stock
  temporal attention + shared CS market tokens). Two heads: (A) categorical
  next-token (cross-entropy per codebook), (B) regression (MSE on next candle
  OHLC / return).

## Incremental milestones (each trains, checkpoints, and is tested)

- **M0 — Pipeline + baseline:** ingest→panel→features→splits→DataLoader, plus a
  ridge-regression return baseline to give a RankIC floor and prove no leakage.
  **DONE (2026-07-09):** panel `(4153, 30, 10)`, 25 walk-forward splits, ridge
  floor **RankIC 0.0062 / RankICIR 0.0230**; 22 tests passing on `main`.
  Plan: `docs/superpowers/plans/2026-07-08-m0-data-pipeline.md`.
- **M1 — TS-only VQ-VAE:** reconstruction + VQ losses; monitor codebook
  perplexity. Proves the tokenizer + training loop + checkpoint/resume + eval.
  Concretized: sliding p=4-day window → small Transformer encoder (CLS) → EMA
  codebook → decoder reconstructs the window; one token per day. Design:
  `docs/superpowers/specs/2026-07-09-m1-ts-vqvae-design.md`.
  **DONE (2026-07-09):** held-out recon MSE 1.46 vs mean-baseline 3.81, perplexity
  78.5, 53% of 512 codes used; 39 tests passing. Plan:
  `docs/superpowers/plans/2026-07-09-m1-ts-vqvae.md`.
- **M2 — Add CS module → full Dual VQ-VAE tokenizer;** ablation switches
  (w/o-TS, w/o-CS). **Freeze** the tokenizer. Concretized: per-day cross-sectional
  encoder (stock-ID embeddings, masked) + **cross-attention fusion before
  quantization** + joint training. Design:
  `docs/superpowers/specs/2026-07-09-m2-dual-vqvae-design.md`.
  First build (2026-07-09, two-codebook: TS token per stock-day + CS token per day)
  worked but diverged from intent. **REDEFINED (2026-07-10):** single fused token
  per stock via a **staged** pipeline — Phase 1 TS VQ-VAE (=M1), Phase 2 windowed CS
  VQ-VAE (new `cs_p` hyperparameter), Phase 3 fuse the frozen *continuous* encoder
  outputs with residual cross-attention (market→stock) → single fusion codebook →
  joint whole-market decoder. Redefined design:
  `docs/superpowers/specs/2026-07-10-m2-staged-dual-vqvae-design.md` (supersedes the
  two-token version; that `DualVQVAE`/`train-dual` code is removed).
  **DONE (2026-07-10):** staged Phases 1-3 built (`train-tokenizer`/`train-cs`/
  `train-fusion`). Fusion token: held-out recon 1.94 vs baseline 3.81, perplexity
  86.7, **80% codes used** (healthy — the fused single codebook works far better
  than the collapsed CS-alone). Frozen `checkpoints_fusion/last.pt` → M3 via
  `FusionVQVAE.encode → ids[B,N]`. Plans:
  `docs/superpowers/plans/2026-07-10-m2-phase2-cs-vqvae.md`,
  `2026-07-10-m2-phase3-fusion.md`.
- **M3 — Transformer predictor** on frozen fusion tokens. Concretized: **categorical
  next-token only** this milestone (per-stock causal GPT over the one-token-per-day
  fusion-token sequences, weights shared, no cross-stock attention; CE over the
  512-vocab). Eval: next-token accuracy/perplexity vs a marginal baseline + light
  multi-step token rollout. Regression/RankIC head deferred; `use_fusion` ablation
  optional. Design: `docs/superpowers/specs/2026-07-10-m3-predictor-design.md`.
- **M4 — Optional joint fine-tune** end-to-end + full metric suite + ablations.
  Emits the frozen **Transformer world-model artifact** consumed by Spec 2.

## Training & reproducibility (matters because of Colab)

Config-driven YAML (data / model / train / eval), single source of truth, hashed
for caching + run naming. Trainer: AMP mixed precision, gradient clipping, AdamW
(lr 1e-4, weight decay 0.05, warmup + decay), deterministic seeds, and
**full-state checkpointing** (model + optimizer + scheduler + RNG + step) to
persistent storage (Drive / bucket) so a killed Colab session resumes cleanly.
Walk-forward: train/eval per window, metrics aggregated across windows.
TensorBoard logging.

## Evaluation

RankIC, RankICIR, plain IC, return MSE/MAE, reconstruction error, next-token
accuracy/perplexity, and **codebook perplexity/usage** (to catch codebook
collapse). Compared against the M0 baseline and the w/o-TS / w/o-CS ablations.
No trading backtest yet (Spec 2).

## Error handling & pitfalls

- **Data:** missing tickers/dates, delistings/survivorship (note & mitigate),
  NaNs from indicator warmup (masking), splits/dividends (`auto_adjust`),
  lookahead (assert targets strictly future).
- **VQ:** codebook collapse — monitor perplexity, use EMA or dead-code reinit,
  keep the diversity loss.
- **Training:** NaNs (grad clip, lr), overfitting (early stop on val RankIC).
- **Interface stability:** define a saved world-model artifact format (frozen
  Transformer + tokenizer + metadata) as the contract for Spec 2.

## Testing strategy (TDD)

- **Unit:** feature causality (shuffling *future* rows must not change past
  features); split non-overlap / no-leak; VQ straight-through gradient shapes;
  forward-pass shapes on a tiny synthetic panel; RankIC=1 on perfectly-ranked
  input.
- **Integration:** tiny end-to-end run (few stocks/days, 1–2 epochs) that trains
  and survives checkpoint→resume.
- **Golden:** a synthetic dataset with a known predictable factor should beat the
  baseline (and the world model should reconstruct/roll it out).

## Proposed project layout

```
bubble_bi/
  configs/            # YAML: data, model, train, eval
  data/  ingest.py universe.py features.py panel.py splits.py dataset.py
  models/ vqvae.py ts_module.py cs_module.py tokenizer.py transformer.py storm.py
  train/  trainer.py losses.py
  eval/   metrics.py evaluate.py
  cli.py              # ingest | build-panel | train-tokenizer | train-transformer | evaluate
  tests/
```

## Defaults (config-overridable)

| Setting | Default |
|---|---|
| Market / bars | US equities, daily, via yfinance (auto_adjust) |
| Universe | ~30 tickers (DJ30-like), configurable |
| History | max available daily |
| Window W | 64 days |
| Tokenization granularity | daily (patch p=1) → "next candle" = next token |
| Codebook | K=512, dim H=128 |
| Features | ~30 causal TA indicators, expandable toward Alpha158 |
| Framework | plain PyTorch + a small Trainer (no Lightning) |
| Training scheme | two-stage (tokenizer → freeze → Transformer), optional joint at M4 |

## Verification (end-to-end)

1. `python -m bubble_bi.cli ingest` → per-ticker parquet appears and is
   incremental on re-run.
2. `python -m bubble_bi.cli build-panel` → aligned feature panel + shifted
   targets; causality unit tests pass.
3. `pytest` → unit + integration (tiny end-to-end + checkpoint/resume) green.
4. `train-tokenizer` on a tiny config → reconstruction loss decreases, codebook
   perplexity stays healthy; checkpoint resumes after an interrupt.
5. `train-transformer` on frozen tokens → next-token accuracy above chance,
   RankIC above the M0 ridge baseline on the held-out walk-forward window; a
   generative rollout reconstructs a known synthetic factor.

## Interface to Spec 2 (RL)

M4 emits a versioned artifact: frozen tokenizer + frozen Transformer + config +
the hidden-state extractor. Spec 2's RL agent consumes the Transformer hidden
state as its observation and (optionally) the generative head for
imagination-based rollouts.
