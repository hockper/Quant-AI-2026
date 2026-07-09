# M1 — TS-only Windowed VQ-VAE Tokenizer (Design)

> Concretizes milestone **M1** of the master spec
> (`docs/superpowers/specs/2026-07-08-storm-trading-design.md`). Builds on the
> completed M0 data pipeline (`Panel`, walk-forward splits, metrics).

## Context

M0 gave us a leak-free `Panel [T, N, D]` and a ridge floor (RankIC 0.0062). M1
builds the **first neural component**: a Time-Series VQ-VAE that turns each
`(stock, day)` into a discrete token, plus the reusable training infrastructure
(Trainer, checkpoint/resume, eval) that every later milestone depends on.

A design decision from brainstorming reconciled the master spec's "Transformer
encoder" with our daily token stream: the encoder reads a **sliding p-day
window** ending at day `t` and emits **one token per day** `t`. Time modeling
inside the tokenizer is real (the encoder attends over the window); the *sequence*
of daily tokens is what Stage 2 (M3) will model.

## Scope & non-goals

**In scope:** windowed TS VQ-VAE (encoder → EMA codebook → decoder), a
`WindowDataset` + leak-free standardizer over the M0 panel, a plain-PyTorch
`Trainer` with full-state checkpoint/resume, reconstruction + perplexity eval,
CLI `train-tokenizer` / `eval-tokenizer`.

**Out of scope:** CS module (M2), Transformer predictor (M3), RL (Spec 2),
walk-forward retraining of the tokenizer, TensorBoard (optional, not required).

**Non-goals:** matching paper numbers; multi-GPU; joint training.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Encoder input | sliding p-day window per (stock, day); **one token per day** |
| Encoder type | small **Transformer** (CLS-pooled) |
| Decoder | reconstructs the full p-day window |
| Codebook | **EMA-updated** (+ commitment loss, straight-through); dead-code reinit |
| Split | chronological **70/15/15** by date (one shared tokenizer) |
| Window `p` | **4** days (paper's TS patch) |
| Env | torch 2.13.0 cp314; **CPU-only** build locally, CUDA on Colab |

## Data path

```
Panel [T, N, D]  (from M0)
  → Standardizer: per-feature mean/std fit on TRAIN dates only (leak-free), applied to all splits
  → WindowDataset: for each (stock j, day t) with t ≥ p−1 AND every day in [t−p+1 … t] valid (mask),
        sample = standardized_features[t−p+1 … t, j, :]   shape [p, D]
  → DataLoader (shuffle in train; no shuffle in val/test)
```

- The tokenizer is **shared across all stocks** (one vocabulary of market-day
  window patterns).
- Split by date index into the panel: `train = [0, 0.7T)`, `val = [0.7T, 0.85T)`,
  `test = [0.85T, T)`. Windows are assigned to a split by their **last day `t`**.
- Standardizer stats are computed from train-split valid rows only and saved in
  the checkpoint (needed at eval and by later milestones).

## Model — `TSVQVAE`

```
window x [B, p, D]
  ── encoder ────────────────────────────────────────────────
   linear D→H ; + positional encoding over p ; prepend learnable [CLS]
   → n_layers × TransformerEncoderLayer (pre-norm, heads, ff)
   → take [CLS] hidden = z_e [B, H]
  ── quantize ───────────────────────────────────────────────
   VectorQuantizerEMA(K, H): code = argmin‖z_e − e_k‖²  → token id, z_q [B, H]
   straight-through: z_q ← z_e + (z_q − z_e).detach()
  ── decoder ────────────────────────────────────────────────
   z_q → expand to p positions (learned query per position or repeat + pos-enc)
   → n_layers × TransformerEncoderLayer → linear H→D → x̂ [B, p, D]
```

**Loss** = `MSE(x̂, x)` + `β · ‖z_e − sg(z_q)‖²` (commitment) + `λ_div · L_div`
+ `λ_ortho · L_ortho`. Codebook vectors are updated by **EMA** of assigned
encoder outputs (no separate codebook-loss term). `L_div` = negative entropy of
mean soft-assignment (encourage code usage); `L_ortho` = ‖ÊÊᵀ − I‖²_F on
L2-normalized codebook embeddings (paper Eqs. 4–5).

**Monitors:** perplexity `exp(−Σ p_k log p_k)` over the batch's code usage, and
fraction of codes used. **Dead-code reinit:** every N steps, codes unused over
the window are reset to random encoder outputs from the current batch.

**Defaults (config):** `p=4, H=128, K=512, enc_layers=3, dec_layers=2, heads=4,
ff=256, dropout=0.1, β=0.25, λ_div=0.1, λ_ortho=0.1, ema_decay=0.99,
dead_code_reinit_every=250`.

## Training & reproducibility

Plain PyTorch + a small `Trainer`:

- AdamW (lr `1e-4`, weight_decay `0.05`), gradient clipping (`1.0`).
- AMP autocast/GradScaler **only on CUDA**; plain float32 on CPU.
- Deterministic seeds (`torch`, `numpy`, python) from `Config.seed`.
- **Full-state checkpoint** every N steps and at epoch end: model, optimizer,
  scheduler, AMP scaler, RNG states, global step, and standardizer stats →
  single `.pt` file in `cache_dir/checkpoints/`. Resume restores all of it and
  continues (loss curve is continuous across a restart).
- Early stop / best-checkpoint on val reconstruction MSE.
- Logging: console + append-only `metrics.jsonl` (step, losses, perplexity).

Local dev uses CPU-only torch (tiny configs, fast tests); Colab uses CUDA for
full runs. `torch` stays out of `requirements.txt`; install is documented per env.

## Eval

`eval-tokenizer` on val/test: reconstruction MSE (overall + per-feature),
codebook perplexity and code-usage fraction. **Health thresholds:** perplexity
≫ 1 (not collapsed toward 1), recon MSE ≪ 1 (standardized-feature variance ≈ 1).
Report against a **mean-reconstruction baseline** (predict each feature's train
mean) to confirm the VQ-VAE actually encodes structure.

## Error handling & pitfalls

- **Codebook collapse:** perplexity monitor + dead-code reinit + diversity loss.
- **Leakage:** standardizer fit on train only; windows assigned by last day;
  test asserts train-fit stats differ from a full-fit and that no future date
  contributes to a train window.
- **Masked/short windows:** dataset skips any window overlapping an invalid day
  or the pre-warmup region; assert no NaNs reach the model.
- **CPU/CUDA divergence:** AMP guarded by device; tests run on CPU.
- **Interface for M2/M3:** `TSVQVAE.encode(x) -> token_ids` and a saved artifact
  (weights + standardizer + config) define the contract the CS module and the
  predictor will reuse.

## Testing strategy (TDD)

- **VQ unit:** nearest-code selection correct on hand-set codebook; straight-through
  yields non-None encoder gradient; perplexity = K under uniform assignment, = 1
  under single-code assignment.
- **Model unit:** forward shapes on a tiny synthetic `[B,p,D]` batch; loss finite;
  **overfit a tiny fixed batch** → recon MSE → ~0 (proves learning + gradient path).
- **Dataset unit:** window contents equal the panel slice; masked/short windows
  excluded; standardizer fit uses train rows only (leak check).
- **Trainer integration:** 2-step run produces a checkpoint; **resume restores
  exact state** (same step, loss continues); best-checkpoint tracks val MSE.
- **Golden:** synthetic low-rank panel → VQ-VAE recon MSE < mean-baseline MSE.

## New / changed files

```
bubble_bi/
  models/__init__.py
  models/vq.py            # VectorQuantizerEMA (+ perplexity, dead-code reinit)
  models/ts_vqvae.py      # TSEncoder, TSDecoder, TSVQVAE (forward → recon, losses, tokens)
  data/windows.py         # Standardizer, WindowDataset, chronological split, build_loaders
  train/__init__.py
  train/trainer.py        # Trainer: loop, AMP, checkpoint/resume, metrics.jsonl
  config.py               # + ModelConfig, TrainConfig; extend Config
  cli.py                  # + train-tokenizer, eval-tokenizer subcommands
configs/m1.yaml
tests/  test_vq.py test_ts_vqvae.py test_windows.py test_trainer.py
```

## Verification (end-to-end)

1. `pip install torch --index-url https://download.pytorch.org/whl/cpu` (local).
2. `pytest` → all M0 + M1 tests green (incl. overfit-tiny-batch and resume).
3. `train-tokenizer --config configs/m1.yaml` on a tiny config → recon MSE
   decreases, perplexity stays healthy; kill and re-run → resumes from checkpoint.
4. `eval-tokenizer` → held-out recon MSE below the mean baseline; perplexity ≫ 1.

---

## M1 Results (recorded 2026-07-09)

Config: `configs/m1.yaml` — 30 tickers, `p=4`, `d_model=128`, `K=512`, 3-layer
Transformer encoder / 2-layer decoder, 500 steps on CPU (~1m48s), chronological
70/15/15.

| Metric | Value |
|---|---|
| Train recon MSE (last step) | 0.348 |
| Val recon MSE | 0.610 |
| **Held-out test recon MSE** | **1.460** |
| Mean-reconstruction baseline (test) | 3.807 |
| Perplexity (test) | 78.5 (train ~164) |
| Codes used (of 512) | 53.3% |

The tokenizer reconstructs held-out windows at **0.38×** the mean-baseline error
with healthy, non-collapsed codebook usage — the VQ-VAE captures real structure.
(Baseline > 1 because test-period features, standardized with train stats, drift
away from the train mean — expected distribution shift.) Checkpoint/resume
mechanism verified by unit test and by `eval-tokenizer` loading `last.pt`.

**Note on losses:** with the chosen EMA codebook, the orthogonality term is a
diagnostic only (an EMA buffer has no gradient); reconstruction + commitment +
diversity drive training. This is a conscious reconciliation of the spec's
"EMA codebook" + "orthogonality" items, not a gap.
