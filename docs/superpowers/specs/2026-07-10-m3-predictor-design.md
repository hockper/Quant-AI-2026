# M3 — Next-Token Transformer (categorical world model) Design

> Concretizes milestone **M3** of the master spec
> (`docs/superpowers/specs/2026-07-08-storm-trading-design.md`). Consumes the
> frozen redefined-M2 fusion tokenizer
> (`docs/superpowers/specs/2026-07-10-m2-staged-dual-vqvae-design.md`).

## Context

The redefined M2 emits **one fused, market-aware token per (stock, day)**
(`FusionVQVAE.encode(batch) → ids[B,N]`, frozen at
`artifacts/cache/checkpoints_fusion/last.pt`). M3 is a **causal GPT over each
stock's token sequence over time** that predicts the **next day's token**
(categorical, cross-entropy over the 512-code vocab). This is the "world model"
from the project's original framing: predict the next market-state token; later
(Spec 2) drop the head and feed the Transformer's hidden state to an RL agent.

**Decision (from brainstorming):** categorical **next-token only** this milestone
(no regression/RankIC head — deferred). The tokens already carry the fused spatial
/ cross-sectional state, so the predictor is **per-stock temporal only** (weights
shared across stocks; **no cross-stock attention**).

## Architecture

```
Frozen M2 tokenizer  →  pre-encode the panel ONCE  →  token grid ids[T, N]  (int, -1 = invalid stock-day)
  per stock j: contiguous run of valid daily token ids
  windowed to length W (=64):  tokens = run[s : s+W]   targets = run[s+1 : s+W+1]

NextTokenPredictor (per-stock causal GPT, weights shared across stocks):
  Embedding(vocab=512 → H) + positional[W]
  → N_layers causal decoder-only Transformer (triangular mask) → h [B, W, H]
  → Linear(H → 512) logits [B, W, 512]
  loss = cross_entropy(logits, targets)            # teacher-forced over all W positions
  NO cross-stock attention — the fusion token already carries the market.

Spec-2 handoff: freeze the GPT, drop the Linear head, expose h as the RL observation.
```

## Data

- **Pre-encode (`token_grid.py`):** load the frozen `FusionVQVAE`; iterate the
  panel day-by-day in batches (block `[N, window_len, D]`, `window_len=max(p,cs_p)`),
  call `encode` → per-day `ids[N]`; assemble `ids[T, N]` aligned to `panel.dates`,
  setting `-1` where the stock's block isn't fully valid (warmup / missing). Cache
  to `artifacts/cache/tokens.npz` (grid + dates + tickers + a config/checkpoint
  hash for invalidation).
- **`TokenSeqDataset`:** for a chronological day-range (train/val/test split by day),
  yields per-stock windows `{"tokens": LongTensor[W], "targets": LongTensor[W]}`
  from **contiguous valid runs** (all `W+1` tokens `!= -1`); a window belongs to a
  split by its **last target day**. Uses the same `chronological_split` fractions.
- The regression target (`panel.target`) is **not** used in M3 (categorical only).

## Model (`models/predictor.py`) — Llama-3-style decoder

A **Llama-3-style** causal decoder (not `nn.TransformerEncoderLayer`): RMSNorm
pre-norm, rotary positional embeddings (RoPE), SwiGLU feed-forward, and **bias-free**
linears throughout. GQA is configurable (default = full multi-head); KV-cache is
**not** implemented (training is teacher-forced; the light rollout recomputes).

`NextTokenPredictor(cfg, vocab)`:
- `Embedding(vocab, cfg.d_model)` — **no** learned positional embedding (RoPE
  supplies position).
- `cfg.pred_layers` × `LlamaBlock` (pre-norm residual):
  ```
  h = h + Attention(RMSNorm(h))     # bias-free q/k/v/o; RoPE applied to q,k; causal mask;
                                     # GQA-capable: cfg.n_kv_heads KV heads (default = cfg.heads)
  h = h + SwiGLU(RMSNorm(h))         # W2( SiLU(W1 x) ⊙ W3 x ), bias-free, hidden = cfg.ff
  ```
- final `RMSNorm`, then bias-free `Linear(d_model, vocab)` head.
- Building blocks (each small, unit-tested): `RMSNorm(dim, eps)`;
  `RotaryEmbedding(head_dim, max_len=cfg.pred_window, theta=cfg.rope_theta)` →
  precomputed cos/sin, applied to q,k as the standard rotate-half; `SwiGLU(dim, hidden)`.
  Attention uses `n_heads=cfg.heads`, `head_dim=d_model//heads`, repeats KV heads
  when `n_kv_heads < heads` (GQA), and a causal mask so position `t` never sees `t+1`.
- `forward(batch: {"tokens":[B,W], "targets":[B,W]}) → dict` with `loss` (CE),
  `recon_loss` (= same CE, for Trainer/metrics compatibility), `perplexity`
  (`exp(CE)`), `accuracy` (top-1 next-token), `logits`.
- `dead_code_reinit_every = 10**9` and a no-op `reinit_dead_codes(out)` so the
  existing `Trainer` runs it unmodified (it expects those on any model).

## Training

Reuse the `Trainer` (dict batches, AMP-on-CUDA, grad-clip, full-state checkpoint/
resume, `MetricsLogger`) unchanged. Loss is the CE. `_scalars(out)` logs
`loss`/`recon_loss`/`perplexity`/`accuracy` densely. Chronological split; AdamW
(lr 1e-4, wd 0.05). Config adds `pred_window: int = 64`, `pred_layers: int = 4`,
`n_kv_heads: int = 0` (0 ⇒ full multi-head = `heads`; set `<heads` for GQA),
`rope_theta: float = 10000.0`; reuses `d_model`, `heads`, `ff` (SwiGLU hidden),
`dropout`; vocab = `fusion_codebook_size`.

## Eval (`eval/predictor_eval.py`)

`evaluate_predictor(model, loader, device) → dict`:
- **next-token top-1 accuracy** and **perplexity** on held-out test;
- **marginal baseline accuracy** — always predict the train-set most-frequent
  next token; the GPT must beat it;
- **light multi-step rollout** (`rollout_accuracy(model, loader, device, horizon)`):
  from each window, free-run `argmax` `horizon` steps and report token accuracy at
  the horizon (token-level only — no candle decode, which would need all N stocks'
  tokens jointly through the M2 `JointDecoder`).

## `use_fusion` ablation (optional)

Optional secondary comparison (a separate config, not gated): train a
`use_fusion=false` tokenizer (`train-fusion` with `use_fusion: false` → TS-only
tokens), build its token grid, train a second predictor, and compare held-out
next-token accuracy. If the market-aware fusion tokens are more predictable, the
market context earns its place. Run only if time allows.

## CLI

- `tokenize --config configs/m3.yaml` → builds/caches `tokens.npz` from the frozen
  fusion checkpoint.
- `train-predictor` / `eval-predictor` (auto-build the token grid if missing;
  `--run-name` for metrics history; checkpoints at `cache/checkpoints_predictor/`).
- `configs/m3.yaml`.

## Error handling & pitfalls

- **Frozen tokenizer required:** `tokenize`/`train-predictor` raise a clear error if
  `checkpoints_fusion/last.pt` is missing (run `train-fusion` first).
- **-1 (invalid) tokens never enter a window:** `TokenSeqDataset` requires all
  `W+1` tokens valid; asserted by test.
- **Causal masking:** position `t` must not see `t+1`; a test checks that perturbing
  a *future* token in the window doesn't change an earlier position's logits.
- **Grid staleness:** `tokens.npz` stores a hash of the fusion checkpoint + key
  config; rebuild when it changes.
- **Vocab bound:** all grid tokens are in `[0, 512)` (or `-1`); asserted.

## Testing (TDD)

- **token grid:** `build_token_grid` on a tiny synthetic panel returns `ids[T,N]`
  in `[-1, vocab)`; invalid stock-days are `-1`; cache round-trips.
- **TokenSeqDataset:** window contents = grid slice; `targets` = tokens shifted by
  one; windows with any `-1` excluded; split-by-last-day correct.
- **Llama blocks:** `RMSNorm` gives unit-RMS output; `RotaryEmbedding` preserves
  vector norm and differs by position; `SwiGLU` output shape; attention has **no
  bias** params; GQA (`n_kv_heads<heads`) runs and matches full-MHA shapes.
- **model:** forward shapes `[B,W,vocab]`; causal mask (future token change doesn't
  affect earlier-position logits); overfits a tiny batch (accuracy → 1).
- **eval:** accuracy/perplexity finite; marginal baseline computed; a model that
  memorised a tiny set beats the marginal baseline; rollout runs.
- **CLI:** `tokenize` writes `tokens.npz`; `train-predictor` (after a tiny
  `train-fusion`) writes a checkpoint + run folder; `eval-predictor` writes metrics.

## Verification (end-to-end)

1. `pytest` → all prior + new tests green.
2. Ensure the fusion tokenizer exists (`train-tokenizer` → `train-cs` →
   `train-fusion`, from M2) — reuse the existing `checkpoints_fusion/last.pt`.
3. `tokenize --config configs/m3.yaml` → `tokens.npz`.
4. `train-predictor --config configs/m3.yaml --run-name pred` → CE falls,
   next-token accuracy rises; checkpoint + run folder written.
5. `eval-predictor --config configs/m3.yaml --run-name pred` → held-out next-token
   accuracy **above the marginal baseline**; perplexity reported; rollout accuracy
   reported.

## Interface to Spec 2 (RL)

`checkpoints_predictor/last.pt` (frozen GPT) + a `hidden_state(batch)` extractor
(the Transformer output `h` before the head) is the RL observation encoder. The
categorical head enables token-level imagination rollouts; full candle decode is a
later concern (needs the M2 `JointDecoder` over all stocks).

## Defaults

| Setting | Default |
|---|---|
| Block style | **Llama-3**: RMSNorm + RoPE + SwiGLU + bias-free |
| Context window `pred_window` (W) | 64 |
| Predictor layers `pred_layers` | 4 |
| `d_model` H / heads / ff (SwiGLU hidden) | 128 / 4 / 256 |
| `n_kv_heads` (GQA) | 0 ⇒ full multi-head; KV-cache deferred |
| `rope_theta` | 10000.0 |
| Vocab | `fusion_codebook_size` (512) |
| Optim/Trainer | AdamW lr 1e-4 / wd 0.05 / grad-clip 1.0 |
| Ablation | optional `use_fusion=false` config, run if time |
