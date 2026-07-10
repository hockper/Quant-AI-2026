# M2 (redefined) — Staged Dual VQ-VAE with Cross-Attention Fusion (Design)

> **Supersedes** `docs/superpowers/specs/2026-07-09-m2-dual-vqvae-design.md` (the
> two-token version). During M3 brainstorming we found the merged M2 diverged from
> the intended architecture: it produced **two** token streams (TS per stock-day +
> CS per day, two codebooks), whereas the intent is **one** fused token per stock.
> This redefinition replaces it. The two-token `DualVQVAE` / `train-dual` /
> `eval-dual` and their tests are removed.

## Context

We want a tokenizer that emits **exactly one token per (stock, day)** — a
market-aware summary of each stock — trained in **stages** so each piece is
verified before the next. Phase 1 (TS) is literally M1. Phase 2 adds a
**windowed** cross-sectional encoder. Phase 3 fuses the two **continuous** frozen
encoder outputs by cross-attention, quantizes once, and reconstructs the whole
market with a joint decoder. The frozen Phase-3 tokens are what M3 consumes.

## Architecture (three phases)

```
PHASE 1 — TS VQ-VAE (== M1, reused unchanged)
  ts_in [B,N,p,D] → TS enc → z_ts → VQ(codebook_TS) → TS dec → reconstruct p-day window ; MSE + VQ
  Verify recon↓ + perplexity.  Carry forward: the frozen TS ENCODER (continuous z_ts). Codebook_TS/dec are scaffolding.

PHASE 2 — CS VQ-VAE (windowed, new)
  cs_in [B,N,cs_p,D] → CS enc(market over cs_p days, masked) → z_cs → VQ(codebook_CS) → CS dec → reconstruct field ; MSE + VQ
  Verify recon↓ + perplexity.  Carry forward: the frozen CS ENCODER (continuous z_cs).

PHASE 3 — Fusion tokenizer (encoders frozen; fuse CONTINUOUS outputs)
  z_ts = frozenTSenc(ts_in)                 → [B,N,H]      (requires_grad=False)
  z_cs = frozenCSenc(cs_in)                 → [B,H]        (requires_grad=False)
  fused  = z_ts + cross-attn(query=z_ts, kv=z_cs)          → [B,N,H]   # residual; market→stock; single market kv
  tokens = VQ(codebook_FUSION)(fused[B*N,H]) → z_q [B,N,H], ids [B,N]  # the ONE final token
  recon  = joint_decoder(z_q, valid) → whole market [B,N,p,D]         # cross-stock attn, then temporal expand
  Loss = masked_recon(recon, ts_in) + commit + λ_div·diversity        # only fusion + codebook_FUSION + joint dec train
  encode(batch) → ids [B,N]                                           # frozen artifact for M3
```

**Only `codebook_FUSION` produces the M3 token.** The TS/CS codebooks exist only to
give Phases 1–2 a verifiable reconstruction objective; Phase 3 uses the encoders'
continuous pre-quantization outputs.

## Data

`DayDataset` (generalized) yields, per day `t`: `block [N, L, D]` where
`L = max(p, cs_p)` trailing days, plus `valid [N]` (stock valid iff **all L** days
are valid). Consumers slice:
- `ts_in = block[:, -p:, :]` → `[B,N,p,D]`
- `cs_in = block[:, -cs_p:, :]` → `[B,N,cs_p,D]`

Phase 1 keeps using M1's per-stock `WindowDataset` (`[B,p,D]`). Phases 2 & 3 use
`DayDataset` with `window_len = cs_p` (Phase 2) and `max(p,cs_p)` (Phase 3).
Reuses the M1 `Standardizer` (fit on train) and `chronological_split`. Invalid
stocks are zero-filled and excluded from attention (key-padding) and every loss.

## Components / files

- **Phase 1 (reuse):** `models/ts_vqvae.py` `TSVQVAE`, `data/windows.py`
  `WindowDataset`/`build_loaders`, CLI `train-tokenizer`/`eval-tokenizer`. Unchanged.
- **`data/windows.py` (modify):** `DayDataset` gains `window_len` (default = `p`);
  `build_day_loaders(panel, cfg, window_len)` yields `[N, window_len, D]` blocks.
- **`models/cross_sectional.py` (modify):** `CSEncoder` takes a **windowed** field
  `x [B,N,cs_p,D]` + `valid [B,N]`: embed each (stock, day) vector + stock-ID
  embedding + day-position embedding, flatten to `[B, N*cs_p, H]`, prepend `[CLS]`,
  masked `TransformerEncoder` (invalid stocks masked across their `cs_p` days),
  return `[CLS]` = `z_cs [B,H]`. `CSFieldDecoder`: `z_q_cs [B,H]` → stock-ID +
  day-position queries → Transformer → `H→D` → reconstruct `[B,N,cs_p,D]`.
- **`models/cs_vqvae.py` (new):** `CSVQVAE` = `CSEncoder` + `VectorQuantizerEMA`
  (`cs_codebook_size`) + `CSFieldDecoder`. `forward(batch) → {recon, loss,
  recon_loss, commit, diversity, perplexity, ids, z_e}`; masked recon over valid
  stocks. `encode`/`reconstruct` helpers.
- **`models/fusion.py` (rewrite):** `MarketToStockFusion(cfg)` —
  `forward(z_ts [B,N,H], z_cs [B,H]) → [B,N,H]` = `z_ts + cross-attn(q=z_ts,
  kv=z_cs.unsqueeze(1))`, `fusion_layers` blocks, residual, no mask (single market
  key). Bidirectional `CrossAttentionFusion` is removed.
- **`models/joint_decoder.py` (new):** `JointDecoder(cfg, d_out, n_stocks)` —
  `forward(z_q [B,N,H], valid) → [B,N,p,D]`: `+ stock_id`, cross-stock
  `TransformerEncoder` (masked), then per-stock temporal expand
  (`pos[1,p,H]` + Transformer + `H→D`).
- **`models/fusion_vqvae.py` (new; replaces `dual_vqvae.py`):** `FusionVQVAE(cfg,
  d_in, n_stocks)` holds a frozen `TSEncoder`, a frozen `CSEncoder`, a
  `MarketToStockFusion`, `VectorQuantizerEMA(fusion_codebook_size)`, and a
  `JointDecoder`. `load_frozen(ts_ckpt, cs_ckpt)` loads the two encoders and sets
  `requires_grad=False`. `forward(batch) → {recon, loss, recon_loss, commit,
  diversity, perplexity, ids, z_e}`. `encode(batch) → ids [B,N]`.
  `reinit_dead_codes(out)`. `use_fusion=False` bypasses CS (fused = z_ts) as the
  ablation. **`dual_vqvae.py` and `tests/test_dual_vqvae.py` are deleted.**
- **`train/trainer.py` (modify):** build the optimizer over
  `[p for p in model.parameters() if p.requires_grad]` so frozen encoders don't
  update (and AMP/grad-clip see only trainable params).
- **`config.py` (modify):** add `cs_p: int = 5`, `fusion_codebook_size: int = 512`;
  **remove** `active_modules` and its uses. Keep `use_fusion`, `cs_codebook_size`,
  `codebook_size`, `p`.
- **`eval/tokenizer_eval.py` (modify):** add `evaluate_cs(model, loader, device)`
  and `evaluate_fusion(model, loader, device)` (recon MSE vs mean baseline,
  perplexity, code-usage). **Remove `evaluate_dual`.**
- **`cli.py` (modify):** add `train-cs`/`eval-cs` (Phase 2, `DayDataset`
  `window_len=cs_p`) and `train-fusion`/`eval-fusion` (Phase 3, loads frozen TS+CS
  checkpoints, `window_len=max(p,cs_p)`). **Remove `train-dual`/`eval-dual`.** Each
  supports `--run-name` (metrics history, unchanged). Phase-3 checkpoints under
  `checkpoints/fusion_last.pt`; Phase 1/2 under `checkpoints/ts_last.pt`,
  `checkpoints/cs_last.pt` (distinct so phases don't clobber).
- **`configs/` :** `m2_cs.yaml`, `m2_fusion.yaml` (Phase 2 / Phase 3 configs;
  Phase 1 reuses `configs/m1.yaml`).

## Training & freezing

Each phase is a separate CLI run + checkpoint. Phase 3 constructs `FusionVQVAE`,
calls `load_frozen(ts_last.pt, cs_last.pt)` (encoders → `eval()` + `requires_grad
=False`), and the Trainer optimizes only trainable params. AMP-on-CUDA, grad-clip,
dict-batch, MetricsLogger, dead-code reinit — all unchanged. `cs_p` and `p` are
independent windows; joint loss weights 1.0.

## Eval / ablation

- Phase 1: M1's `eval-tokenizer` (recon vs baseline, perplexity, code usage).
- Phase 2: `eval-cs` — CS field recon MSE vs mean baseline, perplexity, code usage.
- Phase 3: `eval-fusion` — whole-market recon MSE vs mean baseline, **fusion**
  perplexity + code usage; **RankIC is NOT here** (that's M3's regression head).
- Ablation: `use_fusion=false` (Phase 3) = tokens from TS only, to measure whether
  the windowed market context improves the tokens.

## Error handling & pitfalls

- **Single-key fusion degeneracy:** the residual `z_ts +` is required — with a
  single market key, pure cross-attention would return an identical value for every
  stock. Asserted by a test (two stocks with different `z_ts` get different fused).
- **Frozen leakage:** Phase 3 must not update the encoders — a test checks their
  weights are unchanged after a train step.
- **Masking:** invalid stocks never influence a valid stock's output or the loss
  (tested for CS encoder, joint decoder).
- **Codebook collapse:** perplexity monitor + dead-code reinit + diversity, per
  codebook (as before).
- **Window alignment:** `DayDataset` validity uses `L=max(p,cs_p)`; a test checks
  `ts_in`/`cs_in` slices match the panel.

## Testing (TDD)

- `DayDataset(window_len)`: block contents + `valid` match the panel; days with
  `<2` valid skipped; `ts_in`/`cs_in` slices correct.
- `CSEncoder` (windowed): forward shapes; invalid stocks ignored; `CSFieldDecoder`
  shapes; `CSVQVAE` overfits a tiny batch (recon → small).
- `MarketToStockFusion`: shapes; **residual keeps stocks distinct** (different
  `z_ts` → different fused even with one shared market key).
- `JointDecoder`: shapes; masked invariance (invalid stock doesn't change a valid
  stock's output).
- `FusionVQVAE`: `load_frozen` sets `requires_grad=False`; a train step leaves
  encoder weights unchanged but updates fusion/codebook/decoder; `encode` returns
  `ids [B,N]` long; `use_fusion=False` bypasses CS; overfits a tiny batch.
- `Trainer`: optimizes only trainable params (frozen params unchanged) — reuse the
  frozen check above via a tiny `FusionVQVAE`.
- CLI: `train-cs`→`cs_last.pt` + run folder; `train-fusion` (after ts/cs
  checkpoints exist)→`fusion_last.pt`; `eval-fusion` writes metrics.

## Verification (end-to-end)

1. `pytest` → all surviving M0/M1/metrics tests + new M2 tests green; the deleted
   two-token tests are gone.
2. `train-tokenizer --config configs/m1.yaml --run-name ts` (Phase 1) → `ts_last.pt`.
3. `train-cs --config configs/m2_cs.yaml --run-name cs` (Phase 2) → `cs_last.pt`;
   `eval-cs` recon below baseline, perplexity ≫ 1.
4. `train-fusion --config configs/m2_fusion.yaml --run-name fusion` (Phase 3, loads
   the two frozen encoders) → `fusion_last.pt`; `eval-fusion` whole-market recon
   below baseline, fusion perplexity healthy.
5. `plot-metrics --run-name ts cs fusion` overlays the three phases' curves.

## Interface to M3

Phase 3's `fusion_last.pt` (frozen TS+CS encoders + fusion + `codebook_FUSION` +
joint decoder) is the world-model tokenizer. M3 calls `FusionVQVAE.encode(batch) →
ids [B,N]` to build the per-stock token sequences, drops the joint decoder, and
trains the predictor Transformer on them.

## Phase 2 results (recorded 2026-07-10)

Config `configs/m2_cs.yaml` — 30 tickers, `cs_p=5`, `d_model=128`, `K_cs=512`,
500 steps CPU (~10.5 min).

| Metric | Value |
|---|---|
| Held-out CS field recon MSE | **3.55** |
| Mean baseline | 3.80 |
| Perplexity (held-out) | 9.0 (train ~24) |
| Codes used | 7.6% |

The windowed CS is a **weak reconstructor** (0.93× baseline) — compressing a
`30×5×10` market field into one token per day is a hard bottleneck, on par with
M2's single-day CS. Phase 2's purpose is met (the CS encoder trains standalone and
beats the baseline), and Phase 3 consumes the **continuous** encoder output (not
the quantized token), so the codebook's low utilisation here is not fatal. The
open question — does the market context actually help — is answered at Phase 3
(fusion, `use_fusion` ablation) and M3. 65 tests passing.

## Defaults

| Setting | Default |
|---|---|
| `p` (TS window) | 4 |
| `cs_p` (CS window, NEW) | 5 |
| `d_model` H | 128 |
| `codebook_size` (TS) | 512 |
| `cs_codebook_size` (CS) | 512 |
| `fusion_codebook_size` (final) | 512 |
| `fusion_layers` | 2 |
| `use_fusion` | true |
| Optim/Trainer | AdamW lr 1e-4 / wd 0.05 / grad-clip 1.0 / EMA 0.99 |
