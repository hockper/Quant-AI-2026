# M2 — Cross-Sectional Module → Dual VQ-VAE with Cross-Attention Fusion (Design)

> Concretizes milestone **M2** of the master spec
> (`docs/superpowers/specs/2026-07-08-storm-trading-design.md`). Builds on M0
> (pipeline) and M1 (TS windowed VQ-VAE: `VectorQuantizerEMA`, `TSEncoder`/
> `TSDecoder`, `Trainer`, `Standardizer`, chronological split).

## Context

M1 gave a TS tokenizer (one token per stock-day from its trailing p-day window).
M2 adds the **Cross-Sectional (CS) module** (one "market-state" token per day),
**cross-attention fusion** that lets the two views inform each other before
quantization, and joint training of the whole **Dual VQ-VAE**. This reintroduces
the paper's *feature-fusion* cross-attention — but **not** its prior/posterior
return head, which the M3 Transformer still replaces. The frozen Dual tokenizer
is the contract M3 consumes.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| CS input | single-day cross-section `[N, D]` (paper-style), one CS token/day |
| Assembly / training | **joint** — TS + CS + fusion trained together in one run |
| Fusion | cross-attention **before** quantization, bidirectional, masked |
| CS enc/dec identity | learned stock-ID embeddings (fixed N-stock universe) |
| Ablations | `w/o-TS`, `w/o-CS` (drop a module); plus a `use_fusion` flag |

## Data — one dataset indexed by day

```
DayDataset[t] → windows [N, p, D]   (each stock's trailing p-day window ending at t; standardized)
             → valid   [N] bool     (stock valid iff its whole window is valid)
CS input for day t = windows[:, -1, :]  = that day's [N, D] cross-section (no extra data)
Batch of B days → windows [B, N, p, D], valid [B, N].  Days with <2 valid stocks are skipped.
```

Reuses the M1 `Standardizer` (fit on train dates only) and `chronological_split`
(split by day `t`). Invalid stocks are zero-filled in `windows` and excluded via
`valid` from attention (as key-padding) and from every reconstruction loss.

## Model — `DualVQVAE`

```
windows [B,N,p,D], valid [B,N]
  ├─ TS encoder (reused M1 TSEncoder), per stock over p days:  z_ts  [B,N,H]
  └─ CS encoder (new), across stocks on day t (masked):        z_cs  [B,H]
        │
   CrossAttentionFusion (before quant, within each day, `fusion_layers` blocks):
     fused_ts[i] = z_ts[i] + xattn(q=z_ts[i], kv=z_cs)                 # stock ← market
     fused_cs    = z_cs    + xattn(q=z_cs,   kv=z_ts, key_pad=~valid)  # market ← stocks
        │   (skipped when use_fusion=False or only one module active)
  ├─ VQ_TS(fused_ts→[B*N,H]) → TS token/stock-day → TSDecoder → recon_win [B,N,p,D]
  └─ VQ_CS(fused_cs)         → CS token/day        → CSDecoder → recon_xs  [B,N,D]
Loss = Σ_active ( masked_recon + commit + λ_div·diversity )     (orthogonality = diagnostic)
```

- **CS encoder:** `Linear(D→H)` per stock + learned `stock_id_embed[N,H]`; prepend
  `[CLS]`; `TransformerEncoder(src_key_padding_mask = ~valid)`; take `[CLS]` = `z_cs`.
- **CS decoder:** `z_q_cs` broadcast + `stock_id_query[N,H]` → `TransformerEncoder`
  → `Linear(H→D)` → `[B,N,D]`. The single market token is deliberately lossy
  per-stock (captures market level/dispersion; idiosyncratic detail is TS's job).
- **Fusion:** `nn.MultiheadAttention` (batch_first), one module per direction,
  residual; repeated `fusion_layers` times. TS branch queries per stock against
  the single CS latent (no kv mask); CS branch queries the market latent against
  the N stock latents (`key_padding_mask = ~valid`).
- **Masked recon:** TS loss averages squared error over valid stock-days; CS loss
  over valid stocks. Both ignore zero-filled invalids.
- **Refactor (reuse):** `TSVQVAE` (M1) grows a `reconstruct(x) → {recon, commit,
  diversity, perplexity, z_e, ids}` method (no loss); its `forward` stays a thin
  wrapper so M1 tests are untouched. `DualVQVAE` uses the raw `TSEncoder`/
  `TSDecoder`/`VectorQuantizerEMA` pieces and computes masked losses itself.
- **`active_modules: ["ts","cs"]`** and **`use_fusion`** drive assembly + ablations.
- **`reinit_dead_codes(out)`** hook resets each active codebook from its fused `z_e`.

## Trainer generalization (small, in place)

Make the M1 `Trainer` batch-format-agnostic: a `_to_device(batch)` and
`_batch_size(batch)` that accept a tensor **or** a dict of tensors, and call a
`model.reinit_dead_codes(out)` hook (falling back to the single-`vq` path) so both
codebooks get dead-code reinit. `DayDataset` yields `{"windows", "valid"}` dicts.
Checkpoint/resume, AMP-on-CUDA, and metrics.jsonl are unchanged.

## Eval / ablations / freeze

`eval-dual` reports TS recon MSE and CS recon MSE (each vs its mean-reconstruction
baseline), both perplexities, and both code-usage fractions on held-out test.
Run three configs: full (`ts+cs`, fusion on), `w/o-TS` (cs only), `w/o-CS`
(ts only) — and optionally fusion-off — to show each part earns its place. Freeze
the trained `DualVQVAE` as the artifact M3 consumes (both codebooks + standardizer
+ config + `encode(batch) → (ts_tokens [B,N], cs_tokens [B])`).

## Error handling & pitfalls

- **Masking correctness:** a stock zero-filled and excluded from attention and
  loss must not change any valid stock's output — asserted by test.
- **Codebook collapse:** perplexity monitor + per-codebook dead-code reinit +
  diversity loss (as M1).
- **Cross-attn in-place/autograd:** VQ already clones its codebook (M1 fix); the
  fusion uses standard MHA (no buffer mutation).
- **Fixed universe:** stock-ID embeddings assume the configured N stocks; noted as
  a later generalization.
- **Ablation integrity:** `w/o-CS` (ts-only, fusion off) must reproduce M1's TS
  path exactly — asserted by test.

## Testing (TDD)

- **CS module:** forward shapes with a padding mask; masked reconstruction ignores
  invalid stocks; permutation of valid stocks permutes the decoder output slots
  consistently (stock-ID identity works).
- **Fusion:** output shapes; fully-masked-except-one behaves; residual identity
  when kv is zero.
- **DualVQVAE:** combined loss = sum of active-module losses; `active_modules`
  builds only requested sub-modules; `use_fusion=False` bypasses fusion; overfit a
  tiny dual batch (both recon losses → near zero); `reinit_dead_codes` touches both
  codebooks.
- **DayDataset:** window/mask contents match the panel; days with <2 valid skipped;
  CS input equals the last day of each window.
- **Trainer:** trains on a dict batch; checkpoint→resume restores state.

## New / changed files

```
bubble_bi/models/cross_sectional.py   # CSEncoder, CSDecoder
bubble_bi/models/fusion.py            # CrossAttentionFusion
bubble_bi/models/dual_vqvae.py        # DualVQVAE (assembly, masked losses, encode, reinit hook)
bubble_bi/models/ts_vqvae.py          # + reconstruct() method (forward unchanged)
bubble_bi/data/windows.py             # + DayDataset, build_day_loaders
bubble_bi/train/trainer.py            # batch-format-agnostic + reinit hook
bubble_bi/eval/tokenizer_eval.py      # + evaluate_dual
bubble_bi/config.py                   # + cs_codebook_size, fusion_layers, active_modules, use_fusion
bubble_bi/cli.py                      # + train-dual, eval-dual
configs/m2.yaml
tests/  test_cross_sectional.py test_fusion.py test_dual_vqvae.py test_day_dataset.py
```

## Defaults (config-overridable)

| Setting | Default |
|---|---|
| Reused | `p=4, d_model(H)=128, K_ts=512` |
| CS codebook | `cs_codebook_size=512` |
| Fusion | `fusion_layers=2, heads=4`, `use_fusion=true` |
| Active modules | `["ts","cs"]` |
| Universe N | 30 |
| Batch (days) | 64 |
| Joint loss weights | 1.0 TS, 1.0 CS |
| Optim/Trainer | as M1 (AdamW lr 1e-4, wd 0.05, grad-clip 1.0, EMA 0.99) |

## Verification (end-to-end)

1. `pytest` → all M0 + M1 + M2 tests green (incl. masked recon, ablation-equals-M1,
   dict-batch resume).
2. `train-dual --config configs/m2.yaml` (tiny/real) → both TS and CS recon
   decrease; both perplexities healthy; checkpoint written.
3. `eval-dual` → held-out TS and CS recon below their mean baselines; report the
   two ablations to show TS, CS, and fusion each contribute.

---

## M2 Results (recorded 2026-07-09)

Config: `configs/m2.yaml` — 30 tickers, `p=4`, `d_model=128`, `K_ts=K_cs=512`,
`fusion_layers=2`, joint TS+CS+fusion, 500 steps on CPU (~8 min).

| Module | Held-out recon MSE | Mean baseline | Perplexity | Codes used |
|---|---|---|---|---|
| **TS** | **1.669** | 3.807 | 142.5 | 70.9% |
| **CS** | **3.507** | 3.819 | 15.6 | 12.9% |

- **TS is the workhorse** — recon at 0.44× baseline with rich codebook usage
  (comparable to M1's standalone 1.46; fusion + shared joint capacity cost a little).
- **CS is a weak signal** — recon only marginally below baseline (0.92×) with low
  perplexity. Expected: compressing 30 stocks × 10 features into *one* discrete
  token per day is a hard bottleneck, and standardized data has a low mean
  baseline. The CS token captures modest market-level/dispersion structure; its
  real value (as market context for M3) is to be judged by M3 ablations.
- **Ablation switches** (`active_modules=[ts]` / `[cs]`) confirmed to train and
  evaluate end-to-end. A quick 150-step check leaves both in VQ cold-start
  (under-trained, not comparable to the 500-step run): TS-only still beat baseline
  (2.99 vs 3.81); CS-only collapsed (perplexity 1.0) — it needs the full step
  budget (and TS/fusion context helps CS avoid collapse in the joint run).

57 tests passing. The trained `last.pt` (full dual) is the frozen tokenizer M3
consumes via `DualVQVAE.encode(batch) → (ts_tokens [B,N], cs_tokens [B])`.

**Open follow-up for M3:** CS codebook under-use suggests trying a smaller CS
codebook, a lighter CS reconstruction target, or longer training — revisit if M3
finds the CS tokens uninformative.
