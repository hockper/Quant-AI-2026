# Bubble Bi — STORM-inspired trading world model

A discrete world model for markets: a staged VQ-VAE tokenizer turns each
`(stock, day)` into **one market-aware token**, and a **Llama-3-style causal
Transformer** predicts the next token.

Inspired by *STORM: A Spatio-Temporal Factor Model Based on Dual Vector Quantized
Variational Autoencoders for Financial Trading* (`STORM.pdf`), but deliberately
adapted: the Dual VQ-VAE is used purely as a **tokenizer**, and its tokens feed a
separate predictor Transformer rather than the paper's factor/return head.

## Pipeline

| Stage | Command | What it does |
|---|---|---|
| Data | `ingest` → `build-panel` | yfinance → leak-free `[T, N, D]` panel |
| Baseline | `baseline` | walk-forward ridge (RankIC floor) |
| Phase 1 | `train-tokenizer` | TS VQ-VAE (per-stock p-day window) |
| Phase 2 | `train-cs` | windowed cross-sectional VQ-VAE (the market) |
| Phase 3 | `train-fusion` | fuse the frozen encoders → **one token per stock-day** |
| Tokens | `tokenize` | frozen tokenizer → cached token grid |
| Predictor | `train-predictor` | Llama-3 GPT over the token sequences |
| Analysis | `plot-metrics` | loss/perplexity curves, multi-run overlays |

Each stage has a matching `eval-*`. Add **`--resume`** to any `train-*` command to
continue an interrupted run from its last checkpoint.

Every training run writes a per-run folder (`artifacts/<cache>/runs/<name>/`) with
a dense `metrics.jsonl`, a `metrics.csv`, `meta.json`, `eval.json` and PNG plots —
so experiments can be compared with `plot-metrics --run-name a b c`.

## Local

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only
.venv/bin/python -m pytest -q
.venv/bin/python -m bubble_bi.cli baseline --config configs/m0.yaml
```

## Colab (GPU / TPU)

Open `notebooks/bubble_bi_colab.ipynb` in Colab. It mounts Drive, pulls this repo,
installs deps, **detects the runtime (CPU / GPU / TPU)**, runs the test suite, then
drives the whole pipeline. Artifacts persist under
`/content/drive/MyDrive/bubble_bi/artifacts/`, so each expensive stage is trained
once and reused across sessions.

`torch` is intentionally **absent from `requirements.txt`** so Colab's pre-installed
CUDA/XLA build is preserved. TPU support goes through `torch_xla` (see
`bubble_bi/runtime.py`); GPU is the recommended target since the models are small.

## Layout

```
bubble_bi/
  data/     ingest, features, panel, splits, windows, token_grid
  models/   vq, ts_vqvae, cross_sectional, cs_vqvae, fusion, joint_decoder,
            fusion_vqvae, llama, predictor
  train/    trainer (checkpoint/resume, metrics), metrics_logger
  eval/     metrics (RankIC), tokenizer_eval, predictor_eval
  viz/      plots
  runtime.py  CPU/GPU/TPU detection + XLA-aware dispatch
  colab.py    Drive-aware config loader
  cli.py
```

Designs and implementation plans live in `docs/superpowers/{specs,plans}/`.
