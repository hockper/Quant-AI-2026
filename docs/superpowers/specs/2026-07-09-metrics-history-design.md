# Metrics History + Plots (Design)

> A small cross-cutting feature on top of M0–M2. Not a milestone; adds
> observability so training/eval results can be analyzed and compared across runs
> (as feature sets and module sizes change).

## Context

The `Trainer` currently writes a sparse `metrics.jsonl` (only `step`, `loss`,
`recon`, `perplexity`, `val_mse`, and only at `val_every`). We want a **complete
per-step history of every loss component** for both training and eval, persisted
per run, exported as JSONL **and** CSV, with a command that renders PNG plots and
can **overlay multiple runs** for comparison. No model math changes — only
surfacing and persisting values already computed.

## Goals & non-goals

**Goals:** dense training-loss history (all components), val history, eval
results; per-run folders that don't clobber; CSV for external tools; matplotlib
plots incl. multi-run overlays.

**Non-goals:** live dashboards / TensorBoard; hyperparameter search; changing what
losses are computed; distributed logging.

## Run layout

Checkpoints stay at `artifacts/<cache>/checkpoints/last.pt` (so `eval-*` is
unchanged). Metrics live per run:

```
artifacts/<cache>/runs/<run_name>/
  metrics.jsonl   # per log step:
                  #   {"phase":"train","step":s, "loss":.., "recon_loss":.., "perplexity":..,
                  #    "ts_recon":.., "cs_recon":.., "ts_commit":.., "ts_diversity":..,
                  #    "cs_commit":.., "cs_diversity":.., "ts_perplexity":.., "cs_perplexity":..}
                  #   {"phase":"val","step":s, "val_mse":..}
  metrics.csv     # same records, tidy columns = union of all keys (missing -> blank)
  meta.json       # {active_modules, d_model, codebook_size, cs_codebook_size, fusion_layers,
                  #  n_features(D), n_stocks(N), max_steps, log_every, final_train, final_val}
  eval.json       # written by eval-* with --run-name: per-module recon/baseline/perplexity/codes
  plots/*.png     # produced by plot-metrics
```

`run_name` comes from `--run-name`; default derived from config
(`dual_<active-modules>_<max_steps>` for `train-dual`, `tsvqvae_<max_steps>` for
`train-tokenizer`). Same name → same folder (re-run overwrites); different names
coexist for comparison.

## Components

- **`bubble_bi/train/metrics_logger.py` — `MetricsLogger(run_dir)`**
  - `log(record: dict) -> None`: append one JSON line to `metrics.jsonl` and keep
    it in an in-memory list.
  - `to_csv() -> None`: write `metrics.csv` with columns = sorted union of all
    record keys (blank where a record lacks a key).
  - `write_meta(meta: dict) -> None`: write `meta.json`.
  - `records: list[dict]` attribute for tests.

- **`bubble_bi/train/trainer.py` (modify)**
  - `Trainer.__init__` gains optional `run_dir=None` (defaults to `ckpt_dir`); it
    builds a `MetricsLogger(run_dir)`.
  - New `TrainConfig.log_every: int = 10`.
  - Training loop: every `log_every` steps, build a **train record** from all
    *scalar* entries of the model's output dict via a helper
    `_scalars(out) -> dict[str, float]` (0-dim tensors / floats only; skips
    `ts_z_e`/`cs_z_e`), tagged `phase="train"` with `step`; log it.
  - Every `val_every` steps: log `{"phase":"val","step":s,"val_mse":v}`.
  - On finish: `logger.to_csv()`. (`meta.json` is written by the CLI, which knows
    the config.) Returns the final metrics dict as today.

- **`bubble_bi/models/dual_vqvae.py` + `bubble_bi/models/ts_vqvae.py` (modify)**
  - Surface per-module commit/diversity in the output dict so they are recorded:
    DualVQVAE adds `ts_commit`, `ts_diversity`, `cs_commit`, `cs_diversity`;
    `TSVQVAE.forward` already returns `commit`/`diversity` (unchanged). These are
    the scalars already used to build `loss`; only the dict grows.

- **`bubble_bi/viz/plots.py` — plotting (matplotlib, `Agg` backend)**
  - `plot_run(run_dir: str) -> list[str]`: reads `metrics.jsonl`; writes
    `plots/losses.png` (total + per-module recon + commit/diversity vs step),
    `plots/perplexity.png` (ts/cs perplexity vs step), `plots/val.png` (val_mse vs
    step, with train recon overlaid). Returns the PNG paths.
  - `plot_compare(run_dirs: list[str], out_dir: str) -> list[str]`: overlays each
    run's total-loss and val curves → `compare_loss.png`, `compare_val.png`.

- **`bubble_bi/cli.py` (modify)**
  - `--run-name` optional arg (used by `train-dual`, `train-tokenizer`,
    `eval-dual`, `eval-tokenizer`).
  - `train-*`: compute `run_dir = cache_dir/runs/<run_name>`, pass to `Trainer`,
    and after training write `meta.json` (config summary + final train/val).
  - `eval-*`: if `--run-name` given, write `eval.json` into that run folder.
  - New command `plot-metrics --run-name A [B C ...]`: one name → `plot_run`;
    several → `plot_compare` into the first run's `plots/`.

- **`requirements.txt`**: add `matplotlib`.

## Error handling

- Empty/missing `metrics.jsonl` → plotting raises a clear `FileNotFoundError` with
  the path (don't emit blank plots).
- Records with differing key sets → CSV fills missing cells blank; plotting skips a
  series with no data points.
- Headless: force `matplotlib.use("Agg")` at import in `viz/plots.py`.
- Back-compat: existing Trainer callers (M1 tests) that pass no `run_dir` keep
  writing metrics into the checkpoint dir; `log_every` default keeps them working.

## Testing (TDD)

- `MetricsLogger`: `log` appends JSONL lines and populates `records`; `to_csv`
  writes a header = union of keys with blanks for missing; `write_meta` writes JSON.
- `_scalars`: extracts float scalars, skips multi-element tensors (`z_e`).
- Trainer: after a short dual run, `metrics.jsonl` has **≥2 train records**
  (dense, at `log_every`) each containing `ts_recon` and `cs_recon`, plus ≥1 val
  record; `metrics.csv` exists with those columns.
- `plot_run` / `plot_compare`: produce non-empty PNG files from a tiny synthetic
  `metrics.jsonl` (assert files exist and size > 0).
- CLI: `train-dual --run-name r1` creates `runs/r1/{metrics.jsonl,metrics.csv,meta.json}`;
  `eval-dual --run-name r1` writes `runs/r1/eval.json`; `plot-metrics --run-name r1`
  writes PNGs under `runs/r1/plots/`.

## Verification (end-to-end)

1. `pip install -r requirements.txt` (adds matplotlib).
2. `pytest` → all prior + new tests green.
3. `train-dual --config configs/m2.yaml --run-name dual_full` → run folder with
   dense `metrics.jsonl` + `metrics.csv` + `meta.json`.
4. `eval-dual --config configs/m2.yaml --run-name dual_full` → `eval.json`.
5. `plot-metrics --run-name dual_full` → PNGs under `runs/dual_full/plots/`.
6. (Later) a second run under a different `--run-name`, then
   `plot-metrics --run-name dual_full other_run` → overlaid comparison plots.

## Defaults

| Setting | Default |
|---|---|
| `log_every` | 10 steps |
| Run folder | `artifacts/<cache>/runs/<run_name>/` |
| `run_name` | derived from config, override with `--run-name` |
| Plot backend | matplotlib `Agg` |
| Single-run plots | losses, perplexity, val |
| Multi-run plots | overlaid total-loss, val |
