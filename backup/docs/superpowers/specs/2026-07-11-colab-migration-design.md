# Colab Migration (CPU / GPU / TPU) — Design

> Enables fine-tuning and revising the whole stack on Google Colab with real
> compute. Builds on the completed M0–M3 pipeline.

## Context

The project (M0 pipeline, M1 TS VQ-VAE, redefined M2 staged fusion tokenizer,
metrics/plots, M3 Llama-3 predictor; 91 tests) outgrew local CPU — the M2 fusion
stage alone took ~10 min. We move training to Colab while keeping the code, tests
and version control as the source of truth.

**The loop:** edit locally → commit → push to GitHub → `!git pull` in Colab →
train on GPU/TPU → checkpoints/metrics/plots land in Google Drive → inspect inline
→ tweak → repeat.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Code transport | **GitHub remote** `https://github.com/hockper/Quant-AI-2026` (pushed; local supersedes) |
| Artifact persistence | **Mount Google Drive** for `artifacts/` (survives session death) |
| Resume | **Explicit `--resume`** flag (never silently continue an old run) |
| Runtime | **Full three-way: CPU / GPU / TPU** (TPU via `torch_xla`) |
| Notebook style | Drives the **Python API** directly (functions return metrics; inline plots) |

## Runtime abstraction — `bubble_bi/runtime.py` (new)

PyTorch treats TPU fundamentally differently: it needs `torch_xla`, a different
optimizer step, explicit `mark_step()`, and **no CUDA `GradScaler`**. A single
module isolates all of it:

```python
detect_runtime() -> "tpu" | "cuda" | "cpu"
    # tpu  : torch_xla importable AND an XLA device is available
    # cuda : torch.cuda.is_available()
    # cpu  : otherwise

resolve_device(name) -> torch.device
    # "auto" -> detect_runtime(); "tpu" -> xm.xla_device(); else torch.device(name)

is_xla(device) -> bool                       # device.type == "xla"

optimizer_step(opt, scaler, device) -> None
    # xla        -> xm.optimizer_step(opt)
    # cuda + amp -> scaler.step(opt); scaler.update()
    # otherwise  -> opt.step()

mark_step(device) -> None                    # xla -> xm.mark_step(); else no-op
save_state(state, path, device) -> None      # xla -> xm.save(); else torch.save()
```

## Trainer changes (runtime-aware)

- **AMP is CUDA-only.** Today `GradScaler(device.type)` / `autocast(device.type)`
  would break on an XLA device. Guard both: `use_amp = cfg.amp and device.type ==
  "cuda"`, build the scaler with `"cuda"` (disabled when not CUDA), and only enter
  `autocast` on CUDA.
- **Optimizer step** goes through `runtime.optimizer_step(...)` (replacing the
  direct `scaler.step`/`scaler.update`). Gradient clipping still runs after
  `scaler.unscale_` on CUDA; on XLA/CPU it clips the raw grads.
- **`mark_step(device)`** after each step (no-op off TPU).
- **Checkpoints** save via `runtime.save_state(...)` (`xm.save` on XLA).
- `resolve_device` moves to `runtime.py`; `trainer.py` re-exports it so existing
  imports (`from bubble_bi.train.trainer import resolve_device`) keep working.

**Note:** DataLoaders are used unchanged on TPU (tensors are moved with `.to(device)`).
`MpDeviceLoader` is a throughput optimization, not a correctness requirement —
deliberately out of scope.

## `--resume`

The `Trainer` already persists model + optimizer + scaler + RNG + global step +
standardizer and can `load_checkpoint`. The CLI never used it. Each `train_*`
function gains `resume: bool = False`; when true and the stage's `last.pt` exists,
it loads that state before `train()`, so `global_step` continues. `main` gains a
`--resume` flag. Without it, training starts fresh — so fine-tuning experiments
never silently inherit old weights.

## `bubble_bi/colab.py`

```python
load_colab_config(config_path, drive_root, **overrides) -> Config
    # loads the YAML, then points data.raw_dir/cache_dir at
    #   <drive_root>/artifacts/{raw,cache}, sets train.device = "auto",
    #   and applies any **overrides (e.g. batch_size=256, max_steps=5000)
```
No `colab_*.yaml` duplicates to keep in sync — the notebook overrides in memory.

## Drive layout

```
/content/drive/MyDrive/bubble_bi/artifacts/
    raw/     # ingested parquet (ingest once)
    cache/   # panel.npz, tokens.npz, checkpoints_{,cs,fusion,predictor}/, runs/
```

## `notebooks/bubble_bi_colab.ipynb` (committed)

1. Mount Drive; `git clone` (or `pull`) `Quant-AI-2026`.
2. `pip install -r requirements.txt` — **torch is deliberately absent from
   requirements**, so Colab's pre-installed CUDA/XLA torch is preserved.
3. Print `detect_runtime()` and the device.
4. **Run `pytest`** — the migration's verification gate (91 tests on Colab's
   Python 3.11/3.12 + pandas 2.x, vs our 3.14 + pandas 3.0).
5. Pipeline stages (each one cell, calling the Python API):
   `ingest → build-panel → baseline → train-tokenizer → train-cs → train-fusion
   → tokenize → train-predictor` (+ the matching `eval-*`).
6. Inspection: `plot_run` / `plot_compare` inline.
7. Fine-tuning cells: override hyperparameters, give the run a new `run_name`,
   compare curves.

## Error handling & pitfalls

- **TPU without torch_xla:** `detect_runtime()` only reports `"tpu"` if
  `torch_xla` imports *and* an XLA device is obtainable; otherwise it degrades to
  cuda/cpu rather than crashing.
- **AMP on non-CUDA:** guarded (see Trainer changes) — this is the concrete bug
  that would otherwise break the TPU path.
- **Resume mismatch:** if the checkpoint's architecture differs from the config,
  `load_state_dict` raises loudly (better than silently wrong weights).
- **Drive I/O:** checkpoints are 8–20 MB and written every `ckpt_every` steps —
  fine over Drive.
- **pandas/py version drift:** verified by running the test suite on Colab.

## Testing (TDD)

- `detect_runtime()` returns `"cpu"` on this machine; `resolve_device("cpu"/"auto")`
  works; `is_xla` is False for CPU.
- **XLA dispatch without a TPU:** with a *stubbed* `xm` module and a fake
  `device.type == "xla"`, assert `optimizer_step` calls `xm.optimizer_step`,
  `mark_step` calls `xm.mark_step`, and `save_state` calls `xm.save`. This proves
  the branching without owning a TPU.
- **CUDA/CPU dispatch:** `optimizer_step` calls `scaler.step` when CUDA+amp, else
  `opt.step()`.
- **`--resume`:** train N steps → construct a fresh Trainer → resume → `global_step`
  continues from N and training proceeds to 2N.
- **`load_colab_config`:** rewrites `raw_dir`/`cache_dir` under the drive root and
  applies overrides.
- The existing 91 tests stay green (Trainer refactor must not regress them).

## Verification (end-to-end)

1. Local: `pytest` green (91 + new tests).
2. Push; in Colab open `notebooks/bubble_bi_colab.ipynb`.
3. Cell prints the detected runtime (switch Colab runtime GPU↔CPU↔TPU to confirm
   each is reported and used).
4. `pytest` passes **inside Colab** (proves the Python/pandas port).
5. Run the pipeline on GPU; confirm checkpoints/metrics appear under Drive.
6. Kill a training cell mid-run, re-run with `resume=True`, confirm it continues
   from the saved step rather than restarting.

## Out of scope

`MpDeviceLoader` TPU throughput tuning; multi-core TPU (`xmp.spawn`); Docker;
CI. The TPU path is correctness-focused; GPU remains the primary target (our
models are small, so a T4/A100 will likely outperform a TPU here).
