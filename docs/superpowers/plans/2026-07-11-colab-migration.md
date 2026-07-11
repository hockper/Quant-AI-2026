# Colab Migration (CPU / GPU / TPU) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the project run correctly on Colab's CPU, GPU, and TPU runtimes; add `--resume` for interrupted GPU runs; ship a Colab notebook that drives the pipeline from Google Drive.

**Architecture:** A new `bubble_bi/runtime.py` isolates every device difference (`detect_runtime`, `resolve_device`, `optimizer_step`, `mark_step`, `save_state`). The `Trainer` routes its optimizer step / checkpoint save through it and guards AMP to CUDA-only (it would otherwise break on XLA). The CLI gains `--resume`. A `colab.py` helper points configs at Drive, and a committed notebook runs the whole pipeline.

**Tech Stack:** PyTorch 2.13 (+ `torch_xla` on Colab TPU only), existing `bubble_bi` package.

## Global Constraints

- Run everything as `.venv/bin/python` from repo root `/home/hockper/Documents/Code/Bubble Bi`.
- **AMP is CUDA-only.** `GradScaler`/`autocast` must never be constructed with an XLA device type.
- **TPU via `torch_xla`**: `xm.xla_device()`, `xm.optimizer_step(opt)`, `xm.mark_step()`, `xm.save()`. No `MpDeviceLoader`, no `xmp.spawn` (out of scope).
- `detect_runtime()` degrades gracefully: reports `"tpu"` only if `torch_xla` imports **and** an XLA device is obtainable; else `"cuda"`; else `"cpu"`.
- **`torch` stays OUT of `requirements.txt`** so Colab's pre-installed CUDA/XLA torch is preserved.
- `--resume` is **explicit** — without it, training starts fresh.
- The existing 91 tests must stay green.
- TDD; frequent commits.

---

### Task 1: `bubble_bi/runtime.py`

**Files:**
- Create: `bubble_bi/runtime.py`
- Test: `tests/test_runtime.py`

**Interfaces:**
- Produces, in `bubble_bi/runtime.py`:
  - `detect_runtime() -> str` (`"tpu"|"cuda"|"cpu"`)
  - `resolve_device(name: str = "auto") -> torch.device`
  - `is_xla(device) -> bool`
  - `optimizer_step(opt, scaler, device) -> None`
  - `mark_step(device) -> None`
  - `save_state(state: dict, path: str, device) -> None`
  - `_xla()` — internal; returns the `torch_xla.core.xla_model` module or `None` (monkeypatched in tests).

- [ ] **Step 1: Write the failing test**

`tests/test_runtime.py`:
```python
import types

import torch

from bubble_bi import runtime


def test_detect_runtime_is_cpu_here():
    # this dev machine has no CUDA and no torch_xla
    assert runtime.detect_runtime() == "cpu"


def test_resolve_device_cpu_and_auto():
    assert runtime.resolve_device("cpu").type == "cpu"
    assert runtime.resolve_device("auto").type in {"cpu", "cuda"}
    assert runtime.is_xla(torch.device("cpu")) is False


def test_optimizer_step_cpu_calls_opt_step():
    stepped = []
    opt = types.SimpleNamespace(step=lambda: stepped.append(1))
    runtime.optimizer_step(opt, None, torch.device("cpu"))
    assert stepped == [1]


def test_optimizer_step_cuda_uses_scaler():
    calls = []
    scaler = types.SimpleNamespace(is_enabled=lambda: True,
                                   step=lambda o: calls.append("step"),
                                   update=lambda: calls.append("update"))
    runtime.optimizer_step(object(), scaler, torch.device("cuda"))
    assert calls == ["step", "update"]


def test_xla_dispatch_with_stubbed_xm(monkeypatch, tmp_path):
    """Proves the TPU code path without owning a TPU."""
    calls = []
    fake_xm = types.SimpleNamespace(
        optimizer_step=lambda opt: calls.append("optimizer_step"),
        mark_step=lambda: calls.append("mark_step"),
        save=lambda state, path: calls.append("save"),
    )
    monkeypatch.setattr(runtime, "_xla", lambda: fake_xm)
    dev = types.SimpleNamespace(type="xla")          # stand-in XLA device
    assert runtime.is_xla(dev) is True
    runtime.optimizer_step(object(), None, dev)
    runtime.mark_step(dev)
    runtime.save_state({"a": 1}, str(tmp_path / "x.pt"), dev)
    assert calls == ["optimizer_step", "mark_step", "save"]


def test_save_state_cpu_writes_file(tmp_path):
    p = tmp_path / "s.pt"
    runtime.save_state({"a": 1}, str(p), torch.device("cpu"))
    assert p.exists()
    assert torch.load(str(p), weights_only=False)["a"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_runtime.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'bubble_bi.runtime'`).

- [ ] **Step 3: Write implementation**

`bubble_bi/runtime.py`:
```python
from __future__ import annotations

import torch


def _xla():
    """Return torch_xla's xla_model module, or None when unavailable."""
    try:
        import torch_xla.core.xla_model as xm  # noqa: PLC0415
        return xm
    except Exception:
        return None


def detect_runtime() -> str:
    xm = _xla()
    if xm is not None:
        try:
            xm.xla_device()
            return "tpu"
        except Exception:
            pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = detect_runtime()
    if name == "tpu":
        xm = _xla()
        if xm is None:
            raise RuntimeError("device 'tpu' requested but torch_xla is not installed")
        return xm.xla_device()
    return torch.device(name)


def is_xla(device) -> bool:
    return getattr(device, "type", None) == "xla"


def optimizer_step(opt, scaler, device) -> None:
    if is_xla(device):
        _xla().optimizer_step(opt)
    elif getattr(device, "type", None) == "cuda" and scaler is not None and scaler.is_enabled():
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()


def mark_step(device) -> None:
    if is_xla(device):
        _xla().mark_step()


def save_state(state: dict, path: str, device) -> None:
    if is_xla(device):
        _xla().save(state, path)
    else:
        torch.save(state, path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_runtime.py -v`
Expected: PASS (6 tests). The stubbed-`xm` test proves the TPU dispatch without a TPU.

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/runtime.py tests/test_runtime.py
git commit -m "feat: runtime abstraction (detect CPU/GPU/TPU, XLA-aware dispatch)"
```

---

### Task 2: Make the Trainer runtime-aware

**Files:**
- Modify: `bubble_bi/train/trainer.py`
- Test: `tests/test_trainer_runtime.py`

**Interfaces:**
- Consumes: `runtime.detect_runtime/resolve_device/optimizer_step/mark_step/save_state` (Task 1).
- Produces: `Trainer` uses the runtime dispatchers; AMP is CUDA-only (`use_amp = cfg.amp and device.type == "cuda"`; the `GradScaler` is always built with `"cuda"` and disabled off-CUDA; `autocast` is entered only when `use_amp`). `trainer.resolve_device` is re-exported from `runtime` so existing imports keep working.

- [ ] **Step 1: Write the failing test**

`tests/test_trainer_runtime.py`:
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bubble_bi.config import TrainConfig
from bubble_bi.train.trainer import Trainer, set_seed, resolve_device


class _M(nn.Module):
    def __init__(self, d=6):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.dead_code_reinit_every = 10 ** 9

    def forward(self, batch):
        x = batch["block"]
        mse = ((self.lin(x) - x) ** 2).mean()
        return {"loss": mse, "recon_loss": mse, "perplexity": torch.tensor(1.0)}

    def reinit_dead_codes(self, out):
        pass


class _DS(Dataset):
    def __init__(self, n=32, N=5, L=4, D=6):
        self.b = torch.randn(n, N, L, D)
        self.v = torch.ones(n, N, dtype=torch.bool)

    def __len__(self):
        return self.b.shape[0]

    def __getitem__(self, i):
        return {"block": self.b[i], "valid": self.v[i]}


def _loaders():
    ld = DataLoader(_DS(), batch_size=8, shuffle=True, drop_last=True)
    return {"train": ld, "val": ld, "test": ld}


def test_resolve_device_reexported_from_trainer():
    assert resolve_device("cpu").type == "cpu"


def test_amp_disabled_off_cuda(tmp_path):
    set_seed(0)
    cfg = TrainConfig(max_steps=2, batch_size=8, val_every=2, ckpt_every=2,
                      device="cpu", amp=True)          # amp requested but device is CPU
    tr = Trainer(_M(), _loaders(), cfg, str(tmp_path), device="cpu")
    assert tr.use_amp is False                          # must not enable AMP off CUDA
    assert tr.scaler.is_enabled() is False
    tr.train()                                          # still trains fine
    assert tr.global_step == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_trainer_runtime.py -v`
Expected: FAIL — `test_amp_disabled_off_cuda` fails because `GradScaler(self.device.type, ...)` is currently built with the raw device type (and `use_amp` logic/scaler construction isn't the guarded form). If it happens to pass, the implementation step still applies the XLA-safe form.

- [ ] **Step 3: Edit `bubble_bi/train/trainer.py`**

Replace the imports block at the top:
```python
from __future__ import annotations

import contextlib
import random
from pathlib import Path

import numpy as np
import torch

from bubble_bi.runtime import (detect_runtime, is_xla, mark_step, optimizer_step,
                               resolve_device, save_state)
from bubble_bi.train.metrics_logger import MetricsLogger
```

Delete the old local `resolve_device` function from `trainer.py` (it now comes from `runtime`; the import above re-exports it for `from bubble_bi.train.trainer import resolve_device`).

In `Trainer.__init__`, replace the AMP/scaler lines:
```python
        self.use_amp = bool(cfg.amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
```
with:
```python
        self.use_amp = bool(cfg.amp) and self.device.type == "cuda"
        # Always construct the scaler for "cuda"; disabled off-CUDA so it is a no-op
        # (constructing it with an XLA device type would fail).
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
```

In `Trainer.train`, replace the forward/backward/step block:
```python
                xb = _to_device(xb, self.device)
                self.opt.zero_grad()
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    out = model(xb)
                    loss = out["loss"]
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.global_step += 1
```
with:
```python
                xb = _to_device(xb, self.device)
                self.opt.zero_grad()
                amp_ctx = (torch.autocast(device_type="cuda") if self.use_amp
                           else contextlib.nullcontext())
                with amp_ctx:
                    out = model(xb)
                    loss = out["loss"]
                self.scaler.scale(loss).backward()
                if self.use_amp:
                    self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer_step(self.opt, self.scaler, self.device)
                mark_step(self.device)
                self.global_step += 1
```

In `Trainer.save_checkpoint`, replace the final line `torch.save(state, path)` with:
```python
        save_state(state, path, self.device)
```

- [ ] **Step 4: Run the new test + the whole trainer suite**

Run: `.venv/bin/python -m pytest tests/test_trainer_runtime.py tests/test_trainer.py tests/test_trainer_dual.py tests/test_trainer_metrics.py tests/test_trainer_frozen.py -v`
Expected: PASS — CPU training behaves exactly as before (scaler disabled ⇒ `scale()` is identity and `optimizer_step` calls `opt.step()`), and AMP can never be enabled off CUDA.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all previous tests still green.

- [ ] **Step 6: Commit**

```bash
git add bubble_bi/train/trainer.py tests/test_trainer_runtime.py
git commit -m "feat: runtime-aware Trainer (CUDA-only AMP, XLA optimizer step/save)"
```

---

### Task 3: `--resume` on the train commands

**Files:**
- Modify: `bubble_bi/cli.py`
- Test: `tests/test_resume.py`

**Interfaces:**
- Consumes: `Trainer.load_checkpoint(path)` (already exists).
- Produces: `train_tokenizer(cfg, run_name=None, resume=False)`, `train_cs(...)`, `train_fusion(...)`, `train_predictor(...)` all accept `resume: bool = False`; `main` gains a `--resume` flag threaded into all four.

- [ ] **Step 1: Write the failing test**

`tests/test_resume.py`:
```python
import json

import numpy as np
import pandas as pd

from bubble_bi.config import Config, DataConfig, FeatureConfig, ModelConfig, TrainConfig
from bubble_bi.cli import train_tokenizer


def _write_raw(raw, n=320, N=6):
    rng = np.random.default_rng(0)
    for k in range(N):
        dates = pd.bdate_range("2015-01-01", periods=n)
        c = pd.Series(100 + np.cumsum(rng.normal(size=n)), index=dates)
        v = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
        df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": v}, index=dates)
        df.index.name = "date"
        df.to_parquet(f"{raw}/T{k}.parquet")


def _cfg(tmp_path, max_steps):
    return Config(
        data=DataConfig(tickers=[f"T{k}" for k in range(6)], raw_dir=str(tmp_path / "raw"),
                        cache_dir=str(tmp_path / "cache"), min_history=50),
        features=FeatureConfig(),
        model=ModelConfig(p=4, d_model=16, codebook_size=16, enc_layers=1,
                          dec_layers=1, heads=2, ff=32, dropout=0.0),
        train=TrainConfig(max_steps=max_steps, batch_size=8, val_every=100,
                          ckpt_every=1, log_every=1, device="cpu", amp=False),
    )


def _train_steps_logged(run_dir):
    lines = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    return [r["step"] for r in lines if r["phase"] == "train"]


def test_resume_continues_instead_of_restarting(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "cache").mkdir()
    _write_raw(tmp_path / "raw")
    run = tmp_path / "cache" / "runs" / "a"

    m1 = train_tokenizer(_cfg(tmp_path, 4), run_name="a")
    assert m1["step"] == 4
    assert _train_steps_logged(run) == [1, 2, 3, 4]

    # resume with a bigger budget: must do ONLY steps 5 and 6 (not restart at 1)
    m2 = train_tokenizer(_cfg(tmp_path, 6), run_name="a", resume=True)
    assert m2["step"] == 6
    assert _train_steps_logged(run) == [1, 2, 3, 4, 5, 6]   # a restart would append 1..6 again
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_resume.py -v`
Expected: FAIL — `train_tokenizer() got an unexpected keyword argument 'resume'`.

- [ ] **Step 3: Edit `bubble_bi/cli.py`**

Add a helper (below `_write_eval_json`):
```python
def _maybe_resume(trainer, ckpt_dir: Path, resume: bool) -> None:
    ckpt = Path(ckpt_dir) / "last.pt"
    if resume and ckpt.exists():
        trainer.load_checkpoint(str(ckpt))
        print(f"resumed from {ckpt} at step {trainer.global_step}")
```

Change the four training functions. For `train_tokenizer`, change the signature and add the resume call after the `Trainer(...)` construction:
```python
def train_tokenizer(cfg: Config, run_name: str | None = None, resume: bool = False) -> dict:
```
and immediately after `trainer = Trainer(...)` insert:
```python
    _maybe_resume(trainer, ckpt_dir, resume)
```

Apply the identical two edits to `train_cs`, `train_fusion`, and `train_predictor` (each already defines a local `ckpt_dir` and a `trainer`):
```python
def train_cs(cfg: Config, run_name: str | None = None, resume: bool = False) -> dict:
    ...
    trainer = Trainer(...)
    _maybe_resume(trainer, ckpt_dir, resume)
    ...

def train_fusion(cfg: Config, run_name: str | None = None, resume: bool = False) -> dict:
    ...
    trainer = Trainer(...)
    _maybe_resume(trainer, ckpt_dir, resume)
    ...

def train_predictor(cfg: Config, run_name: str | None = None, resume: bool = False) -> dict:
    ...
    trainer = Trainer(...)
    _maybe_resume(trainer, ckpt_dir, resume)
    ...
```

In `main`, add the flag and thread it through:
```python
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run from its last checkpoint")
```
and update the four dispatch branches:
```python
    elif args.command == "train-tokenizer":
        train_tokenizer(cfg, run_name=run_name, resume=args.resume)
    ...
    elif args.command == "train-cs":
        train_cs(cfg, run_name=run_name, resume=args.resume)
    ...
    elif args.command == "train-fusion":
        train_fusion(cfg, run_name=run_name, resume=args.resume)
    ...
    elif args.command == "train-predictor":
        train_predictor(cfg, run_name=run_name, resume=args.resume)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_resume.py -v`
Expected: PASS. The logged step sequence `[1..6]` (rather than `[1..4, 1..6]`) proves it continued rather than restarting.

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/cli.py tests/test_resume.py
git commit -m "feat: --resume for interrupted training runs"
```

---

### Task 4: `bubble_bi/colab.py`

**Files:**
- Create: `bubble_bi/colab.py`
- Test: `tests/test_colab_config.py`

**Interfaces:**
- Consumes: `load_config`, `Config`.
- Produces: `load_colab_config(config_path: str, drive_root: str, **overrides) -> Config` — points `data.raw_dir`/`data.cache_dir` at `<drive_root>/artifacts/{raw,cache}`, sets `train.device = "auto"`, and applies `**overrides` to whichever config section owns each field (raises `AttributeError` on an unknown field).

- [ ] **Step 1: Write the failing test**

`tests/test_colab_config.py`:
```python
import pytest

from bubble_bi.colab import load_colab_config


def _write_cfg(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  tickers: [AAPL]\ntrain:\n  batch_size: 64\n  device: cpu\n")
    return str(p)


def test_points_at_drive_and_auto_device(tmp_path):
    cfg = load_colab_config(_write_cfg(tmp_path), "/drive/bubble_bi")
    assert cfg.data.raw_dir == "/drive/bubble_bi/artifacts/raw"
    assert cfg.data.cache_dir == "/drive/bubble_bi/artifacts/cache"
    assert cfg.train.device == "auto"


def test_overrides_apply_to_the_right_section(tmp_path):
    cfg = load_colab_config(_write_cfg(tmp_path), "/drive/bubble_bi",
                            batch_size=256, max_steps=5000, d_model=256)
    assert cfg.train.batch_size == 256
    assert cfg.train.max_steps == 5000
    assert cfg.model.d_model == 256


def test_unknown_override_raises(tmp_path):
    with pytest.raises(AttributeError):
        load_colab_config(_write_cfg(tmp_path), "/drive/bubble_bi", nonsense=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_colab_config.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'bubble_bi.colab'`).

- [ ] **Step 3: Write implementation**

`bubble_bi/colab.py`:
```python
from __future__ import annotations

from pathlib import Path

from bubble_bi.config import Config, load_config


def load_colab_config(config_path: str, drive_root: str, **overrides) -> Config:
    """Load a config, point its paths at Google Drive, and apply overrides.

    Keeps a single set of YAML configs — the notebook overrides in memory rather
    than maintaining parallel `colab_*.yaml` files.
    """
    cfg = load_config(config_path)
    root = Path(drive_root)
    cfg.data.raw_dir = str(root / "artifacts" / "raw")
    cfg.data.cache_dir = str(root / "artifacts" / "cache")
    cfg.train.device = "auto"

    for key, value in overrides.items():
        for section in (cfg.train, cfg.model, cfg.data, cfg.features, cfg.splits):
            if hasattr(section, key):
                setattr(section, key, value)
                break
        else:
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                raise AttributeError(f"unknown config field: {key!r}")
    return cfg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_colab_config.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/colab.py tests/test_colab_config.py
git commit -m "feat: load_colab_config (Drive paths + in-memory overrides)"
```

---

### Task 5: Colab notebook + README

**Files:**
- Create: `notebooks/bubble_bi_colab.ipynb`
- Create: `README.md`

**Interfaces:**
- Consumes: `load_colab_config` (Task 4), `detect_runtime`/`resolve_device` (Task 1), the CLI Python API (`run_ingest`, `build_panel_from_raw`, `run_baseline`, `train_tokenizer`, `train_cs`, `train_fusion`, `tokenize_panel`, `train_predictor`, the `eval_*` functions, `plot_metrics`).
- Produces: a runnable Colab notebook and a repo README documenting the workflow.

- [ ] **Step 1: Write `notebooks/bubble_bi_colab.ipynb`**

```python
import json
from pathlib import Path

REPO = "https://github.com/hockper/Quant-AI-2026.git"

cells = [
    ("markdown", [
        "# Bubble Bi — Colab runner\n",
        "\n",
        "Runs the full world-model pipeline (data -> tokenizers -> next-token predictor) on Colab.\n",
        "\n",
        "**Pick a runtime first:** Runtime -> Change runtime type -> CPU / GPU (T4) / TPU.\n",
        "The code detects it automatically. GPU is the recommended target.\n",
    ]),
    ("markdown", ["## 1. Mount Drive + get the code"]),
    ("code", [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "DRIVE = '/content/drive/MyDrive/bubble_bi'\n",
        "!mkdir -p {DRIVE}/artifacts/raw {DRIVE}/artifacts/cache\n",
        "print('drive ready:', DRIVE)\n",
    ]),
    ("code", [
        f"REPO = '{REPO}'\n",
        "import os\n",
        "if os.path.isdir('/content/Quant-AI-2026'):\n",
        "    !cd /content/Quant-AI-2026 && git pull --ff-only\n",
        "else:\n",
        "    !git clone {REPO} /content/Quant-AI-2026\n",
        "%cd /content/Quant-AI-2026\n",
    ]),
    ("code", [
        "# NOTE: torch is deliberately NOT in requirements.txt so Colab's\n",
        "# pre-installed CUDA/XLA torch is preserved.\n",
        "!pip install -q -r requirements.txt\n",
    ]),
    ("markdown", ["## 2. Runtime check + tests (the migration's proof)"]),
    ("code", [
        "import torch\n",
        "from bubble_bi.runtime import detect_runtime, resolve_device\n",
        "\n",
        "rt = detect_runtime()\n",
        "print('torch  :', torch.__version__)\n",
        "print('runtime:', rt)\n",
        "print('device :', resolve_device('auto'))\n",
        "if rt == 'cpu':\n",
        "    print('\\nTip: Runtime -> Change runtime type -> GPU for real speed.')\n",
    ]),
    ("code", [
        "# 91+ tests. If these pass here, the port to Colab's Python/pandas is proven.\n",
        "!python -m pytest -q\n",
    ]),
    ("markdown", ["## 3. Config (points at Drive; override anything here)"]),
    ("code", [
        "from bubble_bi.colab import load_colab_config\n",
        "\n",
        "DRIVE = '/content/drive/MyDrive/bubble_bi'\n",
        "cfg = load_colab_config('configs/m3.yaml', DRIVE)   # m3 config covers every stage\n",
        "print(cfg.data.raw_dir)\n",
        "print(cfg.data.cache_dir)\n",
        "print('device:', cfg.train.device)\n",
    ]),
    ("markdown", ["## 4. Data (run once; cached on Drive)"]),
    ("code", [
        "from bubble_bi.cli import run_ingest, build_panel_from_raw, run_baseline\n",
        "\n",
        "run_ingest(cfg)                 # yfinance -> Drive parquet\n",
        "panel = build_panel_from_raw(cfg)\n",
        "print('panel:', panel.features.shape)\n",
        "run_baseline(cfg)               # ridge RankIC floor\n",
    ]),
    ("markdown", [
        "## 5. Staged tokenizer\n",
        "\n",
        "Phase 1 TS -> Phase 2 CS -> Phase 3 fusion (frozen encoders). Pass `resume=True`\n",
        "to continue a run that a Colab disconnect killed.\n",
    ]),
    ("code", [
        "from bubble_bi.cli import train_tokenizer, train_cs, train_fusion\n",
        "\n",
        "cfg.train.batch_size = 256      # scale up on GPU\n",
        "cfg.train.max_steps = 2000\n",
        "\n",
        "train_tokenizer(cfg, run_name='ts_gpu')            # add resume=True after a disconnect\n",
        "train_cs(cfg, run_name='cs_gpu')\n",
        "train_fusion(cfg, run_name='fusion_gpu')\n",
    ]),
    ("code", [
        "from bubble_bi.cli import eval_tokenizer, eval_cs, eval_fusion\n",
        "\n",
        "eval_tokenizer(cfg, run_name='ts_gpu')\n",
        "eval_cs(cfg, run_name='cs_gpu')\n",
        "eval_fusion(cfg, run_name='fusion_gpu')\n",
    ]),
    ("markdown", ["## 6. Token grid + next-token predictor"]),
    ("code", [
        "from bubble_bi.cli import tokenize_panel, train_predictor, eval_predictor\n",
        "\n",
        "tokenize_panel(cfg)                                 # frozen tokenizer -> tokens.npz\n",
        "train_predictor(cfg, run_name='pred_gpu')           # resume=True to continue\n",
        "eval_predictor(cfg, run_name='pred_gpu')\n",
    ]),
    ("markdown", ["## 7. Plots + comparing experiments"]),
    ("code", [
        "from bubble_bi.cli import plot_metrics\n",
        "from IPython.display import Image, display\n",
        "\n",
        "paths = plot_metrics(cfg, ['pred_gpu'])\n",
        "for p in paths:\n",
        "    display(Image(filename=p))\n",
    ]),
    ("markdown", [
        "## 8. Fine-tuning loop\n",
        "\n",
        "Change hyperparameters, give the run a NEW name, then overlay the curves:\n",
    ]),
    ("code", [
        "cfg2 = load_colab_config('configs/m3.yaml', DRIVE,\n",
        "                         pred_layers=6, d_model=256, batch_size=256, max_steps=4000)\n",
        "train_predictor(cfg2, run_name='pred_big')\n",
        "eval_predictor(cfg2, run_name='pred_big')\n",
        "\n",
        "plot_metrics(cfg, ['pred_gpu', 'pred_big'])   # overlaid comparison\n",
    ]),
]


def _cell(kind, src):
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": src}
    return {"cell_type": "code", "metadata": {}, "source": src,
            "execution_count": None, "outputs": []}


nb = {
    "cells": [_cell(k, s) for k, s in cells],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

Path("notebooks").mkdir(exist_ok=True)
Path("notebooks/bubble_bi_colab.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote notebooks/bubble_bi_colab.ipynb")
```

Run that snippet once (e.g. `.venv/bin/python - <<'PY' … PY`) to generate the notebook file.

- [ ] **Step 2: Verify the notebook is valid JSON/nbformat**

Run:
```bash
.venv/bin/python -c "import json;nb=json.load(open('notebooks/bubble_bi_colab.ipynb'));print(nb['nbformat'], len(nb['cells']), 'cells')"
```
Expected: prints `4 <N> cells` with no exception.

- [ ] **Step 3: Write `README.md`**

```markdown
# Bubble Bi — STORM-inspired trading world model

A discrete world model for markets: a staged VQ-VAE tokenizer turns each
`(stock, day)` into one market-aware token, and a Llama-3-style causal
Transformer predicts the next token.

## Pipeline

| Stage | Command | What it does |
|---|---|---|
| Data | `ingest` → `build-panel` | yfinance → leak-free `[T, N, D]` panel |
| Baseline | `baseline` | walk-forward ridge (RankIC floor) |
| Phase 1 | `train-tokenizer` | TS VQ-VAE (per-stock window) |
| Phase 2 | `train-cs` | windowed cross-sectional VQ-VAE (market) |
| Phase 3 | `train-fusion` | fuse frozen encoders → **one token per stock-day** |
| Tokens | `tokenize` | frozen tokenizer → token grid |
| Predictor | `train-predictor` | Llama-3 GPT over token sequences |
| Analysis | `plot-metrics` | loss/perplexity curves, multi-run overlays |

Add `--resume` to any `train-*` command to continue an interrupted run.

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
`/content/drive/MyDrive/bubble_bi/artifacts/`.

`torch` is intentionally absent from `requirements.txt` so Colab's pre-installed
CUDA/XLA build is preserved.

## Docs

Designs and implementation plans live in `docs/superpowers/{specs,plans}/`.
```

- [ ] **Step 4: Run the full suite (nothing should break)**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests green.

- [ ] **Step 5: Commit**

```bash
git add notebooks/bubble_bi_colab.ipynb README.md
git commit -m "docs: Colab notebook + README"
```

---

### Task 6: Push and verify on Colab (manual)

**Files:** none.

- [ ] **Step 1: Push the branch's work to GitHub**

After merging to `main` (see the finishing step), run:
```bash
git push origin main
```
Expected: `hockper/Quant-AI-2026` updated.

- [ ] **Step 2: Verify on Colab — GPU**

Open `notebooks/bubble_bi_colab.ipynb` from the repo in Colab, set Runtime → GPU, and run cells 1–2.
Expected: `runtime: cuda`, `device: cuda:0`, and **`pytest` passes** (this is the proof that the code ports to Colab's Python 3.11/3.12 + pandas 2.x).

- [ ] **Step 3: Verify on Colab — CPU and TPU**

Switch Runtime → CPU, re-run the runtime cell → expect `runtime: cpu`.
Switch Runtime → TPU, re-run → expect `runtime: tpu` and a `device: xla:*`. Then run
a short training cell (e.g. `cfg.train.max_steps = 20; train_tokenizer(cfg, run_name='tpu_smoke')`)
to confirm the XLA path actually trains.

- [ ] **Step 4: Verify resume**

Start `train_predictor(cfg, run_name='pred_gpu')` with a large `max_steps`, interrupt the cell,
then re-run with `resume=True`.
Expected: prints `resumed from …/last.pt at step <N>` and continues rather than restarting.

- [ ] **Step 5: Record the outcome**

Append the observed runtime detections (cpu/cuda/tpu), whether Colab `pytest` passed,
and any version fixes needed to the design doc
(`docs/superpowers/specs/2026-07-11-colab-migration-design.md`), then commit.

---

## Self-Review

**Spec coverage:**
- `runtime.py` (detect_runtime, resolve_device, is_xla, optimizer_step, mark_step, save_state) → Task 1 ✅
- Trainer runtime-aware: **CUDA-only AMP** (the concrete XLA-breaking bug), optimizer-step dispatch, `mark_step`, `save_state`, `resolve_device` re-export → Task 2 ✅
- `--resume` on all four train commands + `main` flag → Task 3 ✅
- `load_colab_config` (Drive paths, in-memory overrides, no `colab_*.yaml` duplicates) → Task 4 ✅
- Colab notebook (mount → pull → install → **runtime print** → **pytest** → pipeline → plots → fine-tuning) + README → Task 5 ✅
- GitHub remote (already pushed) + Colab verification of CPU/GPU/TPU + resume → Task 6 ✅
- `torch` stays out of `requirements.txt` → unchanged; asserted in the notebook comment and README ✅
- Out of scope (MpDeviceLoader, xmp.spawn, Docker, CI) → not planned, as specified ✅

**Placeholder scan:** none. Task 6 is explicitly manual (it needs a Colab runtime, which cannot be exercised locally).

**Type consistency:** `runtime.optimizer_step(opt, scaler, device)` / `mark_step(device)` / `save_state(state, path, device)` (Task 1) are called with exactly those signatures in the Trainer (Task 2). `Trainer.load_checkpoint(path)` (existing) is used by `_maybe_resume(trainer, ckpt_dir, resume)` (Task 3). `load_colab_config(config_path, drive_root, **overrides)` (Task 4) is called in the notebook (Task 5). The notebook calls the CLI Python API with the signatures those functions actually have (`train_*(cfg, run_name=..., resume=...)`, `eval_*(cfg, run_name=...)`, `plot_metrics(cfg, [names])`, `tokenize_panel(cfg)`).

**Note on Task 2's failing test:** `test_amp_disabled_off_cuda` may already pass on CPU with the current code (since `GradScaler("cpu", enabled=False)` happens to construct). It is kept as a **regression lock** on the CUDA-only invariant — the substantive change is that the scaler is never built with an XLA device type and `autocast` is never entered off CUDA, which is what makes the TPU path viable.
