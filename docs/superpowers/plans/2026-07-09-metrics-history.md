# Metrics History + Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist a complete per-step history of every training/eval loss component in per-run folders (JSONL + CSV + meta), and render matplotlib PNG plots (single run + multi-run overlay) for analysis.

**Architecture:** A small `MetricsLogger` writes JSONL/CSV/meta into a per-run folder. The `Trainer` logs a train record (every scalar in the model output dict) every `log_every` steps plus a val record every `val_every`. Models surface their per-module commit/diversity so they're recorded. A `viz/plots.py` module renders curves; the CLI adds `--run-name` and a `plot-metrics` command.

**Tech Stack:** matplotlib (Agg backend) + the existing numpy/torch `bubble_bi` package.

## Global Constraints

- Run everything as `.venv/bin/python` from repo root `/home/hockper/Documents/Code/Bubble Bi`.
- No model math changes — only surface already-computed scalars and persist them.
- Checkpoints stay at `artifacts/<cache>/checkpoints/last.pt` (eval path unchanged); metrics live in `artifacts/<cache>/runs/<run_name>/`.
- Back-compat: `Trainer(run_dir=None)` defaults to the checkpoint dir; existing M1/M2 trainer tests keep passing.
- `log_every` default 10; matplotlib forced to the `Agg` backend at import.
- TDD; frequent commits.

---

### Task 0: requirements + `log_every` config

**Files:**
- Modify: `requirements.txt`
- Modify: `bubble_bi/config.py`
- Test: `tests/test_config_metrics.py`

**Interfaces:**
- Produces: `TrainConfig.log_every: int = 10`.

- [ ] **Step 1: Write the failing test**

`tests/test_config_metrics.py`:
```python
from bubble_bi.config import TrainConfig, load_config


def test_train_config_log_every_default():
    assert TrainConfig().log_every == 10


def test_load_config_parses_log_every(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  tickers: [AAPL]\ntrain:\n  log_every: 5\n")
    assert load_config(str(p)).train.log_every == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config_metrics.py -v`
Expected: FAIL (`AttributeError: ... 'log_every'`).

- [ ] **Step 3: Implement**

In `bubble_bi/config.py`, add to `TrainConfig` after `ckpt_every`:
```python
    log_every: int = 10
```

In `requirements.txt`, add a line:
```
matplotlib
```

- [ ] **Step 4: Install matplotlib + run test**

Run: `.venv/bin/python -m pip install matplotlib` then
`.venv/bin/python -m pytest tests/test_config_metrics.py -v`
Expected: install succeeds (cp314 wheels); PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add requirements.txt bubble_bi/config.py tests/test_config_metrics.py
git commit -m "feat: TrainConfig.log_every + matplotlib dependency"
```

---

### Task 1: MetricsLogger

**Files:**
- Create: `bubble_bi/train/metrics_logger.py`
- Test: `tests/test_metrics_logger.py`

**Interfaces:**
- Produces, in `bubble_bi/train/metrics_logger.py`:
  - `class MetricsLogger` — `__init__(run_dir)`; `log(record: dict) -> None` (append JSONL + keep in `self.records`); `to_csv() -> None` (columns = sorted union of keys, blanks for missing); `write_meta(meta: dict) -> None`. Attributes `run_dir: Path`, `records: list[dict]`.

- [ ] **Step 1: Write the failing test**

`tests/test_metrics_logger.py`:
```python
import json

from bubble_bi.train.metrics_logger import MetricsLogger


def test_log_appends_jsonl_and_records(tmp_path):
    ml = MetricsLogger(str(tmp_path))
    ml.log({"phase": "train", "step": 1, "loss": 0.5})
    ml.log({"phase": "val", "step": 1, "val_mse": 0.7})
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["loss"] == 0.5
    assert len(ml.records) == 2


def test_to_csv_uses_union_of_keys(tmp_path):
    ml = MetricsLogger(str(tmp_path))
    ml.log({"step": 1, "loss": 0.5})
    ml.log({"step": 2, "val_mse": 0.7})
    ml.to_csv()
    header = (tmp_path / "metrics.csv").read_text().splitlines()[0]
    assert set(header.split(",")) == {"loss", "step", "val_mse"}


def test_write_meta(tmp_path):
    ml = MetricsLogger(str(tmp_path))
    ml.write_meta({"active_modules": ["ts", "cs"], "max_steps": 10})
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["max_steps"] == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_metrics_logger.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

`bubble_bi/train/metrics_logger.py`:
```python
from __future__ import annotations

import csv
import json
from pathlib import Path


class MetricsLogger:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.records: list[dict] = []

    def log(self, record: dict) -> None:
        self.records.append(record)
        with open(self.jsonl_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    def to_csv(self) -> None:
        if not self.records:
            return
        cols = sorted({k for r in self.records for k in r})
        with open(self.run_dir / "metrics.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for r in self.records:
                writer.writerow(r)

    def write_meta(self, meta: dict) -> None:
        with open(self.run_dir / "meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_metrics_logger.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/train/metrics_logger.py tests/test_metrics_logger.py
git commit -m "feat: MetricsLogger (jsonl + csv + meta)"
```

---

### Task 2: Surface per-module commit/diversity in DualVQVAE

**Files:**
- Modify: `bubble_bi/models/dual_vqvae.py`
- Test: `tests/test_dual_vqvae.py` (add one test)

**Interfaces:**
- Produces: `DualVQVAE.forward` output dict additionally contains `ts_commit`, `ts_diversity`, `cs_commit`, `cs_diversity` (scalar tensors) when the module is active.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_dual_vqvae.py`:
```python
def test_forward_exposes_per_module_commit_diversity():
    model = DualVQVAE(_cfg(), d_in=6, n_stocks=5)
    out = model(_batch())
    for k in ["ts_commit", "ts_diversity", "cs_commit", "cs_diversity"]:
        assert k in out
        assert out[k].ndim == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dual_vqvae.py::test_forward_exposes_per_module_commit_diversity -v`
Expected: FAIL (`KeyError`/assert on missing key).

- [ ] **Step 3: Implement**

In `bubble_bi/models/dual_vqvae.py` `forward`, extend the two `out.update(...)` calls:

TS branch:
```python
            out.update(ts_recon=ts_recon, ts_perplexity=q["perplexity"],
                       ts_commit=q["commit"], ts_diversity=q["diversity"],
                       ts_z_e=z_ts.reshape(B * N, -1).detach())
```
CS branch:
```python
            out.update(cs_recon=cs_recon, cs_perplexity=q["perplexity"],
                       cs_commit=q["commit"], cs_diversity=q["diversity"],
                       cs_z_e=z_cs.detach())
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_dual_vqvae.py -v`
Expected: PASS (7 tests). (`TSVQVAE.forward` already returns `commit`/`diversity`, so single-tokenizer runs are covered too.)

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/models/dual_vqvae.py tests/test_dual_vqvae.py
git commit -m "feat: surface per-module commit/diversity for metrics history"
```

---

### Task 3: Trainer dense logging + run_dir

**Files:**
- Modify: `bubble_bi/train/trainer.py`
- Test: `tests/test_trainer_metrics.py`

**Interfaces:**
- Consumes: `MetricsLogger` (Task 1); `DualVQVAE` (for the test).
- Produces: `Trainer.__init__(..., run_dir=None)` (defaults to `ckpt_dir`); attribute `Trainer.logger: MetricsLogger`; module-level `_scalars(out) -> dict[str,float]`. Training writes `{"phase":"train","step",...scalars}` every `log_every` and `{"phase":"val","step","val_mse"}` every `val_every`; `metrics.csv` written on finish. `train()` still returns a summary dict with `step`, `recon`, `perplexity`, `val_mse`.

- [ ] **Step 1: Write the failing test**

`tests/test_trainer_metrics.py`:
```python
import json

import torch
from torch.utils.data import DataLoader, Dataset

from bubble_bi.config import ModelConfig, TrainConfig
from bubble_bi.models.dual_vqvae import DualVQVAE
from bubble_bi.train.trainer import Trainer, set_seed, _scalars


class _DayDS(Dataset):
    def __init__(self, n=64, N=5, p=4, D=6):
        self.w = torch.randn(n, N, p, D)
        self.v = torch.ones(n, N, dtype=torch.bool)

    def __len__(self):
        return self.w.shape[0]

    def __getitem__(self, i):
        return {"windows": self.w[i], "valid": self.v[i]}


def _loaders():
    ld = DataLoader(_DayDS(), batch_size=8, shuffle=True, drop_last=True)
    return {"train": ld, "val": ld, "test": ld}


def _model():
    return DualVQVAE(ModelConfig(p=4, d_model=16, codebook_size=16, cs_codebook_size=16,
                                 enc_layers=1, dec_layers=1, fusion_layers=1, heads=2,
                                 ff=32, dropout=0.0), d_in=6, n_stocks=5)


def test_scalars_skips_non_scalar_tensors():
    out = {"loss": torch.tensor(1.5), "ts_recon": torch.tensor(0.3),
           "ts_z_e": torch.randn(4, 16), "ids": torch.zeros(4, dtype=torch.long)}
    s = _scalars(out)
    assert s["loss"] == 1.5 and s["ts_recon"] == 0.3
    assert "ts_z_e" not in s


def test_trainer_writes_dense_history(tmp_path):
    set_seed(0)
    cfg = TrainConfig(max_steps=25, batch_size=8, val_every=10, ckpt_every=25,
                      log_every=10, device="cpu", amp=False)
    tr = Trainer(_model(), _loaders(), cfg, str(tmp_path / "ck"),
                 run_dir=str(tmp_path / "run"), device="cpu")
    tr.train()
    lines = [json.loads(x) for x in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    train_recs = [r for r in lines if r["phase"] == "train"]
    val_recs = [r for r in lines if r["phase"] == "val"]
    assert len(train_recs) >= 2                    # steps 10, 20
    assert "ts_recon" in train_recs[0] and "cs_recon" in train_recs[0]
    assert len(val_recs) >= 1
    assert (tmp_path / "run" / "metrics.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_trainer_metrics.py -v`
Expected: FAIL (`ImportError: cannot import name '_scalars'` / `run_dir` unexpected kwarg).

- [ ] **Step 3: Implement**

In `bubble_bi/train/trainer.py`, add the import at top:
```python
from bubble_bi.train.metrics_logger import MetricsLogger
```

Add a module-level helper (below `_batch_size`):
```python
def _scalars(out: dict) -> dict:
    result = {}
    for k, v in out.items():
        if isinstance(v, (int, float)):
            result[k] = float(v)
        elif torch.is_tensor(v) and v.ndim == 0:
            result[k] = float(v.detach())
    return result
```

Replace the tail of `Trainer.__init__` (the `metrics_path` line) — change:
```python
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.ckpt_dir / "metrics.jsonl"
```
to:
```python
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.logger = MetricsLogger(run_dir if run_dir is not None else ckpt_dir)
```
and change the signature `def __init__(self, model, loaders, cfg, ckpt_dir, standardizer=None, device=None):` to:
```python
    def __init__(self, model, loaders, cfg, ckpt_dir, standardizer=None, device=None, run_dir=None):
```

Delete the old `_log` method entirely.

Replace the whole `train` method body with:
```python
    def train(self) -> dict:
        cfg = self.cfg
        model = self.model
        summary: dict = {}
        model.train()
        while self.global_step < cfg.max_steps:
            for xb in self.loaders["train"]:
                if self.global_step >= cfg.max_steps:
                    break
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

                if self.global_step % _dead_every(model) == 0:
                    if hasattr(model, "reinit_dead_codes"):
                        model.reinit_dead_codes(out)
                    else:
                        model.vq.reset_dead_codes(out["z_e"].detach())

                scal = _scalars(out)
                summary = {"step": self.global_step,
                           "recon": scal.get("recon_loss", scal.get("loss", float("nan"))),
                           "perplexity": scal.get("perplexity", float("nan"))}
                if self.global_step % cfg.log_every == 0:
                    self.logger.log({"phase": "train", "step": self.global_step, **scal})
                if self.global_step % cfg.val_every == 0:
                    val = self.evaluate("val")
                    self.best_val = min(self.best_val, val)
                    summary["val_mse"] = val
                    self.logger.log({"phase": "val", "step": self.global_step, "val_mse": val})
                    model.train()
                if self.global_step % cfg.ckpt_every == 0:
                    self.save_checkpoint(str(self.ckpt_dir / "last.pt"))
        if "val_mse" not in summary:
            summary["val_mse"] = self.evaluate("val")
        self.save_checkpoint(str(self.ckpt_dir / "last.pt"))
        self.logger.to_csv()
        return summary
```

- [ ] **Step 4: Run new + existing trainer tests**

Run: `.venv/bin/python -m pytest tests/test_trainer_metrics.py tests/test_trainer.py tests/test_trainer_dual.py -v`
Expected: PASS (dense history works; M1/M2 trainer tests still green — they pass no `run_dir`, so metrics go to the checkpoint dir, and the returned summary still has `step`/`val_mse`).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/train/trainer.py tests/test_trainer_metrics.py
git commit -m "feat: Trainer dense metrics logging + run_dir + CSV export"
```

---

### Task 4: Plotting (viz/plots.py)

**Files:**
- Create: `bubble_bi/viz/__init__.py`
- Create: `bubble_bi/viz/plots.py`
- Test: `tests/test_plots.py`

**Interfaces:**
- Produces, in `bubble_bi/viz/plots.py`:
  - `plot_run(run_dir: str) -> list[str]` — writes `plots/losses.png`, `plots/perplexity.png`, `plots/val.png`; returns paths. Raises `FileNotFoundError` if no `metrics.jsonl`.
  - `plot_compare(run_dirs: list[str], out_dir: str) -> list[str]` — writes `compare_loss.png`, `compare_val.png`.

- [ ] **Step 1: Write the failing test**

`tests/test_plots.py`:
```python
import json
from pathlib import Path

import pytest

from bubble_bi.viz.plots import plot_run, plot_compare


def _write_history(run_dir):
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    recs = []
    for s in (10, 20, 30):
        recs.append({"phase": "train", "step": s, "loss": 1.0 / s,
                     "ts_recon": 0.5 / s, "cs_recon": 0.9 / s,
                     "ts_perplexity": s, "cs_perplexity": s / 2, "recon_loss": 0.5 / s})
    recs.append({"phase": "val", "step": 20, "val_mse": 0.4})
    with open(Path(run_dir) / "metrics.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


def test_plot_run_writes_pngs(tmp_path):
    run = tmp_path / "r1"
    _write_history(run)
    paths = plot_run(str(run))
    assert len(paths) == 3
    for p in paths:
        assert Path(p).exists() and Path(p).stat().st_size > 0


def test_plot_compare_writes_pngs(tmp_path):
    _write_history(tmp_path / "r1")
    _write_history(tmp_path / "r2")
    paths = plot_compare([str(tmp_path / "r1"), str(tmp_path / "r2")], str(tmp_path / "cmp"))
    assert len(paths) == 2
    for p in paths:
        assert Path(p).exists() and Path(p).stat().st_size > 0


def test_plot_run_missing_history_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        plot_run(str(tmp_path / "nope"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_plots.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

`bubble_bi/viz/__init__.py`: empty file.

`bubble_bi/viz/plots.py`:
```python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _read(run_dir: str):
    p = Path(run_dir) / "metrics.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"no metrics history at {p}")
    recs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    train = [r for r in recs if r.get("phase") == "train"]
    val = [r for r in recs if r.get("phase") == "val"]
    return train, val


def _series(recs, key):
    xs = [r["step"] for r in recs if key in r]
    ys = [r[key] for r in recs if key in r]
    return xs, ys


def _line_plot(series, title, xlabel, ylabel, path, markers=False):
    fig, ax = plt.subplots()
    drew = False
    for label, (xs, ys) in series.items():
        if xs:
            ax.plot(xs, ys, marker="o" if markers else None, label=label)
            drew = True
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if drew:
        ax.legend()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_run(run_dir: str) -> list[str]:
    train, val = _read(run_dir)
    out_dir = Path(run_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_keys = ["loss", "ts_recon", "cs_recon", "recon_loss",
                 "ts_commit", "cs_commit", "ts_diversity", "cs_diversity"]
    paths = [
        _line_plot({k: _series(train, k) for k in loss_keys},
                   "training losses", "step", "loss", out_dir / "losses.png"),
        _line_plot({k: _series(train, k) for k in ["ts_perplexity", "cs_perplexity", "perplexity"]},
                   "codebook perplexity", "step", "perplexity", out_dir / "perplexity.png"),
        _line_plot({"val_mse": _series(val, "val_mse"), "train_recon": _series(train, "recon_loss")},
                   "validation", "step", "mse", out_dir / "val.png", markers=True),
    ]
    return paths


def plot_compare(run_dirs: list[str], out_dir: str) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    loss_series, val_series = {}, {}
    for rd in run_dirs:
        train, val = _read(rd)
        name = Path(rd).name
        loss_series[name] = _series(train, "loss")
        val_series[name] = _series(val, "val_mse")
    return [
        _line_plot(loss_series, "total loss (compare)", "step", "loss", out / "compare_loss.png"),
        _line_plot(val_series, "val_mse (compare)", "step", "val_mse", out / "compare_val.png", markers=True),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_plots.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/viz/ tests/test_plots.py
git commit -m "feat: metrics plotting (single-run curves + multi-run overlay)"
```

---

### Task 5: CLI wiring (--run-name, meta.json, eval.json, plot-metrics)

**Files:**
- Modify: `bubble_bi/cli.py`
- Test: `tests/test_metrics_cli.py`

**Interfaces:**
- Consumes: `MetricsLogger`, `plot_run`/`plot_compare`, `build_day_loaders`, `DualVQVAE`, `Trainer`, `evaluate_dual`.
- Produces: `train_dual(cfg, run_name=None)`, `eval_dual(cfg, run_name=None)`, `train_tokenizer(cfg, run_name=None)`, `eval_tokenizer(cfg, run_name=None)`, `plot_metrics(cfg, run_names: list[str])`; `main` gains `--run-name` (nargs="+") and a `plot-metrics` command.

- [ ] **Step 1: Write the failing test**

`tests/test_metrics_cli.py`:
```python
from pathlib import Path

import numpy as np
import pandas as pd

from bubble_bi.config import Config, DataConfig, FeatureConfig, ModelConfig, TrainConfig
from bubble_bi.cli import train_dual, eval_dual, plot_metrics


def _write_raw(raw, n=320, N=6):
    rng = np.random.default_rng(0)
    for k in range(N):
        dates = pd.bdate_range("2015-01-01", periods=n)
        c = pd.Series(100 + np.cumsum(rng.normal(size=n)), index=dates)
        v = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
        df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": v}, index=dates)
        df.index.name = "date"
        df.to_parquet(f"{raw}/T{k}.parquet")


def _cfg(tmp_path):
    return Config(
        data=DataConfig(tickers=[f"T{k}" for k in range(6)], raw_dir=str(tmp_path / "raw"),
                        cache_dir=str(tmp_path / "cache"), min_history=50),
        features=FeatureConfig(),
        model=ModelConfig(p=4, d_model=16, codebook_size=16, cs_codebook_size=16,
                          enc_layers=1, dec_layers=1, fusion_layers=1, heads=2, ff=32,
                          dropout=0.0),
        train=TrainConfig(max_steps=8, batch_size=8, val_every=8, ckpt_every=8,
                          log_every=4, device="cpu", amp=False),
    )


def test_train_eval_plot_run_folder(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "cache").mkdir()
    _write_raw(tmp_path / "raw")
    cfg = _cfg(tmp_path)
    train_dual(cfg, run_name="r1")
    run = tmp_path / "cache" / "runs" / "r1"
    assert (run / "metrics.jsonl").exists()
    assert (run / "metrics.csv").exists()
    assert (run / "meta.json").exists()
    eval_dual(cfg, run_name="r1")
    assert (run / "eval.json").exists()
    plot_metrics(cfg, ["r1"])
    assert (run / "plots" / "losses.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_metrics_cli.py -v`
Expected: FAIL (`TypeError: train_dual() got an unexpected keyword argument 'run_name'` / `ImportError: plot_metrics`).

- [ ] **Step 3: Implement**

In `bubble_bi/cli.py`, add imports (near the M2 imports):
```python
from bubble_bi.viz.plots import plot_run, plot_compare
```

Add a run-name helper (below `_load_or_build_panel`):
```python
def _run_dir(cfg: Config, run_name: str) -> Path:
    return Path(cfg.data.cache_dir) / "runs" / run_name


def _default_dual_run(cfg: Config) -> str:
    return f"dual_{'-'.join(cfg.model.active_modules)}_{cfg.train.max_steps}"
```

Replace `train_tokenizer` with a `run_name`-aware version:
```python
def train_tokenizer(cfg: Config, run_name: str | None = None) -> dict:
    set_seed(cfg.seed)
    panel = _load_or_build_panel(cfg)
    loaders, std = build_loaders(panel, cfg)
    model = TSVQVAE(cfg.model, d_in=panel.features.shape[2])
    run_name = run_name or f"tsvqvae_{cfg.train.max_steps}"
    ckpt_dir = Path(cfg.data.cache_dir) / "checkpoints"
    trainer = Trainer(model, loaders, cfg.train, str(ckpt_dir), standardizer=std,
                      run_dir=str(_run_dir(cfg, run_name)))
    metrics = trainer.train()
    trainer.logger.write_meta({"model": "tsvqvae", "d_model": cfg.model.d_model,
                               "codebook_size": cfg.model.codebook_size,
                               "n_features": int(panel.features.shape[2]),
                               "max_steps": cfg.train.max_steps,
                               "log_every": cfg.train.log_every, "final": metrics})
    print(f"trained {metrics['step']} steps | recon {metrics['recon']:.4f} "
          f"| val_mse {metrics['val_mse']:.4f} | ppl {metrics['perplexity']:.1f}")
    return metrics
```

Replace `eval_tokenizer` to write `eval.json` when `run_name` is given — change its signature and add the write before `return`:
```python
def eval_tokenizer(cfg: Config, run_name: str | None = None) -> dict:
```
and, just before `return result`, insert:
```python
    if run_name:
        import json
        rd = _run_dir(cfg, run_name)
        rd.mkdir(parents=True, exist_ok=True)
        with open(rd / "eval.json", "w") as fh:
            json.dump(result, fh, indent=2)
```

Replace `train_dual` with a `run_name`-aware version:
```python
def train_dual(cfg: Config, run_name: str | None = None) -> dict:
    set_seed(cfg.seed)
    panel = _load_or_build_panel(cfg)
    loaders, std = build_day_loaders(panel, cfg)
    model = DualVQVAE(cfg.model, d_in=panel.features.shape[2], n_stocks=len(panel.tickers))
    run_name = run_name or _default_dual_run(cfg)
    ckpt_dir = Path(cfg.data.cache_dir) / "checkpoints"
    trainer = Trainer(model, loaders, cfg.train, str(ckpt_dir), standardizer=std,
                      run_dir=str(_run_dir(cfg, run_name)))
    metrics = trainer.train()
    trainer.logger.write_meta({
        "model": "dual", "active_modules": cfg.model.active_modules,
        "d_model": cfg.model.d_model, "codebook_size": cfg.model.codebook_size,
        "cs_codebook_size": cfg.model.cs_codebook_size, "fusion_layers": cfg.model.fusion_layers,
        "n_features": int(panel.features.shape[2]), "n_stocks": len(panel.tickers),
        "max_steps": cfg.train.max_steps, "log_every": cfg.train.log_every, "final": metrics,
    })
    print(f"trained {metrics['step']} steps | recon {metrics['recon']:.4f} "
          f"| val_mse {metrics['val_mse']:.4f} | ppl {metrics['perplexity']:.1f}")
    return metrics
```

Replace `eval_dual` to write `eval.json` when `run_name` given — change signature and add the write before `return result`:
```python
def eval_dual(cfg: Config, run_name: str | None = None) -> dict:
```
and before `return result`:
```python
    if run_name:
        import json
        rd = _run_dir(cfg, run_name)
        rd.mkdir(parents=True, exist_ok=True)
        with open(rd / "eval.json", "w") as fh:
            json.dump(result, fh, indent=2)
```

Add `plot_metrics`:
```python
def plot_metrics(cfg: Config, run_names: list[str]) -> list[str]:
    dirs = [str(_run_dir(cfg, n)) for n in run_names]
    if len(dirs) == 1:
        paths = plot_run(dirs[0])
    else:
        paths = plot_compare(dirs, str(Path(dirs[0]) / "plots"))
    print("wrote plots:")
    for p in paths:
        print(f"  {p}")
    return paths
```

In `main`, add the `--run-name` arg and the new command, and thread `run_name`:
```python
    parser.add_argument("command", choices=["ingest", "build-panel", "baseline",
                                            "train-tokenizer", "eval-tokenizer",
                                            "train-dual", "eval-dual", "plot-metrics"])
    parser.add_argument("--config", default="configs/m0.yaml")
    parser.add_argument("--run-name", nargs="+", default=None)
```
Update the dispatch branches:
```python
    elif args.command == "train-tokenizer":
        train_tokenizer(cfg, run_name=(args.run_name[0] if args.run_name else None))
    elif args.command == "eval-tokenizer":
        eval_tokenizer(cfg, run_name=(args.run_name[0] if args.run_name else None))
    elif args.command == "train-dual":
        train_dual(cfg, run_name=(args.run_name[0] if args.run_name else None))
    elif args.command == "eval-dual":
        eval_dual(cfg, run_name=(args.run_name[0] if args.run_name else None))
    elif args.command == "plot-metrics":
        plot_metrics(cfg, args.run_name or [_default_dual_run(cfg)])
```

- [ ] **Step 4: Run new test + full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: ALL tests pass (M0 + M1 + M2 + metrics).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/cli.py tests/test_metrics_cli.py
git commit -m "feat: --run-name, per-run meta/eval json, plot-metrics command"
```

---

### Task 6: Real end-to-end check (manual)

**Files:** none.

- [ ] **Step 1: Train with a run name (reuses the M2 panel cache)**

Run: `.venv/bin/python -m bubble_bi.cli train-dual --config configs/m2.yaml --run-name dual_full`
Expected: `artifacts/cache/runs/dual_full/` has `metrics.jsonl` (dense — a train record every 10 steps + val records), `metrics.csv`, `meta.json`.

- [ ] **Step 2: Eval + plot**

```bash
.venv/bin/python -m bubble_bi.cli eval-dual --config configs/m2.yaml --run-name dual_full
.venv/bin/python -m bubble_bi.cli plot-metrics --config configs/m2.yaml --run-name dual_full
```
Expected: `runs/dual_full/eval.json` written; `runs/dual_full/plots/{losses,perplexity,val}.png` created. Open a PNG to confirm curves render.

---

## Self-Review

**Spec coverage:**
- dense per-step history with all components → Task 3 (`_scalars`) + Task 2 (surface commit/diversity) ✅
- JSONL + CSV + meta.json → Task 1 (`MetricsLogger`) + Task 5 (meta) ✅
- eval.json → Task 5 ✅
- per-run folders / run_name → Task 5 (`_run_dir`, defaults) ✅
- matplotlib plots single + overlay → Task 4 ✅
- `plot-metrics` command + `--run-name` → Task 5 ✅
- `log_every` config + matplotlib dep → Task 0 ✅
- back-compat (Trainer run_dir default) → Task 3 ✅
- headless Agg backend, missing-history error → Task 4 ✅

**Placeholder scan:** none; every code step is complete. Task 6 is explicit manual verification.

**Type consistency:** `MetricsLogger.log/to_csv/write_meta` (Task 1) used by Trainer (Task 3) and CLI (Task 5). `_scalars` (Task 3) consumes the model output dict extended in Task 2. `plot_run(run_dir)->list[str]` / `plot_compare(run_dirs,out_dir)->list[str]` (Task 4) called by `plot_metrics` (Task 5). `Trainer(..., run_dir=None)` and `trainer.logger` (Task 3) used by `train_dual`/`train_tokenizer` (Task 5). `train_dual(cfg, run_name=None)` / `eval_dual(cfg, run_name=None)` signatures (Task 5) match the test and `main` dispatch.

**Note:** with `max_steps < log_every` (e.g. M1's `test_trainer` at 5 steps) no *train* record is logged, only a val record — that's fine; those tests assert the returned summary and `global_step`, not the metrics file. Task 3's test uses `max_steps=25, log_every=10` to exercise dense logging.
