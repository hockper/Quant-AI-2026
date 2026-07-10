# M2 Phase 2 — Windowed CS VQ-VAE (+ cleanup) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the superseded two-token M2 with the first two pieces of the staged tokenizer: generalize the day-indexed dataset to arbitrary window length, and build a standalone **windowed** Cross-Sectional VQ-VAE (Phase 2) that encodes a `cs_p`-day market field and reconstructs it.

**Architecture:** Delete the two-token `DualVQVAE`/`train-dual` code. Generalize `DayDataset` to yield `[N, window_len, D]` blocks. Add a windowed `CSFieldEncoder`/`CSFieldDecoder` and a `CSVQVAE` that trains alone (encode the `cs_p`-day cross-section → quantize → reconstruct the field). Phase 1 (TS VQ-VAE) is M1, unchanged. Phase 3 (fusion) is a separate later plan.

**Tech Stack:** PyTorch 2.13+cpu, existing `bubble_bi` package.

## Global Constraints

- Run everything as `.venv/bin/python` from repo root `/home/hockper/Documents/Code/Bubble Bi`.
- Redefined M2 emits ONE fused token per stock (Phase 3); this plan builds the CS half only.
- The superseded two-token `DualVQVAE`, `train-dual`/`eval-dual`, `evaluate_dual`, and `active_modules` are removed.
- New config: `cs_p` (CS window, default 5), `fusion_codebook_size` (default 512, used in Phase 3).
- Masked losses: invalid stocks are zero-filled and excluded from attention (key-padding) and every loss.
- EMA codebook; orthogonality stays a diagnostic; recon + commitment + diversity drive training.
- TDD; frequent commits.

---

### Task 0: Remove two-token M2 + config fields

**Files:**
- Delete: `bubble_bi/models/dual_vqvae.py`, `tests/test_dual_vqvae.py`, `tests/test_dual_cli.py`
- Modify: `bubble_bi/config.py`, `bubble_bi/cli.py`, `bubble_bi/eval/tokenizer_eval.py`, `tests/test_config_m2.py`, `tests/test_metrics_cli.py`
- Test: `tests/test_config_m2.py`

**Interfaces:**
- Produces: `ModelConfig` gains `cs_p: int = 5`, `fusion_codebook_size: int = 512`; loses `active_modules`. `cli` loses `train_dual`/`eval_dual`/`_default_dual_run`; `tokenizer_eval` loses `evaluate_dual`.

- [ ] **Step 1: Rewrite `tests/test_config_m2.py`**

```python
from bubble_bi.config import load_config, ModelConfig


def test_model_config_new_fields(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "data:\n  tickers: [AAPL]\n"
        "model:\n  cs_p: 8\n  fusion_codebook_size: 256\n  cs_codebook_size: 128\n"
    )
    cfg = load_config(str(p))
    assert cfg.model.cs_p == 8
    assert cfg.model.fusion_codebook_size == 256
    assert cfg.model.cs_codebook_size == 128
    assert cfg.model.codebook_size == 512      # default preserved


def test_model_config_defaults():
    m = ModelConfig()
    assert m.cs_p == 5
    assert m.fusion_codebook_size == 512
    assert not hasattr(m, "active_modules")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config_m2.py -v`
Expected: FAIL (`cs_p` missing / `active_modules` still present).

- [ ] **Step 3: Edit `bubble_bi/config.py`**

In `ModelConfig`, replace the block:
```python
    cs_codebook_size: int = 512
    fusion_layers: int = 2
    use_fusion: bool = True
    active_modules: list[str] = field(default_factory=lambda: ["ts", "cs"])
```
with:
```python
    cs_codebook_size: int = 512
    cs_p: int = 5
    fusion_codebook_size: int = 512
    fusion_layers: int = 2
    use_fusion: bool = True
```

- [ ] **Step 4: Delete superseded files and references**

```bash
git rm bubble_bi/models/dual_vqvae.py tests/test_dual_vqvae.py tests/test_dual_cli.py
```

In `bubble_bi/eval/tokenizer_eval.py`, delete the entire `evaluate_dual` function.

In `bubble_bi/cli.py`: delete the functions `train_dual`, `eval_dual`, `_default_dual_run`; delete the imports `from bubble_bi.models.dual_vqvae import DualVQVAE` and `evaluate_dual` (keep `evaluate_tokenizer`); remove the `"train-dual"`, `"eval-dual"` choices and their `elif` branches in `main`.

- [ ] **Step 5: Fix `tests/test_metrics_cli.py` to use the TS tokenizer**

Replace its body with:
```python
import numpy as np
import pandas as pd

from bubble_bi.config import Config, DataConfig, FeatureConfig, ModelConfig, TrainConfig
from bubble_bi.cli import train_tokenizer, eval_tokenizer, plot_metrics


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
        model=ModelConfig(p=4, d_model=16, codebook_size=16, enc_layers=1,
                          dec_layers=1, heads=2, ff=32, dropout=0.0),
        train=TrainConfig(max_steps=8, batch_size=8, val_every=8, ckpt_every=8,
                          log_every=4, device="cpu", amp=False),
    )


def test_train_eval_plot_run_folder(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "cache").mkdir()
    _write_raw(tmp_path / "raw")
    cfg = _cfg(tmp_path)
    train_tokenizer(cfg, run_name="r1")
    run = tmp_path / "cache" / "runs" / "r1"
    assert (run / "metrics.jsonl").exists()
    assert (run / "meta.json").exists()
    eval_tokenizer(cfg, run_name="r1")
    assert (run / "eval.json").exists()
    plot_metrics(cfg, ["r1"])
    assert (run / "plots" / "losses.png").exists()
```

- [ ] **Step 6: Run config test + full suite**

Run: `.venv/bin/python -m pytest tests/test_config_m2.py -v` then `.venv/bin/python -m pytest -q`
Expected: config test PASS; full suite green (the deleted two-token tests are gone; `test_day_dataset` still passes since `DayDataset` is unchanged so far).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: remove two-token M2; add cs_p/fusion_codebook_size config"
```

---

### Task 1: Generalize DayDataset to window_len

**Files:**
- Modify: `bubble_bi/data/windows.py`
- Modify: `tests/test_day_dataset.py`

**Interfaces:**
- Produces: `DayDataset(std_features, mask, window_len, day_range, min_valid=2)` yields `{"block": FloatTensor[N, window_len, D], "valid": BoolTensor[N]}`. `build_day_loaders(panel, cfg, window_len) -> (dict[str,DataLoader], Standardizer)`.

- [ ] **Step 1: Rewrite `tests/test_day_dataset.py`**

```python
import numpy as np
import torch

from bubble_bi.data.windows import DayDataset, build_day_loaders


def test_block_contents_and_mask():
    T, N, D, L = 12, 3, 2, 5
    feats = np.arange(T * N * D, dtype=np.float32).reshape(T, N, D)
    mask = np.ones((T, N), dtype=bool)
    mask[6, 1] = False
    ds = DayDataset(feats, mask, window_len=L, day_range=(0, T), min_valid=2)
    # first valid day is t = L-1 = 4; days touching day 6 for stock 1 are excluded for that stock
    item = ds[0]                                  # t = 4
    assert item["block"].shape == (N, L, D)
    assert item["valid"].tolist() == [True, True, True]
    # block for stock 0 at t=4 is rows 0..4
    assert torch.allclose(item["block"][0], torch.tensor(feats[0:5, 0, :]))


def test_days_with_too_few_valid_skipped():
    T, N, D, L = 8, 3, 1, 4
    feats = np.zeros((T, N, D), dtype=np.float32)
    mask = np.ones((T, N), dtype=bool)
    mask[:, 1] = False
    mask[:, 2] = False
    ds = DayDataset(feats, mask, window_len=L, day_range=(0, T), min_valid=2)
    assert len(ds) == 0


def test_build_day_loaders_window_len():
    from bubble_bi.config import Config, DataConfig, FeatureConfig, ModelConfig
    from bubble_bi.data.panel import build_panel
    import pandas as pd

    per = {}
    rng = np.random.default_rng(0)
    for k in range(5):
        dates = pd.bdate_range("2015-01-01", periods=300)
        c = pd.Series(100 + np.cumsum(rng.normal(size=300)), index=dates)
        v = rng.integers(1_000_000, 5_000_000, size=300).astype(float)
        per[f"T{k}"] = pd.DataFrame(
            {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": v}, index=dates
        )
    panel = build_panel(per, DataConfig(tickers=list(per), min_history=50), FeatureConfig())
    cfg = Config(data=DataConfig(tickers=list(per), min_history=50), model=ModelConfig())
    cfg.train.batch_size = 8
    loaders, std = build_day_loaders(panel, cfg, window_len=7)
    batch = next(iter(loaders["train"]))
    assert batch["block"].shape[1:] == (5, 7, panel.features.shape[2])
    assert std.mean is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_day_dataset.py -v`
Expected: FAIL (`DayDataset` signature uses `p`, returns key `windows`; `build_day_loaders` takes no `window_len`).

- [ ] **Step 3: Edit `bubble_bi/data/windows.py`**

Replace the `DayDataset` class and `build_day_loaders` with:
```python
class DayDataset(Dataset):
    def __init__(self, std_features: np.ndarray, mask: np.ndarray, window_len: int,
                 day_range, min_valid: int = 2):
        self.X = std_features
        self.mask = mask
        self.L = window_len
        self.N = mask.shape[1]
        self.D = std_features.shape[2]
        lo, hi = day_range
        days = []
        for t in range(max(lo, window_len - 1), hi):
            n_valid = sum(mask[t - window_len + 1:t + 1, j].all() for j in range(self.N))
            if n_valid >= min_valid:
                days.append(t)
        self.days = days

    def __len__(self) -> int:
        return len(self.days)

    def __getitem__(self, i: int) -> dict:
        t, L, N, D = self.days[i], self.L, self.N, self.D
        block = np.zeros((N, L, D), dtype=np.float32)
        valid = np.zeros(N, dtype=bool)
        for j in range(N):
            if self.mask[t - L + 1:t + 1, j].all():
                block[j] = self.X[t - L + 1:t + 1, j, :]
                valid[j] = True
        return {"block": torch.from_numpy(block), "valid": torch.from_numpy(valid)}


def build_day_loaders(panel, cfg, window_len: int):
    T = len(panel.dates)
    tr, va, te = chronological_split(T, cfg.data.train_frac, cfg.data.val_frac)
    std = Standardizer().fit(panel.features, panel.mask, tr)
    Xs = std.transform(panel.features)
    bs, nw = cfg.train.batch_size, cfg.train.num_workers

    def mk(rng, shuffle):
        return DataLoader(DayDataset(Xs, panel.mask, window_len, rng), batch_size=bs,
                          shuffle=shuffle, num_workers=nw, drop_last=shuffle)

    return {"train": mk(tr, True), "val": mk(va, False), "test": mk(te, False)}, std
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_day_dataset.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/data/windows.py tests/test_day_dataset.py
git commit -m "feat: DayDataset yields [N, window_len, D] blocks (generalized)"
```

---

### Task 2: Windowed CS encoder + field decoder

**Files:**
- Modify (replace contents): `bubble_bi/models/cross_sectional.py`
- Modify (rewrite): `tests/test_cross_sectional.py`

**Interfaces:**
- Consumes: `ModelConfig` (with `cs_p`), `_encoder_stack` from `ts_vqvae`.
- Produces:
  - `CSFieldEncoder(cfg, d_in, n_stocks)` — `forward(x: [B,N,cs_p,D], valid: [B,N]) -> [B,H]`.
  - `CSFieldDecoder(cfg, d_out, n_stocks)` — `forward(z_q: [B,H]) -> [B,N,cs_p,D]`.

- [ ] **Step 1: Rewrite `tests/test_cross_sectional.py`**

```python
import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.cross_sectional import CSFieldEncoder, CSFieldDecoder


def _cfg():
    return ModelConfig(cs_p=3, d_model=16, enc_layers=1, dec_layers=1, heads=2,
                       ff=32, dropout=0.0)


def test_cs_field_shapes():
    cfg = _cfg()
    enc = CSFieldEncoder(cfg, d_in=5, n_stocks=6)
    dec = CSFieldDecoder(cfg, d_out=5, n_stocks=6)
    x = torch.randn(2, 6, cfg.cs_p, 5)
    valid = torch.ones(2, 6, dtype=torch.bool)
    z = enc(x, valid)
    assert z.shape == (2, 16)
    out = dec(z)
    assert out.shape == (2, 6, cfg.cs_p, 5)


def test_cs_field_encoder_ignores_invalid_stocks():
    torch.manual_seed(0)
    cfg = _cfg()
    enc = CSFieldEncoder(cfg, d_in=5, n_stocks=4).eval()
    valid = torch.tensor([[True, True, False, True]])
    x1 = torch.randn(1, 4, cfg.cs_p, 5)
    x2 = x1.clone()
    x2[0, 2] = 999.0                              # garbage in the invalid stock across all cs_p days
    with torch.no_grad():
        z1 = enc(x1, valid)
        z2 = enc(x2, valid)
    assert torch.allclose(z1, z2, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cross_sectional.py -v`
Expected: FAIL (`ImportError: cannot import name 'CSFieldEncoder'`).

- [ ] **Step 3: Replace `bubble_bi/models/cross_sectional.py`**

```python
from __future__ import annotations

import torch
import torch.nn as nn

from bubble_bi.config import ModelConfig
from bubble_bi.models.ts_vqvae import _encoder_stack


class CSFieldEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, d_in: int, n_stocks: int):
        super().__init__()
        self.embed = nn.Linear(d_in, cfg.d_model)
        self.stock_id = nn.Parameter(torch.zeros(1, n_stocks, 1, cfg.d_model))
        self.day_pos = nn.Parameter(torch.zeros(1, 1, cfg.cs_p, cfg.d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.enc = _encoder_stack(cfg, cfg.enc_layers)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        B, N, P, D = x.shape
        h = self.embed(x) + self.stock_id + self.day_pos     # [B, N, P, H]
        h = h.reshape(B, N * P, -1)                          # [B, N*P, H]
        cls = self.cls.expand(B, 1, -1)
        h = torch.cat([cls, h], dim=1)                       # [B, 1+N*P, H]
        pad_stock = (~valid).unsqueeze(-1).expand(B, N, P).reshape(B, N * P)
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        pad = torch.cat([cls_pad, pad_stock], dim=1)
        h = self.enc(h, src_key_padding_mask=pad)
        return h[:, 0]                                        # [B, H]


class CSFieldDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig, d_out: int, n_stocks: int):
        super().__init__()
        self.n_stocks = n_stocks
        self.cs_p = cfg.cs_p
        self.stock_id = nn.Parameter(torch.zeros(1, n_stocks, 1, cfg.d_model))
        self.day_pos = nn.Parameter(torch.zeros(1, 1, cfg.cs_p, cfg.d_model))
        self.dec = _encoder_stack(cfg, cfg.dec_layers)
        self.out = nn.Linear(cfg.d_model, d_out)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        B = z_q.shape[0]
        N, P = self.n_stocks, self.cs_p
        q = z_q.view(B, 1, 1, -1) + self.stock_id + self.day_pos   # [B, N, P, H]
        h = self.dec(q.reshape(B, N * P, -1)).reshape(B, N, P, -1)
        return self.out(h)                                          # [B, N, P, D]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cross_sectional.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/models/cross_sectional.py tests/test_cross_sectional.py
git commit -m "feat: windowed CS field encoder/decoder (cs_p days, masked)"
```

---

### Task 3: CSVQVAE

**Files:**
- Create: `bubble_bi/models/cs_vqvae.py`
- Test: `tests/test_cs_vqvae.py`

**Interfaces:**
- Consumes: `CSFieldEncoder`/`CSFieldDecoder` (Task 2), `VectorQuantizerEMA`, `ModelConfig`.
- Produces: `CSVQVAE(cfg, d_in, n_stocks)` — `forward(batch: {"block":[B,N,L,D],"valid":[B,N]}) -> dict` with `recon [B,N,cs_p,D]`, `loss`, `recon_loss`, `commit`, `diversity`, `perplexity`, `ids [B,N]... ` (actually `ids [B]`? no — one CS token per DAY: `ids [B]`), `z_e`. `reinit_dead_codes(out)`. Note: the CS token is **per day** (one market token), so `ids` shape is `[B]`.

- [ ] **Step 1: Write the failing test**

`tests/test_cs_vqvae.py`:
```python
import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.cs_vqvae import CSVQVAE


def _cfg(**kw):
    base = dict(cs_p=3, d_model=16, cs_codebook_size=16, enc_layers=1, dec_layers=1,
                heads=2, ff=32, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


def _batch(B=3, N=5, L=3, D=6):
    block = torch.randn(B, N, L, D)
    valid = torch.ones(B, N, dtype=torch.bool)
    valid[0, 4] = False
    return {"block": block, "valid": valid}


def test_forward_shapes_and_loss():
    model = CSVQVAE(_cfg(), d_in=6, n_stocks=5)
    out = model(_batch())
    assert out["recon"].shape == (3, 5, 3, 6)
    assert out["ids"].shape == (3,)               # one market token per day
    assert torch.isfinite(out["loss"])


def test_invalid_stock_does_not_change_loss():
    torch.manual_seed(0)
    model = CSVQVAE(_cfg(), d_in=6, n_stocks=5).eval()
    b1 = _batch()
    b2 = {"block": b1["block"].clone(), "valid": b1["valid"].clone()}
    b2["block"][0, 4] = 123.0
    with torch.no_grad():
        assert torch.allclose(model(b1)["recon_loss"], model(b2)["recon_loss"], atol=1e-4)


def test_overfits_tiny_batch():
    torch.manual_seed(0)
    model = CSVQVAE(_cfg(), d_in=6, n_stocks=5).train()
    batch = _batch(B=6)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(1500):
        opt.zero_grad()
        out = model(batch)
        out["loss"].backward()
        opt.step()
        if first is None:
            first = out["recon_loss"].item()
    assert out["recon_loss"].item() < 0.5 * first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cs_vqvae.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write implementation**

`bubble_bi/models/cs_vqvae.py`:
```python
from __future__ import annotations

import torch
import torch.nn as nn

from bubble_bi.config import ModelConfig
from bubble_bi.models.cross_sectional import CSFieldDecoder, CSFieldEncoder
from bubble_bi.models.vq import VectorQuantizerEMA


def _masked_field_mse(recon: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # recon/target [B, N, P, D] ; valid [B, N]
    err = ((recon - target) ** 2).mean(dim=(2, 3))     # [B, N]
    v = valid.float()
    return (err * v).sum() / v.sum().clamp(min=1.0)


class CSVQVAE(nn.Module):
    def __init__(self, cfg: ModelConfig, d_in: int, n_stocks: int):
        super().__init__()
        self.cs_p = cfg.cs_p
        self.enc = CSFieldEncoder(cfg, d_in, n_stocks)
        self.vq = VectorQuantizerEMA(cfg.cs_codebook_size, cfg.d_model,
                                     cfg.ema_decay, commitment_beta=cfg.beta_commit)
        self.dec = CSFieldDecoder(cfg, d_in, n_stocks)
        self.lambda_div = cfg.lambda_div
        self.dead_code_reinit_every = cfg.dead_code_reinit_every

    def forward(self, batch: dict) -> dict:
        x = batch["block"][:, :, -self.cs_p:, :]       # [B, N, cs_p, D]
        valid = batch["valid"]
        z_e = self.enc(x, valid)                        # [B, H]
        q = self.vq(z_e)
        recon = self.dec(q["z_q"])                      # [B, N, cs_p, D]
        recon_loss = _masked_field_mse(recon, x, valid)
        loss = recon_loss + q["commit"] + self.lambda_div * q["diversity"]
        return {"recon": recon, "loss": loss, "recon_loss": recon_loss,
                "commit": q["commit"], "diversity": q["diversity"],
                "perplexity": q["perplexity"], "ids": q["ids"], "z_e": z_e}

    def reinit_dead_codes(self, out: dict) -> None:
        self.vq.reset_dead_codes(out["z_e"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cs_vqvae.py -v`
Expected: PASS (3 tests). If the overfit test is flaky at 1500 steps, raise to 2500 (VQ cold-start).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/models/cs_vqvae.py tests/test_cs_vqvae.py
git commit -m "feat: CSVQVAE (windowed market field autoencoder, one token per day)"
```

---

### Task 4: CS eval + CLI (train-cs / eval-cs)

**Files:**
- Modify: `bubble_bi/eval/tokenizer_eval.py`
- Modify: `bubble_bi/cli.py`
- Create: `configs/m2_cs.yaml`
- Test: `tests/test_cs_cli.py`

**Interfaces:**
- Consumes: `CSVQVAE`, `Trainer`, `build_day_loaders`, `_load_or_build_panel`.
- Produces: `evaluate_cs(model, loader, device) -> dict` (`recon_mse`, `mean_baseline_mse`, `perplexity`, `codes_used_frac`); `train_cs(cfg, run_name=None) -> dict`, `eval_cs(cfg, run_name=None) -> dict`; `main` gains `train-cs`, `eval-cs`. CS checkpoints go to `cache_dir/checkpoints_cs/last.pt`.

- [ ] **Step 1: Write the failing test**

`tests/test_cs_cli.py`:
```python
import numpy as np
import pandas as pd

from bubble_bi.config import Config, DataConfig, FeatureConfig, ModelConfig, TrainConfig
from bubble_bi.cli import train_cs, eval_cs


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
        model=ModelConfig(p=4, cs_p=3, d_model=16, cs_codebook_size=16, enc_layers=1,
                          dec_layers=1, heads=2, ff=32, dropout=0.0),
        train=TrainConfig(max_steps=8, batch_size=8, val_every=8, ckpt_every=8,
                          log_every=4, device="cpu", amp=False),
    )


def test_train_then_eval_cs(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "cache").mkdir()
    _write_raw(tmp_path / "raw")
    cfg = _cfg(tmp_path)
    m = train_cs(cfg, run_name="cs")
    assert m["step"] == 8
    assert (tmp_path / "cache" / "checkpoints_cs" / "last.pt").exists()
    ev = eval_cs(cfg, run_name="cs")
    assert np.isfinite(ev["recon_mse"])
    assert 0.0 <= ev["codes_used_frac"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cs_cli.py -v`
Expected: FAIL (`ImportError: cannot import name 'train_cs'`).

- [ ] **Step 3: Add `evaluate_cs` to `bubble_bi/eval/tokenizer_eval.py`**

```python
@torch.no_grad()
def evaluate_cs(model, loader, device) -> dict:
    model.eval()
    se, base, n, ppl, batches = 0.0, 0.0, 0, 0.0, 0
    used = set()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        x = batch["block"][:, :, -model.cs_p:, :]
        valid = batch["valid"].float()
        denom = float(valid.sum().clamp(min=1.0))
        se += float(out["recon_loss"]) * denom
        base += float((x.pow(2).mean(dim=(2, 3)) * valid).sum())
        n += denom
        ppl += float(out["perplexity"]); batches += 1
        used.update(out["ids"].tolist())
    return {"recon_mse": se / max(n, 1), "mean_baseline_mse": base / max(n, 1),
            "perplexity": ppl / max(batches, 1), "codes_used_frac": len(used) / model.vq.K}
```

- [ ] **Step 4: Add to `bubble_bi/cli.py`**

Add imports (near the model imports):
```python
from bubble_bi.eval.tokenizer_eval import evaluate_cs
from bubble_bi.models.cs_vqvae import CSVQVAE
```

Add functions (below `eval_tokenizer`):
```python
def train_cs(cfg: Config, run_name: str | None = None) -> dict:
    set_seed(cfg.seed)
    panel = _load_or_build_panel(cfg)
    loaders, std = build_day_loaders(panel, cfg, window_len=cfg.model.cs_p)
    model = CSVQVAE(cfg.model, d_in=panel.features.shape[2], n_stocks=len(panel.tickers))
    run_name = run_name or f"cs_{cfg.train.max_steps}"
    ckpt_dir = Path(cfg.data.cache_dir) / "checkpoints_cs"
    trainer = Trainer(model, loaders, cfg.train, str(ckpt_dir), standardizer=std,
                      run_dir=str(_run_dir(cfg, run_name)))
    metrics = trainer.train()
    trainer.logger.write_meta({"model": "cs", "cs_p": cfg.model.cs_p,
                               "cs_codebook_size": cfg.model.cs_codebook_size,
                               "n_features": int(panel.features.shape[2]),
                               "n_stocks": len(panel.tickers),
                               "max_steps": cfg.train.max_steps, "final": metrics})
    print(f"[cs] trained {metrics['step']} steps | recon {metrics['recon']:.4f} "
          f"| val_mse {metrics['val_mse']:.4f} | ppl {metrics['perplexity']:.1f}")
    return metrics


def eval_cs(cfg: Config, run_name: str | None = None) -> dict:
    set_seed(cfg.seed)
    panel = _load_or_build_panel(cfg)
    loaders, std = build_day_loaders(panel, cfg, window_len=cfg.model.cs_p)
    device = resolve_device(cfg.train.device)
    model = CSVQVAE(cfg.model, d_in=panel.features.shape[2],
                    n_stocks=len(panel.tickers)).to(device)
    ckpt = Path(cfg.data.cache_dir) / "checkpoints_cs" / "last.pt"
    if ckpt.exists():
        import torch

        state = torch.load(str(ckpt), map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    result = evaluate_cs(model, loaders["test"], device)
    print(f"[cs] recon {result['recon_mse']:.4f} (baseline {result['mean_baseline_mse']:.4f}) "
          f"| ppl {result['perplexity']:.1f} | codes {result['codes_used_frac']:.2%}")
    if run_name:
        _write_eval_json(cfg, run_name, result)
    return result
```

Extend the command list and dispatch in `main`:
```python
    parser.add_argument("command", choices=["ingest", "build-panel", "baseline",
                                            "train-tokenizer", "eval-tokenizer",
                                            "train-cs", "eval-cs", "plot-metrics"])
```
and before `return 0`:
```python
    elif args.command == "train-cs":
        train_cs(cfg, run_name=run_name)
    elif args.command == "eval-cs":
        eval_cs(cfg, run_name=run_name)
```

- [ ] **Step 5: Write `configs/m2_cs.yaml`**

```yaml
data:
  tickers: [AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, V, JNJ, WMT,
            PG, HD, BAC, XOM, CVX, KO, PEP, DIS, CSCO, INTC,
            VZ, T, MRK, PFE, ABT, NKE, MCD, CAT, BA, IBM]
  start: "2010-01-01"
  min_history: 252
  train_frac: 0.7
  val_frac: 0.15
features:
  ma_windows: [5, 10, 20]
model:
  cs_p: 5
  d_model: 128
  cs_codebook_size: 512
  enc_layers: 3
  dec_layers: 2
  heads: 4
  ff: 256
train:
  lr: 0.0001
  batch_size: 64
  max_steps: 500
  val_every: 100
  ckpt_every: 100
  log_every: 10
  device: auto
seed: 42
```

- [ ] **Step 6: Run new test + full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: ALL tests pass.

- [ ] **Step 7: Commit**

```bash
git add bubble_bi/eval/tokenizer_eval.py bubble_bi/cli.py configs/m2_cs.yaml tests/test_cs_cli.py
git commit -m "feat: CS eval + train-cs/eval-cs CLI"
```

---

### Task 5: Real Phase-2 run + record (manual)

**Files:** none.

- [ ] **Step 1: Train the CS tokenizer on real data**

Run: `.venv/bin/python -m bubble_bi.cli train-cs --config configs/m2_cs.yaml --run-name cs`
Expected: reconstruction falls; checkpoint at `artifacts/cache/checkpoints_cs/last.pt`; run folder `artifacts/cache/runs/cs/`.

- [ ] **Step 2: Evaluate on held-out test**

Run: `.venv/bin/python -m bubble_bi.cli eval-cs --config configs/m2_cs.yaml --run-name cs`
Expected: prints CS field recon MSE vs baseline, perplexity, code usage. A recon below baseline confirms the windowed market encoder captures cross-sectional structure. Note the perplexity to compare against M2's single-day CS (which collapsed to 15.6) — the cs_p window should help.

- [ ] **Step 3: Record the result**

Append the CS recon/baseline/perplexity/codes to the redefined-M2 design doc
(`docs/superpowers/specs/2026-07-10-m2-staged-dual-vqvae-design.md`) under a
"Phase 2 results" note, and commit:
```bash
git add -A && git commit -m "docs: record M2 Phase-2 (windowed CS VQ-VAE) results"
```

---

## Self-Review

**Spec coverage (this plan = Phases 1&2 of the redefined M2):**
- remove two-token M2 + config (`cs_p`, `fusion_codebook_size`, drop `active_modules`) → Task 0 ✅
- `DayDataset` windowed `[N, window_len, D]` → Task 1 ✅
- windowed CS encoder (stock-ID + day-position, masked) + field decoder → Task 2 ✅
- `CSVQVAE` (one market token per day, masked field recon) → Task 3 ✅
- CS eval (recon vs baseline, perplexity, codes) + `train-cs`/`eval-cs` CLI + config → Task 4 ✅
- real Phase-2 run + record → Task 5 ✅
- Phase 1 (TS VQ-VAE) = M1, already built (reused via `train-tokenizer`) ✅
- **Phase 3 (fusion) is deliberately deferred to Plan 2** (after Phase 2 is verified) — stated in the header.

**Placeholder scan:** none; Task 5 is explicit manual verification.

**Type consistency:** `DayDataset(..., window_len, ...)` returns `{"block","valid"}` (Task 1) consumed by `CSVQVAE.forward` (Task 3) and `evaluate_cs` (Task 4). `CSFieldEncoder(cfg,d_in,n_stocks).forward(x[B,N,cs_p,D],valid)->[B,H]` / `CSFieldDecoder(...).forward(z[B,H])->[B,N,cs_p,D]` (Task 2) used by `CSVQVAE` (Task 3). `build_day_loaders(panel, cfg, window_len)` (Task 1) called in Task 4 with `window_len=cfg.model.cs_p`. `evaluate_cs` uses `model.cs_p` and `model.vq.K` (defined on `CSVQVAE`, Task 3). `_run_dir`/`_write_eval_json` already exist in cli from the metrics feature.

**Note:** the CS token is one-per-**day** (`ids [B]`), which is correct for Phase 2's standalone market autoencoder; Phase 3 (Plan 2) produces the one-per-**stock** fused token that M3 consumes.
