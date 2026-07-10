# M2 — Dual VQ-VAE (CS module + cross-attention fusion) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a cross-sectional tokenizer and cross-attention fusion, assembling a `DualVQVAE` that jointly tokenizes each stock-day (TS) and each day's market cross-section (CS), with fusion before quantization and w/o-TS / w/o-CS / fusion-off ablations.

**Architecture:** A day-indexed dataset yields `[B,N,p,D]` windows + validity. A TS encoder (reused M1 `TSEncoder`) makes per-stock latents; a new CS encoder makes one masked market latent per day; a bidirectional cross-attention fusion mixes them; each fused latent is quantized by its own EMA codebook and decoded (TS→window, CS→cross-section) with masked reconstruction losses, trained jointly by the (lightly generalized) M1 `Trainer`.

**Tech Stack:** PyTorch 2.13+cpu (CUDA on Colab), plus the existing M0/M1 `bubble_bi` package.

## Global Constraints

- Run everything as `.venv/bin/python` from repo root `/home/hockper/Documents/Code/Bubble Bi`.
- **Fusion before quantization**, bidirectional, masked for invalid stocks.
- **Joint training**: TS + CS + fusion in one run; masked reconstruction losses (invalid stock-days excluded from attention and loss).
- **Reuse** `VectorQuantizerEMA`, `TSEncoder`, `TSDecoder`, `_encoder_stack`, `Standardizer`, `chronological_split`, and the `Trainer`. Do NOT modify M1's `TSVQVAE` (compose primitives instead).
- Ablations via `active_modules` (`["ts","cs"]`) and `use_fusion`; fusion active only when both modules present AND `use_fusion`.
- EMA codebook → orthogonality stays a diagnostic (no gradient); recon + commitment + diversity drive training.
- Defaults: `p=4, d_model=128, K_ts=512, cs_codebook_size=512, fusion_layers=2, heads=4, N=30`, day-batch 64, AdamW lr 1e-4 / wd 0.05 / grad-clip 1.0 / EMA 0.99.
- TDD; frequent commits.

---

### Task 0: Config additions + m2.yaml

**Files:**
- Modify: `bubble_bi/config.py`
- Create: `configs/m2.yaml`
- Test: `tests/test_config_m2.py`

**Interfaces:**
- Consumes: existing `ModelConfig`, `load_config`.
- Produces: `ModelConfig` gains `cs_codebook_size: int = 512`, `fusion_layers: int = 2`, `use_fusion: bool = True`, `active_modules: list[str]` (default `["ts","cs"]`).

- [ ] **Step 1: Write the failing test**

`tests/test_config_m2.py`:
```python
from bubble_bi.config import load_config


def test_model_config_m2_fields(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "data:\n  tickers: [AAPL]\n"
        "model:\n  cs_codebook_size: 128\n  fusion_layers: 1\n"
        "  use_fusion: false\n  active_modules: [cs]\n"
    )
    cfg = load_config(str(p))
    assert cfg.model.cs_codebook_size == 128
    assert cfg.model.fusion_layers == 1
    assert cfg.model.use_fusion is False
    assert cfg.model.active_modules == ["cs"]
    # defaults preserved
    assert cfg.model.codebook_size == 512


def test_model_config_m2_defaults():
    from bubble_bi.config import ModelConfig
    m = ModelConfig()
    assert m.cs_codebook_size == 512
    assert m.fusion_layers == 2
    assert m.use_fusion is True
    assert m.active_modules == ["ts", "cs"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config_m2.py -v`
Expected: FAIL (`AttributeError: ... 'cs_codebook_size'`).

- [ ] **Step 3: Add fields to `ModelConfig` in `bubble_bi/config.py`**

Insert after the `dead_code_reinit_every: int = 250` line inside `ModelConfig`:
```python
    cs_codebook_size: int = 512
    fusion_layers: int = 2
    use_fusion: bool = True
    active_modules: list[str] = field(default_factory=lambda: ["ts", "cs"])
```
(`field` is already imported in `config.py`.)

- [ ] **Step 4: Write `configs/m2.yaml`**

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
  p: 4
  d_model: 128
  codebook_size: 512
  cs_codebook_size: 512
  enc_layers: 3
  dec_layers: 2
  fusion_layers: 2
  heads: 4
  ff: 256
  active_modules: [ts, cs]
  use_fusion: true
train:
  lr: 0.0001
  batch_size: 64
  max_steps: 500
  val_every: 100
  ckpt_every: 100
  device: auto
seed: 42
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config_m2.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add bubble_bi/config.py configs/m2.yaml tests/test_config_m2.py
git commit -m "feat: M2 model-config fields (cs codebook, fusion, ablation flags)"
```

---

### Task 1: DayDataset + build_day_loaders

**Files:**
- Modify: `bubble_bi/data/windows.py`
- Test: `tests/test_day_dataset.py`

**Interfaces:**
- Consumes: `Standardizer`, `chronological_split` (M1); `Panel` (M0); `Config`.
- Produces, in `bubble_bi/data/windows.py`:
  - `class DayDataset(Dataset)` — `__init__(std_features, mask, p, day_range, min_valid=2)`; `__getitem__` returns `{"windows": FloatTensor[N,p,D], "valid": BoolTensor[N]}`; skips days with `<min_valid` valid stocks; invalid stocks zero-filled.
  - `build_day_loaders(panel, cfg) -> tuple[dict[str, DataLoader], Standardizer]` (keys `train`/`val`/`test`; batch = days).

- [ ] **Step 1: Write the failing test**

`tests/test_day_dataset.py`:
```python
import numpy as np
import torch

from bubble_bi.data.windows import DayDataset, build_day_loaders


def test_day_windows_and_mask_and_cs_slice():
    T, N, D, p = 12, 3, 2, 4
    feats = np.arange(T * N * D, dtype=np.float32).reshape(T, N, D)
    mask = np.ones((T, N), dtype=bool)
    mask[6, 1] = False                       # stock 1 invalid on day 6
    ds = DayDataset(feats, mask, p=p, day_range=(0, T), min_valid=2)
    # find the sample whose last day is t=6
    for i in range(len(ds)):
        item = ds[i]
        # reconstruct which day via stock 0's last row (unique increasing values)
        pass
    item = ds[3]  # day t = p-1+3 = 6
    assert item["windows"].shape == (N, p, D)
    assert item["valid"].tolist() == [True, False, True]
    # invalid stock is zero-filled
    assert torch.all(item["windows"][1] == 0)
    # CS input = last day of each window equals feats[6] for valid stocks
    cs = item["windows"][:, -1, :]
    assert torch.allclose(cs[0], torch.tensor(feats[6, 0, :]))
    assert torch.allclose(cs[2], torch.tensor(feats[6, 2, :]))


def test_days_with_too_few_valid_are_skipped():
    T, N, D, p = 8, 3, 1, 4
    feats = np.zeros((T, N, D), dtype=np.float32)
    mask = np.ones((T, N), dtype=bool)
    mask[:, 1] = False
    mask[:, 2] = False                       # only stock 0 ever valid -> <2 valid
    ds = DayDataset(feats, mask, p=p, day_range=(0, T), min_valid=2)
    assert len(ds) == 0


def test_build_day_loaders_batches_days():
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
    cfg = Config(data=DataConfig(tickers=list(per), min_history=50), model=ModelConfig(p=4))
    cfg.train.batch_size = 8
    loaders, std = build_day_loaders(panel, cfg)
    batch = next(iter(loaders["train"]))
    assert batch["windows"].shape[1:] == (5, 4, panel.features.shape[2])
    assert batch["valid"].shape[1] == 5
    assert std.mean is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_day_dataset.py -v`
Expected: FAIL (`ImportError: cannot import name 'DayDataset'`).

- [ ] **Step 3: Add to `bubble_bi/data/windows.py`**

Append:
```python
class DayDataset(Dataset):
    def __init__(self, std_features: np.ndarray, mask: np.ndarray, p: int,
                 day_range, min_valid: int = 2):
        self.X = std_features
        self.mask = mask
        self.p = p
        self.N = mask.shape[1]
        self.D = std_features.shape[2]
        lo, hi = day_range
        days = []
        for t in range(max(lo, p - 1), hi):
            n_valid = sum(mask[t - p + 1:t + 1, j].all() for j in range(self.N))
            if n_valid >= min_valid:
                days.append(t)
        self.days = days

    def __len__(self) -> int:
        return len(self.days)

    def __getitem__(self, i: int) -> dict:
        t = self.days[i]
        p, N, D = self.p, self.N, self.D
        windows = np.zeros((N, p, D), dtype=np.float32)
        valid = np.zeros(N, dtype=bool)
        for j in range(N):
            if self.mask[t - p + 1:t + 1, j].all():
                windows[j] = self.X[t - p + 1:t + 1, j, :]
                valid[j] = True
        return {"windows": torch.from_numpy(windows),
                "valid": torch.from_numpy(valid)}


def build_day_loaders(panel, cfg):
    T = len(panel.dates)
    tr, va, te = chronological_split(T, cfg.data.train_frac, cfg.data.val_frac)
    std = Standardizer().fit(panel.features, panel.mask, tr)
    Xs = std.transform(panel.features)
    p = cfg.model.p
    bs = cfg.train.batch_size
    nw = cfg.train.num_workers
    loaders = {
        "train": DataLoader(DayDataset(Xs, panel.mask, p, tr), batch_size=bs,
                            shuffle=True, num_workers=nw, drop_last=True),
        "val": DataLoader(DayDataset(Xs, panel.mask, p, va), batch_size=bs,
                          shuffle=False, num_workers=nw),
        "test": DataLoader(DayDataset(Xs, panel.mask, p, te), batch_size=bs,
                           shuffle=False, num_workers=nw),
    }
    return loaders, std
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_day_dataset.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/data/windows.py tests/test_day_dataset.py
git commit -m "feat: DayDataset + build_day_loaders (day-indexed dual batches)"
```

---

### Task 2: CS encoder + decoder

**Files:**
- Create: `bubble_bi/models/cross_sectional.py`
- Test: `tests/test_cross_sectional.py`

**Interfaces:**
- Consumes: `ModelConfig`; `_encoder_stack` from `bubble_bi/models/ts_vqvae.py`.
- Produces, in `bubble_bi/models/cross_sectional.py`:
  - `class CSEncoder(nn.Module)` — `__init__(cfg, d_in, n_stocks)`; `forward(x: [B,N,D], valid: [B,N] bool) -> [B,H]`.
  - `class CSDecoder(nn.Module)` — `__init__(cfg, d_out, n_stocks)`; `forward(z_q: [B,H]) -> [B,N,D]`.

- [ ] **Step 1: Write the failing test**

`tests/test_cross_sectional.py`:
```python
import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.cross_sectional import CSEncoder, CSDecoder


def _cfg():
    return ModelConfig(p=4, d_model=16, enc_layers=1, dec_layers=1, heads=2,
                       ff=32, dropout=0.0)


def test_cs_encoder_decoder_shapes():
    cfg = _cfg()
    enc = CSEncoder(cfg, d_in=5, n_stocks=6)
    dec = CSDecoder(cfg, d_out=5, n_stocks=6)
    x = torch.randn(3, 6, 5)
    valid = torch.ones(3, 6, dtype=torch.bool)
    z = enc(x, valid)
    assert z.shape == (3, 16)
    out = dec(z)
    assert out.shape == (3, 6, 5)


def test_cs_encoder_ignores_invalid_stocks():
    torch.manual_seed(0)
    cfg = _cfg()
    enc = CSEncoder(cfg, d_in=5, n_stocks=4).eval()
    valid = torch.tensor([[True, True, False, True]])
    x1 = torch.randn(1, 4, 5)
    x2 = x1.clone()
    x2[0, 2] = 999.0                         # garbage in the invalid stock slot
    with torch.no_grad():
        z1 = enc(x1, valid)
        z2 = enc(x2, valid)
    assert torch.allclose(z1, z2, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cross_sectional.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write implementation**

`bubble_bi/models/cross_sectional.py`:
```python
from __future__ import annotations

import torch
import torch.nn as nn

from bubble_bi.config import ModelConfig
from bubble_bi.models.ts_vqvae import _encoder_stack


class CSEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, d_in: int, n_stocks: int):
        super().__init__()
        self.embed = nn.Linear(d_in, cfg.d_model)
        self.stock_id = nn.Parameter(torch.zeros(1, n_stocks, cfg.d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.enc = _encoder_stack(cfg, cfg.enc_layers)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        h = self.embed(x) + self.stock_id             # [B, N, H]
        B = h.shape[0]
        cls = self.cls.expand(B, 1, -1)
        h = torch.cat([cls, h], dim=1)                # [B, N+1, H]
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        pad = torch.cat([cls_pad, ~valid], dim=1)     # True = ignore
        h = self.enc(h, src_key_padding_mask=pad)
        return h[:, 0]                                 # [B, H]


class CSDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig, d_out: int, n_stocks: int):
        super().__init__()
        self.stock_query = nn.Parameter(torch.zeros(1, n_stocks, cfg.d_model))
        self.dec = _encoder_stack(cfg, cfg.dec_layers)
        self.out = nn.Linear(cfg.d_model, d_out)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        h = z_q.unsqueeze(1) + self.stock_query       # [B, N, H]
        return self.out(self.dec(h))                  # [B, N, D]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cross_sectional.py -v`
Expected: PASS (2 tests). The masking test proves invalid stocks don't leak into the market token.

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/models/cross_sectional.py tests/test_cross_sectional.py
git commit -m "feat: cross-sectional encoder/decoder (stock-ID embeddings, masked)"
```

---

### Task 3: Cross-attention fusion

**Files:**
- Create: `bubble_bi/models/fusion.py`
- Test: `tests/test_fusion.py`

**Interfaces:**
- Consumes: `ModelConfig`.
- Produces, in `bubble_bi/models/fusion.py`:
  - `class CrossAttentionFusion(nn.Module)` — `__init__(cfg)`; `forward(z_ts: [B,N,H], z_cs: [B,H], valid: [B,N] bool) -> (fused_ts [B,N,H], fused_cs [B,H])`.

- [ ] **Step 1: Write the failing test**

`tests/test_fusion.py`:
```python
import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.fusion import CrossAttentionFusion


def _cfg():
    return ModelConfig(d_model=16, heads=2, fusion_layers=2, dropout=0.0)


def test_fusion_shapes():
    fus = CrossAttentionFusion(_cfg())
    z_ts = torch.randn(3, 6, 16)
    z_cs = torch.randn(3, 16)
    valid = torch.ones(3, 6, dtype=torch.bool)
    ft, fc = fus(z_ts, z_cs, valid)
    assert ft.shape == (3, 6, 16)
    assert fc.shape == (3, 16)


def test_fused_cs_ignores_masked_stocks():
    torch.manual_seed(0)
    fus = CrossAttentionFusion(_cfg()).eval()
    valid = torch.tensor([[True, True, False, True]])
    z_ts1 = torch.randn(1, 4, 16)
    z_ts2 = z_ts1.clone()
    z_ts2[0, 2] = 50.0                       # change the masked stock's latent (a key)
    z_cs = torch.randn(1, 16)
    with torch.no_grad():
        _, fc1 = fus(z_ts1, z_cs, valid)
        _, fc2 = fus(z_ts2, z_cs, valid)
    assert torch.allclose(fc1, fc2, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fusion.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write implementation**

`bubble_bi/models/fusion.py`:
```python
from __future__ import annotations

import torch
import torch.nn as nn

from bubble_bi.config import ModelConfig


class CrossAttentionFusion(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        H, heads, L = cfg.d_model, cfg.heads, cfg.fusion_layers
        mk = lambda: nn.MultiheadAttention(H, heads, dropout=cfg.dropout, batch_first=True)
        self.ts_attn = nn.ModuleList([mk() for _ in range(L)])
        self.cs_attn = nn.ModuleList([mk() for _ in range(L)])
        self.ts_norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(L)])
        self.cs_norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(L)])

    def forward(self, z_ts: torch.Tensor, z_cs: torch.Tensor, valid: torch.Tensor):
        cs = z_cs.unsqueeze(1)                        # [B, 1, H]
        pad = ~valid                                  # [B, N] True = ignore
        for ts_attn, cs_attn, tn, cn in zip(self.ts_attn, self.cs_attn,
                                            self.ts_norm, self.cs_norm):
            a, _ = ts_attn(z_ts, cs, cs)              # stock <- market
            z_ts = tn(z_ts + a)
            b, _ = cs_attn(cs, z_ts, z_ts, key_padding_mask=pad)   # market <- stocks
            cs = cn(cs + b)
        return z_ts, cs.squeeze(1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fusion.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/models/fusion.py tests/test_fusion.py
git commit -m "feat: bidirectional cross-attention fusion (masked)"
```

---

### Task 4: DualVQVAE assembly

**Files:**
- Create: `bubble_bi/models/dual_vqvae.py`
- Test: `tests/test_dual_vqvae.py`

**Interfaces:**
- Consumes: `TSEncoder`, `TSDecoder` (`ts_vqvae`), `VectorQuantizerEMA` (`vq`), `CSEncoder`/`CSDecoder` (`cross_sectional`), `CrossAttentionFusion` (`fusion`), `ModelConfig`.
- Produces, in `bubble_bi/models/dual_vqvae.py`:
  - `class DualVQVAE(nn.Module)` — `__init__(cfg, d_in, n_stocks)`; `forward(batch: dict) -> dict` with `loss`, `recon_loss`, `perplexity`, and per-module `ts_recon`/`cs_recon`/`ts_perplexity`/`cs_perplexity`/`ts_z_e`/`cs_z_e` when active. `encode(batch) -> (ts_tokens|None, cs_tokens|None)`. `reinit_dead_codes(out) -> None`. Attribute `dead_code_reinit_every`.

- [ ] **Step 1: Write the failing test**

`tests/test_dual_vqvae.py`:
```python
import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.dual_vqvae import DualVQVAE


def _cfg(**kw):
    base = dict(p=4, d_model=16, codebook_size=16, cs_codebook_size=16, enc_layers=1,
                dec_layers=1, fusion_layers=1, heads=2, ff=32, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


def _batch(B=3, N=5, p=4, D=6):
    windows = torch.randn(B, N, p, D)
    valid = torch.ones(B, N, dtype=torch.bool)
    valid[0, 4] = False
    return {"windows": windows, "valid": valid}


def test_dual_forward_has_both_losses():
    model = DualVQVAE(_cfg(), d_in=6, n_stocks=5)
    out = model(_batch())
    assert torch.isfinite(out["loss"])
    assert "ts_recon" in out and "cs_recon" in out
    assert "ts_perplexity" in out and "cs_perplexity" in out


def test_active_modules_ts_only_builds_no_cs():
    model = DualVQVAE(_cfg(active_modules=["ts"]), d_in=6, n_stocks=5)
    assert not hasattr(model, "cs_enc")
    assert model.use_fusion is False
    out = model(_batch())
    assert "ts_recon" in out and "cs_recon" not in out


def test_use_fusion_false_skips_fusion():
    model = DualVQVAE(_cfg(use_fusion=False), d_in=6, n_stocks=5)
    assert not hasattr(model, "fusion")
    out = model(_batch())
    assert torch.isfinite(out["loss"])


def test_invalid_stock_does_not_change_losses():
    torch.manual_seed(0)
    model = DualVQVAE(_cfg(), d_in=6, n_stocks=5).eval()
    b1 = _batch()
    b2 = {"windows": b1["windows"].clone(), "valid": b1["valid"].clone()}
    b2["windows"][0, 4] = 123.0              # stock 4 in row 0 is invalid
    with torch.no_grad():
        o1 = model(b1)
        o2 = model(b2)
    assert torch.allclose(o1["ts_recon"], o2["ts_recon"], atol=1e-4)
    assert torch.allclose(o1["cs_recon"], o2["cs_recon"], atol=1e-4)


def test_encode_returns_token_shapes():
    model = DualVQVAE(_cfg(), d_in=6, n_stocks=5)
    ts_tok, cs_tok = model.encode(_batch(B=3, N=5))
    assert ts_tok.shape == (3, 5) and ts_tok.dtype == torch.long
    assert cs_tok.shape == (3,) and cs_tok.dtype == torch.long


def test_reinit_dead_codes_runs_for_both():
    model = DualVQVAE(_cfg(), d_in=6, n_stocks=5)
    out = model(_batch())
    model.reinit_dead_codes(out)             # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dual_vqvae.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write implementation**

`bubble_bi/models/dual_vqvae.py`:
```python
from __future__ import annotations

import torch
import torch.nn as nn

from bubble_bi.config import ModelConfig
from bubble_bi.models.cross_sectional import CSDecoder, CSEncoder
from bubble_bi.models.fusion import CrossAttentionFusion
from bubble_bi.models.ts_vqvae import TSDecoder, TSEncoder
from bubble_bi.models.vq import VectorQuantizerEMA


def _masked_mse(recon: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    err = (recon - target) ** 2
    err = err.reshape(err.shape[0], err.shape[1], -1).mean(-1)   # [B, N]
    v = valid.float()
    return (err * v).sum() / v.sum().clamp(min=1.0)


class DualVQVAE(nn.Module):
    def __init__(self, cfg: ModelConfig, d_in: int, n_stocks: int):
        super().__init__()
        self.active = list(cfg.active_modules)
        self.lambda_div = cfg.lambda_div
        self.dead_code_reinit_every = cfg.dead_code_reinit_every
        self.use_fusion = bool(cfg.use_fusion) and ("ts" in self.active and "cs" in self.active)
        if "ts" in self.active:
            self.ts_enc = TSEncoder(cfg, d_in)
            self.ts_vq = VectorQuantizerEMA(cfg.codebook_size, cfg.d_model,
                                            cfg.ema_decay, commitment_beta=cfg.beta_commit)
            self.ts_dec = TSDecoder(cfg, d_in)
        if "cs" in self.active:
            self.cs_enc = CSEncoder(cfg, d_in, n_stocks)
            self.cs_vq = VectorQuantizerEMA(cfg.cs_codebook_size, cfg.d_model,
                                            cfg.ema_decay, commitment_beta=cfg.beta_commit)
            self.cs_dec = CSDecoder(cfg, d_in, n_stocks)
        if self.use_fusion:
            self.fusion = CrossAttentionFusion(cfg)

    def _encode_latents(self, windows, valid):
        B, N, p, D = windows.shape
        z_ts = z_cs = None
        if "ts" in self.active:
            z_ts = self.ts_enc(windows.reshape(B * N, p, D)).reshape(B, N, -1)
        if "cs" in self.active:
            z_cs = self.cs_enc(windows[:, :, -1, :], valid)
        if self.use_fusion:
            z_ts, z_cs = self.fusion(z_ts, z_cs, valid)
        return z_ts, z_cs

    def forward(self, batch: dict) -> dict:
        windows, valid = batch["windows"], batch["valid"]
        B, N, p, D = windows.shape
        z_ts, z_cs = self._encode_latents(windows, valid)
        loss = windows.new_zeros(())
        out: dict = {}
        if "ts" in self.active:
            q = self.ts_vq(z_ts.reshape(B * N, -1))
            recon = self.ts_dec(q["z_q"]).reshape(B, N, p, D)
            ts_recon = _masked_mse(recon, windows, valid)
            loss = loss + ts_recon + q["commit"] + self.lambda_div * q["diversity"]
            out.update(ts_recon=ts_recon, ts_perplexity=q["perplexity"],
                       ts_z_e=z_ts.reshape(B * N, -1).detach())
        if "cs" in self.active:
            cs_target = windows[:, :, -1, :]
            q = self.cs_vq(z_cs)
            recon = self.cs_dec(q["z_q"])
            cs_recon = _masked_mse(recon, cs_target, valid)
            loss = loss + cs_recon + q["commit"] + self.lambda_div * q["diversity"]
            out.update(cs_recon=cs_recon, cs_perplexity=q["perplexity"],
                       cs_z_e=z_cs.detach())
        out["loss"] = loss
        out["recon_loss"] = out.get("ts_recon", out.get("cs_recon"))
        out["perplexity"] = out.get("ts_perplexity", out.get("cs_perplexity"))
        return out

    @torch.no_grad()
    def encode(self, batch: dict):
        windows, valid = batch["windows"], batch["valid"]
        B, N, p, D = windows.shape
        z_ts, z_cs = self._encode_latents(windows, valid)
        ts_tok = cs_tok = None
        if "ts" in self.active:
            ts_tok = self.ts_vq(z_ts.reshape(B * N, -1))["ids"].reshape(B, N).long()
        if "cs" in self.active:
            cs_tok = self.cs_vq(z_cs)["ids"].long()
        return ts_tok, cs_tok

    def reinit_dead_codes(self, out: dict) -> None:
        if "ts" in self.active and "ts_z_e" in out:
            self.ts_vq.reset_dead_codes(out["ts_z_e"])
        if "cs" in self.active and "cs_z_e" in out:
            self.cs_vq.reset_dead_codes(out["cs_z_e"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dual_vqvae.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/models/dual_vqvae.py tests/test_dual_vqvae.py
git commit -m "feat: DualVQVAE (fusion before quant, masked joint recon, ablations)"
```

---

### Task 5: Generalize Trainer for dict batches + reinit hook

**Files:**
- Modify: `bubble_bi/train/trainer.py`
- Test: `tests/test_trainer_dual.py`

**Interfaces:**
- Consumes: `DualVQVAE` (Task 4); existing `Trainer`, `set_seed`.
- Produces: `Trainer` handles a dict batch (`_to_device`, `_batch_size`) and calls `model.reinit_dead_codes(out)` when present (else the single-`vq` path). No public signature change.

- [ ] **Step 1: Write the failing test**

`tests/test_trainer_dual.py`:
```python
import torch
from torch.utils.data import DataLoader, Dataset

from bubble_bi.config import ModelConfig, TrainConfig
from bubble_bi.models.dual_vqvae import DualVQVAE
from bubble_bi.train.trainer import Trainer, set_seed


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


def test_trainer_handles_dict_batches(tmp_path):
    set_seed(0)
    cfg = TrainConfig(max_steps=4, batch_size=8, val_every=4, ckpt_every=4,
                      device="cpu", amp=False)
    tr = Trainer(_model(), _loaders(), cfg, str(tmp_path), device="cpu")
    metrics = tr.train()
    assert tr.global_step == 4
    assert "val_mse" in metrics


def test_trainer_dual_resume(tmp_path):
    set_seed(0)
    cfg = TrainConfig(max_steps=3, batch_size=8, val_every=3, ckpt_every=3,
                      device="cpu", amp=False)
    tr = Trainer(_model(), _loaders(), cfg, str(tmp_path), device="cpu")
    tr.train()
    fresh = Trainer(_model(), _loaders(), cfg, str(tmp_path), device="cpu")
    fresh.load_checkpoint(str(tmp_path / "last.pt"))
    assert fresh.global_step == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_trainer_dual.py -v`
Expected: FAIL (dict batch has no `.to`, or reinit path breaks) — typically `AttributeError: 'dict' object has no attribute 'to'`.

- [ ] **Step 3: Edit `bubble_bi/train/trainer.py`**

Add two module-level helpers (below `_dead_every`):
```python
def _to_device(batch, device):
    if isinstance(batch, dict):
        return {k: v.to(device) for k, v in batch.items()}
    return batch.to(device)


def _batch_size(batch) -> int:
    if isinstance(batch, dict):
        return next(iter(batch.values())).shape[0]
    return batch.shape[0]
```

In `Trainer.train`, replace `xb = xb.to(self.device)` with:
```python
                xb = _to_device(xb, self.device)
```

In `Trainer.train`, replace the dead-code reinit block:
```python
                if self.global_step % _dead_every(model) == 0:
                    model.vq.reset_dead_codes(out["z_e"].detach())
```
with:
```python
                if self.global_step % _dead_every(model) == 0:
                    if hasattr(model, "reinit_dead_codes"):
                        model.reinit_dead_codes(out)
                    else:
                        model.vq.reset_dead_codes(out["z_e"].detach())
```

In `Trainer.evaluate`, replace `xb = xb.to(self.device)` with `xb = _to_device(xb, self.device)` and replace `n += xb.shape[0]` with `n += _batch_size(xb)`.

- [ ] **Step 4: Run test + M1 trainer test to verify both green**

Run: `.venv/bin/python -m pytest tests/test_trainer_dual.py tests/test_trainer.py -v`
Expected: PASS (M1 tensor-batch path still works; dual dict-batch path works).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/train/trainer.py tests/test_trainer_dual.py
git commit -m "feat: Trainer handles dict batches + multi-codebook reinit hook"
```

---

### Task 6: Dual eval + CLI (train-dual / eval-dual)

**Files:**
- Modify: `bubble_bi/eval/tokenizer_eval.py`
- Modify: `bubble_bi/cli.py`
- Test: `tests/test_dual_cli.py`

**Interfaces:**
- Consumes: `DualVQVAE`, `Trainer`, `build_day_loaders`, `_load_or_build_panel`, `Config`.
- Produces:
  - `evaluate_dual(model, loader, device) -> dict` in `tokenizer_eval.py` with `ts_recon_mse`, `ts_baseline_mse`, `ts_perplexity`, `ts_codes_used`, and `cs_*` counterparts (only for active modules).
  - `train_dual(cfg) -> dict`, `eval_dual(cfg) -> dict` in `cli.py`; `main` gains `train-dual`, `eval-dual`.

- [ ] **Step 1: Write the failing test**

`tests/test_dual_cli.py`:
```python
import numpy as np
import pandas as pd

from bubble_bi.config import Config, DataConfig, FeatureConfig, ModelConfig, TrainConfig
from bubble_bi.cli import train_dual, eval_dual


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
                          device="cpu", amp=False),
    )


def test_train_then_eval_dual(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "cache").mkdir()
    _write_raw(tmp_path / "raw")
    cfg = _cfg(tmp_path)
    m = train_dual(cfg)
    assert m["step"] == 8
    assert (tmp_path / "cache" / "checkpoints" / "last.pt").exists()
    ev = eval_dual(cfg)
    assert np.isfinite(ev["ts_recon_mse"]) and np.isfinite(ev["cs_recon_mse"])
    assert 0.0 <= ev["ts_codes_used"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dual_cli.py -v`
Expected: FAIL (`ImportError: cannot import name 'train_dual'`).

- [ ] **Step 3: Add `evaluate_dual` to `bubble_bi/eval/tokenizer_eval.py`**

Append:
```python
@torch.no_grad()
def evaluate_dual(model, loader, device) -> dict:
    model.eval()
    agg = {}
    for mod in model.active:
        agg[mod] = {"se": 0.0, "base": 0.0, "count": 0.0, "ppl": 0.0, "batches": 0, "used": set()}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        windows, valid = batch["windows"], batch["valid"]
        vf = valid.float()
        denom = vf.sum().clamp(min=1.0)
        if "ts" in model.active:
            g = agg["ts"]
            g["se"] += float(out["ts_recon"]) * float(denom)
            g["base"] += float(((windows ** 2).mean(dim=(2, 3)) * vf).sum())
            g["count"] += float(denom)
            g["ppl"] += float(out["ts_perplexity"]); g["batches"] += 1
            ts_tok, _ = model.encode(batch)
            g["used"].update(ts_tok[valid].tolist())
        if "cs" in model.active:
            g = agg["cs"]
            cs_t = windows[:, :, -1, :]
            g["se"] += float(out["cs_recon"]) * float(denom)
            g["base"] += float(((cs_t ** 2).mean(dim=2) * vf).sum())
            g["count"] += float(denom)
            g["ppl"] += float(out["cs_perplexity"]); g["batches"] += 1
            _, cs_tok = model.encode(batch)
            g["used"].update(cs_tok.tolist())
    result = {}
    if "ts" in model.active:
        g = agg["ts"]
        result.update(ts_recon_mse=g["se"] / max(g["count"], 1),
                      ts_baseline_mse=g["base"] / max(g["count"], 1),
                      ts_perplexity=g["ppl"] / max(g["batches"], 1),
                      ts_codes_used=len(g["used"]) / model.ts_vq.K)
    if "cs" in model.active:
        g = agg["cs"]
        result.update(cs_recon_mse=g["se"] / max(g["count"], 1),
                      cs_baseline_mse=g["base"] / max(g["count"], 1),
                      cs_perplexity=g["ppl"] / max(g["batches"], 1),
                      cs_codes_used=len(g["used"]) / model.cs_vq.K)
    return result
```

- [ ] **Step 4: Add to `bubble_bi/cli.py`**

Add imports near the M1 imports:
```python
from bubble_bi.data.windows import build_day_loaders
from bubble_bi.eval.tokenizer_eval import evaluate_dual
from bubble_bi.models.dual_vqvae import DualVQVAE
```

Add functions below `eval_tokenizer`:
```python
def train_dual(cfg: Config) -> dict:
    set_seed(cfg.seed)
    panel = _load_or_build_panel(cfg)
    loaders, std = build_day_loaders(panel, cfg)
    model = DualVQVAE(cfg.model, d_in=panel.features.shape[2], n_stocks=len(panel.tickers))
    ckpt_dir = Path(cfg.data.cache_dir) / "checkpoints"
    trainer = Trainer(model, loaders, cfg.train, str(ckpt_dir), standardizer=std)
    metrics = trainer.train()
    print(f"trained {metrics['step']} steps | recon {metrics['recon']:.4f} "
          f"| val_mse {metrics['val_mse']:.4f} | ppl {metrics['perplexity']:.1f}")
    return metrics


def eval_dual(cfg: Config) -> dict:
    set_seed(cfg.seed)
    panel = _load_or_build_panel(cfg)
    loaders, std = build_day_loaders(panel, cfg)
    device = resolve_device(cfg.train.device)
    model = DualVQVAE(cfg.model, d_in=panel.features.shape[2],
                      n_stocks=len(panel.tickers)).to(device)
    ckpt = Path(cfg.data.cache_dir) / "checkpoints" / "last.pt"
    if ckpt.exists():
        import torch

        state = torch.load(str(ckpt), map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    result = evaluate_dual(model, loaders["test"], device)
    for mod in model.active:
        print(f"[{mod}] recon {result[f'{mod}_recon_mse']:.4f} "
              f"(baseline {result[f'{mod}_baseline_mse']:.4f}) | "
              f"ppl {result[f'{mod}_perplexity']:.1f} | codes {result[f'{mod}_codes_used']:.2%}")
    return result
```

Extend the command list and dispatch in `main`:
```python
    parser.add_argument("command", choices=["ingest", "build-panel", "baseline",
                                            "train-tokenizer", "eval-tokenizer",
                                            "train-dual", "eval-dual"])
```
and before `return 0`:
```python
    elif args.command == "train-dual":
        train_dual(cfg)
    elif args.command == "eval-dual":
        eval_dual(cfg)
```

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: ALL tests pass (M0 + M1 + M2).

- [ ] **Step 6: Commit**

```bash
git add bubble_bi/eval/tokenizer_eval.py bubble_bi/cli.py tests/test_dual_cli.py
git commit -m "feat: dual eval + train-dual/eval-dual CLI"
```

---

### Task 7: Real smoke run + ablations + record (manual verification)

**Files:** none created; exercises the CLI on the real cached panel.

- [ ] **Step 1: Train the dual tokenizer on real data**

Run: `.venv/bin/python -m bubble_bi.cli train-dual --config configs/m2.yaml`
Expected: both TS and CS reconstruction fall over steps; a checkpoint appears at `artifacts/cache/checkpoints/last.pt`. (Uses the M0 panel cache; rebuilds if missing.)

- [ ] **Step 2: Evaluate on held-out test**

Run: `.venv/bin/python -m bubble_bi.cli eval-dual --config configs/m2.yaml`
Expected: for both `ts` and `cs`, recon MSE **below** its baseline; perplexities ≫ 1; healthy code usage.

- [ ] **Step 3: Run the two ablations**

Create `configs/m2_wo_cs.yaml` and `configs/m2_wo_ts.yaml` copied from `configs/m2.yaml` but with `model.active_modules: [ts]` and `[cs]` respectively (and a distinct `data.cache_dir` so checkpoints don't collide, e.g. `artifacts/cache_wo_cs` / `artifacts/cache_wo_ts`). Train + eval each:
```bash
.venv/bin/python -m bubble_bi.cli train-dual --config configs/m2_wo_cs.yaml
.venv/bin/python -m bubble_bi.cli eval-dual  --config configs/m2_wo_cs.yaml
.venv/bin/python -m bubble_bi.cli train-dual --config configs/m2_wo_ts.yaml
.venv/bin/python -m bubble_bi.cli eval-dual  --config configs/m2_wo_ts.yaml
```
Expected: each ablation trains and evaluates its single module; `w/o-CS` reproduces the TS-only (M1-like) path.

- [ ] **Step 4: Record results**

Append the observed TS/CS recon vs baseline, perplexities, code usage (full run + both ablations) to the M2 design doc (`docs/superpowers/specs/2026-07-09-m2-dual-vqvae-design.md`) and mark the master spec's M2 bullet **DONE**. Commit:
```bash
git add -A && git commit -m "docs: record M2 dual tokenizer results + ablations"
```

---

## Self-Review

**Spec coverage:**
- CS single-day encoder (stock-ID, masked) + decoder → Task 2 ✅
- cross-attention fusion before quantization, bidirectional, masked → Task 3 ✅
- DualVQVAE joint masked losses, active_modules, use_fusion, encode, reinit hook → Task 4 ✅
- day-indexed DayDataset + build_day_loaders → Task 1 ✅
- Trainer generalization (dict batches + multi-codebook reinit) → Task 5 ✅
- dual eval (recon vs baseline, perplexity, code usage) + CLI → Task 6 ✅
- config flags (cs_codebook_size, fusion_layers, use_fusion, active_modules) → Task 0 ✅
- ablations (w/o-TS, w/o-CS) + real smoke + freeze artifact → Task 7 ✅ (the trained checkpoint IS the frozen artifact consumed by M3)

**Placeholder scan:** No TBD/TODO; every code step is complete. Task 7 is explicit manual verification (drives the CLI).

**Type consistency:** `DualVQVAE.forward` returns `loss`/`recon_loss`/`perplexity` (Trainer-compatible, Task 4) consumed by the Trainer (Task 5) and `evaluate_dual` (Task 6). `reinit_dead_codes(out)` (Task 4) matches the Trainer's `hasattr` hook (Task 5). `build_day_loaders` returns `(loaders_dict, standardizer)` (Task 1) as used in Task 6. `CSEncoder(cfg, d_in, n_stocks)` / `CSDecoder(cfg, d_out, n_stocks)` (Task 2) and `CrossAttentionFusion(cfg)` (Task 3) match `DualVQVAE.__init__` usage (Task 4). `_encoder_stack` and `TSEncoder`/`TSDecoder` are imported from M1's `ts_vqvae` unchanged.

**Deferred-with-note:** the overfit-to-zero check is omitted for the dual model (fusion + CS bottleneck make a near-zero target unreliable in a fast unit test); learning is instead proven by the real smoke run (Task 7) and the masked-invariance / shape tests. Auto-resume of `train-dual` from `last.pt` is not wired (starts fresh), same as M1 — the resume *mechanism* is covered by Task 5's test.
