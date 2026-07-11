# M2 Phase 3 — Fusion Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase-3 fusion tokenizer: freeze the Phase-1 TS and Phase-2 CS encoders, fuse their continuous outputs (market→stock residual cross-attention), quantize once (`codebook_FUSION`), and reconstruct the whole market with a joint decoder — producing the single per-stock token M3 consumes.

**Architecture:** A `MarketToStockFusion` adds market context to each stock's TS latent by residual cross-attention over the single market vector. A `JointDecoder` reconstructs the whole market `[B,N,p,D]` by cross-stock attention then temporal expansion. `FusionVQVAE` wires frozen `TSEncoder` + frozen `CSFieldEncoder` → fusion → single VQ → joint decoder; only the fusion, codebook, and decoder train. New `train-fusion`/`eval-fusion` CLI load the two frozen checkpoints.

**Tech Stack:** PyTorch 2.13+cpu, existing `bubble_bi` package (Plan 1 pieces).

## Global Constraints

- Run everything as `.venv/bin/python` from repo root `/home/hockper/Documents/Code/Bubble Bi`.
- Fusion is market→stock with a **residual** and a **single** market key: `fused = z_ts + cross-attn(query=z_ts, kv=z_cs.unsqueeze(1))` (residual required — single-key cross-attention alone is degenerate).
- Phase 3 fuses the **continuous** frozen-encoder outputs; only `codebook_FUSION` produces the M3 token.
- Frozen encoders: `load_frozen` sets `requires_grad=False` and `.eval()`; the Trainer optimizes only `requires_grad` params.
- Masked losses/attention: invalid stocks are zero-filled, key-padding-masked, and excluded from the loss.
- Checkpoints: TS `cache/checkpoints/last.pt`, CS `cache/checkpoints_cs/last.pt`, fusion `cache/checkpoints_fusion/last.pt`.
- TDD; frequent commits. Do not merge the branch (M3 follows).

---

### Task 1: MarketToStockFusion (rewrite fusion.py)

**Files:**
- Modify (replace contents): `bubble_bi/models/fusion.py`
- Modify (rewrite): `tests/test_fusion.py`

**Interfaces:**
- Consumes: `ModelConfig` (`d_model`, `heads`, `fusion_layers`, `dropout`).
- Produces: `MarketToStockFusion(cfg)` — `forward(z_ts: [B,N,H], z_cs: [B,H]) -> [B,N,H]`. The old bidirectional `CrossAttentionFusion` is removed.

- [ ] **Step 1: Rewrite `tests/test_fusion.py`**

```python
import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.fusion import MarketToStockFusion


def _cfg():
    return ModelConfig(d_model=16, heads=2, fusion_layers=2, dropout=0.0)


def test_fusion_shapes():
    fus = MarketToStockFusion(_cfg())
    z_ts = torch.randn(3, 6, 16)
    z_cs = torch.randn(3, 16)
    out = fus(z_ts, z_cs)
    assert out.shape == (3, 6, 16)


def test_residual_keeps_stocks_distinct():
    # With a single shared market key, the attention output is identical for every
    # stock; the residual z_ts is what keeps distinct stocks distinct.
    torch.manual_seed(0)
    fus = MarketToStockFusion(_cfg()).eval()
    z_ts = torch.randn(1, 4, 16)
    z_cs = torch.randn(1, 16)
    with torch.no_grad():
        out = fus(z_ts, z_cs)
    # different input stocks -> different fused rows
    assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fusion.py -v`
Expected: FAIL (`ImportError: cannot import name 'MarketToStockFusion'`).

- [ ] **Step 3: Replace `bubble_bi/models/fusion.py`**

```python
from __future__ import annotations

import torch
import torch.nn as nn

from bubble_bi.config import ModelConfig


class MarketToStockFusion(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        H, heads, L = cfg.d_model, cfg.heads, cfg.fusion_layers

        def mk():
            return nn.MultiheadAttention(H, heads, dropout=cfg.dropout, batch_first=True)

        self.attn = nn.ModuleList([mk() for _ in range(L)])
        self.norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(L)])

    def forward(self, z_ts: torch.Tensor, z_cs: torch.Tensor) -> torch.Tensor:
        kv = z_cs.unsqueeze(1)                        # [B, 1, H]  single market key
        h = z_ts
        for attn, norm in zip(self.attn, self.norm):
            a, _ = attn(h, kv, kv)                    # [B, N, H]  each stock reads the market
            h = norm(h + a)                           # residual keeps stocks distinct
        return h
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fusion.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/models/fusion.py tests/test_fusion.py
git commit -m "feat: MarketToStockFusion (residual market->stock cross-attention)"
```

---

### Task 2: JointDecoder

**Files:**
- Create: `bubble_bi/models/joint_decoder.py`
- Test: `tests/test_joint_decoder.py`

**Interfaces:**
- Consumes: `ModelConfig` (`p`, `d_model`, `dec_layers`, `heads`, `ff`, `dropout`), `_encoder_stack` from `ts_vqvae`.
- Produces: `JointDecoder(cfg, d_out, n_stocks)` — `forward(z_q: [B,N,H], valid: [B,N]) -> [B,N,p,D]`.

- [ ] **Step 1: Write the failing test**

`tests/test_joint_decoder.py`:
```python
import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.joint_decoder import JointDecoder


def _cfg():
    return ModelConfig(p=4, d_model=16, dec_layers=1, heads=2, ff=32, dropout=0.0)


def test_joint_decoder_shapes():
    dec = JointDecoder(_cfg(), d_out=5, n_stocks=6)
    z_q = torch.randn(3, 6, 16)
    valid = torch.ones(3, 6, dtype=torch.bool)
    out = dec(z_q, valid)
    assert out.shape == (3, 6, 4, 5)


def test_joint_decoder_masks_invalid_stock_keys():
    # Changing a masked stock's token must not change a valid stock's reconstruction.
    torch.manual_seed(0)
    dec = JointDecoder(_cfg(), d_out=5, n_stocks=4).eval()
    valid = torch.tensor([[True, True, False, True]])
    z1 = torch.randn(1, 4, 16)
    z2 = z1.clone()
    z2[0, 2] = 50.0                               # perturb the invalid stock's token
    with torch.no_grad():
        o1 = dec(z1, valid)
        o2 = dec(z2, valid)
    # valid stocks (0,1,3) unchanged
    assert torch.allclose(o1[:, [0, 1, 3]], o2[:, [0, 1, 3]], atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_joint_decoder.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write implementation**

`bubble_bi/models/joint_decoder.py`:
```python
from __future__ import annotations

import torch
import torch.nn as nn

from bubble_bi.config import ModelConfig
from bubble_bi.models.ts_vqvae import _encoder_stack


class JointDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig, d_out: int, n_stocks: int):
        super().__init__()
        self.p = cfg.p
        self.stock_id = nn.Parameter(torch.zeros(1, n_stocks, cfg.d_model))
        self.cross = _encoder_stack(cfg, cfg.dec_layers)          # across stocks
        self.day_pos = nn.Parameter(torch.zeros(1, cfg.p, cfg.d_model))
        self.temporal = _encoder_stack(cfg, cfg.dec_layers)       # over days, per stock
        self.out = nn.Linear(cfg.d_model, d_out)

    def forward(self, z_q: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        B, N, H = z_q.shape
        h = z_q + self.stock_id
        h = self.cross(h, src_key_padding_mask=~valid)            # [B, N, H]
        h = h.reshape(B * N, 1, H) + self.day_pos                 # [B*N, p, H]
        h = self.temporal(h)                                      # [B*N, p, H]
        return self.out(h).reshape(B, N, self.p, -1)              # [B, N, p, D]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_joint_decoder.py -v`
Expected: PASS (2 tests). (Valid queries never attend to the masked key, so their outputs are invariant.)

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/models/joint_decoder.py tests/test_joint_decoder.py
git commit -m "feat: JointDecoder (cross-stock attention + temporal expand)"
```

---

### Task 3: FusionVQVAE

**Files:**
- Create: `bubble_bi/models/fusion_vqvae.py`
- Test: `tests/test_fusion_vqvae.py`

**Interfaces:**
- Consumes: `TSEncoder` (`ts_vqvae`), `CSFieldEncoder` (`cross_sectional`), `MarketToStockFusion` (`fusion`), `JointDecoder` (`joint_decoder`), `VectorQuantizerEMA` (`vq`), `ModelConfig`.
- Produces: `FusionVQVAE(cfg, d_in, n_stocks)` — `forward(batch: {"block":[B,N,L,D],"valid":[B,N]}) -> dict` with `recon [B,N,p,D]`, `loss`, `recon_loss`, `commit`, `diversity`, `perplexity`, `ids [B,N]`, `z_e [B*N,H]`. `encode(batch) -> ids [B,N] long`. `load_frozen(ts_ckpt, cs_ckpt) -> None`. `reinit_dead_codes(out)`. Attribute `dead_code_reinit_every`. `use_fusion=False` bypasses the CS branch.

- [ ] **Step 1: Write the failing test**

`tests/test_fusion_vqvae.py`:
```python
import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.fusion_vqvae import FusionVQVAE


def _cfg(**kw):
    base = dict(p=4, cs_p=3, d_model=16, fusion_codebook_size=16, enc_layers=1,
                dec_layers=1, fusion_layers=1, heads=2, ff=32, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


def _batch(B=3, N=5, L=4, D=6):
    block = torch.randn(B, N, L, D)
    valid = torch.ones(B, N, dtype=torch.bool)
    valid[0, 4] = False
    return {"block": block, "valid": valid}


def test_forward_shapes():
    model = FusionVQVAE(_cfg(), d_in=6, n_stocks=5)
    out = model(_batch())
    assert out["recon"].shape == (3, 5, 4, 6)      # p=4 window, whole market
    assert out["ids"].shape == (3, 5)              # one token per (stock, day)
    assert torch.isfinite(out["loss"])


def test_encode_returns_token_grid():
    model = FusionVQVAE(_cfg(), d_in=6, n_stocks=5)
    ids = model.encode(_batch())
    assert ids.shape == (3, 5) and ids.dtype == torch.long
    assert (ids >= 0).all() and (ids < 16).all()


def test_use_fusion_false_bypasses_cs():
    model = FusionVQVAE(_cfg(use_fusion=False), d_in=6, n_stocks=5)
    assert not hasattr(model, "cs_enc")
    assert not hasattr(model, "fusion")
    out = model(_batch())
    assert torch.isfinite(out["loss"])


def test_load_frozen_freezes_encoders(tmp_path):
    # Build a model, save fake TS/CS checkpoints from its own encoders, load_frozen,
    # then a train step must leave the encoders unchanged.
    torch.manual_seed(0)
    model = FusionVQVAE(_cfg(), d_in=6, n_stocks=5)
    ts_ck = tmp_path / "ts.pt"
    cs_ck = tmp_path / "cs.pt"
    torch.save({"model": {f"enc.{k}": v for k, v in model.ts_enc.state_dict().items()}}, ts_ck)
    torch.save({"model": {f"enc.{k}": v for k, v in model.cs_enc.state_dict().items()}}, cs_ck)
    model.load_frozen(str(ts_ck), str(cs_ck))
    assert all(not p.requires_grad for p in model.ts_enc.parameters())
    assert all(not p.requires_grad for p in model.cs_enc.parameters())

    before = model.ts_enc.embed.weight.clone()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    out = model(_batch())
    out["loss"].backward()
    opt.step()
    assert torch.allclose(before, model.ts_enc.embed.weight)   # frozen, unchanged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fusion_vqvae.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write implementation**

`bubble_bi/models/fusion_vqvae.py`:
```python
from __future__ import annotations

import torch
import torch.nn as nn

from bubble_bi.config import ModelConfig
from bubble_bi.models.cross_sectional import CSFieldEncoder
from bubble_bi.models.fusion import MarketToStockFusion
from bubble_bi.models.joint_decoder import JointDecoder
from bubble_bi.models.ts_vqvae import TSEncoder
from bubble_bi.models.vq import VectorQuantizerEMA


def _masked_mse(recon: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    err = ((recon - target) ** 2).reshape(recon.shape[0], recon.shape[1], -1).mean(-1)  # [B,N]
    v = valid.float()
    return (err * v).sum() / v.sum().clamp(min=1.0)


class FusionVQVAE(nn.Module):
    def __init__(self, cfg: ModelConfig, d_in: int, n_stocks: int):
        super().__init__()
        self.p = cfg.p
        self.cs_p = cfg.cs_p
        self.use_fusion = bool(cfg.use_fusion)
        self.lambda_div = cfg.lambda_div
        self.dead_code_reinit_every = cfg.dead_code_reinit_every
        self.ts_enc = TSEncoder(cfg, d_in)
        if self.use_fusion:
            self.cs_enc = CSFieldEncoder(cfg, d_in, n_stocks)
            self.fusion = MarketToStockFusion(cfg)
        self.vq = VectorQuantizerEMA(cfg.fusion_codebook_size, cfg.d_model,
                                     cfg.ema_decay, commitment_beta=cfg.beta_commit)
        self.dec = JointDecoder(cfg, d_in, n_stocks)

    @staticmethod
    def _load_prefixed(module: nn.Module, state: dict, prefix: str) -> None:
        sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        module.load_state_dict(sub)

    def load_frozen(self, ts_ckpt: str, cs_ckpt: str) -> None:
        ts = torch.load(ts_ckpt, map_location="cpu", weights_only=False)["model"]
        self._load_prefixed(self.ts_enc, ts, "enc.")
        self.ts_enc.eval()
        for p in self.ts_enc.parameters():
            p.requires_grad = False
        if self.use_fusion:
            cs = torch.load(cs_ckpt, map_location="cpu", weights_only=False)["model"]
            self._load_prefixed(self.cs_enc, cs, "enc.")
            self.cs_enc.eval()
            for p in self.cs_enc.parameters():
                p.requires_grad = False

    def _fused(self, block: torch.Tensor, valid: torch.Tensor):
        B, N, L, D = block.shape
        ts_in = block[:, :, -self.p:, :]                              # [B,N,p,D]
        z_ts = self.ts_enc(ts_in.reshape(B * N, self.p, D)).reshape(B, N, -1)
        if self.use_fusion:
            z_cs = self.cs_enc(block[:, :, -self.cs_p:, :], valid)    # [B,H]
            fused = self.fusion(z_ts, z_cs)
        else:
            fused = z_ts
        return ts_in, fused

    def forward(self, batch: dict) -> dict:
        block, valid = batch["block"], batch["valid"]
        B, N = valid.shape
        ts_in, fused = self._fused(block, valid)
        q = self.vq(fused.reshape(B * N, -1))
        recon = self.dec(q["z_q"].reshape(B, N, -1), valid)          # [B,N,p,D]
        recon_loss = _masked_mse(recon, ts_in, valid)
        loss = recon_loss + q["commit"] + self.lambda_div * q["diversity"]
        return {"recon": recon, "loss": loss, "recon_loss": recon_loss,
                "commit": q["commit"], "diversity": q["diversity"],
                "perplexity": q["perplexity"], "ids": q["ids"].reshape(B, N),
                "z_e": fused.reshape(B * N, -1).detach()}

    @torch.no_grad()
    def encode(self, batch: dict) -> torch.Tensor:
        block, valid = batch["block"], batch["valid"]
        B, N = valid.shape
        _, fused = self._fused(block, valid)
        return self.vq(fused.reshape(B * N, -1))["ids"].reshape(B, N).long()

    def reinit_dead_codes(self, out: dict) -> None:
        self.vq.reset_dead_codes(out["z_e"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fusion_vqvae.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/models/fusion_vqvae.py tests/test_fusion_vqvae.py
git commit -m "feat: FusionVQVAE (frozen encoders + fusion + single codebook + joint decoder)"
```

---

### Task 4: Trainer optimizes only trainable params

**Files:**
- Modify: `bubble_bi/train/trainer.py`
- Test: `tests/test_trainer_frozen.py`

**Interfaces:**
- Consumes: `FusionVQVAE` (Task 3), `Trainer`, `set_seed`.
- Produces: `Trainer` builds its optimizer over `[p for p in model.parameters() if p.requires_grad]`.

- [ ] **Step 1: Write the failing test**

`tests/test_trainer_frozen.py`:
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bubble_bi.config import TrainConfig
from bubble_bi.train.trainer import Trainer, set_seed


class _FrozenHalf(nn.Module):
    def __init__(self, d=6):
        super().__init__()
        self.frozen = nn.Linear(d, d)
        self.trainable = nn.Linear(d, d)
        self.dead_code_reinit_every = 10 ** 9
        for p in self.frozen.parameters():
            p.requires_grad = False

    def forward(self, batch):
        x = batch["block"]
        mse = ((self.trainable(self.frozen(x)) - x) ** 2).mean()
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


def test_trainer_excludes_frozen_params(tmp_path):
    set_seed(0)
    model = _FrozenHalf()
    ld = DataLoader(_DS(), batch_size=8, shuffle=True, drop_last=True)
    cfg = TrainConfig(max_steps=2, batch_size=8, val_every=2, ckpt_every=2,
                      device="cpu", amp=False)
    tr = Trainer(model, {"train": ld, "val": ld, "test": ld}, cfg, str(tmp_path), device="cpu")
    opt_ids = {id(p) for g in tr.opt.param_groups for p in g["params"]}
    assert id(model.frozen.weight) not in opt_ids       # frozen excluded from optimizer
    assert id(model.trainable.weight) in opt_ids
    before = model.frozen.weight.clone()
    tr.train()
    assert torch.allclose(before, model.frozen.weight)   # and stays unchanged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_trainer_frozen.py -v`
Expected: FAIL — the current optimizer is built over `model.parameters()`, so `model.frozen.weight` IS in the optimizer's param groups and the `not in opt_ids` assertion fails. (The final "stays unchanged" assertion would pass either way because AdamW skips grad-`None` params — the point of the fix is not handing frozen params to the optimizer at all.)

- [ ] **Step 3: Edit `bubble_bi/train/trainer.py`**

In `Trainer.__init__`, replace:
```python
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
```
with:
```python
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
```

- [ ] **Step 4: Run test + existing trainer tests**

Run: `.venv/bin/python -m pytest tests/test_trainer_frozen.py tests/test_trainer.py tests/test_trainer_dual.py tests/test_trainer_metrics.py -v`
Expected: PASS (frozen unchanged; all-trainable models still train as before).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/train/trainer.py tests/test_trainer_frozen.py
git commit -m "feat: Trainer optimizes only requires_grad params (frozen encoders)"
```

---

### Task 5: Fusion eval + CLI (train-fusion / eval-fusion)

**Files:**
- Modify: `bubble_bi/eval/tokenizer_eval.py`
- Modify: `bubble_bi/cli.py`
- Create: `configs/m2_fusion.yaml`
- Test: `tests/test_fusion_cli.py`

**Interfaces:**
- Consumes: `FusionVQVAE`, `Trainer`, `build_day_loaders`, `_load_or_build_panel`, `train_tokenizer`, `train_cs`.
- Produces: `evaluate_fusion(model, loader, device) -> dict` (`recon_mse`, `mean_baseline_mse`, `perplexity`, `codes_used_frac`); `train_fusion(cfg, run_name=None) -> dict`, `eval_fusion(cfg, run_name=None) -> dict`; `main` gains `train-fusion`, `eval-fusion`. Fusion checkpoints at `cache_dir/checkpoints_fusion/last.pt`; loads TS `cache_dir/checkpoints/last.pt` and CS `cache_dir/checkpoints_cs/last.pt`.

- [ ] **Step 1: Write the failing test**

`tests/test_fusion_cli.py`:
```python
import numpy as np
import pandas as pd

from bubble_bi.config import Config, DataConfig, FeatureConfig, ModelConfig, TrainConfig
from bubble_bi.cli import train_tokenizer, train_cs, train_fusion, eval_fusion


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
        model=ModelConfig(p=4, cs_p=3, d_model=16, codebook_size=16, cs_codebook_size=16,
                          fusion_codebook_size=16, enc_layers=1, dec_layers=1,
                          fusion_layers=1, heads=2, ff=32, dropout=0.0),
        train=TrainConfig(max_steps=8, batch_size=8, val_every=8, ckpt_every=8,
                          log_every=4, device="cpu", amp=False),
    )


def test_staged_train_fusion(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "cache").mkdir()
    _write_raw(tmp_path / "raw")
    cfg = _cfg(tmp_path)
    train_tokenizer(cfg, run_name="ts")           # Phase 1 -> checkpoints/last.pt
    train_cs(cfg, run_name="cs")                  # Phase 2 -> checkpoints_cs/last.pt
    m = train_fusion(cfg, run_name="fusion")      # Phase 3 loads both frozen
    assert m["step"] == 8
    assert (tmp_path / "cache" / "checkpoints_fusion" / "last.pt").exists()
    ev = eval_fusion(cfg, run_name="fusion")
    assert np.isfinite(ev["recon_mse"])
    assert 0.0 <= ev["codes_used_frac"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fusion_cli.py -v`
Expected: FAIL (`ImportError: cannot import name 'train_fusion'`).

- [ ] **Step 3: Add `evaluate_fusion` to `bubble_bi/eval/tokenizer_eval.py`**

```python
@torch.no_grad()
def evaluate_fusion(model, loader, device) -> dict:
    model.eval()
    se, base, n, ppl, batches = 0.0, 0.0, 0, 0.0, 0
    used: set[int] = set()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        ts_in = batch["block"][:, :, -model.p:, :]
        valid = batch["valid"].float()
        denom = float(valid.sum().clamp(min=1.0))
        se += float(out["recon_loss"]) * denom
        base += float((ts_in.pow(2).mean(dim=(2, 3)) * valid).sum())
        n += denom
        ppl += float(out["perplexity"]); batches += 1
        used.update(out["ids"][batch["valid"]].tolist())
    return {"recon_mse": se / max(n, 1), "mean_baseline_mse": base / max(n, 1),
            "perplexity": ppl / max(batches, 1), "codes_used_frac": len(used) / model.vq.K}
```

- [ ] **Step 4: Add to `bubble_bi/cli.py`**

Add imports (near the model imports):
```python
from bubble_bi.eval.tokenizer_eval import evaluate_fusion
from bubble_bi.models.fusion_vqvae import FusionVQVAE
```

Add functions (below `eval_cs`):
```python
def train_fusion(cfg: Config, run_name: str | None = None) -> dict:
    set_seed(cfg.seed)
    panel = _load_or_build_panel(cfg)
    window_len = max(cfg.model.p, cfg.model.cs_p)
    loaders, std = build_day_loaders(panel, cfg, window_len=window_len)
    model = FusionVQVAE(cfg.model, d_in=panel.features.shape[2], n_stocks=len(panel.tickers))
    ts_ckpt = Path(cfg.data.cache_dir) / "checkpoints" / "last.pt"
    cs_ckpt = Path(cfg.data.cache_dir) / "checkpoints_cs" / "last.pt"
    if not ts_ckpt.exists():
        raise FileNotFoundError(f"missing TS checkpoint {ts_ckpt}; run train-tokenizer first")
    if cfg.model.use_fusion and not cs_ckpt.exists():
        raise FileNotFoundError(f"missing CS checkpoint {cs_ckpt}; run train-cs first")
    model.load_frozen(str(ts_ckpt), str(cs_ckpt))
    run_name = run_name or f"fusion_{cfg.train.max_steps}"
    ckpt_dir = Path(cfg.data.cache_dir) / "checkpoints_fusion"
    trainer = Trainer(model, loaders, cfg.train, str(ckpt_dir), standardizer=std,
                      run_dir=str(_run_dir(cfg, run_name)))
    metrics = trainer.train()
    trainer.logger.write_meta({"model": "fusion", "p": cfg.model.p, "cs_p": cfg.model.cs_p,
                               "fusion_codebook_size": cfg.model.fusion_codebook_size,
                               "use_fusion": cfg.model.use_fusion,
                               "n_features": int(panel.features.shape[2]),
                               "n_stocks": len(panel.tickers),
                               "max_steps": cfg.train.max_steps, "final": metrics})
    print(f"[fusion] trained {metrics['step']} steps | recon {metrics['recon']:.4f} "
          f"| val_mse {metrics['val_mse']:.4f} | ppl {metrics['perplexity']:.1f}")
    return metrics


def eval_fusion(cfg: Config, run_name: str | None = None) -> dict:
    set_seed(cfg.seed)
    panel = _load_or_build_panel(cfg)
    window_len = max(cfg.model.p, cfg.model.cs_p)
    loaders, std = build_day_loaders(panel, cfg, window_len=window_len)
    device = resolve_device(cfg.train.device)
    model = FusionVQVAE(cfg.model, d_in=panel.features.shape[2],
                        n_stocks=len(panel.tickers)).to(device)
    ckpt = Path(cfg.data.cache_dir) / "checkpoints_fusion" / "last.pt"
    if ckpt.exists():
        import torch

        state = torch.load(str(ckpt), map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    result = evaluate_fusion(model, loaders["test"], device)
    print(f"[fusion] recon {result['recon_mse']:.4f} (baseline {result['mean_baseline_mse']:.4f}) "
          f"| ppl {result['perplexity']:.1f} | codes {result['codes_used_frac']:.2%}")
    if run_name:
        _write_eval_json(cfg, run_name, result)
    return result
```

Extend the command list and dispatch in `main`:
```python
    parser.add_argument("command", choices=["ingest", "build-panel", "baseline",
                                            "train-tokenizer", "eval-tokenizer",
                                            "train-cs", "eval-cs",
                                            "train-fusion", "eval-fusion", "plot-metrics"])
```
and before `return 0`:
```python
    elif args.command == "train-fusion":
        train_fusion(cfg, run_name=run_name)
    elif args.command == "eval-fusion":
        eval_fusion(cfg, run_name=run_name)
```

- [ ] **Step 5: Write `configs/m2_fusion.yaml`**

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
  cs_p: 5
  d_model: 128
  codebook_size: 512
  cs_codebook_size: 512
  fusion_codebook_size: 512
  enc_layers: 3
  dec_layers: 2
  fusion_layers: 2
  heads: 4
  ff: 256
  use_fusion: true
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
git add bubble_bi/eval/tokenizer_eval.py bubble_bi/cli.py configs/m2_fusion.yaml tests/test_fusion_cli.py
git commit -m "feat: fusion eval + train-fusion/eval-fusion CLI (staged, frozen encoders)"
```

---

### Task 6: Real staged run + record (manual)

**Files:** none.

- [ ] **Step 1: Phase 1 — TS tokenizer (if not already trained this session)**

Run: `.venv/bin/python -m bubble_bi.cli train-tokenizer --config configs/m1.yaml --run-name ts`
Expected: `artifacts/cache/checkpoints/last.pt` (the frozen TS encoder source).

- [ ] **Step 2: Phase 2 — CS tokenizer**

Run: `.venv/bin/python -m bubble_bi.cli train-cs --config configs/m2_cs.yaml --run-name cs`
Expected: `artifacts/cache/checkpoints_cs/last.pt`.

- [ ] **Step 3: Phase 3 — fusion tokenizer (loads both frozen)**

Run: `.venv/bin/python -m bubble_bi.cli train-fusion --config configs/m2_fusion.yaml --run-name fusion`
Then: `.venv/bin/python -m bubble_bi.cli eval-fusion --config configs/m2_fusion.yaml --run-name fusion`
Expected: whole-market recon MSE below the mean baseline; fusion perplexity ≫ 1; healthy code usage. `artifacts/cache/checkpoints_fusion/last.pt` is the frozen tokenizer for M3.

- [ ] **Step 4: (optional) fusion-off ablation**

Copy `configs/m2_fusion.yaml` to `configs/m2_fusion_nofuse.yaml` with `model.use_fusion: false` and a distinct `data.cache_dir: artifacts/cache_nofuse` (so it rebuilds the panel and keeps checkpoints separate). Train + eval to compare recon with/without the market context.

- [ ] **Step 5: Record results**

Append the fusion recon/baseline/perplexity/codes (and the ablation if run) to the redefined-M2 design doc (`docs/superpowers/specs/2026-07-10-m2-staged-dual-vqvae-design.md`) under a "Phase 3 results" note; mark the master spec's M2 bullet **DONE (redefined)**. Commit:
```bash
git add -A && git commit -m "docs: record M2 Phase-3 (fusion tokenizer) results"
```

---

## Self-Review

**Spec coverage (Phase 3):**
- market→stock residual fusion, single key → Task 1 ✅
- joint decoder (cross-stock attn + temporal expand, masked) → Task 2 ✅
- `FusionVQVAE` (frozen TS+CS encoders, continuous fusion, single `codebook_FUSION`, joint decode, `encode`, `use_fusion` bypass, `load_frozen`) → Task 3 ✅
- Trainer optimizes only trainable params → Task 4 ✅
- fusion eval + `train-fusion`/`eval-fusion` CLI + config → Task 5 ✅
- staged real run (ts→cs→fusion) + `use_fusion` ablation + record → Task 6 ✅
- frozen artifact for M3 (`FusionVQVAE.encode → ids [B,N]`) → Task 3 (`encode`) + Task 6 (trained checkpoint) ✅

**Placeholder scan:** none; Task 6 is explicit manual verification.

**Type consistency:** `MarketToStockFusion(cfg).forward(z_ts[B,N,H], z_cs[B,H])->[B,N,H]` (Task 1) used by `FusionVQVAE._fused` (Task 3). `JointDecoder(cfg,d_out,n_stocks).forward(z_q[B,N,H], valid)->[B,N,p,D]` (Task 2) used by `FusionVQVAE.forward` (Task 3). `FusionVQVAE.forward` returns `recon_loss`/`loss`/`perplexity`/`ids[B,N]`/`z_e` (Task 3) consumed by the Trainer (Task 4) and `evaluate_fusion` (Task 5, uses `model.p`, `model.vq.K`). `load_frozen(ts_ckpt, cs_ckpt)` (Task 3) called by `train_fusion` (Task 5) with the Phase-1/2 checkpoint paths. `build_day_loaders(panel, cfg, window_len)` (Plan 1) called with `window_len=max(p,cs_p)` in Task 5. Checkpoint state dicts use the `{"model": state_dict}` shape saved by `Trainer.save_checkpoint`, and `enc.` is the submodule prefix in both `TSVQVAE` (`self.enc`) and `CSVQVAE` (`self.enc`), matching `_load_prefixed(..., "enc.")`.

**Note on AMP + frozen params:** on CPU (`amp=False`) `unscale_`/`step` operate only on the optimizer's param group (trainable), so frozen params are untouched; the Task-4 test asserts this directly.
