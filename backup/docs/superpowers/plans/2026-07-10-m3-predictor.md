# M3 — Next-Token Transformer (Llama-3 style) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a per-stock causal GPT (Llama-3-style block) over the frozen M2 fusion-token sequences that predicts the next day's token, evaluated by next-token accuracy/perplexity vs a marginal baseline.

**Architecture:** Pre-encode the panel once with the frozen `FusionVQVAE` into a token grid `ids[T,N]`. A `TokenSeqDataset` yields per-stock length-W windows. A `NextTokenPredictor` — embedding + N `LlamaBlock`s (RMSNorm + RoPE attention + SwiGLU, bias-free) + a bias-free head — is trained with cross-entropy via the existing `Trainer`.

**Tech Stack:** PyTorch 2.13+cpu, existing `bubble_bi` package (M0–M2).

## Global Constraints

- Run everything as `.venv/bin/python` from repo root `/home/hockper/Documents/Code/Bubble Bi`.
- **Categorical next-token only** (no regression head this milestone). Per-stock temporal only — **no cross-stock attention** (the fusion token already carries the market).
- **Llama-3-style block:** RMSNorm pre-norm, RoPE on q/k, SwiGLU FFN, **bias-free** linears. GQA configurable (`n_kv_heads`, 0 ⇒ full multi-head). **No KV-cache** (teacher-forced training).
- Vocab = `fusion_codebook_size` (512). Frozen tokenizer at `cache/checkpoints_fusion/last.pt` (has model + standardizer).
- Token grid excludes invalid stock-days (`-1`); windows never contain `-1`.
- Reuse the `Trainer` unmodified (predictor exposes `loss`/`recon_loss`/`perplexity`/`accuracy`, `dead_code_reinit_every=10**9`, no-op `reinit_dead_codes`).
- TDD; frequent commits.

---

### Task 0: Config fields

**Files:**
- Modify: `bubble_bi/config.py`
- Test: `tests/test_config_m3.py`

**Interfaces:**
- Produces: `ModelConfig` gains `pred_window: int = 64`, `pred_layers: int = 4`, `n_kv_heads: int = 0`, `rope_theta: float = 10000.0`.

- [ ] **Step 1: Write the failing test**

`tests/test_config_m3.py`:
```python
from bubble_bi.config import ModelConfig, load_config


def test_predictor_config_defaults():
    m = ModelConfig()
    assert m.pred_window == 64
    assert m.pred_layers == 4
    assert m.n_kv_heads == 0
    assert m.rope_theta == 10000.0


def test_predictor_config_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  tickers: [AAPL]\nmodel:\n  pred_window: 32\n  n_kv_heads: 2\n")
    cfg = load_config(str(p))
    assert cfg.model.pred_window == 32
    assert cfg.model.n_kv_heads == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config_m3.py -v`
Expected: FAIL (`pred_window` missing).

- [ ] **Step 3: Implement**

In `bubble_bi/config.py`, add to `ModelConfig` after `fusion_layers`:
```python
    pred_window: int = 64
    pred_layers: int = 4
    n_kv_heads: int = 0
    rope_theta: float = 10000.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config_m3.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/config.py tests/test_config_m3.py
git commit -m "feat: predictor config fields (pred_window/pred_layers/n_kv_heads/rope_theta)"
```

---

### Task 1: Llama primitives — RMSNorm, RoPE, SwiGLU

**Files:**
- Create: `bubble_bi/models/llama.py`
- Test: `tests/test_llama_primitives.py`

**Interfaces:**
- Produces, in `bubble_bi/models/llama.py`:
  - `RMSNorm(dim, eps=1e-6)` — `forward(x) -> x` normalized to unit RMS then scaled.
  - `RotaryEmbedding(head_dim, max_len, theta=10000.0)` — `forward(x: [B,H,T,head_dim]) -> [B,H,T,head_dim]` (RoPE applied over the first `T` positions).
  - `SwiGLU(dim, hidden)` — `forward(x) -> x` (bias-free `w2(silu(w1 x) * w3 x)`).

- [ ] **Step 1: Write the failing test**

`tests/test_llama_primitives.py`:
```python
import torch

from bubble_bi.models.llama import RMSNorm, RotaryEmbedding, SwiGLU


def test_rmsnorm_unit_rms():
    x = torch.randn(4, 8) * 5.0
    y = RMSNorm(8)(x)                                  # weight=1 initially
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones(4), atol=1e-4)


def test_rope_preserves_norm_and_varies_by_position():
    rope = RotaryEmbedding(head_dim=8, max_len=16)
    x = torch.randn(2, 3, 16, 8)                       # [B,H,T,hd]
    y = rope(x)
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-4)   # rotation preserves norm
    # same input vector at different positions -> different outputs
    xc = x.clone()
    xc[:, :, 5] = xc[:, :, 0]
    yc = rope(xc)
    assert not torch.allclose(yc[:, :, 0], yc[:, :, 5], atol=1e-4)


def test_swiglu_shape_and_bias_free():
    m = SwiGLU(8, 16)
    assert m(torch.randn(3, 8)).shape == (3, 8)
    assert all("bias" not in n for n, _ in m.named_parameters())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_llama_primitives.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write implementation**

`bubble_bi/models/llama.py`:
```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_len: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)              # [max_len, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)       # [max_len, head_dim]
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[-2]
        cos = self.cos[:T]                            # [T, head_dim]
        sin = self.sin[:T]
        return x * cos + _rotate_half(x) * sin


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_llama_primitives.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/models/llama.py tests/test_llama_primitives.py
git commit -m "feat: Llama primitives (RMSNorm, RoPE, SwiGLU)"
```

---

### Task 2: LlamaAttention + LlamaBlock

**Files:**
- Modify: `bubble_bi/models/llama.py`
- Test: `tests/test_llama_block.py`

**Interfaces:**
- Consumes: `RMSNorm`, `RotaryEmbedding`, `SwiGLU` (Task 1), `ModelConfig` (`d_model`, `heads`, `n_kv_heads`, `ff`, `dropout`, `pred_window`, `rope_theta`).
- Produces:
  - `LlamaAttention(cfg)` — `forward(x: [B,T,H]) -> [B,T,H]`, bias-free, RoPE, causal, GQA-capable.
  - `LlamaBlock(cfg)` — `forward(x: [B,T,H]) -> [B,T,H]` (pre-norm: `x + attn(norm(x))`, then `x + ffn(norm(x))`).

- [ ] **Step 1: Write the failing test**

`tests/test_llama_block.py`:
```python
import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.llama import LlamaAttention, LlamaBlock


def _cfg(**kw):
    base = dict(d_model=16, heads=4, n_kv_heads=0, ff=32, dropout=0.0,
                pred_window=8, rope_theta=10000.0)
    base.update(kw)
    return ModelConfig(**base)


def test_attention_shape_and_bias_free():
    attn = LlamaAttention(_cfg())
    x = torch.randn(2, 8, 16)
    assert attn(x).shape == (2, 8, 16)
    assert all("bias" not in n for n, _ in attn.named_parameters())


def test_attention_is_causal():
    torch.manual_seed(0)
    attn = LlamaAttention(_cfg()).eval()
    x = torch.randn(1, 8, 16)
    x2 = x.clone()
    x2[0, 7] = 99.0                                    # perturb the LAST position
    with torch.no_grad():
        y = attn(x)
        y2 = attn(x2)
    # earlier positions (0..6) must be unchanged by a future token
    assert torch.allclose(y[:, :7], y2[:, :7], atol=1e-5)


def test_gqa_runs_and_matches_shape():
    attn = LlamaAttention(_cfg(n_kv_heads=2))          # 4 query heads, 2 kv heads
    assert attn(torch.randn(2, 8, 16)).shape == (2, 8, 16)


def test_block_shape():
    blk = LlamaBlock(_cfg())
    assert blk(torch.randn(2, 8, 16)).shape == (2, 8, 16)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_llama_block.py -v`
Expected: FAIL (`ImportError: cannot import name 'LlamaAttention'`).

- [ ] **Step 3: Add to `bubble_bi/models/llama.py`**

```python
class LlamaAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.heads
        self.n_kv = cfg.n_kv_heads if cfg.n_kv_heads > 0 else cfg.heads
        assert self.n_heads % self.n_kv == 0, "heads must be divisible by n_kv_heads"
        self.head_dim = cfg.d_model // cfg.heads
        self.dropout = cfg.dropout
        self.q = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=False)
        self.k = nn.Linear(cfg.d_model, self.n_kv * self.head_dim, bias=False)
        self.v = nn.Linear(cfg.d_model, self.n_kv * self.head_dim, bias=False)
        self.o = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, cfg.pred_window, cfg.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = self.rope(q), self.rope(k)
        if self.n_kv < self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o(out)


class LlamaBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = LlamaAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_llama_block.py -v`
Expected: PASS (4 tests). The causal test confirms `scaled_dot_product_attention(is_causal=True)` masks the future.

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/models/llama.py tests/test_llama_block.py
git commit -m "feat: Llama attention (RoPE, causal, GQA, bias-free) + block"
```

---

### Task 3: NextTokenPredictor

**Files:**
- Create: `bubble_bi/models/predictor.py`
- Test: `tests/test_predictor.py`

**Interfaces:**
- Consumes: `LlamaBlock`, `RMSNorm` (Task 1-2), `ModelConfig`.
- Produces: `NextTokenPredictor(cfg, vocab)` — `forward(batch: {"tokens":[B,W],"targets":[B,W]}) -> dict` with `loss`, `recon_loss`, `perplexity`, `accuracy`, `logits [B,W,vocab]`; `hidden_state(tokens: [B,W]) -> [B,W,H]`; `reinit_dead_codes(out)` (no-op); attribute `dead_code_reinit_every`.

- [ ] **Step 1: Write the failing test**

`tests/test_predictor.py`:
```python
import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.predictor import NextTokenPredictor


def _cfg(**kw):
    base = dict(d_model=16, heads=4, n_kv_heads=0, ff=32, dropout=0.0,
                pred_window=8, pred_layers=2, rope_theta=10000.0)
    base.update(kw)
    return ModelConfig(**base)


def _batch(B=3, W=8, vocab=16):
    tokens = torch.randint(0, vocab, (B, W))
    targets = torch.randint(0, vocab, (B, W))
    return {"tokens": tokens, "targets": targets}


def test_forward_shapes_and_keys():
    model = NextTokenPredictor(_cfg(), vocab=16)
    out = model(_batch())
    assert out["logits"].shape == (3, 8, 16)
    assert torch.isfinite(out["loss"])
    assert 0.0 <= float(out["accuracy"]) <= 1.0


def test_hidden_state_shape():
    model = NextTokenPredictor(_cfg(), vocab=16)
    h = model.hidden_state(torch.randint(0, 16, (2, 8)))
    assert h.shape == (2, 8, 16)


def test_causal_future_does_not_change_earlier_logits():
    torch.manual_seed(0)
    model = NextTokenPredictor(_cfg(), vocab=16).eval()
    toks = torch.randint(0, 16, (1, 8))
    toks2 = toks.clone()
    toks2[0, 7] = (toks[0, 7] + 1) % 16                # change last token
    with torch.no_grad():
        l1 = model({"tokens": toks, "targets": toks})["logits"]
        l2 = model({"tokens": toks2, "targets": toks2})["logits"]
    assert torch.allclose(l1[:, :7], l2[:, :7], atol=1e-5)


def test_overfits_tiny_batch():
    torch.manual_seed(0)
    model = NextTokenPredictor(_cfg(), vocab=16).train()
    batch = _batch(B=4, W=8, vocab=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        out = model(batch)
        out["loss"].backward()
        opt.step()
    assert float(out["accuracy"]) > 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_predictor.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write implementation**

`bubble_bi/models/predictor.py`:
```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bubble_bi.config import ModelConfig
from bubble_bi.models.llama import LlamaBlock, RMSNorm


class NextTokenPredictor(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab: int):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, cfg.d_model)
        self.blocks = nn.ModuleList([LlamaBlock(cfg) for _ in range(cfg.pred_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab, bias=False)
        self.dead_code_reinit_every = 10 ** 9

    def hidden_state(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)

    def forward(self, batch: dict) -> dict:
        tokens, targets = batch["tokens"], batch["targets"]
        logits = self.head(self.hidden_state(tokens))          # [B, W, vocab]
        loss = F.cross_entropy(logits.reshape(-1, self.vocab), targets.reshape(-1))
        with torch.no_grad():
            acc = (logits.argmax(-1) == targets).float().mean()
            ppl = loss.detach().exp()
        return {"loss": loss, "recon_loss": loss.detach(), "perplexity": ppl,
                "accuracy": acc, "logits": logits}

    def reinit_dead_codes(self, out: dict) -> None:
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_predictor.py -v`
Expected: PASS (4 tests). The overfit test confirms the whole Llama stack learns.

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/models/predictor.py tests/test_predictor.py
git commit -m "feat: NextTokenPredictor (Llama-3 decoder over fusion tokens)"
```

---

### Task 4: Token grid pre-encode + TokenSeqDataset

**Files:**
- Create: `bubble_bi/data/token_grid.py`
- Test: `tests/test_token_grid.py`

**Interfaces:**
- Consumes: a frozen `FusionVQVAE` (for `encode`), `chronological_split` (windows.py), numpy/torch.
- Produces, in `bubble_bi/data/token_grid.py`:
  - `build_token_grid(model, std_features: np.ndarray, mask: np.ndarray, window_len: int, device, batch_days: int = 64) -> np.ndarray` — `ids[T, N]` int64, `-1` where a stock's `window_len`-block isn't fully valid.
  - `class TokenSeqDataset(Dataset)` — `__init__(grid: np.ndarray, W: int, day_range)`; `__getitem__` returns `{"tokens": LongTensor[W], "targets": LongTensor[W]}` from contiguous valid runs, assigned to the split by the last target day.
  - `build_token_loaders(grid, cfg) -> dict[str, DataLoader]`.

- [ ] **Step 1: Write the failing test**

`tests/test_token_grid.py`:
```python
import numpy as np
import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.fusion_vqvae import FusionVQVAE
from bubble_bi.data.token_grid import build_token_grid, TokenSeqDataset, build_token_loaders


def _cfg():
    return ModelConfig(p=4, cs_p=3, d_model=16, fusion_codebook_size=16, enc_layers=1,
                       dec_layers=1, fusion_layers=1, heads=2, ff=32, dropout=0.0,
                       pred_window=5)


def test_build_token_grid_shapes_and_invalid():
    torch.manual_seed(0)
    cfg = _cfg()
    model = FusionVQVAE(cfg, d_in=6, n_stocks=4).eval()
    T, N, D, L = 20, 4, 6, 4
    feats = np.random.default_rng(0).normal(size=(T, N, D)).astype(np.float32)
    mask = np.ones((T, N), dtype=bool)
    mask[:3, 0] = False                                # stock 0 invalid on first days
    grid = build_token_grid(model, feats, mask, window_len=L, device="cpu")
    assert grid.shape == (T, N)
    assert grid.dtype == np.int64
    # days before a full L-window are -1; and stock 0 stays -1 while its window overlaps day<3
    assert (grid[:L - 1] == -1).all()
    assert grid[3, 0] == -1                            # window [0..3] includes invalid day 0
    valid_tokens = grid[grid != -1]
    assert (valid_tokens >= 0).all() and (valid_tokens < cfg.fusion_codebook_size).all()


def test_token_seq_dataset_windows_and_shift():
    grid = np.arange(1, 41).reshape(20, 2).astype(np.int64) % 16   # [T=20, N=2], all valid
    ds = TokenSeqDataset(grid, W=5, day_range=(0, 20))
    item = ds[0]
    assert item["tokens"].shape == (5,) and item["targets"].shape == (5,)
    # targets are tokens shifted by one day
    assert torch.equal(item["targets"][:-1], item["tokens"][1:])


def test_token_seq_dataset_excludes_invalid_windows():
    grid = np.ones((12, 1), dtype=np.int64)
    grid[6, 0] = -1                                    # a gap
    ds = TokenSeqDataset(grid, W=4, day_range=(0, 12))
    # no returned window may contain -1
    for i in range(len(ds)):
        assert (ds[i]["tokens"] != -1).all() and (ds[i]["targets"] != -1).all()


def test_build_token_loaders_splits():
    from bubble_bi.config import Config, DataConfig
    grid = np.tile(np.arange(300, dtype=np.int64).reshape(300, 1) % 16, (1, 3))
    cfg = Config(data=DataConfig(tickers=["A", "B", "C"]), model=_cfg())
    cfg.train.batch_size = 8
    loaders = build_token_loaders(grid, cfg)
    assert set(loaders) == {"train", "val", "test"}
    batch = next(iter(loaders["train"]))
    assert batch["tokens"].shape[1] == cfg.model.pred_window
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_token_grid.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write implementation**

`bubble_bi/data/token_grid.py`:
```python
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from bubble_bi.data.windows import chronological_split


@torch.no_grad()
def build_token_grid(model, std_features: np.ndarray, mask: np.ndarray,
                     window_len: int, device, batch_days: int = 64) -> np.ndarray:
    model.eval().to(device)
    T, N, D = std_features.shape
    grid = np.full((T, N), -1, dtype=np.int64)
    days = list(range(window_len - 1, T))
    for i in range(0, len(days), batch_days):
        chunk = days[i:i + batch_days]
        blocks = np.zeros((len(chunk), N, window_len, D), dtype=np.float32)
        valids = np.zeros((len(chunk), N), dtype=bool)
        for bi, t in enumerate(chunk):
            for j in range(N):
                if mask[t - window_len + 1:t + 1, j].all():
                    blocks[bi, j] = std_features[t - window_len + 1:t + 1, j, :]
                    valids[bi, j] = True
        batch = {"block": torch.from_numpy(blocks).to(device),
                 "valid": torch.from_numpy(valids).to(device)}
        ids = model.encode(batch).cpu().numpy()        # [len(chunk), N]
        for bi, t in enumerate(chunk):
            grid[t, valids[bi]] = ids[bi, valids[bi]]
    return grid


class TokenSeqDataset(Dataset):
    def __init__(self, grid: np.ndarray, W: int, day_range):
        self.grid = grid
        self.W = W
        lo, hi = day_range
        T, N = grid.shape
        self.samples: list[tuple[int, int]] = []
        for j in range(N):
            col = grid[:, j]
            t = 0
            while t < T:
                if col[t] == -1:
                    t += 1
                    continue
                start = t
                while t < T and col[t] != -1:
                    t += 1
                # windows within [start, t): need W+1 tokens; assign by last target day
                for s in range(start, t - W):
                    if lo <= s + W < hi:
                        self.samples.append((j, s))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        j, s = self.samples[i]
        toks = self.grid[s:s + self.W, j]
        tgts = self.grid[s + 1:s + self.W + 1, j]
        return {"tokens": torch.from_numpy(np.ascontiguousarray(toks)).long(),
                "targets": torch.from_numpy(np.ascontiguousarray(tgts)).long()}


def build_token_loaders(grid: np.ndarray, cfg) -> dict:
    T = grid.shape[0]
    tr, va, te = chronological_split(T, cfg.data.train_frac, cfg.data.val_frac)
    W = cfg.model.pred_window
    bs, nw = cfg.train.batch_size, cfg.train.num_workers

    def mk(rng, shuffle):
        return DataLoader(TokenSeqDataset(grid, W, rng), batch_size=bs,
                          shuffle=shuffle, num_workers=nw, drop_last=shuffle)

    return {"train": mk(tr, True), "val": mk(va, False), "test": mk(te, False)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_token_grid.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/data/token_grid.py tests/test_token_grid.py
git commit -m "feat: token-grid pre-encode + TokenSeqDataset (per-stock windows)"
```

---

### Task 5: Predictor eval + CLI (tokenize / train-predictor / eval-predictor)

**Files:**
- Create: `bubble_bi/eval/predictor_eval.py`
- Modify: `bubble_bi/cli.py`
- Create: `configs/m3.yaml`
- Test: `tests/test_predictor_cli.py`

**Interfaces:**
- Consumes: `NextTokenPredictor`, `FusionVQVAE`, `build_token_grid`/`build_token_loaders`, `Trainer`, `Standardizer`, `_load_or_build_panel`.
- Produces:
  - `evaluate_predictor(model, loader, device, marginal: int) -> dict` (`accuracy`, `perplexity`, `baseline_accuracy`) and `rollout_accuracy(model, loader, device, horizon=5) -> float` in `predictor_eval.py`; plus `train_marginal_token(loader) -> int`.
  - `tokenize_panel(cfg) -> np.ndarray` (builds + caches `tokens.npz`), `train_predictor(cfg, run_name=None)`, `eval_predictor(cfg, run_name=None)` in `cli.py`; `main` gains `tokenize`, `train-predictor`, `eval-predictor`.

- [ ] **Step 1: Write the failing test**

`tests/test_predictor_cli.py`:
```python
import numpy as np
import pandas as pd

from bubble_bi.config import Config, DataConfig, FeatureConfig, ModelConfig, TrainConfig
from bubble_bi.cli import train_tokenizer, train_cs, train_fusion, tokenize_panel, train_predictor, eval_predictor


def _write_raw(raw, n=360, N=6):
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
                          fusion_codebook_size=16, enc_layers=1, dec_layers=1, fusion_layers=1,
                          heads=2, ff=32, dropout=0.0, pred_window=6, pred_layers=1),
        train=TrainConfig(max_steps=8, batch_size=8, val_every=8, ckpt_every=8,
                          log_every=4, device="cpu", amp=False),
    )


def test_tokenize_train_eval_predictor(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "cache").mkdir()
    _write_raw(tmp_path / "raw")
    cfg = _cfg(tmp_path)
    train_tokenizer(cfg, run_name="ts")
    train_cs(cfg, run_name="cs")
    train_fusion(cfg, run_name="fusion")
    grid = tokenize_panel(cfg)
    assert (tmp_path / "cache" / "tokens.npz").exists()
    assert grid.shape[1] == 6
    m = train_predictor(cfg, run_name="pred")
    assert m["step"] == 8
    assert (tmp_path / "cache" / "checkpoints_predictor" / "last.pt").exists()
    ev = eval_predictor(cfg, run_name="pred")
    assert 0.0 <= ev["accuracy"] <= 1.0
    assert "baseline_accuracy" in ev
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_predictor_cli.py -v`
Expected: FAIL (`ImportError: cannot import name 'tokenize_panel'`).

- [ ] **Step 3: Write `bubble_bi/eval/predictor_eval.py`**

```python
from __future__ import annotations

import torch


def train_marginal_token(loader) -> int:
    counts: dict[int, int] = {}
    for batch in loader:
        for tok in batch["targets"].reshape(-1).tolist():
            counts[tok] = counts.get(tok, 0) + 1
    return max(counts, key=counts.get) if counts else 0


@torch.no_grad()
def evaluate_predictor(model, loader, device, marginal: int) -> dict:
    model.eval()
    correct, total, ce_sum, batches, base_correct = 0, 0, 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        preds = out["logits"].argmax(-1)
        targets = batch["targets"]
        correct += int((preds == targets).sum())
        base_correct += int((targets == marginal).sum())
        total += targets.numel()
        ce_sum += float(out["loss"]); batches += 1
    import math
    return {"accuracy": correct / max(total, 1),
            "baseline_accuracy": base_correct / max(total, 1),
            "perplexity": math.exp(ce_sum / max(batches, 1))}


@torch.no_grad()
def rollout_accuracy(model, loader, device, horizon: int = 5) -> float:
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        targets = batch["targets"].to(device)
        W = tokens.shape[1]
        h = min(horizon, W - 1)
        seq = tokens[:, :W - h].clone()                      # context prefix
        for step in range(h):
            nxt = model({"tokens": seq, "targets": seq})["logits"][:, -1].argmax(-1)
            seq = torch.cat([seq, nxt[:, None]], dim=1)
            correct += int((nxt == targets[:, W - h - 1 + step]).sum())
            total += nxt.numel()
        break                                                # one batch is enough for a sanity check
    return correct / max(total, 1)
```

- [ ] **Step 4: Add to `bubble_bi/cli.py`**

Add imports (near the model imports):
```python
import numpy as np

from bubble_bi.data.token_grid import build_token_grid, build_token_loaders
from bubble_bi.data.windows import Standardizer
from bubble_bi.eval.predictor_eval import evaluate_predictor, rollout_accuracy, train_marginal_token
from bubble_bi.models.predictor import NextTokenPredictor
```

Add functions (below `eval_fusion`):
```python
def _load_frozen_fusion(cfg: Config, device):
    import torch

    panel = _load_or_build_panel(cfg)
    ckpt = Path(cfg.data.cache_dir) / "checkpoints_fusion" / "last.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing fusion tokenizer {ckpt}; run train-fusion first")
    state = torch.load(str(ckpt), map_location=device, weights_only=False)
    model = FusionVQVAE(cfg.model, d_in=panel.features.shape[2], n_stocks=len(panel.tickers))
    model.load_state_dict(state["model"])
    std = Standardizer()
    std.load_state_dict(state["standardizer"])
    return panel, model, std


def tokenize_panel(cfg: Config) -> "np.ndarray":
    device = resolve_device(cfg.train.device)
    panel, model, std = _load_frozen_fusion(cfg, device)
    window_len = max(cfg.model.p, cfg.model.cs_p)
    grid = build_token_grid(model, std.transform(panel.features), panel.mask, window_len, device)
    out = Path(cfg.data.cache_dir) / "tokens.npz"
    np.savez_compressed(out, grid=grid)
    print(f"token grid {grid.shape} → {out}  (valid {(grid != -1).mean():.1%})")
    return grid


def _load_or_build_token_grid(cfg: Config) -> "np.ndarray":
    p = Path(cfg.data.cache_dir) / "tokens.npz"
    if p.exists():
        return np.load(str(p))["grid"]
    return tokenize_panel(cfg)


def train_predictor(cfg: Config, run_name: str | None = None) -> dict:
    set_seed(cfg.seed)
    grid = _load_or_build_token_grid(cfg)
    loaders = build_token_loaders(grid, cfg)
    model = NextTokenPredictor(cfg.model, vocab=cfg.model.fusion_codebook_size)
    run_name = run_name or f"pred_{cfg.train.max_steps}"
    ckpt_dir = Path(cfg.data.cache_dir) / "checkpoints_predictor"
    trainer = Trainer(model, loaders, cfg.train, str(ckpt_dir),
                      run_dir=str(_run_dir(cfg, run_name)))
    metrics = trainer.train()
    trainer.logger.write_meta({"model": "predictor", "pred_window": cfg.model.pred_window,
                               "pred_layers": cfg.model.pred_layers,
                               "vocab": cfg.model.fusion_codebook_size,
                               "max_steps": cfg.train.max_steps, "final": metrics})
    print(f"[pred] trained {metrics['step']} steps | ce {metrics['recon']:.4f} "
          f"| val {metrics['val_mse']:.4f} | ppl {metrics['perplexity']:.1f}")
    return metrics


def eval_predictor(cfg: Config, run_name: str | None = None) -> dict:
    set_seed(cfg.seed)
    grid = _load_or_build_token_grid(cfg)
    loaders = build_token_loaders(grid, cfg)
    device = resolve_device(cfg.train.device)
    model = NextTokenPredictor(cfg.model, vocab=cfg.model.fusion_codebook_size).to(device)
    ckpt = Path(cfg.data.cache_dir) / "checkpoints_predictor" / "last.pt"
    if ckpt.exists():
        import torch

        state = torch.load(str(ckpt), map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    marginal = train_marginal_token(loaders["train"])
    result = evaluate_predictor(model, loaders["test"], device, marginal)
    result["rollout_accuracy"] = rollout_accuracy(model, loaders["test"], device)
    print(f"[pred] acc {result['accuracy']:.2%} (baseline {result['baseline_accuracy']:.2%}) "
          f"| ppl {result['perplexity']:.1f} | rollout {result['rollout_accuracy']:.2%}")
    if run_name:
        _write_eval_json(cfg, run_name, result)
    return result
```

Extend the command list and dispatch in `main`:
```python
    parser.add_argument("command", choices=["ingest", "build-panel", "baseline",
                                            "train-tokenizer", "eval-tokenizer",
                                            "train-cs", "eval-cs",
                                            "train-fusion", "eval-fusion",
                                            "tokenize", "train-predictor", "eval-predictor",
                                            "plot-metrics"])
```
and before `return 0`:
```python
    elif args.command == "tokenize":
        tokenize_panel(cfg)
    elif args.command == "train-predictor":
        train_predictor(cfg, run_name=run_name)
    elif args.command == "eval-predictor":
        eval_predictor(cfg, run_name=run_name)
```

- [ ] **Step 5: Write `configs/m3.yaml`**

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
  pred_window: 64
  pred_layers: 4
  n_kv_heads: 0
train:
  lr: 0.0001
  batch_size: 128
  max_steps: 1000
  val_every: 200
  ckpt_every: 200
  log_every: 20
  device: auto
seed: 42
```

- [ ] **Step 6: Run new test + full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: ALL tests pass.

- [ ] **Step 7: Commit**

```bash
git add bubble_bi/eval/predictor_eval.py bubble_bi/cli.py configs/m3.yaml tests/test_predictor_cli.py
git commit -m "feat: predictor eval + tokenize/train-predictor/eval-predictor CLI"
```

---

### Task 6: Real run + record (manual)

**Files:** none.

- [ ] **Step 1: Ensure the frozen fusion tokenizer exists**

The M2 pipeline already produced `artifacts/cache/checkpoints_fusion/last.pt`. If missing, run `train-tokenizer` (m1.yaml) → `train-cs` (m2_cs.yaml) → `train-fusion` (m2_fusion.yaml) first.

- [ ] **Step 2: Build the token grid**

Run: `.venv/bin/python -m bubble_bi.cli tokenize --config configs/m3.yaml`
Expected: `artifacts/cache/tokens.npz`; prints grid shape `(≈4153, 30)` and the valid fraction.

- [ ] **Step 3: Train + eval the predictor**

```bash
.venv/bin/python -m bubble_bi.cli train-predictor --config configs/m3.yaml --run-name pred
.venv/bin/python -m bubble_bi.cli eval-predictor  --config configs/m3.yaml --run-name pred
```
Expected: CE falls / next-token accuracy rises during training; held-out **accuracy above the marginal baseline**; perplexity and rollout accuracy printed.

- [ ] **Step 4: Record results**

Append the held-out accuracy / baseline / perplexity / rollout to the M3 design doc (`docs/superpowers/specs/2026-07-10-m3-predictor-design.md`) under a "Results" note; mark the master spec's M3 bullet **DONE**. Commit:
```bash
git add -A && git commit -m "docs: record M3 next-token predictor results"
```

---

## Self-Review

**Spec coverage:**
- config `pred_window`/`pred_layers`/`n_kv_heads`/`rope_theta` → Task 0 ✅
- Llama primitives (RMSNorm, RoPE, SwiGLU, bias-free) → Task 1 ✅
- Llama attention (causal, RoPE, GQA) + block → Task 2 ✅
- `NextTokenPredictor` (embedding + blocks + head, CE loss, `hidden_state`, Trainer-compat) → Task 3 ✅
- token grid pre-encode (`-1` invalid) + `TokenSeqDataset` (no `-1`, split-by-last-day) → Task 4 ✅
- eval (accuracy/perplexity, marginal baseline, rollout) + `tokenize`/`train-predictor`/`eval-predictor` CLI + config → Task 5 ✅
- real run + record → Task 6 ✅
- Spec-2 handoff (`hidden_state` extractor) → Task 3 (`hidden_state`) ✅
- ablation is optional (not gated) — run `train-fusion` with `use_fusion: false` into a separate cache and repeat Tasks 5-6; noted, not a required task.

**Placeholder scan:** none; Task 6 is explicit manual verification.

**Type consistency:** `RMSNorm`/`RotaryEmbedding`/`SwiGLU` (Task 1) used by `LlamaAttention`/`LlamaBlock` (Task 2), used by `NextTokenPredictor` (Task 3). `NextTokenPredictor.forward` returns `loss`/`recon_loss`/`perplexity`/`accuracy`/`logits` (Task 3) — the Trainer (`_scalars`, `evaluate`) and `evaluate_predictor` (Task 5) consume these. `build_token_grid(model, std_features, mask, window_len, device)` and `build_token_loaders(grid, cfg)` and `TokenSeqDataset(grid, W, day_range)` (Task 4) are called in `tokenize_panel`/`train_predictor` (Task 5). `FusionVQVAE.encode(batch)->ids[B,N]` (M2) is used by `build_token_grid`. The fusion checkpoint's `state["standardizer"]` is written by `Trainer.save_checkpoint` when `standardizer` is passed (M2 `train_fusion` passes it).

**Note:** M3 reuses the Trainer's `recon_loss`/`val_mse` slots to carry the CE loss (a classifier, not a reconstructor) — intentional, so the metrics/plots stack works unchanged; the printed labels say "ce".
