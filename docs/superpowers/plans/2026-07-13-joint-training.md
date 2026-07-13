# Joint Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train TS, CS, the fusion and the predictor under **one loss, one optimiser, from step zero** — so the tokenizer is shaped by what we actually want (tomorrow) instead of by what it can cheaply rebuild (today).

**Architecture:** Two anchored codebooks (TS and CS each keep their own codebook *and* decoder, each rebuilding its own grid). The cross-attention enriches the TS latent with the market **before** the TS codebook quantises it. A GPT reads a sentence of `(ts_token, cs_token)` pairs. The next-token head reads a **detached** copy of the token vectors, so it can never reshape the vocabulary — the collapse channel is physically severed.

**Tech Stack:** Python 3.14, PyTorch, NumPy, pandas, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-13-joint-training-design.md`. Read it before Task 1.
- Repo root `/home/hockper/Documents/Code/Bubble Bi`; venv `.venv`; branch `main`.
- **Never `pip install torch`.** Colab ships a torch matched to its GPU; a careless install swaps in a CPU-only wheel.
- **No real training on this machine** (the user's PC is too slow). All tests use tiny synthetic panels on CPU and must run in seconds.
- **Never copy STORM's `1e-3` reconstruction weight.** Theirs is a Frobenius norm (a **sum** over millions of elements); ours is a `.mean()`. Copying it deletes the anchor. Start from `recon = 1.0`.
- Code is read by a **non-programmer**: comments explain *why* in plain language and name the trap they prevent.
- Every notebook section ends with `bb.report(...)`, which raises if a check fails.
- Run tests with `.venv/bin/python -m pytest`. Suite is currently **280 passing**.

---

## Two things reading the code revealed that the spec did not know

**1. `attend_to = "days"` is a NO-OP at `cs_days = 1`.**
`VQVAE.context(..., how="days")` averages the market's cells over *companies*, yielding **one key per day**. Our tuned `cs_days = 1` therefore gives the cross-attention **a single key** — and softmax over one key is identically `1.0`. Every company receives the same market vector; no gradient can ever teach it otherwise.

**This is why the attention map was flat.** Not the training. Arithmetic. At `cs_days = 1` the only meaningful choice is `attend_to = "companies"` (30 keys — one per company, which is exactly "a bank can attend to banks").

**2. `context()` re-encodes the grid.** It calls `read_grid()` internally. Under joint training we need *both* the CS summary (for its codebook) and the CS cells (for the fusion) — calling both `read_grid()` and `context()` runs the CS encoder **twice per step**. Task 2 fixes this.

---

## File Structure

| file | responsibility |
|---|---|
| `bubble_bi/settings.py` | **modify** — new `loss` block; `fusion` loses `vocabulary` (there is no fused codebook); reject `attend_to="days"` when `cs["days"] == 1` |
| `bubble_bi/models/vqvae.py` | **modify** — `keys_from_cells()` so the CS encoder runs once |
| `bubble_bi/models/world.py` | **rewrite** — `Tokenizer` (two anchored codebooks, fusion before TS's) and `WorldModel` (two token streams, severed naming channel) |
| `bubble_bi/data/sentences.py` | **rewrite** — serve RAW GRIDS, batched by day-window so CS is encoded once per DAY |
| `bubble_bi/training.py` | **modify** — `train_joint()`: one loss, cold start, revive both codebooks |
| `Bubble_Bi.ipynb` | **modify** — replace the two-stage sections with one joint section |
| `tests/test_world.py`, `tests/test_sentences.py`, `tests/test_training.py`, `tests/test_settings.py` | tests |

---

### Task 1: Settings — the loss block, and the no-op attention

**Files:**
- Modify: `bubble_bi/settings.py`
- Test: `tests/test_settings.py`

**Interfaces:**
- Produces: `DEFAULTS["loss"] = {"predict": 1.0, "naming": 0.1, "recon": 1.0}`
- Produces: `DEFAULTS["fusion"]` keeps `depth`, `attend_to`, `batch`, `commitment`, `diversity`, `decay` — but **`vocabulary` is removed** (there is no fused codebook any more).
- Produces: `DEFAULTS["fusion"]["attend_to"] = "companies"` (was `"days"`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_settings.py`:

```python
def test_a_single_market_key_is_rejected_because_the_attention_would_be_a_no_op():
    """⚠️ THE BUG THAT MADE THE ATTENTION MAP FLAT, and it was arithmetic, not training.

    `attend_to="days"` averages the market's cells over COMPANIES, giving one key per day.
    At `cs["days"] == 1` that is ONE key — and softmax over one key is identically 1.0.
    Every company then receives the identical market vector, and no gradient can ever
    teach it otherwise. The cross-attention is not weak. It does not exist.

    Our tuned settings are exactly this: cs days = 1, attend_to = "days".
    """
    with pytest.raises(ValueError, match="one key"):
        bb.check({"tickers": ["AAPL"], "cs": {"days": 1},
                  "fusion": {"attend_to": "days"}})

    # 30 companies is a real menu: a bank can attend to banks.
    bb.check({"tickers": ["AAPL"], "cs": {"days": 1},
              "fusion": {"attend_to": "companies"}})           # must not raise


def test_the_loss_block_is_the_joint_objective():
    assert DEFAULTS["loss"] == {"predict": 1.0, "naming": 0.1, "recon": 1.0}


def test_there_is_no_fused_codebook_to_configure():
    """TS and CS keep their own codebooks, each ANCHORED by rebuilding its own grid. The
    fused codebook — the only one the predictor ever saw — is the one we watched collapse
    to 10 words of 512. It is deleted, not tuned."""
    assert "vocabulary" not in DEFAULTS["fusion"]
```

- [ ] **Step 2: Run them and watch them fail**

Run: `.venv/bin/python -m pytest tests/test_settings.py -k "no_op or loss_block or fused_codebook" -v`
Expected: FAIL — `check()` accepts the no-op combination, `DEFAULTS["loss"]` still has `candle`, and `fusion` still has `vocabulary`.

- [ ] **Step 3: Rewrite the blocks**

In `bubble_bi/settings.py`, replace the `fusion` and `loss` entries of `DEFAULTS`:

```python
    # Where the market reaches each company — BEFORE its codebook quantises it. After a
    # 9-bit quantisation the fine detail the attention needs is already destroyed, so this
    # is the only place it can happen.
    #
    # `attend_to` decides the menu the market offers:
    #   "companies"  one vector per company  (30 keys) -- a bank can attend to banks
    #   "cells"      every (company, day)             -- richest; identical to "companies"
    #                                                    when cs["days"] == 1
    #   "days"       one vector per market day        -- ⚠️ ONE KEY at cs["days"] == 1,
    #                                                    which makes the attention a no-op
    "fusion": {
        "depth": 2,
        "attend_to": "companies",
        "batch": 32,
    },

    # THE JOINT OBJECTIVE. Everything trains at once, against these.
    #
    #   predict   tomorrow's CANDLE. The real objective — the only target the model cannot
    #             manufacture for itself.
    #   naming    tomorrow's WORD. ⚠️ Historically the loss that ATE THE VOCABULARY: the
    #             model invents its own words and is then graded on guessing them, so
    #             making every day the same word scores perfectly. Measured: 92% accuracy
    #             at perplexity 2.2. It is safe now only because `WorldModel` computes it
    #             from a DETACHED copy of the tokens — see `world.py`. Keep it small: the
    #             candle carries the objective, naming only rides along.
    #   recon     rebuild today's TS grid and today's CS grid. THE ANCHOR. Every codebook
    #             in this project that had one stayed healthy (TS perplexity 157); the one
    #             that did not, collapsed (fusion, 10 of 512).
    #             ⚠️ STORM reports 1e-3 here. DO NOT COPY IT. Their reconstruction is a
    #             Frobenius norm — a SUM over millions of elements — and ours is a .mean().
    #             1e-3 on a sum of 4M terms is an enormous weight on the mean; copying the
    #             number would silently DELETE the anchor.
    "loss": {
        "predict": 1.0,
        "naming": 0.1,
        "recon": 1.0,
    },
```

- [ ] **Step 4: Reject the no-op attention**

Still in `bubble_bi/settings.py`, replace the existing `attend_to` validation with:

```python
    attend = out["fusion"]["attend_to"]
    if attend not in ("days", "companies", "cells"):
        raise ValueError(
            f"`fusion['attend_to']` must be 'days', 'companies' or 'cells' — got {attend!r}."
        )

    # ⚠️ How many keys does the market actually offer? If the answer is ONE, the
    # cross-attention is a NO-OP: softmax over a single key is identically 1.0, every
    # company receives the same market vector, and no gradient can ever change that. This
    # is not a weak attention. It is an absent one, and it is exactly what made our
    # attention map flat while we blamed the training.
    keys = {"days": out["cs"]["days"],
            "companies": len(out["tickers"]),
            "cells": out["cs"]["days"] * len(out["tickers"])}[attend]
    if keys < 2:
        raise ValueError(
            f"`fusion['attend_to'] = {attend!r}` with `cs['days'] = {out['cs']['days']}` and "
            f"{len(out['tickers'])} companies offers the attention exactly one key.\n"
            "     Softmax over ONE key is identically 1.0 — every company would receive the "
            "identical market vector and the cross-attention would be a NO-OP.\n"
            "     Use `attend_to = 'companies'` (one key per company: a bank can attend to "
            "banks)."
        )
```

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: failures in `tests/test_world.py` and anywhere that reads `settings["fusion"]["vocabulary"]` or `settings["loss"]["candle"]` — those are rewritten in Tasks 3–4. Fix only *settings* tests here; leave the rest.

- [ ] **Step 6: Commit**

```bash
git add bubble_bi/settings.py tests/test_settings.py
git commit -m "fix: attend_to='days' at cs_days=1 gives the attention ONE key -- a no-op

Softmax over one key is identically 1.0, so every company received the same
market vector and no gradient could ever change it. That is why the attention
map was flat: arithmetic, not training. check() now refuses it.

Also: the loss block becomes the joint objective (predict/naming/recon), and
`fusion` loses its `vocabulary` -- there is no fused codebook any more."
```

---

### Task 2: The CS encoder must run ONCE, not twice

`Tokenizer` needs the CS **summary** (for its codebook) and the CS **cells** (for the fusion). `context()` calls `read_grid()` internally, so asking for both runs the CS encoder twice — on the biggest grid in the model.

**Files:**
- Modify: `bubble_bi/models/vqvae.py`
- Test: `tests/test_vqvae.py`

**Interfaces:**
- Produces: `VQVAE.keys_from_cells(cells, present=None, how="companies") -> Tensor [B, keys, width]` — the same menu `context()` builds, from cells you have **already** computed.
- `context()` stays, implemented in terms of it, so existing callers are untouched.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_vqvae.py`:

```python
def test_the_market_can_be_encoded_once_and_used_twice():
    """The fusion needs the CS CELLS and the CS codebook needs the CS SUMMARY. `context()`
    re-encodes to get the cells, so asking for both ran the biggest encoder in the model
    TWICE every step. `read_grid()` already returns both — use them."""
    cs = VQVAE(companies=4, days=1, features=6, width=16, heads=2).eval()
    grid = torch.randn(3, 4, 1, 6)

    with torch.no_grad():
        summary, cells = cs.read_grid(grid)
        from_cells = cs.keys_from_cells(cells, None, "companies")
        from_grid = cs.context(grid, None, "companies")

    assert summary.shape == (3, 16)
    assert torch.allclose(from_cells, from_grid, atol=1e-6), (
        "keys_from_cells must build exactly the menu context() builds — it is an "
        "optimisation of that function, not a different one"
    )


def test_keys_from_cells_never_returns_a_single_key():
    """One key is a no-op (softmax over it is 1.0). The model must not be able to build
    one by accident, so this raises rather than silently doing nothing."""
    cs = VQVAE(companies=30, days=1, features=6, width=16, heads=2).eval()
    _, cells = cs.read_grid(torch.randn(2, 30, 1, 6))

    with pytest.raises(ValueError, match="one key"):
        cs.keys_from_cells(cells, None, "days")        # 1 day -> 1 key -> inert

    assert cs.keys_from_cells(cells, None, "companies").shape[1] == 30
```

- [ ] **Step 2: Run them and watch them fail**

Run: `.venv/bin/python -m pytest tests/test_vqvae.py -k "encoded_once or single_key" -v`
Expected: FAIL — `AttributeError: 'VQVAE' object has no attribute 'keys_from_cells'`.

- [ ] **Step 3: Implement it**

In `bubble_bi/models/vqvae.py`, add after `read_grid`:

```python
    def keys_from_cells(self, cells: torch.Tensor, present: torch.Tensor | None = None,
                        how: str = "companies") -> torch.Tensor:
        """The menu the market offers the cross-attention — from cells ALREADY encoded.

        `context()` below re-encodes the grid to get these cells. Under joint training the
        caller already needs `read_grid()`'s summary (for the codebook) AND its cells (for
        the fusion), so going through `context()` would run the biggest encoder in the model
        a second time, every step, for a tensor it is already holding.
        """
        if how not in ("days", "companies", "cells"):
            raise ValueError(
                f"`attend_to` must be 'days', 'companies' or 'cells' — got {how!r}."
            )
        b, c, d, w = cells.shape

        if how == "cells":
            keys = cells.reshape(b, c * d, w)
        elif how == "companies":                 # average over DAYS -> one per company
            keys = cells.mean(dim=2)
        elif present is None:                    # "days": average over COMPANIES
            keys = cells.mean(dim=1)
        else:
            weight = present.to(cells.dtype).unsqueeze(-1).unsqueeze(-1)
            keys = (cells * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)

        # ⚠️ A single key is a NO-OP: softmax over one key is identically 1.0, so every
        # company would receive the same market vector and the attention could never learn
        # anything. `settings.check()` refuses this combination, but a caller reaching this
        # function directly must not be able to build an inert model in silence.
        if keys.shape[1] < 2:
            raise ValueError(
                f"`{how}` over a {c}-company, {d}-day market gives the attention exactly "
                "one key. Softmax over one key is 1.0 — the cross-attention would be a "
                "NO-OP. Use 'companies'."
            )
        return keys
```

and rewrite `context()` to use it (deleting its duplicated body, keeping its docstring):

```python
    def context(self, grid: torch.Tensor, present: torch.Tensor | None = None,
                how: str = "companies") -> torch.Tensor:
        """What CS hands the fusion to attend over: [B, keys, width]. Encodes the grid, then
        builds the menu — see `keys_from_cells` if you have the cells already."""
        _, cells = self.read_grid(grid, present)
        return self.keys_from_cells(cells, present, how)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_vqvae.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/models/vqvae.py tests/test_vqvae.py
git commit -m "perf: encode the market ONCE -- keys_from_cells()

The fusion needs the CS cells and the CS codebook needs the CS summary.
context() re-encodes to get the cells, so asking for both ran the biggest
encoder in the model twice every step. read_grid() already returns both."
```

---

### Task 3: `Tokenizer` — two anchored codebooks, fusion before TS's

**Files:**
- Modify: `bubble_bi/models/world.py` (the `Tokenizer` class)
- Test: `tests/test_world.py`

**Interfaces:**
- Consumes: `VQVAE.read_grid`, `VQVAE.keys_from_cells` (Task 2), `VQVAE.rebuild`, `VQVAE.codebook`, `Fusion`.
- Produces: `Tokenizer(ts, cs, *, depth=2, attend_to="companies", heads=4, dropout=0.1, batch=None, commitment=..., diversity=..., decay=...)` — **`vocabulary` is gone**; it has **no codebook of its own**.
- Produces: `Tokenizer.forward(ts_grid, cs_grid, cs_present=None) -> dict` with keys:
  `ts_token [B]`, `cs_token [B]`, `ts_vector [B,W]`, `cs_vector [B,W]`, `attention [B,keys]`,
  `recon_loss`, `commitment_loss`, `diversity_loss`, `ts_perplexity`, `cs_perplexity`.

- [ ] **Step 1: Write the failing tests**

Replace the fusion-codebook tests in `tests/test_world.py` with:

```python
def _pair():
    ts = VQVAE(companies=1, days=2, features=26, width=32, heads=2, vocabulary=16)
    cs = VQVAE(companies=4, days=1, features=26, width=32, heads=2, vocabulary=16)
    return ts, cs


def test_the_tokenizer_has_no_codebook_of_its_own():
    """The fused codebook is the ONE we watched collapse -- 10 words of 512, while TS (which
    rebuilds its own grid) sat at 157. We delete it rather than keep patching it. TS and CS
    keep theirs, and theirs are ANCHORED."""
    from bubble_bi.models.world import Tokenizer

    ts, cs = _pair()
    tok = Tokenizer(ts, cs, model_size=32, attend_to="companies")

    assert not hasattr(tok, "codebook") or tok.codebook is None
    assert tok.ts.codebook is ts.codebook
    assert tok.cs.codebook is cs.codebook


def test_it_speaks_TWO_words_a_day_and_anchors_both():
    from bubble_bi.models.world import Tokenizer

    ts, cs = _pair()
    tok = Tokenizer(ts, cs, model_size=32, attend_to="companies")
    out = tok(torch.randn(5, 1, 2, 26), torch.randn(5, 4, 1, 26))

    assert out["ts_token"].shape == (5,)
    assert out["cs_token"].shape == (5,)
    assert torch.isfinite(out["recon_loss"])          # rebuilds BOTH grids
    assert out["attention"].shape[1] == 4             # one key per company


def test_the_ts_token_actually_READS_the_market():
    """The cross-attention must sit BEFORE the TS codebook. After a 9-bit quantisation the
    detail it needs is already destroyed, so this is the only place it can happen -- and if
    changing the market does not change the TS vector, the fusion is decorative."""
    from bubble_bi.models.world import Tokenizer

    torch.manual_seed(0)
    ts, cs = _pair()
    tok = Tokenizer(ts, cs, model_size=32, attend_to="companies").eval()

    stock = torch.randn(4, 1, 2, 26)
    with torch.no_grad():
        calm = tok(stock, torch.zeros(4, 4, 1, 26))["ts_vector"]
        panic = tok(stock, torch.randn(4, 4, 1, 26) * 5)["ts_vector"]

    assert not torch.allclose(calm, panic, atol=1e-4), (
        "the same stock, in two completely different markets, produced the same TS vector — "
        "the cross-attention is doing nothing"
    )
```

- [ ] **Step 2: Run them and watch them fail**

Run: `.venv/bin/python -m pytest tests/test_world.py -k "no_codebook_of_its_own or TWO_words or READS_the_market" -v`
Expected: FAIL — the current `Tokenizer` has its own `codebook` and returns a single `token`.

- [ ] **Step 3: Rewrite `Tokenizer`**

In `bubble_bi/models/world.py`, replace the whole `Tokenizer` class:

```python
class Tokenizer(nn.Module):
    """Two words a day: what THIS stock did, and what the MARKET did.

    ⚠️ There is no fused codebook here any more, and its absence is the design.

    Every codebook in this project that was anchored by a reconstruction loss stayed
    healthy — TS reached perplexity 157. The one that was not (the fused codebook, asked
    only to draw tomorrow's candle) collapsed to 10 words out of 512, on every loss balance
    we tried. Under joint training that would get worse, not better, because the naming loss
    is then free to pull on the encoders too.

    So TS and CS keep their OWN codebooks and their OWN decoders, each anchored by rebuilding
    its own grid, and the predictor reads both words. It is also far cheaper: the CS grid is
    identical for every company on a day, so it is encoded once per DAY rather than once per
    company-day.

    The cross-attention sits INSIDE the TS path, before the TS codebook quantises:

        ts_token  =  "what this stock did, GIVEN what the market was doing"
        cs_token  =  "what the market did"

    It has to be there. After a 9-bit quantisation the fine detail the attention needs is
    already gone. Reconstruction keeps the token honest; PREDICTION is what pays for the
    market context, because rebuilding the TS grid does not need CS at all.
    """

    def __init__(self, ts: VQVAE, cs: VQVAE, *, model_size: int, depth: int = 2,
                 attend_to: str = "companies", heads: int = 4, dropout: float = 0.1,
                 batch: int | None = None):
        # `batch` belongs to the data loader, not the model. Accepted and ignored purely so
        # the notebook can splat `**settings["fusion"]` in one go -- the same convention
        # VQVAE follows for its own `batch`/`steps`.
        del batch
        super().__init__()
        self.ts, self.cs = ts, cs
        self.attend_to = attend_to
        self.width = ts.read.out_features
        if model_size != self.width:
            raise ValueError(
                f"`model_size` is {model_size} but the TS/CS encoders were built "
                f"{self.width} wide. The cross-attention needs both sides the same width, "
                "so these must agree."
            )
        self.fusion = Fusion(self.width, depth=depth, heads=heads, dropout=dropout)

    def forward(self, ts_grid: torch.Tensor, cs_grid: torch.Tensor,
                cs_present: torch.Tensor | None = None) -> dict:
        # --- CS: an ordinary VQ-VAE pass. Its codebook is anchored by rebuilding its grid.
        # read_grid gives BOTH the summary (for the codebook) and the cells (for the
        # fusion), so the biggest encoder in the model runs exactly once.
        cs_summary, cs_cells = self.cs.read_grid(cs_grid, cs_present)
        cs_chosen = self.cs.codebook(cs_summary)
        cs_recon = _rebuild_loss(self.cs, cs_chosen["snapped"], cs_grid, cs_present)
        market = self.cs.keys_from_cells(cs_cells, cs_present, self.attend_to)

        # --- TS: encode, THEN read the market, THEN quantise.
        ts_summary, _ = self.ts.read_grid(ts_grid)
        fused, attention = self.fusion(ts_summary, market)
        ts_chosen = self.ts.codebook(fused)
        ts_recon = _rebuild_loss(self.ts, ts_chosen["snapped"], ts_grid, None)

        return {
            "ts_token": ts_chosen["ids"],
            "cs_token": cs_chosen["ids"],
            "ts_vector": ts_chosen["snapped"],
            "cs_vector": cs_chosen["snapped"],
            "attention": attention,
            "recon_loss": ts_recon + cs_recon,
            "commitment_loss": ts_chosen["commitment_loss"] + cs_chosen["commitment_loss"],
            "diversity_loss": ts_chosen["diversity_loss"] + cs_chosen["diversity_loss"],
            "ts_perplexity": ts_chosen["perplexity"],
            "cs_perplexity": cs_chosen["perplexity"],
            "ts_summary": fused,          # pre-quantisation, for reviving dead words
            "cs_summary": cs_summary,
        }


def _rebuild_loss(model: VQVAE, snapped: torch.Tensor, grid: torch.Tensor,
                  present: torch.Tensor | None) -> torch.Tensor:
    """Rebuild the grid from the snapped word and score it. THE ANCHOR.

    Only companies that actually traded are scored -- rewarding the model for "rebuilding" a
    company that did not trade would be meaningless.
    """
    error = (model.rebuild(snapped) - grid).pow(2)
    if present is None:
        return error.mean()
    weight = present.unsqueeze(-1).unsqueeze(-1).to(error.dtype)
    return (error * weight).sum() / weight.expand_as(error).sum().clamp(min=1)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_world.py -k "no_codebook_of_its_own or TWO_words or READS_the_market" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/models/world.py tests/test_world.py
git commit -m "feat: Tokenizer speaks TWO words a day, and both are anchored

TS and CS keep their own codebooks AND decoders, each anchored by rebuilding
its own grid -- the arrangement we MEASURED to be healthy (TS perplexity 157).
The fused codebook, the only one the predictor ever saw, is the one that
collapsed to 10 of 512. It is deleted, not patched.

The cross-attention now sits INSIDE the TS path, before the TS codebook: after
a 9-bit quantisation the detail it needs is already gone."
```

---

### Task 4: `WorldModel` — and the severed collapse channel

**This is the most important task in the plan.**

**Files:**
- Modify: `bubble_bi/models/world.py` (the `WorldModel` class)
- Test: `tests/test_world.py`

**Interfaces:**
- Consumes: `Tokenizer.forward` (Task 3).
- Produces: `WorldModel(tokenizer, sentence=64, depth=4, heads=4, dropout=0.0, predict=1.0, naming=0.1, recon=1.0)`.
- Produces: `WorldModel.forward(batch) -> dict` where `batch` has `ts_grid [B,T,1,ts_days,F]`, `cs_grid [B,T,C,cs_days,F]`, `cs_present [B,T,C]`, `candle [B,T,4]`. Returns `loss`, `naming_loss`, `drawing_loss`, `recon_loss`, `commitment_loss`, `diversity_loss`, `shrugging`, `accuracy`, `persistence`, `ts_perplexity`, `cs_perplexity`, `ts_summary`, `cs_summary`.

- [ ] **Step 1: Write the failing tests — the two that matter most**

Add to `tests/test_world.py`:

```python
def test_the_naming_loss_CANNOT_touch_the_vocabulary():
    """⚠️ THE TEST THIS WHOLE DESIGN EXISTS FOR.

    The model invents its own words and is then graded on guessing them:

        naming = CrossEntropy( GPT(z1..zt),  id(z_{t+1}) )
                               └ the model's ┘ └ ALSO the model's ┘

    Two ways to drive that to zero: learn real dynamics (hard), or make every day the same
    word (trivial). Gradient descent takes the second, and it is a STABLE fixed point -- once
    the vocabulary is dead nothing pulls it back. We measured it: 92% accuracy at perplexity
    2.2.

    So the naming head reads a DETACHED copy of the token vectors. It can make the GPT better
    at predicting the language; it can never make the language easier. This test asserts the
    gradient path is severed -- if it ever reconnects, the codebook dies and the loss curve
    will look FINE while it happens.
    """
    world, batch = _tiny_world()
    world.zero_grad()
    world(batch)["naming_loss"].backward()

    guilty = [name for name, p in world.tokenizer.named_parameters()
              if p.grad is not None and p.grad.abs().sum() > 0]
    assert not guilty, (
        f"the naming loss reached the tokenizer through {guilty[:3]} — it can now buy a "
        "cheap win by collapsing the vocabulary, and it will"
    )


def test_the_anchor_DOES_reach_the_vocabulary():
    """The counterpart. Reconstruction must reach the encoders and codebooks -- it is the
    only thing holding the words apart. A severed naming channel is worthless if the anchor
    is severed too."""
    world, batch = _tiny_world()
    world.zero_grad()
    world(batch)["recon_loss"].backward()

    reached = [name for name, p in world.tokenizer.named_parameters()
               if p.grad is not None and p.grad.abs().sum() > 0]
    assert reached, "reconstruction reaches nothing — the codebooks have no anchor at all"


def test_the_candle_loss_DOES_reach_the_vocabulary():
    """This is the whole point of joint training: the FORECAST shapes the token. If the
    prediction loss cannot reach the tokenizer, we have simply rebuilt the two-stage model
    with extra steps."""
    world, batch = _tiny_world()
    world.zero_grad()
    world(batch)["drawing_loss"].backward()

    reached = [n for n, p in world.tokenizer.named_parameters()
               if p.grad is not None and p.grad.abs().sum() > 0]
    assert reached, "the forecast cannot shape the token — joint training is doing nothing"


def test_it_reports_persistence_and_shrugging_beside_its_own_scores():
    """A number without its floor gets quoted. Persistence is 'tomorrow's word is today's
    word'; shrugging is 'draw the average candle'."""
    world, batch = _tiny_world()
    out = world(batch)
    assert torch.isfinite(out["persistence"])
    assert torch.isfinite(out["shrugging"])
```

And the shared fixture, at the top of `tests/test_world.py`:

```python
def _tiny_world(sentence: int = 6, companies: int = 4, batch: int = 2):
    """A whole joint model over a tiny synthetic sentence. CPU, milliseconds."""
    from bubble_bi.models.world import Tokenizer, WorldModel

    torch.manual_seed(0)
    ts = VQVAE(companies=1, days=2, features=26, width=32, heads=2, vocabulary=16)
    cs = VQVAE(companies=companies, days=1, features=26, width=32, heads=2, vocabulary=16)
    tok = Tokenizer(ts, cs, model_size=32, attend_to="companies")
    world = WorldModel(tok, sentence=sentence, depth=1, heads=2)

    data = {
        "ts_grid": torch.randn(batch, sentence, 1, 2, 26),
        "cs_grid": torch.randn(batch, sentence, companies, 1, 26),
        "cs_present": torch.ones(batch, sentence, companies, dtype=torch.bool),
        "candle": torch.randn(batch, sentence, 4),
    }
    return world, data
```

- [ ] **Step 2: Run them and watch them fail**

Run: `.venv/bin/python -m pytest tests/test_world.py -k "CANNOT_touch or anchor_DOES or candle_loss_DOES or persistence_and_shrugging" -v`
Expected: FAIL — `WorldModel` still expects cached `z_ts`/`market` and has one token stream.

- [ ] **Step 3: Rewrite `WorldModel`**

In `bubble_bi/models/world.py`, replace the whole `WorldModel` class:

```python
class WorldModel(nn.Module):
    """A GPT that reads the market as a sentence of TWO words a day, and everything trains
    at once: the encoders, both codebooks, the fusion, and this.

    ⚠️ THE ONE THING THAT MUST NEVER BE UNDONE — see `forward`.

    This model invents its own vocabulary and is then graded on predicting it. That means
    "make every day the same word" scores perfectly, and it is a STABLE fixed point: once the
    vocabulary is dead, nothing pulls it back. We measured exactly that -- 92% next-token
    accuracy at perplexity 2.2, on a codebook of three live words.

    So the naming head reads a DETACHED copy of the tokens. It trains the GPT to read the
    language; it cannot rewrite the language to be easier. The forecast (`draw`) keeps its
    full gradient into the tokenizer -- that is the entire point of training jointly -- and
    reconstruction anchors the codebooks so the words stay apart.
    """

    def __init__(self, tokenizer: Tokenizer, sentence: int = 64, depth: int = 4,
                 heads: int = 4, dropout: float = 0.0,
                 predict: float = 1.0, naming: float = 0.1, recon: float = 1.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.sentence = sentence
        width = tokenizer.width

        self.predict_weight = predict
        self.naming_weight = naming
        self.recon_weight = recon

        self.ts_words = tokenizer.ts.codebook.words
        self.cs_words = tokenizer.cs.codebook.words

        self.blocks = nn.ModuleList(
            [Block(width, heads=heads, dropout=dropout) for _ in range(depth)])
        self.settle = RMSNorm(width)
        self.guess_ts = nn.Linear(width, self.ts_words, bias=False)   # head A, this stock
        self.guess_cs = nn.Linear(width, self.cs_words, bias=False)   # head A, the market
        self.draw = nn.Sequential(                                     # head B, the candle
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, len(CANDLE)))

    def understand(self, vectors: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            vectors = block(vectors)
        return self.settle(vectors)

    def forward(self, batch: dict) -> dict:
        b, t = batch["ts_grid"].shape[:2]

        # Every day of the sentence becomes two words. Flatten the sentence into the batch so
        # the encoders see them all at once.
        spoken = self.tokenizer(
            batch["ts_grid"].reshape(b * t, *batch["ts_grid"].shape[2:]),
            batch["cs_grid"].reshape(b * t, *batch["cs_grid"].shape[2:]),
            batch["cs_present"].reshape(b * t, -1) if "cs_present" in batch else None,
        )
        ts_ids = spoken["ts_token"].view(b, t)
        cs_ids = spoken["cs_token"].view(b, t)
        # The day's meaning is BOTH words. Summing keeps the sentence one vector per day, so
        # the GPT's sequence length stays T rather than 2T.
        vectors = (spoken["ts_vector"] + spoken["cs_vector"]).view(b, t, -1)

        # ── head B: tomorrow's CANDLE. Full gradient into the tokenizer. This is the whole
        # reason for training jointly: the FORECAST shapes the token.
        thought = self.understand(vectors[:, :-1])
        drawn = self.draw(thought)

        # ── head A: tomorrow's WORDS. ⚠️ READS A DETACHED COPY.
        #
        # `vectors.detach()` severs the gradient path from this loss back into the encoders,
        # the fusion and the codebooks. Naming can make the GPT better at reading the
        # language. It can NEVER make the language easier -- which is the shortcut it took
        # last time, all the way down to three live words.
        #
        # This costs a second pass through the GPT blocks (small -- the encoders dominate).
        # It buys a vocabulary that survives.
        blind = self.understand(vectors.detach()[:, :-1])
        said_ts = self.guess_ts(blind)
        said_cs = self.guess_cs(blind)

        next_ts, next_cs = ts_ids[:, 1:], cs_ids[:, 1:]
        naming = (F.cross_entropy(said_ts.reshape(-1, self.ts_words), next_ts.reshape(-1))
                  + F.cross_entropy(said_cs.reshape(-1, self.cs_words), next_cs.reshape(-1)))

        accuracy = ((said_ts.argmax(-1) == next_ts).float().mean()
                    + (said_cs.argmax(-1) == next_cs).float().mean()) / 2
        # THE HONEST BAR: "tomorrow's word is today's word". Regimes are sticky, so this is
        # hard to beat -- and it is the bar, never zero.
        persistence = ((ts_ids[:, :-1] == next_ts).float().mean()
                       + (cs_ids[:, :-1] == next_cs).float().mean()) / 2

        wanted = batch["candle"][:, 1:]                    # TOMORROW's candle
        drawing = F.mse_loss(drawn, wanted)
        # What you score by shrugging and drawing the average candle. The features are
        # normalised to spread 1, so this is ~1.0 -- which is what makes `drawing` mean
        # anything on its own.
        shrugging = wanted.pow(2).mean()

        loss = (self.predict_weight * drawing
                + self.naming_weight * naming
                + self.recon_weight * spoken["recon_loss"]
                + spoken["commitment_loss"]
                + spoken["diversity_loss"])

        return {
            "loss": loss,
            "drawing_loss": drawing,
            "naming_loss": naming,
            "recon_loss": spoken["recon_loss"],
            "commitment_loss": spoken["commitment_loss"],
            "diversity_loss": spoken["diversity_loss"],
            "shrugging": shrugging,
            "accuracy": accuracy,
            "persistence": persistence,
            "ts_perplexity": spoken["ts_perplexity"],
            "cs_perplexity": spoken["cs_perplexity"],
            "ts_summary": spoken["ts_summary"],          # for reviving dead words
            "cs_summary": spoken["cs_summary"],
            "attention": spoken["attention"].view(b, t, -1),
        }

    def describe(self) -> str:
        return (
            f"A GPT reading {self.sentence} days, two words each — "
            f"{self.ts_words} for the stock, {self.cs_words} for the market. "
            "Everything trains at once."
        )
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_world.py -q`
Expected: PASS. Delete any old test that asserts a single `token` stream or a fused codebook — those describe a model that no longer exists. Do **not** delete tests that assert gradient flow or codebook health.

- [ ] **Step 5: Prove the severed channel has teeth**

Temporarily change `blind = self.understand(vectors.detach()[:, :-1])` to `blind = self.understand(vectors[:, :-1])` (reconnecting the collapse channel), run
`.venv/bin/python -m pytest tests/test_world.py -k CANNOT_touch -q`
Expected: **FAIL**. Revert, confirm `git status` is clean, and record the failure message in your report.

- [ ] **Step 6: Commit**

```bash
git add bubble_bi/models/world.py tests/test_world.py
git commit -m "feat: the naming loss keeps its head but loses its teeth

The model invents its own vocabulary and is then graded on predicting it, so
'make every day the same word' scores perfectly -- and it is a STABLE fixed
point. Measured: 92% accuracy at perplexity 2.2, on three live words.

The naming head now reads a DETACHED copy of the tokens. It can make the GPT
better at reading the language; it can never make the language easier. The
forecast keeps its full gradient into the tokenizer -- that IS joint training --
and reconstruction anchors both codebooks.

A gradient test asserts the severed channel, and it fails the moment anyone
reconnects it."
```

---

### Task 5: `Sentences` — raw grids, and CS encoded once per DAY

Nothing is frozen any more, so nothing can be cached. The dataset must serve **raw grids**. But the CS grid is **identical for every company on a day** — serving it per company-day would encode the biggest grid in the model 30× more often than necessary. **A batch is therefore one time-window across ALL companies at once.**

**Files:**
- Rewrite: `bubble_bi/data/sentences.py`
- Test: `tests/test_sentences.py`

**Interfaces:**
- Consumes: `Batches` (`.arrays`, `.scaler`, `.days`) from `bubble_bi/data/tensors.py`.
- Produces: `make_sentences(batches, settings) -> dict[str, DataLoader]` with keys `"learn"`, `"tune"`, `"test"`. Each item is one **day-window** and yields, for a batch of `B` windows:
  `ts_grid [B, T, N, 1, ts_days, F]`, `cs_grid [B, T, cs_days·N... ]` — see the exact shapes in the code below.
- **`Memory`, `remember()` and the old `Sentences` are deleted.** They exist only to cache frozen encoders, and nothing is frozen now.

- [ ] **Step 1: Write the failing test — the saving the whole design rests on**

Replace `tests/test_sentences.py` with:

```python
import numpy as np
import pytest
import torch

import bubble_bi as bb
from bubble_bi.data.sentences import make_sentences


def test_the_market_is_encoded_once_per_DAY_not_once_per_company_day(tiny_batches,
                                                                     tiny_settings):
    """⚠️ THE SAVING THE WHOLE DESIGN RESTS ON.

    The CS grid is IDENTICAL for every company on a day. Serving it per company-day would
    push the biggest grid in the model through the biggest encoder in the model 30x more
    often than necessary, every single step. A batch is therefore one TIME WINDOW across all
    companies at once: `cs_grid` carries ONE copy per day, not one per company-day.

    If this silently regresses, training becomes ~30x slower and nothing says a word.
    """
    loaders = make_sentences(tiny_batches, tiny_settings)
    item = next(iter(loaders["learn"]))

    b, t = item["ts_grid"].shape[:2]
    companies = len(tiny_settings["tickers"])

    # TS: one grid per COMPANY per day.
    assert item["ts_grid"].shape[:3] == (b, t, companies)
    # CS: ONE grid per day -- no company axis at all.
    assert item["cs_grid"].shape[:2] == (b, t)
    assert item["cs_grid"].shape[2] == companies      # the 30 companies INSIDE the grid
    assert item["cs_grid"].dim() == 5                 # [B, T, C, cs_days, F] -- not 6


def test_a_sentence_never_straddles_a_gap_in_the_data(tiny_batches, tiny_settings):
    """A 'sentence' of days that are not consecutive is not a sentence. If a company stopped
    trading in the middle, the window is not usable and must not be served."""
    loaders = make_sentences(tiny_batches, tiny_settings)
    item = next(iter(loaders["learn"]))
    days = item["days"]                                # [B, T] -- the day index of each step
    assert (days.diff(dim=1) == 1).all(), "a sentence jumped over a missing day"


def test_the_test_period_is_built_but_never_iterated_by_training(tiny_batches, tiny_settings):
    """Unlike the tuning search (which must not even BUILD it), the world model is finally
    scored on `test` at the very end -- so it exists here. It is simply never fed to
    `train_joint`."""
    loaders = make_sentences(tiny_batches, tiny_settings)
    assert set(loaders) == {"learn", "tune", "test"}
```

Add the fixtures (reuse the synthetic panel already in `tests/test_tuning.py` — import it via a shared `conftest.py` if it is not already there; if `tests/conftest.py` does not exist, create it and move `_synthetic_panel`, `tiny_settings` and `tiny_batches` into it so both files share one panel rather than drifting apart).

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_sentences.py -q`
Expected: FAIL — `make_sentences` still takes a `tokenizer` and returns cached latents.

- [ ] **Step 3: Rewrite `sentences.py`**

Replace `bubble_bi/data/sentences.py` entirely:

```python
"""The market as a sentence: one time-window, every company, raw grids.

⚠️ Nothing is cached here any more, and that is the point.

The old version ran the FROZEN encoders once over history and stored their output, which is
what freezing buys you. Under joint training nothing is frozen -- the encoders are being
reshaped by the forecast every step -- so a cache would simply be wrong.

But the CS grid is IDENTICAL for every company on a given day. Serving it per company-day
would push the biggest grid in the model through the biggest encoder in the model thirty
times more often than necessary. So a batch is one TIME WINDOW across ALL companies at once:
the market is carried once per day, and encoded once per day.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from bubble_bi.data.tensors import Batches, _complete_windows
from bubble_bi.models.world import CANDLE

PERIODS = ("learn", "tune", "test")


class Sentences(Dataset):
    """One item = one WINDOW of `length` consecutive days, for every company at once."""

    def __init__(self, batches: Batches, day_pool: np.ndarray, settings: dict):
        arrays = batches.arrays
        self.x = batches.scaler.apply(arrays.x)              # [T, N, F]  normalised
        self.ok = arrays.ok                                   # [T, N]
        self.length = settings["predictor"]["sentence_length"]
        self.ts_days = settings["ts"]["days"]
        self.cs_days = settings["cs"]["days"]
        self.candle = [arrays.names.index(name) for name in CANDLE]

        # A day is usable as the END of a step only if both grids behind it are whole.
        ts_whole = _complete_windows(self.ok, self.ts_days)   # [T, N]
        cs_whole = _complete_windows(self.ok, self.cs_days)   # [T, N]
        market_ok = cs_whole.sum(axis=1) >= 2                 # enough companies traded

        usable = np.zeros(len(self.x), dtype=bool)
        usable[day_pool] = True
        step_ok = usable & market_ok & ts_whole.all(axis=1)

        # An unbroken run of `length` usable steps is a sentence. A window that jumps over a
        # missing day is not a sentence, it is two sentences stapled together.
        run, self.ends = 0, []
        for day, fine in enumerate(step_ok):
            run = run + 1 if fine else 0
            if run >= self.length:
                self.ends.append(day)

        self.cs_present = cs_whole

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, i: int) -> dict:
        end = self.ends[i]
        days = np.arange(end - self.length + 1, end + 1)      # [T]

        # TS: one grid per company per day -> [T, N, 1, ts_days, F]
        ts = np.stack([self.x[d - self.ts_days + 1: d + 1] for d in days])   # [T, ts_days, N, F]
        ts = ts.transpose(0, 2, 1, 3)[:, :, None, :, :]                       # [T, N, 1, ts_days, F]

        # CS: ONE grid per day -> [T, N, cs_days, F].  No company axis outside the grid:
        # this single copy is what every company on that day will read.
        cs = np.stack([self.x[d - self.cs_days + 1: d + 1] for d in days])   # [T, cs_days, N, F]
        cs = cs.transpose(0, 2, 1, 3)                                         # [T, N, cs_days, F]

        present = self.cs_present[days]                                       # [T, N]
        cs = np.where(present[:, :, None, None], cs, 0.0)

        return {
            "ts_grid": torch.from_numpy(np.ascontiguousarray(ts)).float(),
            "cs_grid": torch.from_numpy(np.ascontiguousarray(cs)).float(),
            "cs_present": torch.from_numpy(present.copy()),
            "candle": torch.from_numpy(
                np.ascontiguousarray(self.x[days][:, :, self.candle])).float(),  # [T, N, 4]
            "days": torch.from_numpy(days.copy()),
        }


def make_sentences(batches: Batches, settings: dict) -> dict[str, DataLoader]:
    """The market as sentences — learn, tune and test."""
    size = settings["fusion"]["batch"]
    out = {}
    for period in PERIODS:
        data = Sentences(batches, batches.days[period], settings)
        if len(data) == 0:
            raise ValueError(
                f"No usable {settings['predictor']['sentence_length']}-day sentences in the "
                f"'{period}' period. Either shorten `predictor['sentence_length']` or start "
                "the history earlier."
            )
        out[period] = DataLoader(
            data, batch_size=size, shuffle=(period == "learn"),
            drop_last=(period == "learn" and len(data) > size))
    return out
```

⚠️ Note the shapes the model must now handle: `ts_grid [B, T, N, 1, ts_days, F]` and `candle [B, T, N, 4]` carry a **company axis**, because one window covers every company. `WorldModel.forward` (Task 4) flattens `B·T` — it must now flatten `B·T·N` for the TS side while flattening only `B·T` for CS, and broadcast the single CS token to all N companies of that day. Update `WorldModel.forward` accordingly and extend `_tiny_world` to the new shapes.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_sentences.py tests/test_world.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/data/sentences.py tests/test_sentences.py tests/conftest.py
git commit -m "feat: sentences serve RAW GRIDS, and the market is encoded once per DAY

Nothing is frozen under joint training, so the encoder cache is not merely
unnecessary -- it would be WRONG. But the CS grid is identical for every company
on a day, so a batch is one TIME WINDOW across all companies: the market is
carried, and encoded, once per day rather than once per company-day.

That is a ~30x saving on the biggest encoder in the model, and it is the only
reason a two-token design is cheaper than the one-token design it replaces."
```

---

### Task 6: `train_joint()` — one loss, cold start, both codebooks watched

**Files:**
- Modify: `bubble_bi/training.py`
- Test: `tests/test_training.py`

**Interfaces:**
- Consumes: `WorldModel.forward` (Task 4), `make_sentences` (Task 5).
- Produces: `train_joint(world, loaders, settings, steps=None, revive_every=50, check_every=None, patience=5, quiet=False) -> History`.
- Produces: `score_joint(world, loader, where, limit=30) -> dict` with `drawing`, `shrugging`, `accuracy`, `persistence`, `ts_perplexity`, `cs_perplexity`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_training.py`:

```python
def test_a_joint_run_keeps_BOTH_dictionaries_alive():
    """⚠️ THE TEST THAT WOULD HAVE CAUGHT THE FUSION COLLAPSE.

    Perplexity genuinely STARTS at 1.0 -- one word for everything -- and only climbs out via
    the reconstruction anchor, the diversity loss and dead-word revival. Under joint training
    the naming loss used to be free to push it straight back down. If either dictionary ends
    a run near 1, the anchor or the severed naming channel has failed, and the loss curve
    will look perfectly healthy while it happens.
    """
    from bubble_bi.training import train_joint

    world, loaders, settings = _tiny_joint()
    history = train_joint(world, loaders, settings, steps=60, quiet=True)
    last = history.last()

    assert last["ts_perplexity"] > 2.0, f"the TS dictionary collapsed: {last}"
    assert last["cs_perplexity"] > 2.0, f"the CS dictionary collapsed: {last}"


def test_a_joint_run_reports_both_honest_floors():
    """Persistence for the word, shrugging for the candle. A number without its floor gets
    quoted, and this project has had to walk back two such numbers already."""
    from bubble_bi.training import train_joint

    world, loaders, settings = _tiny_joint()
    last = train_joint(world, loaders, settings, steps=20, quiet=True).last()

    for name in ("drawing", "shrugging", "accuracy", "persistence"):
        assert name in last, f"{name} is not reported"
```

Plus the fixture (in `tests/conftest.py`, beside the synthetic panel):

```python
@pytest.fixture
def _tiny_joint(tiny_batches, tiny_settings):
    """A whole joint model on the synthetic panel. CPU, seconds."""
    from bubble_bi.data.sentences import make_sentences
    from bubble_bi.models import VQVAE
    from bubble_bi.models.world import Tokenizer, WorldModel

    settings = {**tiny_settings,
                "predictor": {"sentence_length": 6, "depth": 1},
                "fusion": {"depth": 1, "attend_to": "companies", "batch": 2}}
    features = len(bb.data.names())
    n = len(settings["tickers"])

    ts = VQVAE(companies=1, features=features, width=16, **settings["ts"])
    cs = VQVAE(companies=n, features=features, width=16, **settings["cs"])
    tok = Tokenizer(ts, cs, model_size=16, **settings["fusion"])
    world = WorldModel(tok, sentence=6, depth=1, heads=2, **settings["loss"])
    return world, make_sentences(tiny_batches, settings), settings
```

- [ ] **Step 2: Run them and watch them fail**

Run: `.venv/bin/python -m pytest tests/test_training.py -k "joint" -v`
Expected: FAIL — `ImportError: cannot import name 'train_joint'`.

- [ ] **Step 3: Implement `train_joint` and `score_joint`**

Add to `bubble_bi/training.py`:

```python
@torch.no_grad()
def score_joint(world, loader, where: torch.device, limit: int = 30) -> dict:
    """How the joint model is doing, against its two HONEST floors.

        drawing  vs  shrugging     -- draw tomorrow's candle, vs draw the average candle
        accuracy vs  persistence   -- name tomorrow's words, vs say today's words again

    Never against zero. This project has had to walk back two numbers that were quoted
    without their floor.
    """
    world.to(where).eval()
    total = {k: 0.0 for k in ("drawing", "shrugging", "accuracy", "persistence",
                              "ts_perplexity", "cs_perplexity")}
    seen = 0
    for i, batch in enumerate(loader):
        if i >= limit:
            break
        out = world(_to(batch, where))
        total["drawing"] += float(out["drawing_loss"])
        total["shrugging"] += float(out["shrugging"])
        total["accuracy"] += float(out["accuracy"])
        total["persistence"] += float(out["persistence"])
        total["ts_perplexity"] += float(out["ts_perplexity"])
        total["cs_perplexity"] += float(out["cs_perplexity"])
        seen += 1
    world.train()
    return {k: v / max(seen, 1) for k, v in total.items()}


def train_joint(world, loaders: dict, settings: dict, steps: int | None = None,
                revive_every: int = 50, check_every: int | None = None,
                patience: int = 5, quiet: bool = False) -> History:
    """Everything at once: the encoders, BOTH codebooks, the fusion and the GPT.

    ⚠️ COLD START, on purpose. There is no reconstruction-only warm-up, not even a short one:
    pretraining would shape the representation for the compress-everything objective this
    whole design exists to escape.

    The price is that the GPT spends its first steps predicting a vocabulary of roughly one
    word (perplexity really does start at 1.0). Watch `ts_perplexity` and `cs_perplexity`
    from step one. If they never open, the cold start has dead-locked -- and the fix is a
    300-step reconstruction-only warm-up, added WITH EVIDENCE rather than on principle.
    """
    steps = steps or settings["steps"]
    check_every = check_every or max(1, steps // 10)
    where = pick_device(settings)
    world.to(where).train()

    optimiser = torch.optim.AdamW(world.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    feed = cycle(loaders["learn"])
    history = History()
    started = time.time()
    best, best_at, best_weights, stale = float("inf"), 0, None, 0
    revived = 0

    for step in range(1, steps + 1):
        out = world(_to(next(feed), where))
        optimiser.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(world.parameters(), 1.0)
        optimiser.step()

        # Dead words get dropped onto real encoder output, so they have somewhere realistic
        # to compete from. Both dictionaries -- there is no fused codebook to revive now.
        if step % revive_every == 0:
            revived += world.tokenizer.ts.codebook.revive_dead_words(
                out["ts_summary"].detach())
            revived += world.tokenizer.cs.codebook.revive_dead_words(
                out["cs_summary"].detach())

        if step % check_every == 0 or step == steps:
            scored = score_joint(world, loaders["tune"], where)
            history.add(step=step, revived=revived, **scored)
            if not quiet:
                print(f"  {step:>6}  candle {scored['drawing']:.3f} "
                      f"(shrug {scored['shrugging']:.3f})   "
                      f"words {scored['accuracy']:.1%} "
                      f"(persist {scored['persistence']:.1%})   "
                      f"perplexity TS {scored['ts_perplexity']:.0f} "
                      f"CS {scored['cs_perplexity']:.0f}")

            # Keep the best model on how well it DRAWS tomorrow -- naming accuracy is the
            # number that flatters a collapsed codebook, so it must never be what we select on.
            if scored["drawing"] < best - 1e-4:
                best, best_at, stale = scored["drawing"], step, 0
                best_weights = {k: v.detach().cpu().clone()
                                for k, v in world.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    if not quiet:
                        print(f"\n  ⏹  Stopped at step {step:,} — tomorrow's candle has not "
                              f"improved for {patience} checks.")
                    break

    history.seconds = time.time() - started
    history.best_step = best_at
    if best_weights is not None:
        world.load_state_dict(best_weights)
    return history
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_training.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/training.py tests/test_training.py tests/conftest.py
git commit -m "feat: train_joint() -- one loss, cold start, both dictionaries watched

Selection is on how well it DRAWS tomorrow, never on naming accuracy: accuracy
is precisely the number that flatters a collapsed codebook (92% on three live
words), so selecting on it would select FOR the collapse.

Perplexity for both codebooks is reported every check, from step one. A cold
start genuinely begins at perplexity 1.0 and has to climb out; if it never
does, we will see it rather than discover it at the end."
```

---

### Task 7: The notebook — one section instead of two

**Files:**
- Modify: `Bubble_Bi.ipynb`
- Test: `tests/test_notebook.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_notebook.py`:

```python
def test_the_notebook_has_no_two_stage_training_left():
    """Sections 8 and 10 trained TS and CS separately, then froze them. That IS the design
    this rewrite replaces -- leaving them would train the tokenizer for the wrong objective
    and then quietly overwrite it."""
    source = "\n".join("".join(c["source"]) for c in _cells())
    assert "bb.train(ts" not in source, "TS is still being trained on its own"
    assert "bb.train(cs" not in source, "CS is still being trained on its own"
    assert "bb.keep.load(ts" not in source, "a separately-trained TS is still being loaded"
    assert "train_joint" in source, "the joint trainer is not in the notebook"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_notebook.py -k two_stage -v`
Expected: FAIL — `bb.train(ts, ...)` is still in the notebook.

- [ ] **Step 3: Replace the training sections**

Using `NotebookEdit` (never hand-edit the JSON), **delete** the cells that train TS alone (`a1d29c5d`, `f389d96c`) and CS alone (`e6e2b727`, `df2a9a5c`), and replace the predictor cell (`df092e68`, `b34dc670`, `cea57592`) with:

**Markdown:**

```markdown
---
## 8. Train everything at once

The tokenizer used to be trained to **rebuild the present**, then frozen, and only then did
the predictor get a say. But rebuilding the present and carrying the future are different
jobs, and we measured the gap three ways — most damningly, our own tuning objective turned
out to be **gameable**: "does today survive the bottleneck?" is trivially maximised by making
the window *be* today, because what we asked the token to preserve was already sitting in its
input. The search found the loophole and drove `days` to the floor.

**Tomorrow is not in the grid.** A token cannot manufacture it. So everything now trains
against it — the encoders, both codebooks, the fusion and the GPT — under one loss.

### Two words a day, and both are anchored

| | |
|---|---|
| `ts_token` | what **this stock** did, *given what the market was doing* |
| `cs_token` | what the **market** did |

Each keeps its own dictionary **and its own decoder**, and must rebuild its own grid. That
reconstruction is the **anchor**: every dictionary in this project that had one stayed
healthy, and the one that did not — the old fused codebook — collapsed to **10 words out of
512**.

### ⚠️ The number to watch is perplexity, from the first line

This model **invents its own vocabulary and is then graded on predicting it.** "Make every
day the same word" scores perfectly. We have watched it happen: **92% accuracy at perplexity
2.2.**

The naming head now reads a **detached** copy of the words — it can learn to *read* the
language, but it can never rewrite the language to be easier. If perplexity still slides,
that protection has failed, and **the loss curve will look perfectly healthy while it does.**

> Judge it against its floors, never against zero:
> **persistence** ("tomorrow's word is today's word") and **shrugging** ("draw the average candle").
```

**Code:**

```python
n_features = len(bb.data.names())
n_companies = len(settings["tickers"])

ts = bb.models.VQVAE(companies=1, features=n_features,
                     width=settings["model_size"], **settings["ts"])
cs = bb.models.VQVAE(companies=n_companies, features=n_features,
                     width=settings["model_size"], **settings["cs"])

tokenizer = bb.models.Tokenizer(ts, cs, model_size=settings["model_size"],
                                **settings["fusion"])
world = bb.models.WorldModel(
    tokenizer,
    sentence=settings["predictor"]["sentence_length"],
    depth=settings["predictor"]["depth"],
    **settings["loss"],
)
book = bb.data.make_sentences(batches, settings)

print(world.describe())
print(f"   {len(book['learn'].dataset):,} sentences to learn from")

world_history = None
if bb.keep.load(world, "world", settings) is None:
    world_history = bb.train_joint(world, book, settings)
    bb.keep.save(world, "world", settings, steps=settings["steps"])
```

**Check:**

```python
bb.verify.joint(world, world_history, book, settings)
```

- [ ] **Step 4: Add `verify.joint`**

In `bubble_bi/verify.py`:

```python
def joint(world, history, book, settings: dict) -> None:
    """Section 8: everything trained at once — and did the language survive?"""
    from bubble_bi.training import pick_device, score_joint

    scored = score_joint(world, book["tune"], pick_device(settings))
    alive = min(scored["ts_perplexity"], scored["cs_perplexity"])
    beats_shrug = scored["drawing"] < scored["shrugging"]
    beats_persist = scored["accuracy"] > scored["persistence"]

    report(
        "8. Everything, at once",
        [
            ("Both dictionaries alive", alive > 20,
             f"TS {scored['ts_perplexity']:.0f}, CS {scored['cs_perplexity']:.0f} words in use"),
            ("Draws tomorrow better than shrugging", beats_shrug,
             f"{scored['drawing']:.3f} vs {scored['shrugging']:.3f}"),
            ("Names tomorrow better than persistence", beats_persist,
             f"{scored['accuracy']:.1%} vs {scored['persistence']:.1%}"),
        ],
        have=f"""
        One model. The encoders, both codebooks, the fusion and the GPT were all
        shaped by the same thing: tomorrow.
        Each day is two words — this stock, and the market.
        """,
        known_problem=(
            None if (alive > 20 and beats_shrug) else
            "The dictionary collapsed, or the model cannot beat shrugging. Perplexity is the "
            "one to read first: if it is near 1, the naming loss found its way back into the "
            "vocabulary and everything else is meaningless."
        ),
    )
```

- [ ] **Step 5: Run the suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS. Confirm the notebook is unexecuted (no `outputs`, no `execution_count`) — `tests/test_notebook.py` already asserts it.

- [ ] **Step 6: Commit**

```bash
git add Bubble_Bi.ipynb bubble_bi/verify.py tests/test_notebook.py
git commit -m "feat: the notebook trains everything at once

The two-stage sections are gone. They trained the tokenizer for the wrong
objective (rebuild the present) and then froze it before the predictor got a
say."
```

---

## Self-review

**Spec coverage:** two anchored codebooks (Task 3), fusion before quantisation (Task 3), severed naming channel (Task 4), cold start (Task 6), CS encoded once per day (Tasks 2 + 5), persistence and shrugging floors (Tasks 4 + 6), the loss block with `recon = 1.0` (Task 1), and every test the spec's Testing section names.

**Two gaps the spec did not know about, added here:** the `attend_to="days"` no-op at `cs_days=1` (Task 1 + Task 2) — which explains the flat attention map — and `context()` re-encoding the market (Task 2).

**Deliberately deferred, and stated in the spec:** the EMA teacher (only if perplexity slides — Task 6 makes that visible), and searching `w_predict` (the tuning machinery already exists; run it once the joint model trains at all).
