# Tokenizer Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An opt-in, two-stage hyperparameter search for the TS and CS tokenizers, scored by what a held-out linear probe can recover from the token — and, first, a fix for the settings that never reached a model at all.

**Architecture:** Three layers. (1) Connect the dead settings and add a structural test that makes "a setting no model reads" impossible to reintroduce. (2) A generic `bb.tuning` module: Optuna TPE + pruning, a pluggable scorer, resume-to-Drive, a head-to-head confirm, and a committed `tuned.json`. (3) A notebook section that, by default, *loads* the tuning rather than running it.

**Tech Stack:** Python 3.14, PyTorch, NumPy, pandas, Optuna (new — pure Python, **no torch dependency**), matplotlib, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-12-tokenizer-tuning-design.md`. Read it before Task 1.
- Repo root `/home/hockper/Documents/Code/Bubble Bi`; venv `.venv`; branch `main`.
- **Never `pip install torch`.** Colab ships a torch matched to its GPU. `requirements.txt` says so at the bottom, and we have already been burned by a CPU-only wheel silently replacing the CUDA build. Optuna is safe: pure Python, no torch dependency.
- **The search must never see the `test` period.** Enforced structurally in Task 4 — the test loader is *not built*, not merely not read.
- **The probe target is TODAY** — the last day of the window the model was just given. Never `arrays.y` (tomorrow's return). TS and CS are autoencoders; scoring them on a forecast would rank noise.
- **No training on the user's machine.** The user's PC is too slow; all real training happens on Colab. Tests must run on CPU in seconds — use tiny synthetic panels.
- Every notebook section ends with `bb.report(...)`, which raises if a check fails.
- Run tests with `.venv/bin/python -m pytest`.

---

## File Structure

| file | responsibility |
|---|---|
| `bubble_bi/settings.py` | **modify** — codebook knobs move into `ts`/`cs`/`fusion`; `loss` shrinks to the predictor's two heads; new `search` block and top-level `weight_decay` |
| `bubble_bi/models/vqvae.py` | **modify** — accept `decay`; `commitment` default `1.0` → `0.25` |
| `bubble_bi/models/world.py` | **modify** — `Tokenizer` passes the fusion codebook's knobs through |
| `bubble_bi/training.py` | **modify** — `weight_decay` from settings; an `on_check` callback so a search can prune |
| `bubble_bi/data/features/__init__.py` | **modify** — `by_family(settings)`: which feature names each family produces |
| `bubble_bi/data/tensors.py` | **modify** — `tuning_loaders()`: one entry's loaders at a given window, **learn + tune only** |
| `bubble_bi/tuning.py` | **new** — the scorer, the two-stage search, the confirm, `tuned.json` |
| `bubble_bi/plots.py` | **modify** — `tuning_importance()`; refactor `kept_by_family` onto `by_family` |
| `bubble_bi/__init__.py` | **modify** — export `tuning` |
| `requirements.txt` | **modify** — add `optuna` |
| `tests/test_settings.py` | **modify** — the structural "no decorative setting" test |
| `tests/test_tuning.py` | **new** |
| `Bubble_Bi.ipynb` | **modify** — new section between the tensors and the TS training |

---

### Task 1: Kill the decorative settings

The bug that motivated the whole spec: `SETTINGS["loss"]` is read by **nothing** outside `settings.py`. `VQVAE` runs on its own hardcoded defaults, and those *contradict* `Codebook`'s — we have trained the entire project at `commitment=1.0`, four times the literature value, while looking at a setting that said otherwise.

**Files:**
- Modify: `bubble_bi/settings.py`
- Modify: `bubble_bi/models/vqvae.py:56-72` (add `decay`), `:99-100` (pass it on)
- Test: `tests/test_settings.py`

**Interfaces:**
- Produces: `DEFAULTS["ts"]` / `["cs"]` gain `commitment: 0.25`, `diversity: 0.1`, `decay: 0.99`, `dropout: 0.1`, `heads: 4`. `DEFAULTS["fusion"]` gains `commitment: 0.25`, `diversity: 0.1`, `decay: 0.99`. `DEFAULTS["loss"]` shrinks to `{"naming": 0.1, "candle": 1.0}`. New `DEFAULTS["search"] = {"run": False, "trials": 12, "steps": 600}` and `DEFAULTS["weight_decay"] = 0.05`.
- Produces: `VQVAE(..., decay: float = 0.99)`.

- [ ] **Step 1: Write the failing structural test**

Add to `tests/test_settings.py`:

```python
import inspect

from bubble_bi.models import VQVAE
from bubble_bi.settings import DEFAULTS


def test_no_setting_in_an_entry_block_is_decorative():
    """A setting no model reads is worse than no setting: it LIES.

    This is the test that would have caught the bug this whole spec exists for.
    `loss['commitment']` sat in SETTINGS for the entire project, was validated on every
    run, and was handed to nothing. We trained at commitment=1.0 while reading 0.25.
    """
    accepted = set(inspect.signature(VQVAE.__init__).parameters)
    for entry in ("ts", "cs"):
        unread = set(DEFAULTS[entry]) - accepted
        assert not unread, (
            f"SETTINGS[{entry!r}] contains settings VQVAE never reads: {sorted(unread)}. "
            "Either wire them up or delete them — a setting that does nothing is a lie."
        )


def test_the_codebook_knobs_actually_reach_the_codebook():
    model = VQVAE(companies=1, features=6, width=16,
                  **{**DEFAULTS["ts"], "commitment": 0.9, "diversity": 0.7, "decay": 0.5})
    assert model.codebook.commitment == 0.9
    assert model.codebook.diversity == 0.7
    assert model.codebook.decay == 0.5


def test_commitment_defaults_to_the_literature_value():
    """0.25, not 1.0. An over-strong commitment pins the encoder to the codebook and is a
    documented cause of the collapse we see in fusion."""
    assert DEFAULTS["ts"]["commitment"] == 0.25
    assert DEFAULTS["cs"]["commitment"] == 0.25
    assert VQVAE(companies=1, days=4, features=6, width=16).codebook.commitment == 0.25
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_settings.py -k "decorative or codebook_knobs or literature" -v`

Expected: FAIL. `test_no_setting_in_an_entry_block_is_decorative` passes vacuously today (the keys aren't there yet), but `test_the_codebook_knobs_actually_reach_the_codebook` fails with `TypeError: __init__() got an unexpected keyword argument 'decay'`, and `test_commitment_defaults_to_the_literature_value` fails with `assert 1.0 == 0.25`.

- [ ] **Step 3: Give VQVAE a `decay` and fix its commitment default**

In `bubble_bi/models/vqvae.py`, change the `__init__` signature (currently lines 56-72):

```python
    def __init__(
        self,
        companies: int,
        days: int,
        features: int,
        *,
        vocabulary: int = 512,
        width: int = 128,
        encoder_depth: int = 3,
        decoder_depth: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        commitment: float = 0.25,
        diversity: float = 0.1,
        decay: float = 0.99,
        batch: int | None = None,
        steps: int | None = None,
    ):
```

and the codebook construction (currently lines 99-100):

```python
        self.codebook = Codebook(words=vocabulary, width=width, decay=decay,
                                 commitment=commitment, diversity=diversity)
```

- [ ] **Step 4: Restructure the settings blocks**

In `bubble_bi/settings.py`, replace the `ts`, `cs`, `fusion`, and `loss` entries of `DEFAULTS`, and add two new keys. The comments matter — they are the only documentation a notebook reader gets.

```python
    # Entry 1 — TS: what THIS stock has been doing (one stock, over time).
    "ts": {
        "days": 15,
        "vocabulary": 512,
        "encoder_depth": 3,
        "decoder_depth": 2,
        "heads": 4,
        "dropout": 0.1,
        "batch": 256,
        "steps": None,        # None -> use the shared `steps`
        # --- the codebook's own knobs. Every one of these reaches Codebook. ---
        # commitment  how hard the encoder is pulled towards a word it already has.
        #   ⚠️ 0.25 is the standard. We ran at 1.0 for the whole project by accident:
        #   `loss["commitment"]` was in SETTINGS, was validated on every run, and was
        #   handed to NOTHING. Too strong a commitment pins the encoder to the codebook
        #   and is a documented cause of collapse.
        "commitment": 0.25,
        "diversity": 0.1,     # punish the dictionary for crowding (STORM eq. 4)
        "decay": 0.99,        # the codebook is a moving average; this is its memory
    },
    # Entry 2 — CS: what the WHOLE MARKET was doing (all stocks, on a day).
    # ~30x FEWER grids than TS, each ~30x bigger. Hence its own, much smaller batch and
    # step budget: at batch 64, 10,000 steps is 243 passes and it overfits badly.
    "cs": {
        "days": 5,
        "vocabulary": 512,
        "encoder_depth": 3,
        "decoder_depth": 2,
        "heads": 4,
        "dropout": 0.1,
        "batch": 64,
        "steps": 2000,
        "commitment": 0.25,
        "diversity": 0.1,
        "decay": 0.99,
    },
    # Where the two entries merge into the single token we keep.
    #   "days"       one vector per market day   (5 keys)  -- what the paper does
    #   "companies"  one vector per company      (30 keys)
    #   "cells"      every (company, day)        (150 keys)
    "fusion": {
        "vocabulary": 512,
        "depth": 2,
        "attend_to": "days",
        "batch": 32,
        "commitment": 0.25,
        "diversity": 0.1,
        "decay": 0.99,
    },
    # The GPT that reads sentences of tokens.
    "predictor": {
        "sentence_length": 64,
        "depth": 4,
    },

    # The PREDICTOR's two heads. (commitment and diversity used to live here, which was
    # the bug: they belong to a codebook, and there are three codebooks.)
    #
    #   naming   predict tomorrow's WORD. ⚠️ This one rewards CHEATING: make every day the
    #            same word and it is trivially satisfied. We watched "accuracy" hit 87%
    #            on a 3-word codebook. The paper keeps it small (0.1); so do we.
    #   candle   draw tomorrow's CANDLE. The anchor: a collapsed token carries no
    #            information, so it cannot draw a candle, which makes the cheat expensive.
    "loss": {
        "naming": 0.1,
        "candle": 1.0,
    },

    # Finding the settings above, by measuring instead of guessing. OFF by default:
    # the search is a one-time act of discovery, and everyone after inherits its answer
    # from `tuned.json`. See docs/superpowers/specs/2026-07-12-tokenizer-tuning-design.md.
    "search": {
        "run": False,
        "trials": 12,
        "steps": 600,         # a CEILING per trial; early stopping usually ends it sooner
    },
```

and, alongside `learning_rate`:

```python
    "weight_decay": 0.05,     # STORM's value. Was hardcoded to 0.01 in training.py.
```

- [ ] **Step 5: Validate the new settings**

Still in `bubble_bi/settings.py`. Extend `_POSITIVE` with the new whole-number settings, and replace the old collapse guard (which read the now-moved `loss["diversity"]`).

Add to `_POSITIVE`:

```python
    ("ts", "heads"), ("cs", "heads"),
    ("search", "trials"), ("search", "steps"),
```

Replace the `loss` validation block (currently `settings.py:184-194`) with:

```python
    for name, weight in out["loss"].items():
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight < 0:
            raise ValueError(
                f"`loss['{name}']` must be a number of 0 or more, got {weight!r}."
            )

    for entry in ("ts", "cs", "fusion"):
        for knob in ("commitment", "diversity", "decay"):
            value = out[entry][knob]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"`{entry}['{knob}']` must be a number of 0 or more, got {value!r}."
                )
        if not 0 < out[entry]["decay"] < 1:
            raise ValueError(
                f"`{entry}['decay']` is the codebook's memory and must sit strictly "
                f"between 0 and 1, got {out[entry]['decay']!r}."
            )
        if out[entry]["commitment"] == 0 and out[entry]["diversity"] == 0:
            raise ValueError(
                f"With both `{entry}['commitment']` and `{entry}['diversity']` at 0, "
                "nothing holds the codebook apart: it will crowd onto a few words, and a "
                "token drawn from a handful of words carries almost no information."
            )

    if not isinstance(out["search"]["run"], bool):
        raise ValueError(
            f"`search['run']` must be True or False, got {out['search']['run']!r}."
        )

    if out["weight_decay"] < 0:
        raise ValueError(f"`weight_decay` must be 0 or more, got {out['weight_decay']!r}.")
```

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_settings.py tests/test_vqvae.py -q`

Expected: PASS. If `tests/test_settings.py` has an existing test asserting `DEFAULTS["loss"]` contains `commitment`, delete it — it was asserting the bug.

- [ ] **Step 7: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`

Expected: PASS. Anything that constructs `VQVAE` with a positional `commitment` or reads `settings["loss"]["diversity"]` must be updated to the new shape.

- [ ] **Step 8: Commit**

```bash
git add bubble_bi/settings.py bubble_bi/models/vqvae.py tests/test_settings.py
git commit -m "fix: the codebook settings were decorative -- nothing read them

SETTINGS['loss']['commitment'] was validated on every run and handed to no
model. VQVAE fell back to its own default of 1.0 while Codebook's was 0.25,
so the entire project trained at four times the literature commitment.

The knobs now live with the codebook that owns them (ts/cs/fusion each have
one), 'decay' is exposed for the first time, and a structural test over
inspect.signature makes a decorative setting impossible to reintroduce."
```

---

### Task 2: Wire the remaining consumers

Three more places read hardcoded numbers where a setting exists: the fusion codebook, the predictor's loss weights, and `weight_decay`.

**Files:**
- Modify: `bubble_bi/models/world.py` (the `Tokenizer` constructor, where the fusion `Codebook` is built)
- Modify: `bubble_bi/training.py:185` and `:389` (`weight_decay=0.01`)
- Test: `tests/test_settings.py`, `tests/test_world.py`

**Interfaces:**
- Consumes: `DEFAULTS["fusion"]`, `DEFAULTS["loss"]`, `DEFAULTS["weight_decay"]` from Task 1.
- Produces: `WorldModel.__init__` accepts every key of `DEFAULTS["loss"]` (under the names `naming` and `candle`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_settings.py`:

```python
def test_no_loss_weight_is_decorative():
    """Same disease as the entry blocks: the predictor's weights must reach the predictor."""
    from bubble_bi.models.world import WorldModel

    accepted = set(inspect.signature(WorldModel.__init__).parameters)
    unread = set(DEFAULTS["loss"]) - accepted
    assert not unread, (
        f"SETTINGS['loss'] contains weights WorldModel never reads: {sorted(unread)}."
    )


def test_the_optimiser_uses_the_weight_decay_setting():
    """It was hardcoded to 0.01 while STORM uses 0.05, and no setting existed at all."""
    import inspect as _inspect

    from bubble_bi import training

    source = _inspect.getsource(training)
    assert "weight_decay=0.01" not in source, (
        "training.py still hardcodes weight_decay=0.01 — it must come from settings."
    )
    assert DEFAULTS["weight_decay"] == 0.05
```

Add to `tests/test_world.py`:

```python
def test_the_fusion_codebook_gets_its_own_knobs():
    """The fusion codebook is the one that COLLAPSES to ~12 words. It had been running on
    class defaults, unexamined, because nothing passed it anything."""
    from bubble_bi.models.world import Tokenizer
    from bubble_bi.settings import DEFAULTS

    settings = {
        **{k: v for k, v in DEFAULTS.items() if k != "tickers"},
        "tickers": ["AAA"],
        "fusion": {**DEFAULTS["fusion"], "commitment": 0.8, "diversity": 0.6, "decay": 0.5},
    }
    ts = VQVAE(companies=1, days=4, features=6, width=16, heads=2)
    cs = VQVAE(companies=3, days=4, features=6, width=16, heads=2)
    tokenizer = Tokenizer(ts, cs, settings)

    assert tokenizer.codebook.commitment == 0.8
    assert tokenizer.codebook.diversity == 0.6
    assert tokenizer.codebook.decay == 0.5
```

- [ ] **Step 2: Run them and watch them fail**

Run: `.venv/bin/python -m pytest tests/test_settings.py::test_no_loss_weight_is_decorative tests/test_settings.py::test_the_optimiser_uses_the_weight_decay_setting tests/test_world.py::test_the_fusion_codebook_gets_its_own_knobs -v`

Expected: FAIL — `WorldModel.__init__` names them `candle_weight`/`naming_weight` (not `candle`/`naming`), `training.py` still says `weight_decay=0.01`, and the fusion codebook gets no knobs.

- [ ] **Step 3: Rename WorldModel's weights to match the settings**

In `bubble_bi/models/world.py`, the `WorldModel.__init__` signature currently ends with `candle_weight: float = 1.0, naming_weight: float = 0.1`. Rename the *parameters* to `candle` and `naming` so `WorldModel(..., **settings["loss"])` splats cleanly — the same pattern `VQVAE(**settings["ts"])` already uses. Keep the attributes as they are so the rest of the class is untouched:

```python
    def __init__(self, ..., dropout: float = 0.0,
                 candle: float = 1.0, naming: float = 0.1):
        ...
        self.candle_weight = candle
        self.naming_weight = naming
```

Update every construction site of `WorldModel` that passed `candle_weight=`/`naming_weight=` — find them with:

```bash
grep -rn "candle_weight\|naming_weight" --include=*.py --include=*.ipynb .
```

- [ ] **Step 4: Pass the fusion codebook its knobs**

In `bubble_bi/models/world.py`, find where `Tokenizer` builds its `Codebook` and hand it the fusion block:

```python
        fusion = settings["fusion"]
        self.codebook = Codebook(
            words=fusion["vocabulary"],
            width=settings["model_size"],
            commitment=fusion["commitment"],
            diversity=fusion["diversity"],
            decay=fusion["decay"],
        )
```

- [ ] **Step 5: Take `weight_decay` from settings**

In `bubble_bi/training.py`, both optimisers (`training.py:184-186` in `train()` and `training.py:388-389` in `train_world()`) currently read `weight_decay=0.01`. Change both to:

```python
        weight_decay=settings["weight_decay"],
```

- [ ] **Step 6: Run the tests**

Run: `.venv/bin/python -m pytest -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add bubble_bi/models/world.py bubble_bi/training.py tests/
git commit -m "fix: wire the last three hardcoded numbers to their settings

The fusion codebook -- the one that collapses to ~12 words -- had been
running on class defaults because nothing ever passed it anything. The
predictor's loss weights and weight_decay were the same story."
```

---

### Task 3: `by_family()` — ask each family what it produces

The tuning scorer needs the *volatility* family's column names. `plots.kept_by_family` already works this out, by building a fake price frame and asking each family module what it emits — but it does it inline, so nothing else can use it.

**Files:**
- Modify: `bubble_bi/data/features/__init__.py`
- Modify: `bubble_bi/data/__init__.py` (export it)
- Modify: `bubble_bi/plots.py:330-340` (use it instead of the inline copy)
- Test: `tests/test_features.py`

**Interfaces:**
- Produces: `bubble_bi.data.by_family(settings: dict) -> dict[str, list[str]]` — e.g. `{"candle": ["gap", "body", "upper_wick", "lower_wick"], "volatility": [...], ...}`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_features.py`:

```python
from bubble_bi.data import FAMILIES, by_family, names
from bubble_bi.settings import DEFAULTS


def test_by_family_covers_every_feature_exactly_once():
    grouped = by_family(DEFAULTS)
    assert set(grouped) == set(FAMILIES)

    flat = [n for group in grouped.values() for n in group]
    assert sorted(flat) == sorted(names()), "a feature is in two families, or in none"


def test_by_family_knows_where_the_candle_and_the_volatility_are():
    grouped = by_family(DEFAULTS)
    assert grouped["candle"] == ["gap", "body", "upper_wick", "lower_wick"]
    assert "realized_vol" in grouped["volatility"]
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_features.py -k by_family -v`

Expected: FAIL with `ImportError: cannot import name 'by_family'`.

- [ ] **Step 3: Implement it**

Add to `bubble_bi/data/features/__init__.py`:

```python
def by_family(settings: dict) -> dict[str, list[str]]:
    """Which feature names each family produces.

    Asked of the family modules themselves rather than kept as a hand-written list, so it
    cannot drift out of date the next time someone adds a feature.
    """
    import numpy as np

    fake = pd.DataFrame(
        {c: np.linspace(1.0, 2.0, 400) for c in ("open", "high", "low", "close", "volume")},
        index=pd.date_range("2020-01-01", periods=400, freq="B"),
    )
    return {name: list(module.build(fake, settings)) for name, module in FAMILIES.items()}
```

In `bubble_bi/data/__init__.py`, add `by_family` to the import from `bubble_bi.data.features` and to `__all__`.

- [ ] **Step 4: Use it in plots, deleting the inline copy**

In `bubble_bi/plots.py`, inside `kept_by_family`, replace the fake-frame block (currently `plots.py:330-340`) with:

```python
    from bubble_bi.data.features import by_family

    kept_by = {
        family: float(np.mean([kept[names.index(n)] for n in columns]))
        for family, columns in by_family(settings).items()
    }
    frame = pd.Series(kept_by).sort_values()
```

and drop the now-unused `import pandas as pd` if it was local to that block (`pd` is already imported at module top — check before deleting).

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_features.py tests/test_plots.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add bubble_bi/data/ bubble_bi/plots.py tests/test_features.py
git commit -m "feat: by_family() -- ask each family which features it produces

Was inline in plots.kept_by_family. The tuning scorer needs the volatility
family's columns too, and a hand-written list would drift the next time
someone adds a feature."
```

---

### Task 4: `tuning_loaders()` — the search cannot see `test`

The search varies `days`, which changes the grids, so the loaders must be rebuilt per trial. This is also where we make the no-lookahead guarantee **structural**: the test loader is not built. Not "not read" — *not built*. A comment saying "don't touch test" is not a defence.

**Files:**
- Modify: `bubble_bi/data/tensors.py`
- Test: `tests/test_tensors.py`

**Interfaces:**
- Consumes: `Batches` (has `.arrays`, `.scaler`, `.days`), `TSGrids`, `CSGrids` — all already in `tensors.py`.
- Produces: `bubble_bi.data.tensors.tuning_loaders(batches: Batches, entry: str, days: int, batch: int) -> dict[str, DataLoader]` with keys exactly `{"learn", "tune"}`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tensors.py` (reuse whatever fixture that file already uses to build a `Batches` from a small synthetic table — call it `batches` below):

```python
import pytest

from bubble_bi.data.tensors import tuning_loaders


def test_the_search_cannot_reach_the_test_days(batches):
    """Not 'does not read test' — CANNOT. The loader is never built.

    A search that could reach the test period would quietly tune on the answer, and the
    only defence that actually holds is not constructing the thing.
    """
    loaders = tuning_loaders(batches, "ts", days=4, batch=8)
    assert set(loaders) == {"learn", "tune"}
    assert "test" not in loaders


def test_tuning_loaders_rebuild_at_a_new_window_length(batches):
    short = tuning_loaders(batches, "ts", days=3, batch=8)
    long = tuning_loaders(batches, "ts", days=6, batch=8)
    assert next(iter(short["learn"]))["grid"].shape[2] == 3
    assert next(iter(long["learn"]))["grid"].shape[2] == 6


def test_tuning_loaders_serve_the_market_grid_for_cs(batches):
    loaders = tuning_loaders(batches, "cs", days=3, batch=4)
    grid = next(iter(loaders["learn"]))["grid"]
    assert grid.shape[1] == len(batches.arrays.tickers)   # every company, together
    assert grid.shape[2] == 3


def test_an_unknown_entry_is_rejected(batches):
    with pytest.raises(ValueError, match="'ts' or 'cs'"):
        tuning_loaders(batches, "fusion", days=3, batch=8)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_tensors.py -k tuning_loaders -v`
Also: `.venv/bin/python -m pytest tests/test_tensors.py -k cannot_reach -v`

Expected: FAIL with `ImportError: cannot import name 'tuning_loaders'`.

- [ ] **Step 3: Implement it**

Add to `bubble_bi/data/tensors.py`, below `make_tensors`:

```python
def tuning_loaders(batches: Batches, entry: str, days: int,
                   batch: int) -> dict[str, DataLoader]:
    """One entry's grids at a given window length — LEARN AND TUNE ONLY.

    The search changes `days`, which changes the grids themselves, so they have to be
    rebuilt for every window it tries.

    ⚠️ `test` is not in the result, and that is the whole point. A search that could reach
    the test days would tune on the answer and then report a wonderful score on it. The
    defence is not a comment telling the next person to be careful — it is that the test
    loader does not exist for them to reach.
    """
    if entry not in ("ts", "cs"):
        raise ValueError(f"`entry` must be 'ts' or 'cs', got {entry!r}.")

    build = TSGrids if entry == "ts" else CSGrids
    scaled = batches.scaler.apply(batches.arrays.x)

    out = {}
    for period in ("learn", "tune"):
        dataset = build(batches.arrays, scaled, batches.days[period], days)
        if len(dataset) == 0:
            raise ValueError(
                f"No usable {days}-day {entry.upper()} grids in the '{period}' period. "
                f"A {days}-day window needs {days} days of run-up on top of what the slow "
                "features already need — try a shorter window, or a longer history."
            )
        out[period] = DataLoader(
            dataset,
            batch_size=batch,
            shuffle=(period == "learn"),
            drop_last=(period == "learn" and len(dataset) > batch),
        )
    return out
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_tensors.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/data/tensors.py tests/test_tensors.py
git commit -m "feat: tuning_loaders() -- learn and tune only, by construction

The search varies `days`, so the grids must be rebuilt per trial. It also
must never see the test period, and the only defence that holds is not
building the loader at all."
```

---

### Task 5: The scorer — does the present day survive the bottleneck?

The heart of the spec. **Not reconstruction** (the best compressor threw the candle away) and **not a forecast** (TS and CS are autoencoders; scoring them on tomorrow would rank noise). We ask what a held-out linear probe can recover about **today** from the token.

The targets come straight out of the grid — `grid[:, :, -1, :]` *is* today — so the search is blind to the future by construction.

**Files:**
- Create: `bubble_bi/tuning.py`
- Modify: `bubble_bi/__init__.py`
- Modify: `requirements.txt`
- Test: `tests/test_tuning.py`

**Interfaces:**
- Consumes: `autopsy._probe(x, y) -> float`, `training.pick_device(settings)`, `training._to(batch, where)`, `data.by_family(settings)` (Task 3), `data.names()`.
- Produces:
  - `bubble_bi.tuning.ALIVE = 0.05`
  - `bubble_bi.tuning.DIRECTION = ["body", "log_return"]`
  - `bubble_bi.tuning.look(model, loader, settings, limit=40) -> tuple[np.ndarray, ...]` → `(ids, summary, direction, volatility)`
  - `bubble_bi.tuning.skill(x, y, seed=0) -> float`
  - `bubble_bi.tuning.score_tokenizer(model, loader, settings) -> dict` with keys `score`, `direction`, `volatility`, `before_quant`, `words_used`, `why`

- [ ] **Step 1: Add optuna to requirements**

In `requirements.txt`, add `optuna` above the torch comment:

```
optuna
```

Then: `.venv/bin/pip install optuna`

Expected: installs `optuna` plus `alembic`, `colorlog`, `sqlalchemy`. **Verify torch was not touched:** `.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"` must print the same as before.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_tuning.py`:

```python
import math

import numpy as np
import pytest
import torch

from bubble_bi import tuning
from bubble_bi.models import VQVAE
from bubble_bi.settings import DEFAULTS


def _loader(grids, batch=8):
    return torch.utils.data.DataLoader(
        [{"grid": g, "present": torch.ones(g.shape[0], dtype=torch.bool)} for g in grids],
        batch_size=batch,
    )


def test_the_shuffled_floor_makes_a_useless_token_score_zero():
    """skill = 0 means 'no better than luck'. A random token knows nothing, so it must
    score ~0 — NOT the R2 that its one-hot width could buy on its own."""
    rng = np.random.default_rng(0)
    token = tuning.one_hot(rng.integers(0, 64, size=600), words=64)
    target = rng.normal(size=(600, 2))                    # unrelated to the token
    assert abs(tuning.skill(token, target)) < 0.05


def test_a_wider_vocabulary_does_not_buy_free_skill():
    """THE confound this floor exists for. A 1024-word one-hot hands the probe 1024
    columns; a 128-word one hands it 128. Raw R2 climbs with vocabulary FOR NOTHING, and
    the search would 'discover' that bigger is better having discovered nothing at all.

    This test fails on the naive implementation (skill = plain R2).
    """
    rng = np.random.default_rng(1)
    target = rng.normal(size=(600, 2))
    narrow = tuning.one_hot(rng.integers(0, 128, size=600), words=128)
    wide = tuning.one_hot(rng.integers(0, 1024, size=600), words=1024)

    assert abs(tuning.skill(wide, target) - tuning.skill(narrow, target)) < 0.1


def test_a_token_that_knows_the_answer_scores_well():
    rng = np.random.default_rng(2)
    ids = rng.integers(0, 8, size=600)
    target = np.c_[ids.astype(float), -ids.astype(float)] + 0.01 * rng.normal(size=(600, 2))
    assert tuning.skill(tuning.one_hot(ids, words=8), target) > 0.9


def test_a_collapsed_codebook_is_rejected_not_ranked():
    """A token drawn from 2 live words carries 1 bit. However well it probes, it is
    useless downstream — and it would DESTROY the predictor's target, which IS the token."""
    model = VQVAE(companies=1, days=4, features=26, width=16, heads=2,
                  vocabulary=512).eval()
    # Every grid identical -> every grid gets the same word.
    same = torch.zeros(4, 1, 4, 26)
    scored = tuning.score_tokenizer(model, _loader([g for g in same]), DEFAULTS)

    assert scored["score"] == -math.inf
    assert "collapsed" in scored["why"]


def test_the_probe_target_is_TODAY_and_never_tomorrow():
    """TS and CS are autoencoders. The target is the LAST DAY of the window they were just
    handed — read straight out of the grid, so nothing from the future can reach it."""
    torch.manual_seed(0)
    model = VQVAE(companies=1, days=4, features=26, width=16, heads=2, vocabulary=8)
    grids = [torch.randn(1, 4, 26) for _ in range(64)]

    _, _, direction, _ = tuning.look(model, _loader(grids), DEFAULTS, limit=99)

    body = tuning.names().index("body")
    expected = np.array([float(g[0, -1, body]) for g in grids])   # last day, `body`
    assert np.allclose(direction[:, 0], expected, atol=1e-5)
```

- [ ] **Step 3: Run them and watch them fail**

Run: `.venv/bin/python -m pytest tests/test_tuning.py -v`

Expected: FAIL with `ImportError: cannot import name 'tuning' from 'bubble_bi'`.

- [ ] **Step 4: Write the scorer**

Create `bubble_bi/tuning.py`:

```python
"""Find the settings that make TS and CS work — by measuring, not by guessing.

WHAT THIS OPTIMISES, AND WHY IT IS NEITHER OF THE OBVIOUS THINGS
----------------------------------------------------------------
NOT reconstruction loss. We already proved it misleads us: handed the candle explicitly,
the best compressor THREW IT AWAY (docs/DECISION-let-the-model-choose.md). Reconstruction
is an equally-weighted MSE over all 26 features, so it is carried by the easy, smooth ones.
Point six knobs at it and you buy a better compressor, not a better token.

NOT a forecast either. TS and CS are AUTOENCODERS — they represent the present and are
never asked to predict. Score them on tomorrow's return and every configuration scores
~0 ± noise, because tomorrow is unpredictable however good the tokenizer is. The search
would rank pure noise and hand back whichever trial got lucky.

So: does the PRESENT DAY survive the bottleneck? The token is 9 bits of a window; the only
honest question is what it chose to keep. And it is the question that matters, because
information destroyed at the tokenizer can NEVER be recovered by any predictor downstream.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from bubble_bi.autopsy import _probe
from bubble_bi.data.features import by_family, names
from bubble_bi.training import _to, pick_device

# Which way did it go today: where it closed against where it opened, and the day's return.
DIRECTION = ["body", "log_return"]

# A codebook using fewer than this share of its words has collapsed.
ALIVE = 0.05


def _columns(settings: dict) -> dict[str, list[int]]:
    """Which columns of the grid are 'direction' and which are 'volatility'."""
    every = names()
    return {
        "direction": [every.index(n) for n in DIRECTION],
        "volatility": [every.index(n) for n in by_family(settings)["volatility"]],
    }


def one_hot(ids: np.ndarray, words: int) -> np.ndarray:
    out = np.zeros((len(ids), words), dtype=np.float32)
    out[np.arange(len(ids)), ids] = 1.0
    return out


def skill(x: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """How much of `y` a linear probe recovers from `x`, ABOVE WHAT LUCK WOULD GIVE IT.

    The floor is not decoration. The token enters the probe one-hot, so a 1024-word
    vocabulary hands the probe 1024 columns and a 128-word one hands it 128 — raw R² would
    climb with `vocabulary` from capacity alone, and the search would 'discover' that
    bigger is better having discovered nothing.

    Shuffling the rows of `x` breaks the pairing while keeping the width, which is exactly
    a capacity-matched floor. Subtract it and the confound is gone.
    """
    rng = np.random.default_rng(seed)
    return _probe(x, y) - _probe(x[rng.permutation(len(x))], y)


@torch.no_grad()
def look(model, loader, settings: dict, limit: int = 40):
    """Run the model over held-out grids. Returns (ids, summary, direction, volatility).

    `direction` and `volatility` are read from the LAST DAY of the very grid the model was
    just given. Nothing from the future is even in the room.
    """
    where = pick_device(settings)
    model.to(where).eval()
    column = _columns(settings)

    ids, summary, direction, volatility = [], [], [], []
    for i, batch in enumerate(loader):
        if i >= limit:
            break
        batch = _to(batch, where)
        out = model(batch)

        grid = batch["grid"]                                  # [B, C, days, F]
        today = grid[:, :, -1, :]                             # [B, C, F]  <- THE PRESENT
        present = batch.get("present")
        weight = (present.unsqueeze(-1).to(today.dtype) if present is not None
                  else torch.ones_like(today[..., :1]))
        # TS has one company, so this is that company. CS has thirty, so this is the
        # market's average — over the ones that actually traded.
        average = (today * weight).sum(1) / weight.sum(1).clamp(min=1)     # [B, F]

        ids.append(out["ids"].cpu().numpy())
        summary.append(out["summary"].detach().cpu().numpy())
        direction.append(average[:, column["direction"]].cpu().numpy())
        volatility.append(average[:, column["volatility"]].cpu().numpy())

    model.train()
    return tuple(np.concatenate(part) for part in (ids, summary, direction, volatility))


def score_tokenizer(model, loader, settings: dict) -> dict:
    """The score one trial gets. Higher is better; -inf means 'thrown out'."""
    ids, summary, direction, volatility = look(model, loader, settings)
    words = model.codebook.words
    used = len(np.unique(ids))

    if used < max(2, int(ALIVE * words)):
        # Not a bad score — NOT RANKED AT ALL. A token from a handful of words carries
        # almost no information, and it would destroy the predictor's target, which IS the
        # token: every day becomes the same word and "predict tomorrow's word" is satisfied
        # by shrugging. We have watched naming accuracy hit 87% on a 3-word codebook.
        return {"score": -math.inf, "direction": float("nan"),
                "volatility": float("nan"), "before_quant": float("nan"),
                "words_used": used, "why": f"codebook collapsed: {used} of {words} words"}

    token = one_hot(ids, words)
    went = skill(token, direction)
    violent = skill(token, volatility)
    return {
        "score": went + violent,
        "direction": went,
        "volatility": violent,
        # The same probe on the CONTINUOUS vector, before the codebook rounded it off. A
        # big gap means the CODEBOOK is destroying the signal, not the encoder.
        "before_quant": skill(summary, direction),
        "words_used": used,
        "why": "",
    }
```

Export it in `bubble_bi/__init__.py` by adding `tuning` to the existing `from bubble_bi import (...)` list.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_tuning.py -v`

Expected: PASS, all six.

- [ ] **Step 6: Commit**

```bash
git add bubble_bi/tuning.py bubble_bi/__init__.py requirements.txt tests/test_tuning.py
git commit -m "feat: score a tokenizer by whether TODAY survives the bottleneck

Not reconstruction (the best compressor threw the candle away) and not a
forecast (an autoencoder scored on tomorrow ranks noise: every config gets
~0 because tomorrow is unpredictable however good the tokenizer is).

The probe target is read from the last day of the grid the model was just
handed, so the search is blind to the future by construction. The shuffled
floor is capacity-matched, which stops a wider vocabulary buying free R2."
```

---

### Task 6: The two-stage search

Twelve trials over six knobs is a lottery. So we shrink the *space*, not the search: knobs whose answer we already know are fixed, and the trials split across the two questions the user actually asked — **get the sizes correct, get the balance right.**

**Files:**
- Modify: `bubble_bi/training.py` (an `on_check` callback so a search can prune)
- Modify: `bubble_bi/tuning.py`
- Test: `tests/test_tuning.py`

**Interfaces:**
- Consumes: `score_tokenizer` (Task 5), `tuning_loaders` (Task 4), `train()` from `bubble_bi.training`.
- Produces:
  - `training.train(..., on_check: Callable[[int, dict], None] | None = None)`
  - `tuning.SPACE: dict` — the two stages and their ranges
  - `tuning.search(entry, batches, settings, scorer=score_tokenizer) -> tuple[dict, pd.DataFrame]` → `(best_settings_for_that_entry, trials_table)`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_tuning.py`:

```python
def test_the_search_space_fixes_the_knobs_we_already_know():
    """decoder_depth is not searched because the decoder is THROWN AWAY when we freeze the
    tokenizer. Searching it would be tuning a part we delete."""
    searched = {k for stage in tuning.SPACE.values() for k in stage}
    assert "decoder_depth" not in searched
    assert "batch" not in searched
    assert searched == {"learning_rate", "commitment", "diversity",
                        "model_size", "vocabulary", "days"}


def test_the_balance_comes_before_the_sizes():
    assert list(tuning.SPACE) == ["balance", "sizes"]
    assert set(tuning.SPACE["balance"]) == {"learning_rate", "commitment", "diversity"}
    assert set(tuning.SPACE["sizes"]) == {"model_size", "vocabulary", "days"}


def test_a_search_returns_a_config_that_check_accepts(tiny_batches, tiny_settings):
    """End-to-end on a 2-trial synthetic run: whatever the search hands back must be a
    settings dict the project will actually accept."""
    from bubble_bi.settings import check

    best, trials = tuning.search("ts", tiny_batches, tiny_settings)

    assert len(trials) == 2
    assert {"score", "direction", "volatility", "words_used"} <= set(trials.columns)
    check({**tiny_settings, "ts": {**tiny_settings["ts"], **best}})


def test_a_killed_search_resumes_instead_of_starting_over(tiny_batches, tiny_settings):
    """Colab WILL disconnect. The study is on disk, so a second call must top up the
    trials that are missing — not run the whole budget again on top of them."""
    tuning.search("ts", tiny_batches, tiny_settings)
    _, again = tuning.search("ts", tiny_batches, tiny_settings)

    assert len(again) == 2, "the resumed search re-ran trials it had already completed"
```

Add these fixtures to `tests/test_tuning.py` — they build a small synthetic panel so the whole test runs on CPU in seconds:

```python
@pytest.fixture
def tiny_settings(tmp_path):
    return {
        **DEFAULTS,
        "tickers": ["AAA", "BBB", "CCC"],
        # The Optuna study lands under data_dir — keep it out of the real artifacts folder,
        # or the resume test would find a stale study from a previous run.
        "data_dir": str(tmp_path),
        "model_size": 16,
        "learning_rate": 1e-3,
        "steps": 20,
        "ts": {**DEFAULTS["ts"], "days": 3, "batch": 8, "vocabulary": 16,
               "encoder_depth": 1, "decoder_depth": 1, "heads": 2, "steps": 20},
        "cs": {**DEFAULTS["cs"], "days": 3, "batch": 4, "vocabulary": 16,
               "encoder_depth": 1, "decoder_depth": 1, "heads": 2, "steps": 20},
        "search": {"run": True, "trials": 2, "steps": 20},
    }


@pytest.fixture
def tiny_batches(tiny_settings):
    """A synthetic panel: 3 companies, 400 days, real features."""
    import pandas as pd

    from bubble_bi.data import add_features, make_tensors

    rng = np.random.default_rng(0)
    frames = []
    for ticker in tiny_settings["tickers"]:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
        frames.append(pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=400, freq="B"),
            "ticker": ticker,
            "open": close * (1 + rng.normal(0, 0.002, 400)),
            "high": close * (1 + abs(rng.normal(0, 0.005, 400))),
            "low": close * (1 - abs(rng.normal(0, 0.005, 400))),
            "close": close,
            "volume": rng.integers(1e6, 5e6, 400).astype(float),
        }))
    table = add_features(pd.concat(frames, ignore_index=True), tiny_settings)
    return make_tensors(table, tiny_settings)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `.venv/bin/python -m pytest tests/test_tuning.py -k "space or balance or search_returns" -v`

Expected: FAIL — `AttributeError: module 'bubble_bi.tuning' has no attribute 'SPACE'`.

- [ ] **Step 3: Let `train()` be interrupted by a search**

In `bubble_bi/training.py`, add a parameter to `train()` (signature at `training.py:148-158`):

```python
def train(
    model,
    loaders: dict,
    settings: dict,
    steps: int | None = None,
    entry: str | None = None,
    revive_every: int = 50,
    check_every: int | None = None,
    patience: int = 5,
    quiet: bool = False,
    on_check=None,
) -> History:
```

Add to its docstring:

```
    on_check: called as on_check(step, scored) at every held-out check. A hyperparameter
              search uses this to give up on a hopeless trial early. It may RAISE to stop
              training — the exception is deliberately not caught.
```

Then, inside the `if step % check_every == 0:` block, immediately after `history.add(...)`, add:

```python
            if on_check is not None:
                on_check(step, scored)      # may raise, on purpose: that is how a search prunes
```

- [ ] **Step 4: Write the search**

Append to `bubble_bi/tuning.py`:

```python
# ---------------------------------------------------------------- the search space
#
# Six knobs, tight informed ranges. What is NOT here matters as much as what is:
#
#   decoder_depth   the decoder is THROWN AWAY when we freeze the tokenizer.
#                   Tuning it is tuning a part we delete.
#   batch           already reasoned from the 30x data-size gap (TS 256 / CS 64).
#   weight_decay    fixed at STORM's 0.05.
#   revive_every    not where the problem is.
#
# Two stages, because the user asked two different questions -- "get the sizes correct,
# the balance right" -- and at twelve trials a blind six-knob search is a lottery.
SPACE = {
    "balance": {
        "learning_rate": ("log", 3e-5, 3e-3),   # never once tested. It is 1e-4 because
                                                # somebody typed 1e-4.
        "commitment": ("log", 0.1, 2.0),        # we ran at 1.0; the standard is 0.25
        "diversity": ("float", 0.0, 1.0),       # the anti-collapse term
    },
    "sizes": {
        "model_size": ("pick", [64, 128, 256]),
        "vocabulary": ("pick", [128, 256, 512, 1024]),
        "days": {"ts": ("pick", [5, 10, 15, 20, 30]),
                 "cs": ("pick", [1, 3, 5, 10])},
    },
}

# `model_size` is a top-level setting; the rest live inside the entry's own block.
_TOP_LEVEL = {"learning_rate", "model_size"}


def _ask(trial, name, rule):
    kind = rule[0]
    if kind == "log":
        return trial.suggest_float(name, rule[1], rule[2], log=True)
    if kind == "float":
        return trial.suggest_float(name, rule[1], rule[2])
    if kind == "pick":
        return trial.suggest_categorical(name, rule[1])
    raise ValueError(f"unknown rule {kind!r} for {name!r}")


def _settle(settings: dict, entry: str, chosen: dict) -> dict:
    """A full settings dict with this trial's choices folded in."""
    out = {**settings, **{k: v for k, v in chosen.items() if k in _TOP_LEVEL}}
    out[entry] = {**settings[entry],
                  **{k: v for k, v in chosen.items() if k not in _TOP_LEVEL}}
    return out


def _run_one(entry, chosen, batches, settings, scorer, features, companies, trial=None):
    """Train one configuration and score it. Returns the scorer's dict."""
    import optuna

    from bubble_bi.data.tensors import tuning_loaders
    from bubble_bi.models import VQVAE
    from bubble_bi.training import train

    live = _settle(settings, entry, chosen)
    block = live[entry]
    loaders = tuning_loaders(batches, entry, block["days"], block["batch"])

    model = VQVAE(
        companies=1 if entry == "ts" else companies,
        features=features,
        width=live["model_size"],
        **block,
    )

    def watch(step, scored):
        if trial is None:
            return
        # Report the REAL objective, not the rebuild loss -- pruning on a signal we have
        # already proved misleading would throw away the good trials.
        trial.report(scorer(model, loaders["tune"], live)["score"], step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    train(model, loaders, live, steps=live["search"]["steps"],
          quiet=True, on_check=watch)
    return scorer(model, loaders["tune"], live)


def search(entry: str, batches, settings: dict, scorer=score_tokenizer):
    """Find good settings for one entry. Returns (best_block, trials_table).

    Two stages: the BALANCE first (learning rate, commitment, diversity, with the sizes
    held at their defaults), then the SIZES with the winning balance held fixed.

    Coordinate descent assumes the two groups barely interact. They do interact -- the
    best learning rate genuinely moves with width -- so this is an approximation, and it
    is the price of a twelve-trial budget. Two things keep it honest: the winner is
    confirmed head-to-head at FULL budget (see `confirm`), and raising `search["trials"]`
    narrows the gap with no change to this code.
    """
    import optuna
    import pandas as pd

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    features = len(names())
    companies = len(settings["tickers"])
    budget = settings["search"]["trials"]
    rows, fixed = [], {}

    for stage, knobs in SPACE.items():
        rules = {name: (rule[entry] if isinstance(rule, dict) else rule)
                 for name, rule in knobs.items()}
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=settings["seed"], n_startup_trials=4),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=3),
            storage=_study_path(settings, entry, stage),
            study_name=f"{entry}-{stage}",
            load_if_exists=True,                 # <- resume. A disconnect costs one trial.
        )

        def objective(trial):
            chosen = {**fixed,
                      **{name: _ask(trial, name, rule) for name, rule in rules.items()}}
            scored = _run_one(entry, chosen, batches, settings, scorer,
                              features, companies, trial=trial)
            for key, value in scored.items():
                trial.set_user_attr(key, value)
            return scored["score"]

        # The study is on disk, so a resumed run must only top up what is missing.
        done = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE])
        study.optimize(objective, n_trials=max(0, budget // 2 - done))

        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            rows.append({"stage": stage, **trial.params,
                         **{k: trial.user_attrs.get(k) for k in
                            ("score", "direction", "volatility",
                             "before_quant", "words_used", "why")}})

        alive = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE and t.value > -math.inf]
        if alive:
            fixed.update(max(alive, key=lambda t: t.value).params)

    table = pd.DataFrame(rows).sort_values("score", ascending=False)
    best = _settle(settings, entry, fixed)
    return {**best[entry], "learning_rate": best["learning_rate"],
            "model_size": best["model_size"]}, table


def _study_path(settings: dict, entry: str, stage: str) -> str:
    """Where the study lives. On Colab `data_dir` is Drive, so a disconnect costs one
    trial rather than the whole session."""
    from pathlib import Path

    folder = Path(settings["data_dir"]) / "search"
    folder.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{folder / f'{entry}-{stage}.db'}"
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_tuning.py -q`

Expected: PASS. The end-to-end test trains two tiny models on CPU; it should take under 30 seconds.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add bubble_bi/tuning.py bubble_bi/training.py tests/test_tuning.py
git commit -m "feat: a two-stage search -- the balance, then the sizes

Twelve trials over six knobs is a lottery, so we shrink the SPACE, not the
search: decoder_depth is not tuned because the decoder is thrown away, batch
follows from the 30x data-size gap, weight_decay is STORM's.

Pruning reports the probe score, not the rebuild loss -- pruning on a signal
we have already proved misleading would throw away the good trials. The
study is SQLite on Drive, so a Colab disconnect costs one trial."
```

---

### Task 7: The confirm, and `tuned.json`

Two things. First, the transfer guard: **a config that wins a 600-step sprint can lose a 5,000-step run** — CS did exactly this, its held-out error climbing 0.90 → 1.03 while its codebook decayed 187 → 141 words. Second, the artifact: the search is a one-time act of discovery and everyone after must inherit its answer without running anything.

**Files:**
- Modify: `bubble_bi/tuning.py`
- Test: `tests/test_tuning.py`

**Interfaces:**
- Produces:
  - `tuning.fingerprint(settings) -> dict`
  - `tuning.confirm(entry, winner, batches, settings, scorer=score_tokenizer) -> dict` → `{"kept": "winner"|"incumbent", "winner": float, "incumbent": float, "settings": dict}`
  - `tuning.save(found: dict, settings: dict, path=TUNED) -> Path`
  - `tuning.apply(typed: dict, path=TUNED) -> tuple[dict, str]` → `(settings_with_tuning_folded_in, a_line_to_print)`
  - `tuning.TUNED: Path` — repo-root `tuned.json`
- Consumes (built in Task 6, use them — do NOT hand-roll the split):
  - `tuning.TOP_LEVEL: frozenset` — exactly `{"learning_rate", "model_size"}`. These two are **top-level settings**, but `search()` returns them (and `tuned.json` stores them) *inside* the entry's flat block. `TOP_LEVEL` is the one place that knows which keys are which.
  - `tuning.settle(settings: dict, entry: str, chosen: dict) -> dict` — folds a flat `search()`-shaped dict back into a full settings dict, putting each key where it belongs.

⚠️ **The trap this task exists to walk into.** A caller who filters the flat block by hand — `{k: v for k, v in tuned_block.items() if k != "model_size"}` — forgets `learning_rate`, leaves it stranded inside `settings["ts"]`, and `check()` rejects it as an unknown setting. The notebook then dies on its first cell. **Always filter on `TOP_LEVEL`, never on a hand-picked key name.**

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_tuning.py`:

```python
def test_apply_leaves_no_top_level_key_stranded_in_an_entry_block(tmp_path):
    """`tuned.json` stores each entry FLAT, so `learning_rate` and `model_size` are sitting
    inside the "ts" object even though they are top-level settings. Filter them out by hand
    and you WILL forget one — then `check()` rejects it and the notebook dies on cell one."""
    import json

    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({
        "found_on": "2026-07-12", "trials": 12,
        "fingerprint": {"tickers": 1, "features": 26, "start": None, "search_steps": 600},
        "score": {},
        "ts": {"vocabulary": 1024, "learning_rate": 4.2e-4, "model_size": 256},
        "cs": {},
    }))

    merged, _ = tuning.apply({"tickers": ["AAA"]}, path=path)

    assert not (tuning.TOP_LEVEL & set(merged["ts"])), (
        "a top-level setting was left inside the entry block — check() will reject it"
    )
    assert merged["learning_rate"] == 4.2e-4      # lifted OUT, not dropped
    assert merged["model_size"] == 256
    assert merged["ts"]["vocabulary"] == 1024
    bb_check(merged)                               # from bubble_bi.settings import check


def test_precedence_defaults_then_tuned_then_what_you_typed(tmp_path):
    """Three layers, most specific wins. A tuned value replaces a default, but a value you
    DELIBERATELY typed in the notebook stands."""
    import json

    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({
        "found_on": "2026-07-12", "trials": 12,
        "fingerprint": {"tickers": 1, "features": 26, "start": None, "search_steps": 600},
        "score": {}, "ts": {"vocabulary": 1024, "commitment": 0.31}, "cs": {},
    }))

    typed = {"tickers": ["AAA"], "ts": {"vocabulary": 256}}
    merged, note = tuning.apply(typed, path=path)

    assert merged["ts"]["vocabulary"] == 256      # you typed it: you win
    assert merged["ts"]["commitment"] == 0.31     # you did not: the tuning wins
    assert "✅" in note


def test_a_stale_tuning_warns_and_does_not_pretend(tmp_path):
    """We have already changed the feature count twice (10 -> 22 -> 26). Silently reusing
    hyperparameters tuned on different data is the exact class of bug we keep catching."""
    import json

    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({
        "found_on": "2026-01-01", "trials": 12,
        "fingerprint": {"tickers": 30, "features": 22, "start": None, "search_steps": 600},
        "score": {}, "ts": {"vocabulary": 1024}, "cs": {},
    }))

    merged, note = tuning.apply({"tickers": ["AAA"]}, path=path)

    assert "STALE" in note
    assert "22" in note and "26" in note           # says WHAT changed, not just "stale"
    assert merged["ts"]["vocabulary"] == 1024      # still used: half-stale beats untuned


def test_no_tuning_file_says_so_plainly(tmp_path):
    merged, note = tuning.apply({"tickers": ["AAA"]}, path=tmp_path / "absent.json")
    assert "No tuned.json" in note
    assert merged == {"tickers": ["AAA"]}


def test_the_confirm_keeps_the_incumbent_when_the_winner_does_not_beat_it(
        tiny_batches, tiny_settings):
    """The transfer guard. A config that wins a 600-step sprint can lose the real run: CS
    did exactly that, its held-out error climbing while its codebook decayed.

    `confirm` scores the winner first, then the incumbent — so a scorer that hands out a
    poor score and then a good one is a winner that flattered to deceive.
    """
    scores = iter([
        {"score": 0.1, "direction": 0.0, "volatility": 0.1, "words_used": 9,
         "before_quant": 0.0, "why": ""},         # the winner,    at FULL budget
        {"score": 0.9, "direction": 0.4, "volatility": 0.5, "words_used": 9,
         "before_quant": 0.0, "why": ""},         # the incumbent, at FULL budget
    ])

    out = tuning.confirm("ts", {**tiny_settings["ts"], "learning_rate": 1e-3,
                                "model_size": 16},
                         tiny_batches, tiny_settings, scorer=lambda *a, **k: next(scores))

    assert out["kept"] == "incumbent"
    assert out["winner"] == 0.1 and out["incumbent"] == 0.9
    assert out["settings"]["vocabulary"] == tiny_settings["ts"]["vocabulary"]
```

- [ ] **Step 2: Run them and watch them fail**

Run: `.venv/bin/python -m pytest tests/test_tuning.py -k "precedence or stale or no_tuning or confirm" -v`

Expected: FAIL — `AttributeError: module 'bubble_bi.tuning' has no attribute 'apply'`.

- [ ] **Step 3: Implement the confirm and the artifact**

Append to `bubble_bi/tuning.py`:

First, add `from pathlib import Path` to the import block at the top of `bubble_bi/tuning.py` (alongside `import math`). Then append:

```python
# ------------------------------------------------------------------- the artifact

TUNED = Path(__file__).resolve().parent.parent / "tuned.json"


def fingerprint(settings: dict) -> dict:
    """What the tuning was found ON. Hyperparameters are only valid for their data."""
    return {
        "tickers": len(settings.get("tickers") or []),
        "features": len(names()),
        "start": settings.get("start"),
        "search_steps": settings.get("search", {}).get("steps"),
    }


def confirm(entry: str, winner: dict, batches, settings: dict, scorer=score_tokenizer):
    """Train the winner AND the incumbent at FULL budget and let the data decide.

    ⚠️ This is the transfer guard, and it exists because we have been fooled by exactly
    this. A configuration that wins a 600-step sprint can lose the real run: CS's held-out
    error bottomed out at step 1,000 and then climbed for nine thousand more, 0.90 -> 1.03,
    while its codebook decayed from 187 words to 141. A short search would have crowned it.

    If the winner does not beat the incumbent here, we KEEP THE INCUMBENT.
    """
    features, companies = len(names()), len(settings["tickers"])
    full = {**settings, "search": {**settings["search"], "steps": settings["steps"]}}

    incumbent = {**settings[entry], "learning_rate": settings["learning_rate"],
                 "model_size": settings["model_size"]}

    scored = {}
    for label, block in (("winner", winner), ("incumbent", incumbent)):
        scored[label] = _run_one(entry, block, batches, full, scorer,
                                 features, companies)["score"]

    kept = "winner" if scored["winner"] > scored["incumbent"] else "incumbent"
    return {"kept": kept, "winner": scored["winner"], "incumbent": scored["incumbent"],
            "settings": winner if kept == "winner" else incumbent}


def save(found: dict, settings: dict, path: Path = TUNED) -> Path:
    """Write the answer, and what it was found on. Committed to the repo, not Drive —
    Drive is private to whoever ran it, and the next person must get this by cloning."""
    import datetime
    import json

    path.write_text(json.dumps({
        "found_on": datetime.date.today().isoformat(),
        "trials": settings["search"]["trials"],
        "fingerprint": fingerprint(settings),
        **found,
    }, indent=2, sort_keys=True) + "\n")
    return path


def apply(typed: dict, path: Path = TUNED) -> tuple[dict, str]:
    """Fold `tuned.json` into the settings the notebook typed. Returns (settings, note).

    Precedence, most specific wins:

        DEFAULTS  <  tuned.json  <  what you typed in the notebook

    A tuned value replaces a default. A value you DELIBERATELY wrote stands. `check()` only
    ever sees the keys actually typed, so it can tell the two apart without guessing.
    """
    import json

    if not path.exists():
        return typed, ("ℹ️  No tuned.json — running on defaults. "
                       "Set search['run'] = True to search for better ones.")

    found = json.loads(path.read_text())
    merged = dict(typed)
    for entry in ("ts", "cs"):
        tuned_block = found.get(entry) or {}
        if not tuned_block:
            continue
        # `tuned.json` stores each entry's block FLAT — the way `search()` returns it — so
        # `learning_rate` and `model_size` are sitting in there even though they are
        # top-level settings, not members of the entry's block. Split on TOP_LEVEL, the one
        # place that knows which is which. Hand-picking the keys here is how you end up
        # leaving `learning_rate` stranded inside `settings["ts"]`, where `check()` rejects
        # it as an unknown setting and the notebook dies on its first cell.
        block = {k: v for k, v in tuned_block.items() if k not in TOP_LEVEL}
        merged[entry] = {**block, **typed.get(entry, {})}       # typed wins
        for shared in TOP_LEVEL:
            if shared in tuned_block and shared not in typed:
                merged[shared] = tuned_block[shared]

    was, now = found["fingerprint"], fingerprint({**typed, "search": {"steps": None}})
    drifted = [f"{k}: tuned on {was[k]}, running {now[k]}"
               for k in ("tickers", "features")
               if was.get(k) != now.get(k) and now.get(k)]
    if drifted:
        note = ("⚠️  tuned.json is STALE — " + "; ".join(drifted) + ".\n"
                "    Using it anyway (half-stale beats untuned), but set "
                "search['run'] = True to re-tune.")
    else:
        note = (f"✅  Using tuned settings — found {found['found_on']} over "
                f"{found['trials']} trials, on this same data.")
    return merged, note
```

Note the `learning_rate` and `model_size` handling: they are top-level settings, but they are *found per entry*, so `save` stores them inside the `ts`/`cs` blocks and `apply` lifts them back out. TS and CS are trained separately, so if the two disagree, the last one applied wins — record that in the notebook output rather than hiding it.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_tuning.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/tuning.py tests/test_tuning.py
git commit -m "feat: the confirm, and tuned.json

The transfer guard: a config that wins a 600-step sprint can lose the real
run. CS did exactly that -- held-out error climbing 0.90 -> 1.03 over nine
thousand steps while its codebook decayed 187 -> 141 words. So the winner
must beat the incumbent HEAD TO HEAD at full budget, or we keep the
incumbent.

tuned.json is committed, not on Drive: the next runner must inherit the
answer by cloning. It carries a fingerprint of the data it was found on,
because we have changed the feature count twice already and silently reusing
stale hyperparameters is the exact bug we keep catching by measuring."
```

---

### Task 8: The notebook section

**Files:**
- Modify: `bubble_bi/plots.py` (add `tuning_importance`)
- Modify: `Bubble_Bi.ipynb` (new section, between the tensors and the TS training)
- Test: `tests/test_plots.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `plots.tuning_importance(trials: pd.DataFrame)` — a bar chart of which knob moved the score.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_plots.py`:

```python
def test_tuning_importance_ranks_the_knob_that_actually_moved_the_score():
    """Even twelve near-random trials answer 'is the learning rate dominating everything?'
    -- which is worth knowing, and is the honest thing a 12-trial screen can tell you."""
    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    from bubble_bi import plots

    rng = np.random.default_rng(0)
    lr = rng.random(20)
    trials = pd.DataFrame({
        "stage": "balance",
        "learning_rate": lr,
        "commitment": rng.random(20),
        "score": 5 * lr + 0.01 * rng.random(20),     # score follows lr, and nothing else
    })
    ranked = plots.tuning_importance(trials)
    assert ranked.index[0] == "learning_rate"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/bin/python -m pytest tests/test_plots.py -k tuning_importance -v`

Expected: FAIL — `AttributeError: module 'bubble_bi.plots' has no attribute 'tuning_importance'`.

- [ ] **Step 3: Implement the chart**

Add to `bubble_bi/plots.py`:

```python
def tuning_importance(trials):
    """Which knob actually moved the score? Returns the ranking, and draws it.

    Rank correlation, not Optuna's fANOVA: at a dozen trials fANOVA fits a forest to
    almost no data and reports confident nonsense. A rank correlation over twelve points is
    crude, and it is honest about being crude.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    usable = trials[np.isfinite(trials["score"])]
    knobs = [c for c in usable.columns
             if c not in {"stage", "score", "direction", "volatility",
                          "before_quant", "words_used", "why"}]
    strength = {}
    for knob in knobs:
        column = pd.to_numeric(usable[knob], errors="coerce")
        if column.notna().sum() > 2 and column.nunique() > 1:
            strength[knob] = abs(column.corr(usable["score"], method="spearman"))

    ranked = pd.Series(strength).dropna().sort_values(ascending=False)
    if ranked.empty:
        print("Not enough completed trials to say which knob mattered.")
        return ranked

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(ranked) + 1.5))
    ax.barh(ranked.index[::-1], ranked.to_numpy()[::-1], color="#26a69a")
    ax.set_xlabel("how strongly this knob moved the score (rank correlation)")
    ax.set_title("Which knob actually mattered", loc="left", fontsize=12)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    plt.show()
    return ranked
```

- [ ] **Step 4: Run the test**

Run: `.venv/bin/python -m pytest tests/test_plots.py -q`

Expected: PASS.

- [ ] **Step 5: Add the notebook section**

Insert into `Bubble_Bi.ipynb` **after** the tensors section and **before** the TS training. Four cells.

**Cell 1 (markdown):**

```markdown
## 6. Getting the settings right

Every number in `SETTINGS` above was a guess. `learning_rate` is `1e-4` because somebody
typed `1e-4`.

So we measure instead. But **not** by asking which settings rebuild the window best — we
already know that lies to us: handed the candle explicitly, the best compressor threw it
away. And **not** by asking which settings predict tomorrow — TS and CS are autoencoders,
they represent the *present*, and scoring them on a forecast would just rank noise.

We ask the only honest question: **does today survive the bottleneck?** A held-out linear
probe reads the token and tries to recover today's direction and today's volatility. That
is what the token is *for*, and whatever it loses here is lost forever — no predictor
downstream can recover information the tokenizer already threw away.

**This is off by default.** The search is a one-time act of discovery; its answer lives in
`tuned.json`, committed to the repo. You inherit it by cloning. Set
`SETTINGS["search"]["run"] = True` only if you want to find it again.
```

**Cell 2 (code) — load or search:**

```python
if settings["search"]["run"]:
    ts_best, ts_trials = bb.tuning.search("ts", batches, settings)
    cs_best, cs_trials = bb.tuning.search("cs", batches, settings)

    ts_verdict = bb.tuning.confirm("ts", ts_best, batches, settings)
    cs_verdict = bb.tuning.confirm("cs", cs_best, batches, settings)

    print(f"TS  head-to-head at full budget: winner {ts_verdict['winner']:+.3f} "
          f"vs incumbent {ts_verdict['incumbent']:+.3f} → kept the {ts_verdict['kept']}")
    print(f"CS  head-to-head at full budget: winner {cs_verdict['winner']:+.3f} "
          f"vs incumbent {cs_verdict['incumbent']:+.3f} → kept the {cs_verdict['kept']}")

    bb.tuning.save({"ts": ts_verdict["settings"], "cs": cs_verdict["settings"],
                    "score": {"ts": ts_trials.iloc[0].to_dict(),
                              "cs": cs_trials.iloc[0].to_dict()}}, settings)
    trials = pd.concat([ts_trials.assign(entry="TS"), cs_trials.assign(entry="CS")])
else:
    trials = None

SETTINGS, note = bb.tuning.apply(SETTINGS)
settings = bb.check(SETTINGS)
print(note)
```

**Cell 3 (code) — what the search found:**

```python
if trials is not None:
    display(trials[["entry", "stage", "score", "direction", "volatility",
                    "before_quant", "words_used"]].round(3))
    bb.plots.tuning_importance(trials)

    best_by_score = trials.iloc[0]
    best_by_direction = trials.sort_values("direction", ascending=False).iloc[0]
    if best_by_score.name != best_by_direction.name:
        print("\n⚠️  The best-scoring config is NOT the one that kept the most DIRECTION.")
        print("    Direction is the scarce quantity — this is a choice, not a detail.")

    print("\n⚠️  {} trials is a SCREEN, not an optimum. It will catch a badly-wrong "
          "setting.\n    It will NOT find the best one. Raise "
          "SETTINGS['search']['trials'] to 60+\n    for a real search — the code is "
          "identical, it just gets better.".format(settings["search"]["trials"]))
```

**Cell 4 (code) — the check:**

```python
import inspect

tuned = bb.tuning.TUNED.exists()
reaches = set(inspect.signature(bb.models.VQVAE.__init__).parameters)
decorative = set(settings["ts"]) - reaches

bb.report(
    "6. The settings",
    [
        ("Every setting reaches a model", not decorative,
         "none are decorative" if not decorative else f"IGNORED: {sorted(decorative)}"),
        ("Commitment is the literature value",
         settings["ts"]["commitment"] <= 0.5,
         f"{settings['ts']['commitment']} (we ran at 1.0 by accident for months)"),
        ("A tuning exists to inherit", tuned,
         "tuned.json" if tuned else "none yet — running on defaults"),
    ],
    have=f"""
    TS  {settings['ts']['vocabulary']} words, width {settings['model_size']},
        {settings['ts']['days']} days back, lr {settings['learning_rate']:.1e}
    CS  {settings['cs']['vocabulary']} words, {settings['cs']['days']} days back
    The search is OFF by default — the next person inherits tuned.json by cloning.
    """,
    known_problem=(
        None if tuned else
        "No tuning has been run yet. The settings are guesses — good ones, but guesses. "
        "Set SETTINGS['search']['run'] = True on a GPU to find better."
    ),
)
```

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`

Expected: PASS, and the count should be ~200 (up from 186).

- [ ] **Step 7: Commit**

```bash
git add bubble_bi/plots.py Bubble_Bi.ipynb tests/test_plots.py
git commit -m "feat: the notebook's tuning section -- off by default

Shows direction and volatility as SEPARATE columns, never a blended number:
direction is the scarce quantity and the choice between them is the user's.
Says plainly that 12 trials is a screen and not an optimum.

Rank correlation for the importance chart, not Optuna's fANOVA -- at a dozen
trials fANOVA fits a forest to almost no data and reports confident nonsense."
```

---

## What the user does next (Colab, GPU)

Not part of the plan's tasks — this is the handoff.

1. Pull, set `SETTINGS["search"]["run"] = True`, run the notebook to the end of section 6 (~15 min per model on a T4).
2. Read the **direction** column. This is the whole point:
   - **Some config keeps today's direction** → the candle was never noise; we were running the wrong hyperparameters. The original plan (predict the next candle, trade direction) stands.
   - **No config keeps it** → direction is destroyed *at the tokenizer*, no predictor can recover it, and the regime pivot in `docs/DECISION-let-the-model-choose.md` is **proven** rather than assumed.
3. Commit the resulting `tuned.json`. Everyone after inherits it by cloning.
4. Then, and only then, the fusion + predictor spec — written against the answer, not a guess.
