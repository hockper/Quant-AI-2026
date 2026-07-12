# Tuning the tokenizer: an opt-in search for TS and CS

**Date:** 2026-07-12
**Status:** design, approved for planning
**Scope:** TS and CS only. Fusion + predictor get their own spec afterwards.

## Why

Two problems, and they compound.

**First, our settings are lying to us.** `SETTINGS["loss"]` — all four weights,
`naming`, `candle`, `commitment`, `diversity` — is read by nothing outside
`settings.py` and the tests. Grep every `.py` and the notebook: no model is ever handed
those numbers. The tokenizer runs on `VQVAE.__init__`'s own hardcoded defaults, and
those *contradict* `Codebook`'s:

| | `Codebook` default | `VQVAE` default | what actually runs | literature |
|---|---|---|---|---|
| `commitment` | 0.25 | **1.0** (wins) | **1.0** | **0.25** |
| `diversity` | 0.1 | 0.1 | 0.1 | — |
| `decay` (EMA) | 0.99 | not exposed | 0.99 | 0.99 |

We have trained the whole project at a commitment four times the standard value, while
looking at a setting that said otherwise. An over-strong commitment pins the encoder to
the codebook and is a documented cause of the collapse we are seeing in fusion
([SQ-VAE, ICML 2022](https://icml.cc/media/icml-2022/Slides/17788.pdf)).

**Second, we have never tuned anything.** `learning_rate` is `1e-4` because someone
typed it. `vocabulary` has never been varied, though the literature is clear that
[larger codebooks reliably improve reconstruction while wider code vectors often
hurt](https://arxiv.org/html/2601.22244v1) — meaning `vocabulary` and `model_size`
are separate axes and our instinct to scale them together is wrong.

## What this is NOT optimising

Two candidate objectives, and **both are wrong**.

**Not "lowest reconstruction loss."** We have already proved it misleads us: given the
candle explicitly, the best compressor *threw it away*
(`docs/DECISION-let-the-model-choose.md`). Reconstruction loss is an equally-weighted MSE
over all 26 features, so it is dominated by the easy, smooth, compressible ones. Pointing
six knobs at it would buy a better compressor and a token no more useful downstream.

**And not "predicts tomorrow" either.** TS and CS are **autoencoders**. They are trained
to represent the *present* window and are never asked to forecast — the predictor does
that. Score a tokenizer on tomorrow's return and every configuration scores ≈ 0 ± noise,
*because tomorrow is unpredictable however good the tokenizer is*. The search would then
rank pure noise and hand back whichever trial got luckiest. Worse, a null result would be
uninterpretable: "tomorrow's return is hard to predict" is a tautology, not a finding
about our tokenizer.

**The right question for an autoencoder is whether the present day survives the
bottleneck.** The token is a 9-bit summary of a window; the only thing we can ask of it is
what it chose to keep. And that is the whole ballgame, because *information destroyed at
the tokenizer can never be recovered by any predictor downstream*.

## The objective

Train on `learn`. Probe on `tune`. **`test` is never touched by the search.**

```
skill(target) = R²(probe: token → target)  −  R²(probe: SHUFFLED token → target)

score = skill(TODAY's direction) + skill(TODAY's volatility)
```

"Today" is the **last day of the window** — the day the token stands for in the
predictor's sentence. `skill = 0` means "no better than luck". The probe is the ridge in
`autopsy._probe`, deliberately linear: if a linear probe cannot find the information, a
codebook of nearest-neighbour lookups will not dig it out either.

This is *not* reconstruction in disguise. Reconstruction asks the decoder to redraw all 26
features with equal weight. This asks a probe to recover the **two quantities anything
downstream actually needs**, and ignores the 20 features that were only ever inflating the
average.

**The shuffled floor is load-bearing, not decoration.** The token enters the probe
one-hot, so a 1024-word vocabulary gives the probe 1024 columns and a 128-word vocabulary
gives it 128. Raw R² would rise with `vocabulary` *from capacity alone*, and the search
would "discover" that bigger is better when it had discovered nothing. Shuffling the token
IDs **at the same vocabulary** produces a capacity-matched floor, and subtracting it
removes the confound exactly.

### The targets

Read straight from the window the model was just given — its **last day**. Nothing from
the future enters the search at all, so the no-lookahead question does not even arise.

| | direction | volatility |
|---|---|---|
| **TS** (one grid per company-day) | that company's `log_return` and `body` today | its `volatility`-family features today |
| **CS** (one grid per day) | the market's mean `log_return` and mean `body` today | the cross-section's dispersion of returns today |

`_probe` already accepts a multi-column target, so each is one ridge fit.

### Both targets carry real signal — which is the point

| | if we had scored *tomorrow* | scoring *today* |
|---|---|---|
| direction | ≈ 0 — **noise; nothing to optimise** | ≈ 5% (`body`) — small, real, **improvable** |
| volatility | ≈ 0.3 | ≈ 50% |

Volatility still has more headroom than direction, so the plain sum leans toward regime.
We keep the plain sum — no arbitrary reweighting — but **report both columns separately
for every trial**, and if the winner-by-score and the winner-by-direction are different
configurations, the notebook shows both and says so. That is the choice the user will
actually want to make, and blending it into one number would hide it.

**This is how the search can settle the open question.** If no configuration anywhere in
the space can make the token keep today's direction, then direction is destroyed *at the
tokenizer*, no downstream predictor can recover it, and the regime pivot is proven rather
than assumed. If some configuration *can*, the candle was never noise — we were simply
running the wrong hyperparameters, and the original plan stands.

### Where "tomorrow" belongs

Not here. The two modules have different jobs and must be tuned on different questions:

| module | trained to produce | tuned on |
|---|---|---|
| **TS / CS** — *this spec* | a token standing for **today** | does today survive the bottleneck |
| **fusion + predictor** — *the next spec* | **tomorrow's token**, and tomorrow's candle | does it forecast |

Tomorrow is the predictor's target, and it gets measured when we tune the predictor. Ask
the tokenizer to forecast and we are grading the wrong module for the wrong thing.

### Rejection, not scoring

A trial whose codebook **collapsed** — fewer than 5% of words alive — scores `−inf`. It is
not ranked, it is thrown out. A token drawn from 12 live words carries ~3.5 bits and is
useless downstream however well it probes.

**This guard is not hygiene. It is what stops the next spec's objective from being a trick
question.** The predictor's target *is tomorrow's token* — so a collapsed codebook does not
merely weaken the predictor, it **destroys its target**. Every day becomes the same word,
"predict tomorrow's token" is satisfied by saying that word forever, and the predictor
scores brilliantly while knowing nothing. We have watched this happen: naming accuracy hit
**87% while the codebook fell to 3 words**.

A tokenizer that collapses hands the predictor a task it can win by shrugging. So the
tokenizer's contract is not just *be informative* — it is **keep tomorrow's token worth
predicting**, and that is enforced here, at the only place it can be.

## Part 0 — Wire the settings to the models

No search can tune a knob that is not connected. Before anything else:

**`settings.py`.** Each codebook's knobs move into the entry that owns them, so the
existing `VQVAE(**settings["ts"])` splat carries them:

```python
"ts":     { ..., "commitment": 0.25, "diversity": 0.1, "decay": 0.99, "dropout": 0.1, "heads": 4 },
"cs":     { ..., "commitment": 0.25, "diversity": 0.1, "decay": 0.99, "dropout": 0.1, "heads": 4 },
"fusion": { ..., "commitment": 0.25, "diversity": 0.1, "decay": 0.99 },
"loss":   { "naming": 0.1, "candle": 1.0 },      # only the PREDICTOR's two heads remain
"search": { "run": False, "trials": 12, "steps": 600 },
"weight_decay": 0.05,                            # was hardcoded 0.01 in training.py
```

`commitment` **1.0 → 0.25** is a correction to the literature value, not a tuning choice.

**`vqvae.py`.** `__init__` gains `decay` and passes it to `Codebook`; its `commitment`
default changes to `0.25` so it agrees with `Codebook`.

**`world.py` / the notebook.** `WorldModel` already accepts `candle_weight` and
`naming_weight`; the notebook must actually pass `settings["loss"]`.

**The validator's collapse guard** currently reads `loss["diversity"]`, which is moving.
It becomes: for each of `ts`/`cs`/`fusion`, reject `diversity == 0` *and* `commitment == 0`
together, since that leaves nothing holding the codebook apart.

### The test that makes this class of bug impossible

Not "does `commitment=0.9` work" but **"is any setting decorative?"** — a structural test
over `inspect.signature`:

```python
def test_no_setting_in_an_entry_block_is_decorative():
    """A setting that no model reads is worse than no setting: it lies."""
    accepted = set(inspect.signature(VQVAE.__init__).parameters)
    for entry in ("ts", "cs"):
        unread = set(DEFAULTS[entry]) - accepted
        assert not unread, f"SETTINGS[{entry!r}] has settings VQVAE never reads: {unread}"
```

plus the same for `loss` against `WorldModel.__init__`, and a behavioural one asserting
`VQVAE(**{**DEFAULTS["ts"], "commitment": 0.9}).codebook.commitment == 0.9`.

## Part 1 — `bubble_bi/tuning.py`: a two-stage search

At 12 trials, a blind six-knob search is a lottery. So we shrink the *space*, not the
search: knobs whose answer we already know are **fixed**, and the twelve trials are split
across the two questions the user actually asked — *"get the sizes correct, the balance
right."*

**Fixed, with reasons:**

| knob | value | why it is not searched |
|---|---|---|
| `decoder_depth` | 2 | **the decoder is thrown away** when the tokenizer is frozen |
| `revive_every` | 50 | not where the problem is |
| grad clip | 1.0 | not where the problem is |
| `batch` | TS 256 / CS 64 | already reasoned from the 30× data-size gap |
| `weight_decay` | 0.05 | STORM's value; we run 0.01 |

**Stage A — the balance** (half the trials; sizes held at their defaults)

| knob | range |
|---|---|
| `learning_rate` | 3e-5 … 3e-3, log |
| `commitment` | 0.1 … 2.0, log |
| `diversity` | 0.0 … 1.0 |

**Stage B — the sizes** (half the trials; balance held at Stage A's winner)

| knob | range |
|---|---|
| `model_size` | 64 / 128 / 256 |
| `vocabulary` | 128 / 256 / 512 / 1024 |
| `days` | TS: 5/10/15/20/30 · CS: 1/3/5/10 |

Coordinate descent assumes the two groups do not interact much. They do interact — the
optimal learning rate genuinely moves with width
([Tensor Programs V](https://arxiv.org/pdf/2203.03466)) — so this is an approximation,
and it is the price of a 12-trial budget. Two mitigations: Stage B's winner is confirmed
head-to-head at full budget (Part 2), and raising `trials` narrows the gap without any
code change.

**Built generic, on purpose.** The fusion + predictor search comes next, and it must be a
new *objective and space* — not a second implementation. So `bb.tuning` takes the scoring
function as an argument, and everything around it is shared: the trial table, the pruner,
the collapse guard, the resume-to-Drive, the head-to-head confirm, and `tuned.json`. This
spec supplies one scorer (`score_tokenizer`); the next spec supplies another.

**Engine:** Optuna. TPE sampler + `MedianPruner`, study persisted to a SQLite file under
`data_dir` (= Drive on Colab) **after every trial**, so a disconnect costs one trial and
not the session. Optuna is pure Python with no torch dependency — it cannot replace
Colab's CUDA build, which is the failure mode we have hit before.

At 12 trials TPE is barely past its startup phase and behaves close to random search.
That is fine and it is stated in the output; the point of choosing TPE is that raising
`trials` to 60 makes the *same code* genuinely better.

**Trial length.** `search["steps"]` (600) is a **ceiling**, not a target: `train()`'s
existing early stopping ends a trial as soon as held-out error stops improving, and
already restores the best checkpoint. So a trial is scored at its best, and CS — which
overfits at 15 passes — stops itself rather than being scored on a ruined model.

### Reported per trial

`score`, and then, separately: `direction`, `volatility`, `words_used`, `rebuild`, and
`before_quant` — the same probe run on the *continuous* summary vector instead of the
token. One extra ridge fit, near-zero cost, and it answers a standing open question: a
large gap between `before_quant` and `direction` means the **codebook** is destroying the
signal, not the encoder.

## Part 2 — The transfer guard

The known trap, and the one that has already caught us: **a config that wins a 600-step
sprint can lose a 5,000-step run.** CS did exactly this — it looked fine early, then its
held-out error *climbed* from 0.90 to 1.03 while its codebook decayed from 187 words to
141.

So the search never gets the last word:

> After the search picks a winner, train **the winner and the current incumbent** at full
> budget, head to head, on the same seed. If the winner does not actually beat the
> incumbent, the notebook **says so and keeps the incumbent.**

Only a config that survives this is written to `tuned.json`.

## Part 3 — The search is opt-in; its answer is an artifact

The search is a one-time act of discovery. **Everyone after inherits the answer without
running anything.**

```python
"search": {"run": False, ...}    # OFF by default. The next runner never searches.
```

| `run` | what the notebook's search cell does |
|---|---|
| `False` (default) | loads `tuned.json`, shows where it came from, applies it. **Seconds, no GPU.** |
| `True` | runs the two-stage search, writes a new `tuned.json`, prints the head-to-head |

### `tuned.json`, committed to the repo

Not Drive — Drive is private to whoever ran it. Committed, so the next person gets the
tuning **by cloning**. It records not just the numbers but what they were found on:

```json
{
  "found_on": "2026-07-12",
  "trials": 12,
  "fingerprint": {"tickers": 30, "features": 26, "start": "2010-01-01", "search_steps": 600},
  "score": {"ts": {"direction": 0.09, "volatility": 0.54, "words_used": 412}, "cs": {"...": "..."}},
  "ts": {"learning_rate": 4.2e-4, "commitment": 0.31, "diversity": 0.4,
         "model_size": 256, "vocabulary": 1024, "days": 15},
  "cs": {"...": "..."}
}
```

### The fingerprint keeps us honest

Tuned hyperparameters are valid **only for the data they were tuned on**. We have already
changed the feature count twice (D=10 → 22 → 26). Silently reusing stale hyperparameters
is precisely the class of bug we keep catching by measuring, so the load path compares the
fingerprint against the live settings and says exactly one of:

```
✅  Using tuned settings — found 2026-07-12 over 12 trials, on this same data.
⚠️  tuned.json was found on 22 features; you are running 26. The tuning is STALE.
    Using it anyway, but set search["run"] = True to re-tune.
ℹ️  No tuned.json — running on defaults. Set search["run"] = True to search.
```

Stale tuning **warns and continues** (via `report(known_problem=...)`, the existing
mechanism for a failure that is understood and written down) rather than raising: a
half-stale tuning still beats untuned defaults, and stopping the notebook dead would
tempt the next person to delete the file.

### Precedence

Three layers, most specific wins:

```
DEFAULTS   <   tuned.json   <   what you typed in the notebook's SETTINGS
```

A tuned value replaces a default, but a value you *deliberately* wrote in the notebook
stands. `check()` already receives only the keys actually typed, so it can tell the
difference without guessing.

## Part 4 — What the notebook shows

One new section, after the tensors and before the TS training:

1. The load-or-search banner above.
2. **The trial table**, sorted by score, with `direction` and `volatility` as separate
   columns — never a single blended number, for the same reason we no longer headline
   "explains 44%".
3. **Which knob actually mattered** — a bar chart of parameter importance. Even 12 random
   trials answer "is learning rate dominating everything?", which is worth knowing.
4. The head-to-head confirm result.
5. The banner that stops this pretending to be more than it is:

   > ⚠️ 12 trials is a **screen**, not an optimum. It will catch a badly-wrong setting.
   > It will not find the best one. Raise `SETTINGS["search"]["trials"]` to 60+ for a
   > real search — the code is identical, it just gets better.

6. A `report()` check block, ending the section like every other.

## Files

| file | change |
|---|---|
| `bubble_bi/settings.py` | move codebook knobs into `ts`/`cs`/`fusion`; shrink `loss`; add `search`; validate; fix the collapse guard |
| `bubble_bi/models/vqvae.py` | accept `decay`; `commitment` default 1.0 → 0.25 |
| `bubble_bi/models/world.py` | `Tokenizer` must pass `settings["fusion"]`'s `commitment`/`diversity`/`decay` to the fusion `Codebook` — today it does not |
| `bubble_bi/training.py` | `weight_decay` is hardcoded `0.01` at `training.py:185` and `:389`; make it a setting, default `0.05` |
| `bubble_bi/tuning.py` | **new** — search space, probe scoring, the two stages, `tuned.json` load/save/fingerprint |
| `bubble_bi/__init__.py` | export `tuning` |
| `bubble_bi/plots.py` | `tuning_importance(study)` bar chart |
| `Bubble_Bi.ipynb` | new section; pass `settings["loss"]` to `WorldModel` |
| `tuned.json` | **new**, repo root, committed |
| `tests/test_tuning.py` | **new** |
| `tests/test_settings.py` | the "no decorative setting" structural test |

## Testing

- **The structural test above** — every key in `DEFAULTS["ts"]`/`["cs"]` is a parameter
  `VQVAE.__init__` accepts; every key in `DEFAULTS["loss"]` is one `WorldModel.__init__`
  accepts. This is the test that would have caught the bug that motivated the spec.
- **Behavioural:** a changed setting changes the model (`commitment=0.9` →
  `model.codebook.commitment == 0.9`).
- **The shuffled floor works:** a probe on a token that carries *no* information about the
  target scores a skill of ≈ 0, not ≈ R²_capacity. Assert with a random token.
- **The capacity confound is neutralised:** a random token at `vocabulary=1024` and one at
  `vocabulary=128` both score skill ≈ 0. Without the shuffled floor, the 1024 case scores
  higher. This test fails on the naive implementation.
- **Collapse is rejected:** a model forced onto 2 words scores `−inf`, not a good number.
- **Precedence:** DEFAULTS < tuned.json < notebook SETTINGS, asserted in all three
  directions.
- **The fingerprint detects staleness:** a `tuned.json` written at 22 features, loaded at
  26, warns and does not silently pass.
- **The search never touches `test`:** assert the test loader is not constructed during a
  search. This is a no-lookahead guarantee and deserves a test, not a comment.
- **The probe targets come from the window, not the future:** assert that corrupting
  `arrays.y` (tomorrow's return) leaves every trial's score bit-for-bit identical. The
  search must be provably blind to the future — and this test is what proves it.
- **End-to-end:** a 2-trial search on a tiny synthetic panel completes, writes a
  `tuned.json` that `check()` accepts, and resumes from a killed study.

## Out of scope

- Fusion and predictor tuning — their own spec, on top of these winners. That is where
  **tomorrow** gets measured, because tomorrow's token is the *predictor's* target, not the
  tokenizer's. Note the dependency runs one way and cannot be reversed: this search must
  land first, because a collapsed tokenizer would leave the predictor with a degenerate
  target and make its whole search meaningless.
- μP / width-aware learning-rate scaling. It is the principled fix for the Stage-A/Stage-B
  interaction and worth revisiting if the head-to-head confirm keeps rejecting winners,
  but it is an architecture change and not worth it at a 12-trial budget.
- Tuning `decoder_depth`, `batch`, `weight_decay`, `revive_every` — fixed above, with
  reasons.
