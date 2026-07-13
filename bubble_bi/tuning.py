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
from pathlib import Path

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


def _is_onehot(x: np.ndarray) -> bool:
    """True when every row of `x` is a token: all zeros except a single 1.

    `skill()` is handed two different shapes of thing — the quantised TOKEN (one-hot)
    and the CONTINUOUS `summary` vector, before the codebook rounded it off. Only the
    first shape gets the cheap treatment below; the check costs one pass over `x` (the
    same order of work `_probe_onehot` does anyway), so there is no reason to make the
    caller say which one it is and risk the two falling out of sync.
    """
    return bool(np.all(x.sum(axis=1) == 1) and np.all((x == 0) | (x == 1)))


def _probe_onehot_ids(ids: np.ndarray, words: int, y: np.ndarray, train: float = 0.7) -> float:
    """The engine underneath `_probe_onehot`, working straight from WORD IDS rather than
    a built one-hot matrix.

    This split matters for the FLOOR, not just this one call: the floor reshuffles the
    row order 64 times, and reshuffling a one-hot matrix means copying the whole
    (rows × words) array 64 times over. An id is one integer per row -- reshuffling THAT
    is 64 tiny copies instead, so the floor stops paying for the one-hot encoding it
    never actually needed.

    Why this is cheap at all: with a one-hot design, `x_fit.T @ x_fit` — the matrix
    `autopsy._probe` builds and solves, at O(rows·words²) to build and O(words³) to
    solve — is almost entirely zeroes. Word A and word B never both fire on the same
    row, so their cross term is exactly 0. The only column that touches every row is the
    intercept (it is 1 everywhere), so the matrix is a DIAGONAL of per-word counts with
    one extra dense border for the intercept — solved exactly with per-word sums, no
    words×words matrix and no words³ solve, ever:

        for word i:  (count_i + λ)·w_i  +  count_i·w_last          = sum_i(y)
        intercept :  Σ_i count_i·w_i    +  (n_fit + λ)·w_last      = Σ(y)

    Eliminate w_i from the first line and substitute into the second (one scalar
    "Schur complement" per target column) and both are solved in a single pass over the
    words. Same ridge λ = 1e-2 as `autopsy._probe`, on purpose: this is an optimisation
    of that exact estimator, not a different one.
    """
    keep = np.isfinite(y).all(axis=1)
    ids, y = ids[keep], y[keep].astype(np.float64)
    if len(ids) < 50:
        return float("nan")

    cut = int(len(ids) * train)
    ids_fit, y_fit = ids[:cut], y[:cut]
    ids_test, y_test = ids[cut:], y[cut:]

    lam = 1e-2                                       # identical to autopsy._probe's ridge
    n_fit = len(ids_fit)

    counts = np.bincount(ids_fit, minlength=words).astype(np.float64)  # rows per word
    sums = np.stack([                                 # per-word sum of y, one column at a time
        np.bincount(ids_fit, weights=y_fit[:, col], minlength=words)
        for col in range(y_fit.shape[1])
    ], axis=1)
    total = y_fit.sum(axis=0)                        # the intercept's own right-hand side

    d = counts + lam                                 # each word's own diagonal entry
    b = counts                                        # the border it shares with the intercept
    c = n_fit + lam                                  # the intercept's own diagonal entry

    # A word never seen in TRAINING (count 0) gets d = λ, b = 0 here, which makes its
    # coefficient exactly 0 below -- its prediction collapses to just the intercept.
    # That is not a special case we added: it is what the dense solve does too, because
    # that word's whole row of x_fit.T @ x_fit is zero except the tiny λ on its diagonal.
    weighted = (b[:, None] * sums / d[:, None]).sum(axis=0)
    schur = c - (b * b / d).sum()
    intercept = (total - weighted) / schur
    per_word = (sums - np.outer(b, intercept)) / d[:, None]

    predicted = per_word[ids_test] + intercept
    left = ((y_test - predicted) ** 2).sum()
    total_var = ((y_test - y_fit.mean(axis=0)) ** 2).sum()
    return float(1 - left / max(total_var, 1e-12))


def _probe_onehot(x: np.ndarray, y: np.ndarray, train: float = 0.7) -> float:
    """THE EXACT SAME ridge probe as `autopsy._probe(x, y)` — same split, same ridge
    strength, same held-out R² — for the case this file actually has: `x` is ONE-HOT
    (every row picks exactly one word). Takes the identical (x, y) shape as `_probe` so
    the two can be compared directly; see `_probe_onehot_ids` for how it is cheap.
    """
    keep = np.isfinite(x).all(axis=1)      # matches `_probe`'s x-finite filter exactly
    return _probe_onehot_ids(x[keep].argmax(axis=1), x.shape[1], y[keep], train)


# How many shuffles the floor is averaged over. Measured, not guessed.
#
# The worst case in the search space is CS at vocabulary=1024: only ~2.5 rows per word
# (2,560 rows / 1,024 words), which is where the floor wobbles the most. Measured at that
# shape, the floor's own standard error is:
#
#     8 reps -> ~0.021     32 reps -> ~0.008     64 reps -> ~0.006     256 reps -> ~0.003
#
# The smallest real effect the search needs to tell apart is `direction` ≈ 0.05 (the
# harder, smaller signal -- `volatility` ≈ 0.5 is not in any danger). 64 reps leaves
# roughly an 8x margin between the floor's wobble and that effect, which is comfortable
# without paying for reps that buy down noise nobody can see anyway. This is only
# affordable at all because `_probe_onehot_ids` makes one rep cost milliseconds, not
# seconds -- at the old dense-matrix cost, these same 64 reps were the entire 38-second
# bill this file exists to cut.
_FLOOR_REPS = 64


def _score_and_floor(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float, float]:
    """Returns (the real score, the shuffled floor, the floor's STANDARD ERROR).

    ⚠️ For a one-hot `x`, the reshuffle is done on the WORD IDS (one integer per row),
    not on the one-hot matrix itself. Reshuffling only breaks the pairing between a row
    and its `y` -- which row got which WIDTH of encoding never changes -- so permuting
    the (rows × words) matrix 64 times over is 64 full copies of an array bought for
    nothing. Permuting `x.argmax(axis=1)` (one int per row) is the same shuffle, at
    a fraction of the memory traffic. This is the second half of the speed fix: the
    first half (`_probe_onehot_ids`) made ONE fit cheap; this makes RESHUFFLING it cheap.
    """
    rng = np.random.default_rng(seed)
    if _is_onehot(x):
        words = x.shape[1]
        ids = x.argmax(axis=1)
        real = _probe_onehot_ids(ids, words, y)
        draws = np.array([_probe_onehot_ids(ids[rng.permutation(len(ids))], words, y)
                           for _ in range(_FLOOR_REPS)])
    else:
        real = _probe(x, y)
        draws = np.array([_probe(x[rng.permutation(len(x))], y) for _ in range(_FLOOR_REPS)])
    floor = float(draws.mean())
    se = float(draws.std(ddof=1) / math.sqrt(_FLOOR_REPS))
    return real, floor, se


def skill(x: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """How much of `y` a linear probe recovers from `x`, ABOVE WHAT LUCK WOULD GIVE IT.

    The floor is not decoration. The token enters the probe one-hot, so a 1024-word
    vocabulary hands the probe 1024 columns and a 128-word one hands it 128 — raw R² would
    climb with `vocabulary` from capacity alone, and the search would 'discover' that
    bigger is better having discovered nothing.

    Shuffling the rows of `x` breaks the pairing while keeping the width, which is exactly
    a capacity-matched floor. Subtract it and the confound is gone.

    ⚠️ ONE shuffle is not enough to trust. At 1024 columns and only a few hundred rows,
    a single reshuffled ridge fit is itself a coin flip -- measured, one draw swings by
    more than the gap this function exists to detect, so a wide and a narrow vocabulary
    could land on either side of "the same" purely by which shuffle happened to land.
    Averaging many shuffles is still exactly the same floor, just measured steadily
    instead of guessed once.
    """
    real, floor, _ = _score_and_floor(x, y, seed)
    return real - floor


def skill_and_noise(x: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """The exact number `skill()` returns, plus the floor's standard error.

    Two trials whose scores differ by less than a couple of these are not distinguishable
    — they are noise apart, not a real difference the search should trust.
    """
    real, floor, se = _score_and_floor(x, y, seed)
    return real - floor, se


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
        return {"score": -math.inf, "direction": float("nan"), "direction_se": float("nan"),
                "volatility": float("nan"), "volatility_se": float("nan"),
                "before_quant": float("nan"),
                "words_used": used, "why": f"codebook collapsed: {used} of {words} words"}

    token = one_hot(ids, words)
    went, went_se = skill_and_noise(token, direction)
    violent, violent_se = skill_and_noise(token, volatility)
    return {
        "score": went + violent,
        "direction": went,
        # The floor is an AVERAGE of shuffles, not the true value -- this is how far that
        # average might still be off. Two trials whose `direction` differs by less than a
        # couple of `direction_se` are not really different; they are noise apart. This
        # matters most for `direction`, whose whole effect (~0.05) is barely bigger than
        # this noise, unlike `volatility` (~0.5) which drowns it out easily.
        "direction_se": went_se,
        "volatility": violent,
        "volatility_se": violent_se,
        # The same probe on the CONTINUOUS vector, before the codebook rounded it off. A
        # big gap means the CODEBOOK is destroying the signal, not the encoder.
        "before_quant": skill(summary, direction),
        "words_used": used,
        "why": "",
    }


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

# `learning_rate` and `model_size` are top-level settings; the rest live inside the
# entry's own block. PUBLIC on purpose -- this is the one place that knows the split,
# and `settle()` below is the one place that uses it. Anyone folding a `search()` result
# back into a settings dict should use `settle()`, not re-derive this set by hand (see
# `search()`'s docstring for the trap that happens when someone does).
TOP_LEVEL = frozenset({"learning_rate", "model_size"})


def _ask(trial, name, rule):
    kind = rule[0]
    if kind == "log":
        return trial.suggest_float(name, rule[1], rule[2], log=True)
    if kind == "float":
        return trial.suggest_float(name, rule[1], rule[2])
    if kind == "pick":
        return trial.suggest_categorical(name, rule[1])
    raise ValueError(f"unknown rule {kind!r} for {name!r}")


def settle(settings: dict, entry: str, chosen: dict) -> dict:
    """A full settings dict with `chosen` folded into the right places.

    `chosen` is FLAT -- exactly the shape `search()` hands back: some of its keys
    (`TOP_LEVEL`) belong at the top of the settings dict, and the rest belong inside
    `settings[entry]`. This function is the one place that knows how to sort them, so
    it is also the one and only correct way to turn a `search()` result into something
    `bubble_bi.settings.check()` will accept. Do not re-split `chosen` by hand elsewhere
    -- see `search()`'s docstring for the bug that causes.
    """
    out = {**settings, **{k: v for k, v in chosen.items() if k in TOP_LEVEL}}
    out[entry] = {**settings[entry],
                  **{k: v for k, v in chosen.items() if k not in TOP_LEVEL}}
    return out


def _run_one(entry, chosen, batches, settings, scorer, features, companies, trial=None):
    """Train one configuration and score it. Returns the scorer's dict."""
    import optuna

    from bubble_bi.data.tensors import tuning_loaders
    from bubble_bi.models import VQVAE
    from bubble_bi.training import train

    live = settle(settings, entry, chosen)
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
    """Find good settings for one entry. Returns (best, trials_table).

    Two stages: the BALANCE first (learning rate, commitment, diversity, with the sizes
    held at their defaults), then the SIZES with the winning balance held fixed.

    Coordinate descent assumes the two groups barely interact. They do interact -- the
    best learning rate genuinely moves with width -- so this is an approximation, and it
    is the price of a twelve-trial budget. Two things keep it honest: the winner is
    confirmed head-to-head at FULL budget (see `confirm`), and raising `search["trials"]`
    narrows the gap with no change to this code.

    ⚠️ `best` IS FLAT, AND THAT IS ON PURPOSE -- READ THIS BEFORE YOU USE IT.
    `best` mixes two different kinds of key, because that is exactly what `SPACE` tunes:
    most keys (`vocabulary`, `days`, `commitment`, `diversity`, ...) belong INSIDE this
    entry's own block (`settings["ts"]` or `settings["cs"]`), but `learning_rate` and
    `model_size` are TOP-LEVEL settings (see `TOP_LEVEL`) -- they live once at the top of
    the settings dict, shared by both entries, not inside either block.

    So you CANNOT do `settings["ts"] = best`. `settings.check()` will reject the result
    with "Unknown setting(s)" the moment `learning_rate` or `model_size` shows up inside
    `settings["ts"]`, where neither belongs. And do not "fix" that by deleting the two
    keys before you assign the rest into the block -- that throws away the very values
    the search just spent a budget finding, and you would ship with whatever
    `learning_rate`/`model_size` the settings dict already had, silently.

    The one correct move is `tuning.settle(settings, entry, best)`: it returns a
    complete settings dict with `best` folded into the right places -- ready to hand
    straight to `settings.check()`.
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
    settled = settle(settings, entry, fixed)
    return {**settled[entry], "learning_rate": settled["learning_rate"],
            "model_size": settled["model_size"]}, table


def _study_path(settings: dict, entry: str, stage: str) -> str:
    """Where the study lives. On Colab `data_dir` is Drive, so a disconnect costs one
    trial rather than the whole session."""
    folder = Path(settings["data_dir"]) / "search"
    folder.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{folder / f'{entry}-{stage}.db'}"


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
    """Write the answer, and what it was found on. Committed to the repo, not Drive --
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
