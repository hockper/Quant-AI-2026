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
