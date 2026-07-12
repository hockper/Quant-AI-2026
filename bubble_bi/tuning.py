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


_FLOOR_REPS = 64  # how many shuffles the floor is averaged over -- see the note below.


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
    rng = np.random.default_rng(seed)
    floor = np.mean([_probe(x[rng.permutation(len(x))], y) for _ in range(_FLOOR_REPS)])
    return _probe(x, y) - floor


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
