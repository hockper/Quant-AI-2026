"""The project's settings live in the notebook, as a plain dict.

This module is the only thing that knows what a valid settings dict looks like.
It fills in defaults, rejects mistakes with a message a human can act on, and
prints a summary of what is about to run.

There is no config file and no config class on purpose: what you read in the
notebook is exactly what runs.
"""

from __future__ import annotations

# Every setting the project understands, with its default.
# Nested blocks mirror the two entries the tokenizer reads a day through.
DEFAULTS: dict = {
    "tickers": None,          # required
    "start": None,
    "end": None,

    # Entry 1 — TS: what THIS stock has been doing (one stock, over time).
    # One grid per company per day, so there are tens of thousands of them.
    "ts": {
        "days": 4,
        "vocabulary": 512,
        "encoder_depth": 3,
        "decoder_depth": 2,
        "batch": 256,
    },
    # Entry 2 — CS: what the WHOLE MARKET was doing (all stocks, on a day).
    # Only ONE grid per day, so there are ~30x fewer of them — and each is ~30x bigger
    # (every company at once). Hence its own, much smaller batch: a batch of 256 would
    # be a tenth of the entire training set, giving barely a handful of steps per pass.
    "cs": {
        "days": 5,
        "vocabulary": 512,
        "encoder_depth": 3,
        "decoder_depth": 2,
        "batch": 64,
    },
    # Where the two entries merge into the single token we keep.
    # `attend_to` decides how fine-grained a menu CS offers the cross-attention:
    #   "days"       one vector per market day   (5 keys)  -- what the paper does
    #   "companies"  one vector per company      (30 keys) -- a bank can attend to banks
    #   "cells"      every (company, day)        (150 keys) -- richest, still cheap
    # The output is ONE vector either way: cross-attention's length follows the QUERY.
    "fusion": {
        "vocabulary": 512,
        "depth": 2,
        "attend_to": "days",
        "batch": 32,
    },
    # The GPT that reads sentences of tokens.
    "predictor": {
        "sentence_length": 64,
        "depth": 4,
    },

    # What the model is being pulled towards, and how hard.
    #
    # These four fight each other, and the balance IS the design:
    #
    #   naming      predict tomorrow's WORD. ⚠️ This is the one that rewards CHEATING:
    #               make every day the same word and it is trivially satisfied. Turn it
    #               up and the codebook collapses -- we watched it fall to 3 words while
    #               "accuracy" shot to 87%. The paper keeps it small (0.1) for this
    #               reason, and so do we.
    #   candle      draw tomorrow's CANDLE. This is the anchor. A collapsed token carries
    #               no information, so it cannot draw a candle -- which makes the cheat
    #               expensive. It also forces the token to carry DIRECTION (the body is
    #               where it closed against where it opened), the one thing both halves
    #               of the tokenizer were throwing away.
    #   commitment  keep the encoder's output close to a word it already has.
    #   diversity   punish the dictionary for crowding onto a few words (STORM eq. 4).
    "loss": {
        "naming": 0.1,
        "candle": 1.0,
        "commitment": 1.0,
        "diversity": 0.1,
    },

    # How to divide history. Strictly by DATE, never at random: the model must be
    # tested on days it has never seen, in the order they actually happened.
    #   learn    the model trains on these
    #   tune     used to check progress and stop at the right time
    #   test     touched once, at the very end. Whatever is left over.
    "split": {
        "learn": 0.70,
        "tune": 0.15,
    },

    # Shared on purpose: TS and CS meet in a cross-attention layer, which needs
    # both sides to be the same width. Splitting this would need extra projection
    # layers -- an architecture change, not a setting.
    "model_size": 128,

    "steps": 2000,
    "learning_rate": 1e-4,
    "seed": 42,
    "data_dir": "artifacts",
}

# Settings that must be a positive whole number.
_POSITIVE = {
    ("ts", "days"), ("ts", "encoder_depth"), ("ts", "decoder_depth"), ("ts", "batch"),
    ("cs", "days"), ("cs", "encoder_depth"), ("cs", "decoder_depth"), ("cs", "batch"),
    ("fusion", "depth"), ("fusion", "batch"),
    ("predictor", "sentence_length"), ("predictor", "depth"),
    ("model_size",), ("steps",),
}
# Settings that must be a vocabulary of at least two words.
_VOCAB = {("ts", "vocabulary"), ("cs", "vocabulary"), ("fusion", "vocabulary")}


def _where(path: tuple[str, ...]) -> str:
    return "".join(f"[{p!r}]" for p in path) if len(path) > 1 else repr(path[0])


def check(settings: dict) -> dict:
    """Validate the notebook's settings and fill in anything left out.

    Returns a complete settings dict. Raises ValueError with a message you can
    act on if something is wrong.
    """
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise ValueError(
            f"Unknown setting(s): {sorted(unknown)}.\n"
            f"Valid settings are: {sorted(DEFAULTS)}"
        )

    out: dict = {}
    for key, default in DEFAULTS.items():
        given = settings.get(key, {} if isinstance(default, dict) else default)
        if isinstance(default, dict):
            extra = set(given) - set(default)
            if extra:
                raise ValueError(
                    f"`{key}` got unknown setting(s): {sorted(extra)}.\n"
                    f"Valid settings for `{key}` are: {sorted(default)}"
                )
            out[key] = {**default, **given}
        else:
            out[key] = given

    tickers = [t.strip().upper() for t in (out["tickers"] or []) if t.strip()]
    if not tickers:
        raise ValueError("`tickers` is empty — list at least one company, e.g. ['AAPL'].")
    seen: set[str] = set()
    out["tickers"] = [t for t in tickers if not (t in seen or seen.add(t))]

    for path in _POSITIVE:
        value = out[path[0]][path[1]] if len(path) > 1 else out[path[0]]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{_where(path)} must be a whole number of at least 1, got {value!r}.")

    for path in _VOCAB:
        value = out[path[0]][path[1]]
        if not isinstance(value, int) or isinstance(value, bool) or value < 2:
            raise ValueError(f"{_where(path)} must be at least 2, got {value!r}.")

    attend = out["fusion"]["attend_to"]
    if attend not in ("days", "companies", "cells"):
        raise ValueError(
            f"`fusion['attend_to']` must be 'days', 'companies' or 'cells' — "
            f"got {attend!r}. It decides how fine-grained a menu the market offers "
            "each company's token to read."
        )

    for name, weight in out["loss"].items():
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight < 0:
            raise ValueError(
                f"`loss['{name}']` must be a number of 0 or more, got {weight!r}."
            )
    if out["loss"]["candle"] == 0 and out["loss"]["diversity"] == 0:
        raise ValueError(
            "With both `loss['candle']` and `loss['diversity']` at 0, nothing stops the "
            "codebook collapsing: the model will make every day the same word, because "
            "a constant word is trivially easy to predict."
        )

    if out["learning_rate"] <= 0:
        raise ValueError(f"`learning_rate` must be above 0, got {out['learning_rate']!r}.")

    learn, tune = out["split"]["learn"], out["split"]["tune"]
    if not (0 < learn < 1) or not (0 < tune < 1):
        raise ValueError(f"`split` fractions must be between 0 and 1, got {out['split']}.")
    if learn + tune >= 1:
        raise ValueError(
            f"`split['learn']` + `split['tune']` = {learn + tune:.2f}, which leaves "
            "nothing for the final test. They must add up to less than 1."
        )

    return out


def device() -> str:
    """Which hardware we will actually run on: 'tpu', 'gpu' or 'cpu'."""
    try:
        import torch
    except ImportError:
        return "cpu"
    try:
        import torch_xla.core.xla_model as xm

        xm.xla_device()
        return "tpu"
    except Exception:
        pass
    return "gpu" if torch.cuda.is_available() else "cpu"


def summary(settings: dict) -> str:
    """A one-glance description of what is about to run."""
    ts, cs = settings["ts"], settings["cs"]
    period = f"from {settings['start']}" if settings["start"] else "all available history"

    ts_line = f"     TS  this stock, {ts['days']} days back".ljust(40)
    cs_line = f"     CS  whole market, {cs['days']} days back".ljust(40)
    return "\n".join([
        f"Running on {device().upper()}",
        f"{len(settings['tickers'])} companies, {period}",
        "",
        "The tokenizer reads each day through two windows:",
        f"{ts_line}→ {ts['vocabulary']} words, depth {ts['encoder_depth']}",
        f"{cs_line}→ {cs['vocabulary']} words, depth {cs['encoder_depth']}",
        f"     ⤷ merged into ONE token out of {settings['fusion']['vocabulary']}",
        "",
        f"Predictor reads sentences of {settings['predictor']['sentence_length']} tokens "
        f"(depth {settings['predictor']['depth']})",
    ])
