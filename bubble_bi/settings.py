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
    "ts": {
        "days": 4,
        "vocabulary": 512,
        "encoder_depth": 3,
        "decoder_depth": 2,
    },
    # Entry 2 — CS: what the WHOLE MARKET was doing (all stocks, on a day).
    "cs": {
        "days": 5,
        "vocabulary": 512,
        "encoder_depth": 3,
        "decoder_depth": 2,
    },
    # Where the two entries merge into the single token we keep.
    "fusion": {
        "vocabulary": 512,
        "depth": 2,
    },
    # The GPT that reads sentences of tokens.
    "predictor": {
        "sentence_length": 64,
        "depth": 4,
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
    "batch_size": 256,
    "learning_rate": 1e-4,
    "seed": 42,
    "data_dir": "artifacts",
}

# Settings that must be a positive whole number.
_POSITIVE = {
    ("ts", "days"), ("ts", "encoder_depth"), ("ts", "decoder_depth"),
    ("cs", "days"), ("cs", "encoder_depth"), ("cs", "decoder_depth"),
    ("fusion", "depth"),
    ("predictor", "sentence_length"), ("predictor", "depth"),
    ("model_size",), ("steps",), ("batch_size",),
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
