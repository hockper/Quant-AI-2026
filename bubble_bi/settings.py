"""The project's settings live in the notebook, as a plain dict.

This module is the only thing that knows what a valid settings dict looks like.
It fills in defaults, rejects mistakes with a message a human can act on, and
prints a summary of what is about to run.

There is no config file and no config class on purpose: what you read in the
notebook is exactly what runs.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# Every setting the project understands, with its default.
# Nested blocks mirror the two entries the tokenizer reads a day through.
DEFAULTS: dict = {
    "tickers": None,          # required
    "start": None,
    "end": None,

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
    "weight_decay": 0.05,     # STORM's value. Was hardcoded to 0.01 in training.py.
    "seed": 42,
    "data_dir": "artifacts",
}

# Settings that must be a positive whole number.
_POSITIVE = {
    ("ts", "days"), ("ts", "encoder_depth"), ("ts", "decoder_depth"), ("ts", "batch"),
    ("ts", "heads"), ("cs", "heads"),
    ("cs", "days"), ("cs", "encoder_depth"), ("cs", "decoder_depth"), ("cs", "batch"),
    ("fusion", "depth"), ("fusion", "batch"),
    ("predictor", "sentence_length"), ("predictor", "depth"),
    ("search", "trials"), ("search", "steps"),
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

    for entry in ("ts", "cs"):
        budget = out[entry].get("steps")
        if budget is not None and (not isinstance(budget, int) or isinstance(budget, bool)
                                   or budget < 1):
            raise ValueError(
                f"`{entry}['steps']` must be a whole number of at least 1, or None to use "
                f"the shared `steps`. Got {budget!r}."
            )

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

    # ⚠️ The transfer guard must not run BACKWARDS.
    #
    # The search trains each trial for a SHORT sprint (`search['steps']`), then `confirm()`
    # re-trains the winner and the incumbent at the REAL budget to check the sprint winner
    # still wins when it actually matters. That only means anything if the real budget is
    # the LONGER of the two.
    #
    # The notebook shipped `steps = 300` against `search['steps'] = 600`, so for two whole
    # GPU runs the "full budget" confirm trained for HALF as long as the sprint it was
    # meant to validate. A guard that re-runs the sprint shorter than the sprint is not a
    # guard — it is the very failure it exists to catch.
    if out["search"]["run"]:
        for entry in ("ts", "cs"):
            real = out[entry]["steps"] or out["steps"]
            if real < out["search"]["steps"]:
                raise ValueError(
                    f"The search would train each {entry.upper()} trial for "
                    f"{out['search']['steps']:,} steps, then 'confirm at full budget' on "
                    f"only {real:,} — SHORTER than the sprint it is meant to validate. The "
                    "transfer guard would run backwards and mean nothing.\n"
                    f"     Raise `steps` (or `{entry}['steps']`) above "
                    f"`search['steps']`, or lower `search['steps']`.\n"
                    "     A short run is fine when you are only reading the notebook — this "
                    "only bites with `search['run'] = True`."
                )

    if out["weight_decay"] < 0:
        raise ValueError(f"`weight_decay` must be 0 or more, got {out['weight_decay']!r}.")

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


def gpu_present() -> bool:
    """Does this MACHINE have an NVIDIA GPU — whatever PyTorch believes?

    ⚠️ The question we spent an afternoon not asking. A Colab **CPU runtime** ships a
    CPU-only torch. So does a machine whose CUDA torch a stray `pip install` overwrote.
    To `torch.version.cuda` the two look IDENTICAL — it is `None` either way — so we
    blamed the pip install every time and sent people off to delete a runtime that was
    never broken.

    One is a wrong menu choice. The other is a wrecked environment. They have completely
    different fixes, and **only the machine can tell them apart.** So we ask it, not torch:
    the driver either exists or it does not.
    """
    if Path("/proc/driver/nvidia/gpus").is_dir():
        return any(Path("/proc/driver/nvidia/gpus").iterdir())
    if shutil.which("nvidia-smi"):
        found = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        return found.returncode == 0 and "GPU" in found.stdout
    return False                    # no driver and no tool: there is no NVIDIA GPU here


def hardware() -> dict:
    """Everything about the hardware, so nobody has to guess why it is slow — or why it
    cannot see the GPU. See `gpu_present()` for the distinction that matters most.
    """
    facts = {"where": "cpu", "torch": None, "built for cuda": None,
             "cuda available": False, "gpu": None, "gpu present": gpu_present(),
             "why": None}
    try:
        import torch
    except ImportError:
        facts["why"] = "PyTorch is not installed at all."
        return facts

    facts["torch"] = torch.__version__
    facts["built for cuda"] = torch.version.cuda          # None => CPU-ONLY BUILD
    facts["cuda available"] = bool(torch.cuda.is_available())
    facts["where"] = device()

    if facts["cuda available"]:
        try:
            facts["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    elif not facts["gpu present"]:
        # THE COMMON CASE, and the one we used to misdiagnose. There is no GPU on this
        # machine at all. Nothing is broken; you are simply on a CPU runtime.
        #
        # Say ONLY what is true here. Whether the wheel happens to be a CPU one is beside
        # the point and is not always so — a perfectly good CUDA build finds no GPU on a
        # CPU runtime either. Naming the wheel would send the reader off to reinstall
        # PyTorch, which is the exact wrong turn this branch exists to stop them taking.
        facts["why"] = (
            "There is NO GPU on this machine — nothing is broken, you are simply on a CPU "
            "runtime.\n"
            "     Fix: Runtime → Change runtime type → T4 GPU, then Runtime → Restart "
            "session.\n"
            "     The restart matters: a kernel that is already running keeps the PyTorch "
            "it started with."
        )
    elif facts["built for cuda"] is None:
        # A GPU is sitting RIGHT THERE and torch cannot talk to it. NOW the pip advice is
        # the correct advice.
        facts["why"] = (
            "This machine HAS a GPU, but this PyTorch was built without CUDA — a CPU-only "
            "wheel. It cannot use that GPU whatever runtime you pick. Something (usually a "
            "pip install) replaced the CUDA build.\n"
            "     Fix: Runtime → Disconnect and delete runtime, start again, and do not "
            "let anything install `torch`."
        )
    else:
        facts["why"] = (
            "This machine has a GPU and this PyTorch was built for CUDA, yet it still "
            "cannot reach it — usually a driver that no longer matches. "
            "Fix: Runtime → Disconnect and delete runtime, then start again."
        )
    return facts


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
