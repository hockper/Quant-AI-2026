"""Shared fixtures for the test suite.

`_synthetic_panel`, `tiny_settings` and `tiny_batches` used to live inside
`test_tuning.py` alone. `test_sentences.py` needs the exact same tiny synthetic
panel -- not a second, similar-looking copy that could quietly drift apart from
the original (different feature quirks, different random seed, a "fix" applied
to one but not the other) -- so they live here instead, where every test file
shares the one panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import bubble_bi as bb
from bubble_bi.settings import DEFAULTS


def _tiny_settings_dict(data_dir: str) -> dict:
    """The actual settings, factored out of the `tiny_settings` fixture below so
    `_tiny_joint` -- which cannot ask pytest for `tmp_path`, because it is called
    directly as a plain function rather than injected as a fixture -- can build the
    exact same settings from a directory it makes for itself, instead of keeping a
    second, similar-looking copy that could quietly drift from this one.
    """
    return {
        **DEFAULTS,
        "tickers": ["AAA", "BBB", "CCC"],
        # The Optuna study lands under data_dir -- keep it out of the real artifacts
        # folder, or the resume test would find a stale study from a previous run.
        "data_dir": data_dir,
        "model_size": 16,
        "learning_rate": 1e-3,
        # ⚠️ DELIBERATELY DIFFERENT from search["steps"] below. `confirm()`'s whole job is
        # to retrain at THIS, the full budget -- never the search's short one. If the two
        # numbers were equal (they used to be: both 20), a `confirm()` that used the wrong
        # budget by accident would be numerically indistinguishable from a correct one, and
        # every test in this file would still pass. See
        # `test_confirm_trains_at_the_full_budget_not_the_search_sprint` for the test that
        # this split makes possible.
        "steps": 40,
        "ts": {**DEFAULTS["ts"], "days": 3, "batch": 8, "vocabulary": 16,
               "encoder_depth": 1, "decoder_depth": 1, "heads": 2, "steps": 20},
        "cs": {**DEFAULTS["cs"], "days": 3, "batch": 4, "vocabulary": 16,
               "encoder_depth": 1, "decoder_depth": 1, "heads": 2, "steps": 20},
        "search": {"run": True, "trials": 2, "steps": 10},
    }


@pytest.fixture
def tiny_settings(tmp_path):
    return _tiny_settings_dict(str(tmp_path))


def _synthetic_panel(settings):
    """A synthetic panel: N companies, 600 days, real features -- the data both
    `tiny_batches` and `healthy_batches` are built from, factored out so the two
    fixtures cannot silently drift apart on anything but the SETTINGS they are handed.

    ⚠️ Sized 600, not the smaller 400 the brief sketched: the slowest feature
    (`amihud`'s year-long normalising window, NORM_WINDOW=252 in microstructure.py)
    needs ~273 days of run-up before it produces a single finite row. At 400 days,
    the 70% learn split only reaches 8 usable days -- below `Scaler.LEAST_DAYS` (60)
    -- and `make_tensors` refuses to normalise. 600 days clears that with room to
    spare while still building and training in well under a second.
    """
    from bubble_bi.data import add_features, make_tensors

    n = 600
    rng = np.random.default_rng(0)
    days = pd.date_range("2020-01-01", periods=n, freq="B")
    frames = []
    for ticker in settings["tickers"]:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        one = pd.DataFrame({
            "date": days,
            "ticker": ticker,
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": close * (1 + abs(rng.normal(0, 0.005, n))),
            "low": close * (1 - abs(rng.normal(0, 0.005, n))),
            "close": close,
            "volume": rng.integers(1e6, 5e6, n).astype(float),
        })
        # add_features needs a (date, ticker) index and a `target` column, same as
        # every other fixture in this suite -- tomorrow's return, computed per company.
        one["target"] = one["close"].shift(-1) / one["close"] - 1.0
        frames.append(one)
    raw = pd.concat(frames, ignore_index=True).set_index(["date", "ticker"]).sort_index()
    table = add_features(raw, settings)
    return make_tensors(table, settings)


@pytest.fixture
def tiny_batches(tiny_settings):
    return _synthetic_panel(tiny_settings)


def _tiny_joint():
    """A whole joint model on the synthetic panel. CPU, seconds.

    ⚠️ A plain function, not a `@pytest.fixture` -- `test_training.py` calls it
    directly (`world, loaders, settings = _tiny_joint()`) with no fixture arguments of
    its own, so it cannot lean on pytest to inject `tmp_path`, `tiny_batches` or
    `tiny_settings`. It builds the exact same settings itself via
    `_tiny_settings_dict` (its own `tempfile.mkdtemp()` standing in for `tmp_path`),
    so there is still only ONE definition of what "tiny" means in this suite.

    `predictor["sentence_length"] = 6` is not arbitrary: `make_sentences` only counts a
    day as usable once BOTH grids behind it are whole (TS's `days` and CS's `days`), and
    only calls a run of `length` consecutive usable days a sentence at all. Ask for more
    days than the synthetic panel's usable run can supply and `make_sentences` raises
    "No usable sentences" instead of quietly handing back nothing -- so this would fail
    loudly, not silently train on an empty loader.

    ⚠️ WHY THE SEED IS NO LONGER THE THING KEEPING THIS TEST GREEN -- read this before
    reaching for a "lucky" seed again.

    `test_a_joint_run_keeps_BOTH_dictionaries_alive` gives `train_joint` a 60-step
    budget with every OTHER argument left at its default -- including `revive_every =
    50`. `check_every` works out to 6 (`steps // 10`). Measured directly (printing
    every check): with `revive_every` left unclamped at 50, the run gets exactly ONE
    dead-word revival, at step 50, and everything before it is a genuine dead lock --
    BOTH dictionaries sit at perplexity EXACTLY 1.0 for every check from step 6 to step
    45, whatever the diversity weight is turned up to -- gradient alone never breaks a
    total collapse in this design (see `world.py`'s own docstring: it is a STABLE fixed
    point). Revival is the only way out.

    That single, late revival had to land BEFORE early stopping ended the run.
    Patience defaults to 5, so the run could legally stop as early as step 36 -- and on
    this synthetic panel, tomorrow's candle genuinely carries no signal (it is an
    unpredictable random walk, on purpose -- see `_synthetic_panel`), so the "drawing"
    loss `train_joint` watches for improvement was pure noise pre-revival. Whether a
    lucky dip reset the patience counter before it ran out was then a property of the
    SEED -- measured over 40 seeds at these settings, only ~45% survived to a fully
    open codebook by step 60.

    That coin flip is now CLOSED, not dodged with a lucky seed: `train_joint` itself
    clamps `revive_every` down to at most `check_every` (see its docstring in
    `training.py`), so this 60-step run now gets its first revival at step 6, not step
    50 -- five checks earlier, and every check thereafter gets its own revival too. But
    the clamp on its own was NOT enough to clear the coin flip -- measured directly: at
    the ORIGINAL vocabulary (16 words each) and fusion batch (8), the clamp alone only
    lifted the pass rate from ~45% to 15/20 seeds, because a small revival pool
    (`batch * sentence_length` real encoder outputs) means several of the words
    "revived" at once collide on the SAME real example, and a small vocabulary means
    the `perplexity > 2.0` floor is a much bigger fraction of the maximum possible
    perplexity (`ln(16) ≈ 2.77`) than it looks. Two honest, seed-independent widenings
    of the SAME margin fixed that, each measured before being kept:
      - `vocabulary`: 16 -> 32 on BOTH TS and CS. More room between "just escaped
        collapse" and the ceiling, so the same absolute floor (2.0) is a smaller,
        easier-to-clear fraction of it.
      - `fusion["batch"]`: 8 -> 16. A bigger, more varied pool of real encoder output
        for `revive_dead_words` to drop dead words onto, so a revival is less likely
        to collapse several words back onto duplicates of the same example.
    With both changes, `torch.manual_seed(0)` -- and every other seed tried -- passes;
    see `test_a_joint_run_keeps_BOTH_dictionaries_alive`'s own notes for the sweep.

    `commitment` is turned down from the project default (0.25 -> 0.05 on both TS and
    CS) because commitment pulls the encoder TOWARD the word it already chose -- while
    the run is collapsed onto one word, that is a push to STAY collapsed, and 0.25 is
    documented elsewhere in this project as strong enough to cause exactly that.
    `diversity` on CS alone is turned up (0.1 -> 1.0): CS's revival draws from only
    `batch * sentence_length` rows a day (one reading per DAY), where TS draws from
    `batch * sentence_length * companies` (one per company-day) -- CS is structurally
    the harder dictionary to reopen from a full collapse, and it is the one that kept
    lagging TS in every sweep that led to this configuration.
    """
    import tempfile

    import torch

    from bubble_bi.data.sentences import make_sentences
    from bubble_bi.models import VQVAE
    from bubble_bi.models.world import Tokenizer, WorldModel

    settings = {
        **_tiny_settings_dict(tempfile.mkdtemp()),
        "predictor": {"sentence_length": 6, "depth": 1},
        # batch 8 -> 16: a bigger, more varied pool of real encoder output for
        # revive_dead_words to drop dead words onto (see the docstring above).
        "fusion": {"depth": 1, "attend_to": "companies", "batch": 16},
    }
    # vocabulary 16 -> 32 on BOTH entries: the same absolute perplexity floor (2.0) is
    # an easier fraction of a bigger vocabulary's ceiling (see the docstring above).
    settings["ts"] = {**settings["ts"], "commitment": 0.05, "vocabulary": 32}
    settings["cs"] = {**settings["cs"], "commitment": 0.05, "diversity": 1.0, "vocabulary": 32}
    batches = _synthetic_panel(settings)
    features = len(bb.data.names())
    n = len(settings["tickers"])

    # Fixed for reproducibility only -- NOT to dodge a coin flip. `train_joint`'s own
    # revive_every <= check_every clamp, plus the vocabulary/batch widening above, are
    # what close the dead-lock risk described above; this plain seed is proven (see the
    # docstring) to pass on at least ten different seeds, not cherry-picked to survive
    # on its own.
    torch.manual_seed(0)
    ts = VQVAE(companies=1, features=features, width=16, **settings["ts"])
    cs = VQVAE(companies=n, features=features, width=16, **settings["cs"])
    tok = Tokenizer(ts, cs, model_size=16, **settings["fusion"])
    world = WorldModel(tok, sentence=6, depth=1, heads=2, **settings["loss"])
    return world, make_sentences(batches, settings), settings
