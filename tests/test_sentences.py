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


def test_a_served_sentence_never_straddles_a_real_trading_gap(tiny_batches, tiny_settings):
    """Plain-language version of the property this test guards:

    A 'sentence' is supposed to be `sentence_length` CONSECUTIVE trading days for a
    company -- a single, unbroken stretch of market history. If a company stopped
    trading partway through that stretch (a halt, a delisting, a hole in the data feed)
    and the window is served anyway, it is not one story any more -- it is two unrelated
    stories stapled together with the seam hidden. The model would be taught that a
    company can vanish for a week and pick up again mid-sentence as if nothing happened,
    which never happens in a real market, and that lie would poison everything the GPT
    learns to predict from it.

    `arrays.ok[day, company]` is the ground truth for "did this company actually trade
    on this day" -- it is exactly what `Sentences` itself is supposed to reason from
    (see `bubble_bi/data/sentences.py`, via `_complete_windows` in
    `bubble_bi/data/tensors.py`). So this test punches a REAL, multi-day hole in one
    company's trading record -- comfortably inside the learn period, comfortably shorter
    than the period itself (so windows can still exist either side of it), and
    comfortably longer than one day (so a bug that only breaks on a MULTI-day gap can't
    hide) -- and then checks every window the loader actually hands out against that
    ground truth: for every day a served window covers, every company must show up as
    having genuinely traded that day.

    ⚠️ An earlier version of this test only asserted `days.diff(dim=1) == 1` -- but
    `days` is built by `Sentences.__getitem__` with `np.arange(...)`, which produces
    consecutive integers BY CONSTRUCTION, every time, no matter what the loader actually
    served. That assertion could never fail; it was testing `np.arange`, not `Sentences`.
    This test was proven to have teeth: with the run-detection in `Sentences.__init__`
    deliberately broken (the "unbroken run" counter made to never reset), this test
    failed with an `AssertionError` pointing at the exact straddled day, while the old
    `days.diff()` assertion kept passing throughout.
    """
    ok = tiny_batches.arrays.ok                              # [T, N] -- ground truth

    # Comfortably inside the stretch of the learn period where, absent the hole punched
    # below, every company trades every day -- so the hole is the ONLY irregularity a
    # served window could possibly be tripped up by.
    gap_lo, gap_hi = 340, 350                                 # a real 10-day hole
    company = 0
    assert ok[gap_lo - 1: gap_hi + 1, :].all(), (
        "test setup assumption broken: expected clean trading data around the gap"
    )
    ok[gap_lo:gap_hi, company] = False                        # this company stopped trading

    loaders = make_sentences(tiny_batches, tiny_settings)

    windows_checked = 0
    for item in loaders["learn"]:
        days = item["days"].numpy()                           # [B, T] -- day index per step
        for row in days:
            for day in row:
                missing = np.where(~ok[day])[0]
                assert missing.size == 0, (
                    f"a sentence was served covering day {day}, but compan"
                    f"{'y' if missing.size == 1 else 'ies'} at index {list(missing)} did "
                    "not trade that day -- this window straddles a gap in the data"
                )
            windows_checked += 1
    assert windows_checked > 0, (
        "the punched gap swallowed every window in the learn period -- widen the "
        "clean stretch either side of the gap so this test can actually prove something"
    )

    # Still a true, worth-stating property -- `days` really is consecutive -- but this is
    # no longer the ONLY thing checked, and on its own it would never have caught the bug
    # this test exists for (see the docstring above).
    for item in loaders["learn"]:
        assert (item["days"].diff(dim=1) == 1).all(), "a sentence jumped over a missing day"


def test_the_test_period_is_built_but_never_iterated_by_training(tiny_batches, tiny_settings):
    """Unlike the tuning search (which must not even BUILD it), the world model is finally
    scored on `test` at the very end -- so it exists here. It is simply never fed to
    `train_joint`."""
    loaders = make_sentences(tiny_batches, tiny_settings)
    assert set(loaders) == {"learn", "tune", "test"}
