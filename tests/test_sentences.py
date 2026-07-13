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


def test_a_sentence_never_straddles_a_gap_in_the_data(tiny_batches, tiny_settings):
    """A 'sentence' of days that are not consecutive is not a sentence. If a company stopped
    trading in the middle, the window is not usable and must not be served."""
    loaders = make_sentences(tiny_batches, tiny_settings)
    item = next(iter(loaders["learn"]))
    days = item["days"]                                # [B, T] -- the day index of each step
    assert (days.diff(dim=1) == 1).all(), "a sentence jumped over a missing day"


def test_the_test_period_is_built_but_never_iterated_by_training(tiny_batches, tiny_settings):
    """Unlike the tuning search (which must not even BUILD it), the world model is finally
    scored on `test` at the very end -- so it exists here. It is simply never fed to
    `train_joint`."""
    loaders = make_sentences(tiny_batches, tiny_settings)
    assert set(loaders) == {"learn", "tune", "test"}
