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

from bubble_bi.settings import DEFAULTS


@pytest.fixture
def tiny_settings(tmp_path):
    return {
        **DEFAULTS,
        "tickers": ["AAA", "BBB", "CCC"],
        # The Optuna study lands under data_dir -- keep it out of the real artifacts
        # folder, or the resume test would find a stale study from a previous run.
        "data_dir": str(tmp_path),
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
