import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch

matplotlib.use("Agg")

import bubble_bi as bb
from bubble_bi.diagnostics import _explained, market_moods, moods_plot


# ------------------------------------------------------ the metric itself

def test_a_perfect_grouping_explains_everything():
    values = np.array([1.0, 1.0, 5.0, 5.0, 9.0, 9.0])
    groups = np.array([0, 0, 1, 1, 2, 2])              # each group is one value
    assert _explained(values, groups) == pytest.approx(1.0)


def test_a_meaningless_grouping_explains_nothing():
    rng = np.random.default_rng(0)
    values = rng.normal(size=600)
    groups = rng.integers(0, 3, 600)                   # groups unrelated to the values
    assert abs(_explained(values, groups)) < 0.05


def test_the_metric_notices_a_partial_pattern():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 2, 400)
    values = groups * 2.0 + rng.normal(0, 1.0, 400)    # group shifts the mean, noisily
    score = _explained(values, groups)
    assert 0.3 < score < 0.7                            # real, but far from perfect


def test_a_group_per_day_would_explain_everything_by_luck():
    """Why the shuffled floor exists. Give every day its own word and the score is a
    perfect 1.0 while knowing nothing — the metric MUST be read against a shuffle."""
    values = np.random.default_rng(0).normal(size=50)
    every_day_its_own = np.arange(50)
    assert _explained(values, every_day_its_own) == pytest.approx(1.0)


# ------------------------------------------------------ end to end on real shapes

@pytest.fixture
def trained_cs():
    rng = np.random.default_rng(0)
    days = pd.date_range("2015-01-01", periods=900, freq="B", name="date")
    frames = []
    for i, ticker in enumerate(("AAA", "BBB", "CCC")):
        # a market with REGIMES: long calm stretches, then a turbulent one
        loud = (np.arange(len(days)) // 150) % 2 == 1
        vol = np.where(loud, 0.04, 0.008)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 1, len(days)) * vol))
        open_ = close * (1 + rng.normal(0, 0.004, len(days)))
        top, bottom = np.maximum(open_, close), np.minimum(open_, close)
        one = pd.DataFrame(
            {"open": open_, "high": top * (1 + rng.uniform(0, 0.01, len(days))),
             "low": bottom * (1 - rng.uniform(0, 0.01, len(days))), "close": close,
             "volume": rng.uniform(1e6, 5e6, len(days))},
            index=days,
        )
        one["target"] = one["close"].shift(-1) / one["close"] - 1.0
        one["ticker"] = ticker
        frames.append(one.reset_index())
    prices = pd.concat(frames).set_index(["date", "ticker"]).sort_index()

    settings = bb.check({"tickers": ["AAA", "BBB", "CCC"], "learning_rate": 3e-3,
                         "cs": {"batch": 32}, "ts": {"batch": 32}})
    batches = bb.data.make_tensors(bb.data.add_features(prices, settings), settings)

    torch.manual_seed(0)
    cs = bb.models.VQVAE(companies=3, features=len(bb.data.names()), width=32, heads=2,
                         **settings["cs"])
    bb.train(cs, batches.cs, settings, steps=120, quiet=True)
    return cs, batches, settings


def test_the_evidence_comes_back_with_a_score_and_a_luck_floor(trained_cs):
    cs, batches, settings = trained_cs
    evidence = market_moods(cs, batches, settings)

    assert set(evidence["scores"].columns) == {"explained by the token", "explained by luck"}
    assert len(evidence["tokens"]) == len(evidence["days"])
    assert evidence["words_used"] >= 1


def test_the_luck_floor_is_actually_low(trained_cs):
    # If shuffling the words scored well, the metric would be worthless.
    cs, batches, settings = trained_cs
    evidence = market_moods(cs, batches, settings)
    assert (evidence["scores"]["explained by luck"] < 0.35).all()


def test_the_words_find_a_volatility_regime_that_really_is_there(trained_cs):
    # The fixture market has deliberate calm and turbulent stretches. If CS cannot find
    # a regime that obvious, it has not learned anything at all.
    cs, batches, settings = trained_cs
    evidence = market_moods(cs, batches, settings)
    real = evidence["scores"].loc["how violently", "explained by the token"]
    luck = evidence["scores"].loc["how violently", "explained by luck"]
    assert real > luck + 0.05


def test_the_plot_prints_the_honest_score_on_the_direction_panel(trained_cs):
    """The direction bars look like a pattern and are noise. The number must be ON the
    chart, or a reader draws exactly the wrong conclusion from it."""
    cs, batches, settings = trained_cs
    fig = moods_plot(market_moods(cs, batches, settings))
    _, right = fig.axes
    printed = " ".join(t.get_text() for t in right.texts)
    assert "luck would give" in printed
    assert "NOISE" in printed
