import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch

matplotlib.use("Agg")           # no window, no display needed

import bubble_bi as bb
from bubble_bi.data.features.candle import rebuild_candles
from bubble_bi.models import VQVAE
from bubble_bi.training import train


@pytest.fixture
def world():
    """A tiny but complete world: prices, features, grids, a briefly-trained model."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2015-01-01", periods=800, freq="B", name="date")
    frames = []
    for ticker, level in (("AAA", 20.0), ("BBB", 300.0)):
        close = level * np.exp(np.cumsum(rng.normal(0, 0.02, len(days))))
        open_ = close * (1 + rng.normal(0, 0.008, len(days)))
        top, bottom = np.maximum(open_, close), np.minimum(open_, close)
        one = pd.DataFrame(
            {"open": open_, "high": top * (1 + rng.uniform(0, 0.02, len(days))),
             "low": bottom * (1 - rng.uniform(0, 0.02, len(days))), "close": close,
             "volume": rng.uniform(1e6, 5e6, len(days))},
            index=days,
        )
        one["target"] = one["close"].shift(-1) / one["close"] - 1.0
        one["ticker"] = ticker
        frames.append(one.reset_index())
    prices = pd.concat(frames).set_index(["date", "ticker"]).sort_index()

    settings = bb.check({"tickers": ["AAA", "BBB"], "learning_rate": 3e-3,
                         "ts": {"batch": 32}, "cs": {"batch": 32}})
    data = bb.data.add_features(prices, settings)
    batches = bb.data.make_tensors(data, settings)

    torch.manual_seed(0)
    model = VQVAE(companies=1, features=len(bb.data.names()), width=32, heads=2,
                  **settings["ts"])
    history = train(model, batches.ts, settings, steps=40, quiet=True)
    return model, batches, prices, settings, history


def test_the_candle_plot_draws_the_real_days_and_the_remembered_ones(world):
    model, batches, prices, settings, _ = world
    fig = bb.plots.remembered(model, batches, prices, settings)
    left, right = fig.axes
    assert "actually happened" in left.get_title(loc="left")
    assert "remembered" in right.get_title(loc="left")
    # one wick line + one body rectangle per day, on each side
    assert len(left.lines) == model.days
    assert len(right.lines) == model.days


def test_the_remembered_candles_are_a_real_inversion_not_a_redraw(world):
    """The plot must show what the MODEL said, not quietly redraw the truth.

    If the two panels were identical, the demo would be a lie. They must differ --
    the whole point is that a single word cannot carry everything.
    """
    model, batches, prices, settings, _ = world
    fig = bb.plots.remembered(model, batches, prices, settings)
    left, right = fig.axes

    real = np.array([line.get_ydata() for line in left.lines])
    said = np.array([line.get_ydata() for line in right.lines])
    assert real.shape == said.shape
    assert not np.allclose(real, said)


def test_the_candle_inversion_is_exact_when_given_the_true_shape():
    # The guarantee the plot rests on: shape -> candle loses nothing.
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    shape = pd.DataFrame(
        {"gap": [0.01, -0.005, 0.0, 0.02, -0.01],
         "body": [0.02, 0.01, -0.03, 0.005, 0.0],
         "upper_wick": [0.004, 0.0, 0.012, 0.001, 0.006],
         "lower_wick": [0.002, 0.007, 0.0, 0.003, 0.001]},
        index=dates,
    )
    back = rebuild_candles(shape, first_close=100.0)
    assert (back["high"] >= back[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (back["low"] <= back[["open", "close"]].min(axis=1) + 1e-9).all()
    assert np.isclose(back["open"].iloc[0], 100.0 * np.exp(0.01))


def test_the_kept_and_lost_chart_names_every_feature(world):
    model, batches, _, settings, _ = world
    fig = bb.plots.kept_and_lost(model, batches, settings, examples=32)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert sorted(labels) == sorted(bb.data.names())


def test_the_progress_chart_shows_the_three_numbers_that_matter(world):
    *_, history = world
    fig = bb.plots.progress(history, "TS")
    loss, ppl, used = fig.axes
    assert "rebuild" in loss.get_title(loc="left")
    assert "perplexity" in ppl.get_title(loc="left")
    assert "vocabulary" in used.get_title(loc="left")


def test_the_loss_chart_shows_learning_AND_held_out_so_overfitting_is_visible(world):
    """A loss curve with only one line cannot tell you the model is memorising."""
    *_, history = world
    labels = [line.get_label() for line in bb.plots.progress(history, "TS").axes[0].lines]
    assert any("learns from" in l for l in labels)
    assert any("never seen" in l for l in labels)
    assert any("guessing" in l for l in labels)


def test_progress_survives_a_model_that_was_loaded_rather_than_trained():
    assert bb.plots.progress(None, "TS") is None


def test_the_family_breakdown_shows_what_the_headline_average_is_hiding(world):
    """'explains 44%' is an average over 26 features, carried by the easy ones. Broken
    apart it says something quite different — and that is the number worth reporting."""
    model, batches, _, settings, _ = world
    fig, frame = bb.plots.kept_by_family(model, batches, settings, examples=48)

    assert sorted(frame.index) == sorted(bb.data.FAMILIES)
    assert "average" in fig.axes[0].get_title(loc="left")
    # every family gets a number, and they are not all the same
    assert frame.notna().all()
