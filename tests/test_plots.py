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
                         "ts": {"batch": 32, "heads": 2}, "cs": {"batch": 32, "heads": 2}})
    data = bb.data.add_features(prices, settings)
    batches = bb.data.make_tensors(data, settings)

    torch.manual_seed(0)
    model = VQVAE(companies=1, features=len(bb.data.names()), width=32,
                  **settings["ts"])
    history = train(model, batches.ts, settings, steps=40, quiet=True)
    return model, batches, prices, settings, history


@pytest.fixture
def joint():
    """A tiny but complete JOINT world: prices, features, grids, sentences, and a
    briefly-trained `WorldModel` -- everything `remembered`/`kept_and_lost`/
    `kept_by_family`/`predicted_candles` now need to route through the REAL chain
    (encode -> cross-attend against that day's market -> quantise), not a bare `VQVAE`
    with no fusion at all.
    """
    rng = np.random.default_rng(0)
    days = pd.date_range("2015-01-01", periods=800, freq="B", name="date")
    frames = []
    for ticker, level in (("AAA", 20.0), ("BBB", 300.0), ("CCC", 60.0)):
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

    settings = bb.check({
        "tickers": ["AAA", "BBB", "CCC"], "learning_rate": 3e-3, "model_size": 32,
        "ts": {"batch": 32, "heads": 2, "vocabulary": 16},
        "cs": {"batch": 16, "heads": 2, "vocabulary": 16},
        "fusion": {"attend_to": "companies", "batch": 8},
        "predictor": {"sentence_length": 6, "depth": 1},
    })
    data = bb.data.add_features(prices, settings)
    batches = bb.data.make_tensors(data, settings)
    book = bb.data.make_sentences(batches, settings)

    torch.manual_seed(0)
    n_features = len(bb.data.names())
    ts = VQVAE(companies=1, features=n_features, width=32, **settings["ts"])
    cs = VQVAE(companies=3, features=n_features, width=32, **settings["cs"])
    tokenizer = bb.models.Tokenizer(ts, cs, model_size=32, **settings["fusion"])
    model = bb.models.WorldModel(tokenizer, sentence=6, depth=1, heads=2, **settings["loss"])
    history = bb.train_joint(model, book, settings, steps=20, quiet=True)
    return tokenizer, model, book, batches, prices, settings, history


def test_the_candle_plot_draws_the_real_days_and_the_remembered_ones(joint):
    tokenizer, _, _, batches, prices, settings, _ = joint
    fig = bb.plots.remembered(tokenizer, batches, prices, settings)
    left, right = fig.axes
    assert "actually happened" in left.get_title(loc="left")
    assert "remembered" in right.get_title(loc="left")
    # one wick line + one body rectangle per day, on each side
    assert len(left.lines) == tokenizer.ts.days
    assert len(right.lines) == tokenizer.ts.days


def test_the_remembered_candles_are_a_real_inversion_not_a_redraw(joint):
    """The plot must show what the MODEL said, not quietly redraw the truth.

    If the two panels were identical, the demo would be a lie. They must differ --
    the whole point is that a single word cannot carry everything.
    """
    tokenizer, _, _, batches, prices, settings, _ = joint
    fig = bb.plots.remembered(tokenizer, batches, prices, settings)
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


def test_the_kept_and_lost_chart_names_every_feature(joint):
    tokenizer, _, _, batches, _, settings, _ = joint
    fig = bb.plots.kept_and_lost(tokenizer, batches, settings, examples=32)
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


def test_the_family_breakdown_shows_what_the_headline_average_is_hiding(joint):
    """'explains 44%' is an average over 26 features, carried by the easy ones. Broken
    apart it says something quite different — and that is the number worth reporting."""
    tokenizer, _, _, batches, _, settings, _ = joint
    fig, frame = bb.plots.kept_by_family(tokenizer, batches, settings, examples=48)

    assert sorted(frame.index) == sorted(bb.data.FAMILIES)
    assert "average" in fig.axes[0].get_title(loc="left")
    # every family gets a number, and they are not all the same
    assert frame.notna().all()


def test_the_predicted_candles_chart_draws_real_and_predicted_side_by_side(joint):
    """The most directly readable output the project produces -- restored after Task 7
    deleted it because `WorldModel.forward` did not yet return `drawn`."""
    _, model, book, batches, prices, settings, _ = joint
    fig = bb.plots.predicted_candles(model, book, batches, prices, settings, show=4)
    left, right = fig.axes
    assert "actually happened" in left.get_title(loc="left")
    assert "predicted" in right.get_title(loc="left")
    assert len(left.lines) == len(right.lines) > 0


def test_the_predicted_candles_are_a_real_forecast_not_a_redraw(joint):
    _, model, book, batches, prices, settings, _ = joint
    fig = bb.plots.predicted_candles(model, book, batches, prices, settings, show=4)
    left, right = fig.axes
    real = np.array([line.get_ydata() for line in left.lines])
    said = np.array([line.get_ydata() for line in right.lines])
    assert real.shape == said.shape
    assert not np.allclose(real, said)


def test_joint_progress_shows_perplexity_and_the_forecast_vs_its_floor(joint):
    *_, history = joint
    fig = bb.plots.joint_progress(history, "Joint")
    ppl, draw = fig.axes
    assert "perplexity" in ppl.get_title(loc="left")
    assert "floor" in draw.get_title(loc="left")
    labels = [line.get_label() for line in ppl.lines]
    assert any("TS" == l for l in labels)
    assert any("CS" == l for l in labels)


def test_joint_progress_survives_a_model_that_was_loaded_rather_than_trained():
    assert bb.plots.joint_progress(None, "Joint") is None


def test_tuning_importance_ranks_the_knob_that_actually_moved_the_score():
    """Even twelve near-random trials answer 'is the learning rate dominating everything?'
    -- which is worth knowing, and is the honest thing a 12-trial screen can tell you."""
    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    from bubble_bi import plots

    rng = np.random.default_rng(0)
    lr = rng.random(20)
    trials = pd.DataFrame({
        "stage": "balance",
        "learning_rate": lr,
        "commitment": rng.random(20),
        "score": 5 * lr + 0.01 * rng.random(20),     # score follows lr, and nothing else
    })
    ranked = plots.tuning_importance(trials)
    assert ranked.index[0] == "learning_rate"


def test_tuning_importance_only_ever_ranks_ACTUAL_KNOBS():
    """⚠️ THE BUG THIS EXISTS FOR.

    `tuning_importance` used a BLOCKLIST -- "any column not in this hardcoded set is a
    knob". So the moment `direction_se` and `volatility_se` were added to the trials table
    (they are OUTPUTS -- the error bars on the score), the chart cheerfully ranked them as
    the second and third most important KNOBS in the search, above `diversity` and
    `learning_rate`. It was correlating an output with the score and calling it a cause.

    The knobs are exactly the keys of `tuning.SPACE`. That is the only source of truth,
    and an allowlist cannot drift the way a blocklist did.
    """
    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    from bubble_bi import plots, tuning

    rng = np.random.default_rng(0)
    lr = rng.random(20)
    trials = pd.DataFrame({
        "entry": "ts", "stage": "balance",
        "learning_rate": lr,
        "commitment": rng.random(20),
        "score": 5 * lr + 0.01 * rng.random(20),
        # the outputs that used to masquerade as knobs -- deliberately made to correlate
        # PERFECTLY with the score, so a blocklist implementation must rank them first
        "direction": 5 * lr,
        "direction_se": 5 * lr,
        "volatility_se": 5 * lr,
        "words_used": (500 * lr).astype(int),
    })
    ranked = plots.tuning_importance(trials)

    knobs = {k for stage in tuning.SPACE.values() for k in stage}
    strays = set(ranked.index) - knobs
    assert not strays, f"ranked things that are NOT knobs: {sorted(strays)}"
    assert ranked.index[0] == "learning_rate"
