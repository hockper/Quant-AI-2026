import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

import bubble_bi as bb
from bubble_bi.autopsy import _probe, report, verdict


# ---------------------------------------------------------------- the probe

def test_the_probe_finds_information_that_is_there():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 8))
    y = (x @ rng.normal(size=(8, 4)))          # y is a linear function of x
    assert _probe(x, y) > 0.95


def test_the_probe_finds_nothing_when_there_is_nothing():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 8))
    y = rng.normal(size=(600, 4))              # unrelated
    assert _probe(x, y) < 0.1


def test_the_probe_is_scored_on_held_out_rows_not_the_ones_it_fitted():
    """A wide vector could otherwise memorise every row and score a perfect 1.0 while
    knowing nothing — which would make the whole autopsy a lie."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 150))            # nearly as many columns as rows
    y = rng.normal(size=(200, 4))              # pure noise
    assert _probe(x, y) < 0.3                  # must NOT report a good fit


def test_the_probe_reports_a_partial_signal_as_partial():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(800, 6))
    y = x[:, :2] @ rng.normal(size=(2, 3)) + rng.normal(size=(800, 3)) * 1.5
    assert 0.15 < _probe(x, y) < 0.8


# ------------------------------------------------------- naming the culprit

def _evidence(fused, onehot, tokens, keys=5, words=64):
    n = len(tokens)
    return {
        "fused": fused,
        "onehot": onehot,
        "tokens": tokens,
        "candle": np.zeros((n, 4)),
        "target": np.zeros((n, 1)),
        "attention": np.full((n, keys), 1 / keys),
        "words": words,
    }


def test_it_blames_the_fusion_when_even_the_continuous_vector_is_empty():
    rng = np.random.default_rng(0)
    n = 500
    e = _evidence(rng.normal(size=(n, 16)), np.zeros((n, 64)), rng.integers(0, 3, n))
    e["candle"] = rng.normal(size=(n, 4))        # unrelated to anything
    assert "CULPRIT: THE FUSION" in verdict(e)


def test_it_blames_the_codebook_when_quantising_destroys_the_information():
    rng = np.random.default_rng(0)
    n = 800
    candle = rng.normal(size=(n, 4))

    fused = np.c_[candle, rng.normal(size=(n, 12))]   # the vector KNOWS the candle
    tokens = rng.integers(0, 3, n)                    # ...but only 3 words survive
    onehot = np.zeros((n, 64), dtype=np.float32)
    onehot[np.arange(n), tokens] = 1.0

    e = _evidence(fused, onehot, tokens)
    e["candle"] = candle
    assert "CULPRIT: THE CODEBOOK" in verdict(e)


def test_it_blames_the_loss_when_both_carry_the_information():
    rng = np.random.default_rng(0)
    n = 800
    # 8 groups, each with its own candle -> BOTH the vector and the token know it
    groups = rng.integers(0, 8, n)
    shapes = rng.normal(size=(8, 4)) * 3
    candle = shapes[groups] + rng.normal(size=(n, 4)) * 0.05

    fused = np.c_[candle, rng.normal(size=(n, 12)) * 0.01]
    onehot = np.zeros((n, 64), dtype=np.float32)
    onehot[np.arange(n), groups] = 1.0

    e = _evidence(fused, onehot, groups)
    e["candle"] = candle
    assert "CULPRIT: THE LOSS" in verdict(e)


def test_it_says_when_the_cross_attention_is_not_choosing():
    rng = np.random.default_rng(0)
    n = 400
    e = _evidence(rng.normal(size=(n, 16)), np.zeros((n, 64)), rng.integers(0, 3, n))
    e["attention"] = np.full((n, 5), 0.2)        # perfectly flat = no choice at all
    assert "FLAT — it is not choosing" in verdict(e)


def test_the_table_names_all_four_things_we_want_to_know():
    rng = np.random.default_rng(0)
    n = 400
    e = _evidence(rng.normal(size=(n, 16)),
                  np.eye(64, dtype=np.float32)[rng.integers(0, 64, n)],
                  rng.integers(0, 64, n))
    e["candle"] = rng.normal(size=(n, 4))
    e["target"] = rng.normal(size=(n, 1))

    table = report(e)
    assert len(table) == 4
    assert any("CONTINUOUS" in i for i in table.index)
    assert any("QUANTISED" in i for i in table.index)
    assert any("TOMORROW" in i for i in table.index)


# ------------------------------------------------------------------ keeping

def test_a_saved_model_comes_back(tmp_path):
    # Two tickers, not one: with a single company the cross-attention has one key,
    # and softmax over one key is a no-op. `check()` now refuses it. (The model
    # below is still built with companies=1 -- it's testing save/load, not fusion.)
    settings = bb.check({"tickers": ["AAA", "BBB"], "data_dir": str(tmp_path)})
    model = bb.models.VQVAE(companies=1, days=4, features=6, vocabulary=16, width=32)

    assert bb.keep.load(model, "ts", settings) is None      # nothing saved yet
    bb.keep.save(model, "ts", settings, steps=100)
    assert bb.keep.trained("ts", settings)

    fresh = bb.models.VQVAE(companies=1, days=4, features=6, vocabulary=16, width=32)
    assert bb.keep.load(fresh, "ts", settings, quiet=True) is not None
    assert torch.allclose(fresh.read.weight, model.read.weight)


def test_loading_weights_that_do_not_fit_is_refused_not_fudged(tmp_path):
    """Silently loading a mismatched model would give you something subtly, invisibly
    wrong -- far worse than an error."""
    # Two tickers, not one: with a single company the cross-attention has one key,
    # and softmax over one key is a no-op. `check()` now refuses it.
    settings = bb.check({"tickers": ["AAA", "BBB"], "data_dir": str(tmp_path)})
    bb.keep.save(bb.models.VQVAE(companies=1, days=4, features=6, vocabulary=16, width=32),
                 "ts", settings)

    different = bb.models.VQVAE(companies=1, days=4, features=6, vocabulary=64, width=32)
    with pytest.raises(RuntimeError, match="does not fit"):
        bb.keep.load(different, "ts", settings, quiet=True)


def test_forgetting_a_model_makes_it_train_again(tmp_path):
    # Two tickers, not one: with a single company the cross-attention has one key,
    # and softmax over one key is a no-op. `check()` now refuses it.
    settings = bb.check({"tickers": ["AAA", "BBB"], "data_dir": str(tmp_path)})
    model = bb.models.VQVAE(companies=1, days=4, features=6, vocabulary=16, width=32)
    bb.keep.save(model, "ts", settings)
    bb.keep.forget("ts", settings)
    assert not bb.keep.trained("ts", settings)
    assert bb.keep.load(model, "ts", settings) is None
