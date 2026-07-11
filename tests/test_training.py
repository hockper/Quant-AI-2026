import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import bubble_bi as bb
from bubble_bi.models import VQVAE
from bubble_bi.training import baseline_rebuild, evaluate, train, word_usage


class _Grids(Dataset):
    """Grids with real structure in them: a few repeating market 'moods'."""

    def __init__(self, n=512, companies=1, days=4, features=6, moods=8, seed=0):
        rng = np.random.default_rng(seed)
        shapes = rng.normal(size=(moods, companies, days, features)).astype(np.float32)
        which = rng.integers(0, moods, n)
        self.x = shapes[which] + 0.1 * rng.normal(
            size=(n, companies, days, features)
        ).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"grid": torch.from_numpy(self.x[i])}


def _loaders(**kw):
    data = _Grids(**kw)
    return {p: DataLoader(data, batch_size=32, shuffle=(p == "learn"))
            for p in ("learn", "tune", "test")}


def _settings():
    return bb.check({"tickers": ["AAA"], "learning_rate": 3e-3})


def test_training_actually_reduces_the_rebuild_error():
    torch.manual_seed(0)
    model = VQVAE(companies=1, days=4, features=6, vocabulary=32, width=32,
                  heads=2, dropout=0.0)
    loaders = _loaders()

    before = evaluate(model, loaders["tune"], torch.device("cpu"))["rebuild"]
    train(model, loaders, _settings(), steps=150, quiet=True)
    after = evaluate(model, loaders["tune"], torch.device("cpu"))["rebuild"]

    assert after < before * 0.6, f"barely learned: {before:.3f} -> {after:.3f}"


def test_the_dictionary_spreads_out_instead_of_collapsing():
    # The failure mode that matters. A VQ-VAE will happily describe everything with
    # one word and report a plausible-looking loss while doing it.
    torch.manual_seed(0)
    model = VQVAE(companies=1, days=4, features=6, vocabulary=32, width=32,
                  heads=2, dropout=0.0)
    loaders = _loaders(moods=8)

    history = train(model, loaders, _settings(), steps=200, quiet=True)
    end = history.rows[-1]

    assert end["perplexity"] > 3, "the dictionary collapsed"
    assert end["words_used"] > 3


def test_a_model_that_beats_guessing_is_actually_saying_something():
    torch.manual_seed(0)
    model = VQVAE(companies=1, days=4, features=6, vocabulary=32, width=32,
                  heads=2, dropout=0.0)
    loaders = _loaders()
    train(model, loaders, _settings(), steps=200, quiet=True)

    scored = evaluate(model, loaders["test"], torch.device("cpu"))
    guessing = baseline_rebuild(loaders["test"])
    assert scored["rebuild"] < guessing * 0.8      # the token carries real information


def test_dead_words_get_revived_during_training():
    torch.manual_seed(0)
    model = VQVAE(companies=1, days=4, features=6, vocabulary=64, width=32,
                  heads=2, dropout=0.0)
    history = train(model, _loaders(moods=4), _settings(), steps=100,
                    revive_every=20, quiet=True)
    assert history.rows[-1]["revived"] > 0        # the collapse WAS fought


def test_history_records_what_actually_happened():
    torch.manual_seed(0)
    model = VQVAE(companies=1, days=4, features=6, vocabulary=16, width=32, heads=2)
    history = train(model, _loaders(), _settings(), steps=60, check_every=20, quiet=True)

    frame = history.frame()
    assert list(frame.index) == [20, 40, 60]
    assert {"rebuild", "perplexity", "words_used", "revived"} <= set(frame.columns)
    assert history.seconds > 0


def test_evaluation_does_not_train_the_model():
    # eval must not move the dictionary, or every "held out" score is a lie.
    torch.manual_seed(0)
    model = VQVAE(companies=1, days=4, features=6, vocabulary=16, width=32, heads=2)
    loaders = _loaders()
    before = model.codebook.dictionary.clone()
    evaluate(model, loaders["test"], torch.device("cpu"))
    assert torch.equal(model.codebook.dictionary, before)


def test_the_model_is_left_in_training_mode_afterwards():
    torch.manual_seed(0)
    model = VQVAE(companies=1, days=4, features=6, vocabulary=16, width=32, heads=2)
    loaders = _loaders()
    evaluate(model, loaders["test"], torch.device("cpu"))
    assert model.training                          # or the next training step is a no-op


def test_word_usage_counts_every_grid():
    torch.manual_seed(0)
    model = VQVAE(companies=1, days=4, features=6, vocabulary=16, width=32, heads=2)
    loaders = _loaders(n=128)
    counts = word_usage(model, loaders["test"])
    assert counts.shape == (16,)
    assert counts.sum() == 128


def test_guessing_the_average_scores_about_one_on_normalised_data():
    # This is what makes `rebuild` mean something on its own: the baseline is 1.0.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, 1, 4, 6)).astype(np.float32)

    class _D(Dataset):
        def __len__(self): return len(x)
        def __getitem__(self, i): return {"grid": torch.from_numpy(x[i])}

    assert baseline_rebuild(DataLoader(_D(), batch_size=32)) == pytest.approx(1.0, abs=0.1)
