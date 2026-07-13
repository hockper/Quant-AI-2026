import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import bubble_bi as bb
from bubble_bi.models import VQVAE
from bubble_bi.training import (baseline_rebuild, evaluate, pick_device, train,
                                word_usage)
from conftest import _tiny_joint


class _Grids(Dataset):
    """Grids with real structure in them: a few repeating market 'moods'.

    Each mood carries its own LEVEL as well as its own shape — as real windows do. Without
    that, "predict the window's average" would have nothing to give away and could not be
    the harder baseline it is meant to be.
    """

    def __init__(self, n=512, companies=1, days=4, features=6, moods=8, seed=0):
        rng = np.random.default_rng(seed)
        shapes = rng.normal(size=(moods, companies, days, features)).astype(np.float32)
        shapes += rng.normal(size=(moods, companies, 1, features)).astype(np.float32)
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
    # Two tickers, not one: with a single company the cross-attention has one key,
    # and softmax over one key is a no-op. `check()` now refuses it. This helper
    # only feeds `train()` a learning rate/step budget -- the models built in these
    # tests set their own `companies=` directly, so this does not change any shape.
    return bb.check({"tickers": ["AAA", "BBB"], "learning_rate": 3e-3})


def _where():
    """The device the TRAINER will pick.

    Never hard-code CPU here. `train()` moves the model to whatever hardware it finds,
    so a test that then evaluates on CPU passes forever on a laptop and fails the first
    time it meets a GPU — which is exactly what happened on Colab.
    """
    return pick_device(_settings())


def test_training_actually_reduces_the_rebuild_error():
    torch.manual_seed(0)
    model = VQVAE(companies=1, days=4, features=6, vocabulary=32, width=32,
                  heads=2, dropout=0.0)
    loaders = _loaders()

    before = evaluate(model, loaders["tune"], _where())["rebuild"]
    train(model, loaders, _settings(), steps=150, quiet=True)
    after = evaluate(model, loaders["tune"], _where())["rebuild"]

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

    scored = evaluate(model, loaders["test"], _where())
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
    model = VQVAE(companies=1, days=4, features=6, vocabulary=16, width=32,
                  heads=2).to(_where())
    loaders = _loaders()
    before = model.codebook.dictionary.clone()          # cloned ON the device
    evaluate(model, loaders["test"], _where())
    assert torch.equal(model.codebook.dictionary, before)


def test_evaluate_takes_the_model_to_the_device_it_was_given():
    """The bug that only ever showed up on Colab.

    `train()` moves the model to the GPU. `evaluate()` used to assume the model was
    already wherever you asked it to work — so evaluating an UNTRAINED model on a GPU
    machine left the weights on the CPU and the data on CUDA. On a laptop the two are
    always the same place and it passed forever.
    """
    model = VQVAE(companies=1, days=4, features=6, vocabulary=16, width=32, heads=2)
    assert next(model.parameters()).device.type == "cpu"      # fresh, still on the CPU

    evaluate(model, _loaders()["test"], _where())
    assert next(model.parameters()).device.type == _where().type


def test_the_model_is_left_in_training_mode_afterwards():
    torch.manual_seed(0)
    model = VQVAE(companies=1, days=4, features=6, vocabulary=16, width=32, heads=2)
    loaders = _loaders()
    evaluate(model, loaders["test"], _where())
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


class _MarketGrids(Dataset):
    """CS-shaped grids: every company at once, one sample per day."""

    def __init__(self, n=256, companies=6, days=5, features=6, moods=6, seed=0):
        rng = np.random.default_rng(seed)
        shapes = rng.normal(size=(moods, companies, days, features)).astype(np.float32)
        self.x = shapes[rng.integers(0, moods, n)] + 0.1 * rng.normal(
            size=(n, companies, days, features)
        ).astype(np.float32)
        self.present = np.ones((n, companies), dtype=bool)
        self.present[:, -1] = False              # one company never trades

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"grid": torch.from_numpy(self.x[i]),
                "present": torch.from_numpy(self.present[i])}


def test_the_same_class_trains_as_cs_on_the_whole_market():
    torch.manual_seed(0)
    model = VQVAE(companies=6, days=5, features=6, vocabulary=32, width=32,
                  heads=2, dropout=0.0)
    data = _MarketGrids()
    loaders = {p: DataLoader(data, batch_size=32, shuffle=(p == "learn"))
               for p in ("learn", "tune", "test")}

    before = evaluate(model, loaders["tune"], _where())["rebuild"]
    history = train(model, loaders, _settings(), steps=200, quiet=True)
    after = history.rows[-1]

    assert after["rebuild"] < before * 0.7          # it learned the market's moods
    assert after["perplexity"] > 3                  # ...without collapsing


def test_a_company_that_did_not_trade_cannot_influence_cs_training():
    # If an absent company leaked into the loss, the model would be scored on
    # rebuilding something that never happened.
    torch.manual_seed(0)
    model = VQVAE(companies=6, days=5, features=6, vocabulary=16, width=32,
                  heads=2, dropout=0.0).eval()
    data = _MarketGrids(n=32)
    batch = next(iter(DataLoader(data, batch_size=8)))

    with torch.no_grad():
        clean = model(batch)["rebuild_loss"]
        poisoned = {k: v.clone() for k, v in batch.items()}
        poisoned["grid"][:, -1] = 999.0             # garbage in the absent company
        assert torch.allclose(clean, model(poisoned)["rebuild_loss"], atol=1e-5)


# ------------------------------------------------- the baseline that actually hurts

def test_the_window_mean_baseline_is_harder_than_predicting_zero():
    """The whole point.

    'Predict zero' is the long-run average of a normalised feature — a WEAK bar. A model
    that knew nothing except 'this window sits above its usual level' would already beat
    it. Predicting the window's OWN average hands that away for free and asks whether the
    model knows anything about the shape INSIDE the window.
    """
    torch.manual_seed(0)
    model = VQVAE(companies=1, days=6, features=6, vocabulary=16, width=32, heads=2)
    scored = evaluate(model, _loaders(days=6)["test"], _where())

    assert scored["window_mean"] < scored["guessing"], (
        "the window-mean bar must be the harder one — if it is not, the fixture has no "
        "level to give away and the test proves nothing"
    )


def test_repeating_the_last_day_is_a_FLATTERING_baseline_not_a_harsh_one():
    """It sounds like the honest floor — 'nothing changed since yesterday' — and it is
    not. One day is a noisy sample, so repeating it 15 times scores WORSE than predicting
    the long-run mean. Reporting against it makes the model look better than it is."""
    torch.manual_seed(0)
    model = VQVAE(companies=1, days=6, features=6, vocabulary=16, width=32, heads=2)
    scored = evaluate(model, _loaders(days=6)["test"], _where())

    assert scored["last_day"] > scored["window_mean"]      # much weaker than the real bar


def test_a_model_that_only_knew_the_window_average_would_beat_the_weak_bar():
    """Proof that the old headline was worthless: a 'model' which does nothing but
    predict the window's own mean already clears the long-run bar comfortably. Any
    '% explained' measured against that bar therefore says nothing."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 1, 6, 6)).astype(np.float32)
    x += rng.normal(size=(200, 1, 1, 6)).astype(np.float32) * 1.5    # give each window a LEVEL

    zero_cost = float((x ** 2).mean())
    mean_cost = float(((x - x.mean(axis=2, keepdims=True)) ** 2).mean())

    assert mean_cost < zero_cost
    assert 1 - mean_cost / zero_cost > 0.3      # knowing only the level "explains" >30%


# ------------------------------------------------------------- early stopping

def test_it_keeps_the_BEST_model_not_the_last_one():
    """The failure this exists for, seen on a real GPU run.

    CS has ~2,600 grids to TS's ~78,000, and they shared one `steps` setting -- so 10,000
    steps was 33 passes over the TS data and 243 over the CS data. Its held-out error
    bottomed out at step 1,000 and then climbed for the next nine thousand (0.90 -> 1.03,
    barely better than guessing) while its codebook decayed from 187 words to 141. Every
    one of those steps made the model worse, and we would have kept the wreckage.
    """
    torch.manual_seed(0)
    model = VQVAE(companies=1, days=4, features=6, vocabulary=16, width=32,
                  heads=2, dropout=0.0)
    loaders = _loaders(n=64)          # tiny -> it will overfit, and fast

    history = train(model, loaders, _settings(), steps=400, check_every=20,
                    patience=3, quiet=True)

    frame = history.frame()
    assert history.best_step > 0
    assert history.best_step in frame.index
    # the kept step must be the best one, not simply the final one
    assert frame.loc[history.best_step, "rebuild"] == frame["rebuild"].min()

    # and the model we were handed back must actually BE that one
    scored = evaluate(model, loaders["tune"], _where())
    assert scored["rebuild"] <= frame["rebuild"].min() + 1e-3


def test_it_gives_up_when_the_held_out_error_stops_improving():
    torch.manual_seed(0)
    model = VQVAE(companies=1, days=4, features=6, vocabulary=16, width=32,
                  heads=2, dropout=0.0)
    history = train(model, _loaders(n=64), _settings(), steps=2000, check_every=20,
                    patience=2, quiet=True)
    assert history.rows[-1]["step"] < 2000, "it should have stopped early"


def test_each_entry_can_have_its_own_step_budget():
    # CS needs far fewer steps than TS, because it has far less data.
    # Two tickers, not one: with a single company the cross-attention has one key,
    # and softmax over one key is a no-op. `check()` now refuses it.
    settings = bb.check({"tickers": ["AAA", "BBB"], "learning_rate": 3e-3,
                         "steps": 500, "cs": {"steps": 40}})
    torch.manual_seed(0)
    model = VQVAE(companies=1, days=4, features=6, vocabulary=16, width=32, heads=2)

    history = train(model, _loaders(), settings, entry="cs", check_every=10,
                    patience=99, quiet=True)
    assert history.rows[-1]["step"] == 40         # CS's own budget, not the shared 500


# ------------------------------------------------------------- early stopping

def test_it_keeps_the_BEST_model_not_the_last_one():
    """The failure this exists for, seen on a real GPU run.

    CS has ~2,600 grids to TS's ~78,000, and they shared one `steps` setting -- so 10,000
    steps was 33 passes over the TS data and 243 over the CS data. Its held-out error
    bottomed out at step 1,000 and then climbed for the next nine thousand (0.90 -> 1.03,
    barely better than guessing) while its codebook decayed from 187 words to 141. Every
    one of those steps made the model worse, and we would have kept the wreckage.
    """
    torch.manual_seed(0)
    model = VQVAE(companies=1, days=4, features=6, vocabulary=16, width=32,
                  heads=2, dropout=0.0)
    loaders = _loaders(n=64)          # tiny -> it overfits, and fast

    history = train(model, loaders, _settings(), steps=300, check_every=20,
                    patience=3, quiet=True)

    frame = history.frame()
    assert history.best_step in frame.index
    # the step we kept must be the BEST one, not simply the final one
    assert frame.loc[history.best_step, "rebuild"] == frame["rebuild"].min()

    # ...and the model handed back must actually BE that one
    scored = evaluate(model, loaders["tune"], _where())
    assert scored["rebuild"] <= frame["rebuild"].min() + 1e-3


def test_it_gives_up_when_the_held_out_error_stops_improving():
    torch.manual_seed(0)
    model = VQVAE(companies=1, days=4, features=6, vocabulary=16, width=32,
                  heads=2, dropout=0.0)
    history = train(model, _loaders(n=64), _settings(), steps=2000, check_every=20,
                    patience=2, quiet=True)
    assert history.rows[-1]["step"] < 2000, "it should have stopped early"


def test_each_entry_can_have_its_own_step_budget():
    # CS needs far fewer steps than TS, because it has far less data to learn from.
    # Two tickers, not one: with a single company the cross-attention has one key,
    # and softmax over one key is a no-op. `check()` now refuses it.
    settings = bb.check({"tickers": ["AAA", "BBB"], "learning_rate": 3e-3,
                         "steps": 500, "cs": {"steps": 40}})
    torch.manual_seed(0)
    model = VQVAE(companies=1, days=4, features=6, vocabulary=16, width=32, heads=2)

    history = train(model, _loaders(), settings, entry="cs", check_every=10,
                    patience=99, quiet=True)
    assert history.rows[-1]["step"] == 40         # CS's own budget, not the shared 500


# ------------------------------------------------------------- joint training

def test_a_joint_run_keeps_BOTH_dictionaries_alive():
    """⚠️ THE TEST THAT WOULD HAVE CAUGHT THE FUSION COLLAPSE.

    Perplexity genuinely STARTS at 1.0 -- one word for everything -- and only climbs out via
    the reconstruction anchor, the diversity loss and dead-word revival. Under joint training
    the naming loss used to be free to push it straight back down. If either dictionary ends
    a run near 1, the anchor or the severed naming channel has failed, and the loss curve
    will look perfectly healthy while it happens.
    """
    from bubble_bi.training import train_joint

    # ⚠️ `_tiny_joint()` seeds itself (see its own docstring): this run's cold start is
    # genuinely bimodal (dead lock vs. escape) on a 60-step budget with only one
    # dead-word revival in it, so the fixture pins `torch.manual_seed` to a value
    # measured to escape with a comfortable margin, rather than leaving that to
    # whatever the global RNG happens to be here.
    world, loaders, settings = _tiny_joint()
    history = train_joint(world, loaders, settings, steps=60, quiet=True)
    last = history.last()

    assert last["ts_perplexity"] > 2.0, f"the TS dictionary collapsed: {last}"
    assert last["cs_perplexity"] > 2.0, f"the CS dictionary collapsed: {last}"


def test_a_joint_run_reports_both_honest_floors():
    """Persistence for the word, shrugging for the candle. A number without its floor gets
    quoted, and this project has had to walk back two such numbers already."""
    from bubble_bi.training import train_joint

    world, loaders, settings = _tiny_joint()
    last = train_joint(world, loaders, settings, steps=20, quiet=True).last()

    for name in ("drawing", "shrugging", "accuracy", "persistence"):
        assert name in last, f"{name} is not reported"
