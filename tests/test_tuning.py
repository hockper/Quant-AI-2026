import math

import numpy as np
import pytest
import torch

from bubble_bi import tuning
from bubble_bi.autopsy import _probe
from bubble_bi.models import VQVAE
from bubble_bi.settings import DEFAULTS


def _loader(grids, batch=8):
    return torch.utils.data.DataLoader(
        [{"grid": g, "present": torch.ones(g.shape[0], dtype=torch.bool)} for g in grids],
        batch_size=batch,
    )


def test_the_shuffled_floor_makes_a_useless_token_score_zero():
    """skill = 0 means 'no better than luck'. A random token knows nothing, so it must
    score ~0 — NOT the R2 that its one-hot width could buy on its own."""
    rng = np.random.default_rng(0)
    token = tuning.one_hot(rng.integers(0, 64, size=600), words=64)
    target = rng.normal(size=(600, 2))                    # unrelated to the token
    assert abs(tuning.skill(token, target)) < 0.05


def test_a_wider_vocabulary_does_not_buy_free_skill():
    """THE confound this floor exists for. A 1024-word one-hot hands the probe 1024
    columns; a 128-word one hands it 128. Raw R2 climbs with vocabulary FOR NOTHING, and
    the search would 'discover' that bigger is better having discovered nothing at all.

    Only 220 rows, on purpose. The capacity confound gets WORSE the closer `words` gets
    to the row count -- at 1024 words and 220 rows the wide token has under a quarter of
    a row per word, so a plain (un-floored) ridge fit can all but memorise it, while the
    narrow one (128 words, ~1.7 rows per word) cannot. That gap is what makes this test
    have teeth: at the row count `score_tokenizer` actually uses (thousands of rows), the
    same two vocabularies are close enough in rows-per-word that plain R2 barely differs,
    and a broken floor would slip through unnoticed. Fewer rows makes the SAME bug loud.

    Verified by deliberately breaking `skill()` to return plain R2 (no floor
    subtraction): at this exact seed the two vocabularies then land 0.48 apart --
    nearly ten times this test's own threshold. See task-5-report.md for the numbers.
    This test fails on the naive implementation; it does not fail on this one.
    """
    rng = np.random.default_rng(0)
    n = 220
    target = rng.normal(size=(n, 2))
    narrow = tuning.one_hot(rng.integers(0, 128, size=n), words=128)
    wide = tuning.one_hot(rng.integers(0, 1024, size=n), words=1024)

    assert abs(tuning.skill(wide, target) - tuning.skill(narrow, target)) < 0.05


def test_the_fast_onehot_probe_matches_the_dense_one_exactly():
    """`_probe_onehot` is an OPTIMISATION of `autopsy._probe`, not a different method --
    it must return the identical number, to the last bit that floating point allows.
    This is the proof: run both on the same small case (small enough that the slow,
    dense path is still instant) and demand they agree to 1e-6."""
    rng = np.random.default_rng(7)
    n, words = 300, 16
    ids = rng.integers(0, words, size=n)
    y = rng.normal(size=(n, 2))
    x = tuning.one_hot(ids, words)

    assert abs(tuning._probe_onehot(x, y) - _probe(x, y)) < 1e-6


def test_the_fast_onehot_probe_matches_the_dense_one_when_a_word_is_never_trained_on():
    """The degenerate case: a word that shows up ONLY in the held-out 30%, never in the
    70% the probe is fit on. The dense path handles this by giving that word's row of
    (X^T X) nothing but a tiny ridge term, so its coefficient comes out ~0 and its
    prediction collapses to the intercept. The fast path must land on that exact number,
    not merely something reasonable-looking, or the optimisation is wrong."""
    rng = np.random.default_rng(3)
    n, words = 300, 16
    cut = int(n * 0.7)
    ids = rng.integers(0, words - 1, size=n)          # words 0..14, never word 15
    ids[cut + 2] = words - 1                          # word 15 appears twice --
    ids[cut + 10] = words - 1                         # -- both times AFTER the cut
    assert (ids[:cut] == words - 1).sum() == 0        # never seen in training, by construction
    y = rng.normal(size=(n, 2))
    x = tuning.one_hot(ids, words)

    assert abs(tuning._probe_onehot(x, y) - _probe(x, y)) < 1e-6


def test_a_token_that_knows_the_answer_scores_well():
    rng = np.random.default_rng(2)
    ids = rng.integers(0, 8, size=600)
    target = np.c_[ids.astype(float), -ids.astype(float)] + 0.01 * rng.normal(size=(600, 2))
    assert tuning.skill(tuning.one_hot(ids, words=8), target) > 0.9


def test_a_collapsed_codebook_is_rejected_not_ranked():
    """A token drawn from 2 live words carries 1 bit. However well it probes, it is
    useless downstream — and it would DESTROY the predictor's target, which IS the token."""
    model = VQVAE(companies=1, days=4, features=26, width=16, heads=2,
                  vocabulary=512).eval()
    # Every grid identical -> every grid gets the same word.
    same = torch.zeros(4, 1, 4, 26)
    scored = tuning.score_tokenizer(model, _loader([g for g in same]), DEFAULTS)

    assert scored["score"] == -math.inf
    assert "collapsed" in scored["why"]
    # The se fields exist on every path, collapsed or not, so the notebook never has to
    # special-case a missing key -- a thrown-out trial just carries nan, like the rest.
    assert math.isnan(scored["direction_se"])
    assert math.isnan(scored["volatility_se"])


def test_score_tokenizer_reports_how_noisy_its_own_floor_is():
    """The floor is an AVERAGE of 64 shuffles, not the true value. `direction_se` and
    `volatility_se` say how far that average might still be off, so a reader can tell
    whether two trials' scores are genuinely different or just noise apart -- a question
    nobody could answer before this number existed."""
    torch.manual_seed(1)
    model = VQVAE(companies=1, days=4, features=26, width=16, heads=2, vocabulary=8)
    grids = [torch.randn(1, 4, 26) for _ in range(300)]
    scored = tuning.score_tokenizer(model, _loader(grids), DEFAULTS)

    assert scored["words_used"] >= 2         # otherwise this hit the collapsed path instead
    for key in ("direction_se", "volatility_se"):
        assert np.isfinite(scored[key])
        assert scored[key] >= 0


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


@pytest.fixture
def tiny_batches(tiny_settings):
    """A synthetic panel: 3 companies, 600 days, real features.

    ⚠️ Sized 600, not the smaller 400 the brief sketched: the slowest feature
    (`amihud`'s year-long normalising window, NORM_WINDOW=252 in microstructure.py)
    needs ~273 days of run-up before it produces a single finite row. At 400 days,
    the 70% learn split only reaches 8 usable days -- below `Scaler.LEAST_DAYS` (60)
    -- and `make_tensors` refuses to normalise. 600 days clears that with room to
    spare while still building and training in well under a second.
    """
    import pandas as pd

    from bubble_bi.data import add_features, make_tensors

    n = 600
    rng = np.random.default_rng(0)
    days = pd.date_range("2020-01-01", periods=n, freq="B")
    frames = []
    for ticker in tiny_settings["tickers"]:
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
    table = add_features(raw, tiny_settings)
    return make_tensors(table, tiny_settings)


def test_the_search_space_fixes_the_knobs_we_already_know():
    """decoder_depth is not searched because the decoder is THROWN AWAY when we freeze the
    tokenizer. Searching it would be tuning a part we delete."""
    searched = {k for stage in tuning.SPACE.values() for k in stage}
    assert "decoder_depth" not in searched
    assert "batch" not in searched
    assert searched == {"learning_rate", "commitment", "diversity",
                        "model_size", "vocabulary", "days"}


def test_the_balance_comes_before_the_sizes():
    assert list(tuning.SPACE) == ["balance", "sizes"]
    assert set(tuning.SPACE["balance"]) == {"learning_rate", "commitment", "diversity"}
    assert set(tuning.SPACE["sizes"]) == {"model_size", "vocabulary", "days"}


def test_a_search_returns_a_config_that_check_accepts(tiny_batches, tiny_settings):
    """End-to-end on a 2-trial synthetic run: whatever the search hands back must be a
    settings dict the project will actually accept.

    `best` is a FLAT dict: it mixes the entry's own knobs (commitment, diversity,
    vocabulary, days, ...) with the two top-level ones the search also tunes
    (learning_rate, model_size) -- because that is what `SPACE` actually searches.
    `tuning.settle()` is the one documented way to fold that flat dict back into a full
    settings dict -- see `test_settle_splits_a_flat_search_result_back_apart` for the
    round trip this relies on, and `search()`'s own docstring for why hand-splitting it
    (as this test used to) is exactly the mistake that function exists to prevent.
    """
    from bubble_bi.settings import check

    best, trials = tuning.search("ts", tiny_batches, tiny_settings)

    assert len(trials) == 2
    assert {"score", "direction", "volatility", "words_used"} <= set(trials.columns)
    check(tuning.settle(tiny_settings, "ts", best))


def test_settle_splits_a_flat_search_result_back_apart():
    """`settle()` is the ONE place a caller should turn a `search()` result back into a
    full settings dict -- this is the test that makes it safe to trust.

    `search()` hands back a FLAT dict on purpose (see its docstring), which means a
    caller who does not know about `settle()` is one keystroke from the Task 7 bug: filter
    out `model_size` before assigning the rest into `settings["ts"]` and forget
    `learning_rate` too, and a stray `learning_rate` is left sitting inside `settings["ts"]`
    -- where `check()` does not expect it and raises "Unknown setting(s)". `settle()` makes
    that mistake structurally impossible: nobody hand-picks which keys are top-level ever
    again, because there is exactly one function that knows, and everybody calls it.
    """
    from bubble_bi.settings import DEFAULTS, check

    # Shaped exactly like a `search()` return: the entry's own knobs plus the two
    # top-level ones, all in one flat dict.
    flat = {"commitment": 0.31, "diversity": 0.5, "vocabulary": 512, "days": 10,
            "learning_rate": 5e-4, "model_size": 256}

    settled = tuning.settle({**DEFAULTS, "tickers": ["AAA"]}, "ts", flat)

    check(settled)                                        # (a) check() accepts it
    assert settled["learning_rate"] == 5e-4                # (b) top-level keys land...
    assert settled["model_size"] == 256                    #     ...at the TOP level
    assert not (tuning.TOP_LEVEL & set(settled["ts"]))     # (c) never inside the block


def test_a_killed_search_resumes_instead_of_starting_over(tiny_batches, tiny_settings):
    """Colab WILL disconnect. The study is on disk, so a second call must top up the
    trials that are missing -- not run the whole budget again on top of them."""
    tuning.search("ts", tiny_batches, tiny_settings)
    _, again = tuning.search("ts", tiny_batches, tiny_settings)

    assert len(again) == 2, "the resumed search re-ran trials it had already completed"


def test_apply_leaves_no_top_level_key_stranded_in_an_entry_block(tmp_path):
    """`tuned.json` stores each entry FLAT, so `learning_rate` and `model_size` are sitting
    inside the "ts" object even though they are top-level settings. Filter them out by hand
    and you WILL forget one — then `check()` rejects it and the notebook dies on cell one."""
    import json

    from bubble_bi.settings import check as bb_check

    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({
        "found_on": "2026-07-12", "trials": 12,
        "fingerprint": {"tickers": 1, "features": 26, "start": None, "search_steps": 600},
        "score": {},
        "ts": {"vocabulary": 1024, "learning_rate": 4.2e-4, "model_size": 256},
        "cs": {},
    }))

    merged, _ = tuning.apply({"tickers": ["AAA"]}, path=path)

    assert not (tuning.TOP_LEVEL & set(merged["ts"])), (
        "a top-level setting was left inside the entry block — check() will reject it"
    )
    assert merged["learning_rate"] == 4.2e-4      # lifted OUT, not dropped
    assert merged["model_size"] == 256
    assert merged["ts"]["vocabulary"] == 1024
    bb_check(merged)                               # from bubble_bi.settings import check


def test_precedence_defaults_then_tuned_then_what_you_typed(tmp_path):
    """Three layers, most specific wins. A tuned value replaces a default, but a value you
    DELIBERATELY typed in the notebook stands."""
    import json

    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({
        "found_on": "2026-07-12", "trials": 12,
        "fingerprint": {"tickers": 1, "features": 26, "start": None, "search_steps": 600},
        "score": {}, "ts": {"vocabulary": 1024, "commitment": 0.31}, "cs": {},
    }))

    typed = {"tickers": ["AAA"], "ts": {"vocabulary": 256}}
    merged, note = tuning.apply(typed, path=path)

    assert merged["ts"]["vocabulary"] == 256      # you typed it: you win
    assert merged["ts"]["commitment"] == 0.31     # you did not: the tuning wins
    assert "✅" in note


def test_a_typed_top_level_setting_beats_the_tuned_one(tmp_path):
    """The entry-block precedence (typed beats tuned) is covered above, but
    `learning_rate`/`model_size` take a DIFFERENT path through `apply()` -- they are lifted
    out of the entry blocks and merged at the top of the settings dict, by separate code.
    That path needs its own proof: typing `learning_rate` in the notebook must beat
    whatever `tuned.json` says, exactly the same as it does inside `ts`/`cs`."""
    import json

    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({
        "found_on": "2026-07-12", "trials": 12,
        "fingerprint": {"tickers": 1, "features": 26, "start": None, "search_steps": 600},
        "score": {}, "ts": {"learning_rate": 9e-4}, "cs": {},
    }))

    typed = {"tickers": ["AAA"], "learning_rate": 1e-4}
    merged, note = tuning.apply(typed, path=path)

    assert merged["learning_rate"] == 1e-4        # you typed it: you win, even at the top


def test_a_disagreement_on_a_shared_setting_is_reported_not_swallowed(tmp_path):
    """TS and CS are searched SEPARATELY (see `search()`), and can come back wanting
    different values for a TOP_LEVEL setting. The reviewer's exact repro: TS wants
    `learning_rate=1e-4`, CS wants `9e-4`. The old code let whichever entry ran LAST win in
    total silence -- this test is the one that would have caught it: it demands the note
    actually SAYS a disagreement happened, not just that some number came out."""
    import json

    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({
        "found_on": "2026-07-12", "trials": 12,
        "fingerprint": {"tickers": 1, "features": 26, "start": None, "search_steps": 600},
        "score": {},
        "ts": {"learning_rate": 1e-4, "vocabulary": 512},
        "cs": {"learning_rate": 9e-4, "vocabulary": 128},
    }))

    merged, note = tuning.apply({"tickers": ["AAA"]}, path=path)

    assert merged["learning_rate"] == 1e-4, "TS has ~30x more grids -- its answer must win"
    assert "disagree" in note.lower()
    assert "0.0001" in note                       # TS's value, named
    assert "0.0009" in note                       # CS's DISCARDED value, named too


def test_a_model_size_disagreement_names_the_cross_attention_reason(tmp_path):
    """`model_size` is not merely a preference the way `learning_rate` is -- TS and CS meet
    in a cross-attention layer that needs both sides the SAME width, and `Tokenizer` raises
    if they are not. The warning for THIS key must say so, not give a generic shrug."""
    import json

    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({
        "found_on": "2026-07-12", "trials": 12,
        "fingerprint": {"tickers": 1, "features": 26, "start": None, "search_steps": 600},
        "score": {},
        "ts": {"model_size": 256, "vocabulary": 512},
        "cs": {"model_size": 128, "vocabulary": 256},
    }))

    merged, note = tuning.apply({"tickers": ["AAA"]}, path=path)

    assert merged["model_size"] == 256
    assert "256" in note and "128" in note
    assert "cross-attention" in note.lower()
    assert "SETTINGS" in note                     # tells the user how to overrule it


def test_a_stale_tuning_warns_and_does_not_pretend(tmp_path):
    """We have already changed the feature count twice (10 -> 22 -> 26). Silently reusing
    hyperparameters tuned on different data is the exact class of bug we keep catching."""
    import json

    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({
        "found_on": "2026-01-01", "trials": 12,
        "fingerprint": {"tickers": 30, "features": 22, "start": None, "search_steps": 600},
        "score": {}, "ts": {"vocabulary": 1024}, "cs": {},
    }))

    merged, note = tuning.apply({"tickers": ["AAA"]}, path=path)

    assert "STALE" in note
    assert "22" in note and "26" in note           # says WHAT changed, not just "stale"
    assert merged["ts"]["vocabulary"] == 1024      # still used: half-stale beats untuned


def test_no_tuning_file_says_so_plainly(tmp_path):
    merged, note = tuning.apply({"tickers": ["AAA"]}, path=tmp_path / "absent.json")
    assert "No tuned.json" in note
    assert merged == {"tickers": ["AAA"]}


def test_save_round_trips_through_apply(tmp_path):
    """`save()` writes exactly what `apply()` later reads back -- this is the round trip
    that makes that promise a fact instead of an assumption. Nothing here touches the
    real `tuned.json` at the repo root: `save()` is handed a `tmp_path` file, same as
    every other test in this file."""
    from bubble_bi.settings import check as bb_check

    settings = {**DEFAULTS, "tickers": ["AAA", "BBB"]}
    found = {
        "ts": {"vocabulary": 256, "commitment": 0.3, "learning_rate": 5e-4,
               "model_size": 128},
        "cs": {"vocabulary": 128, "commitment": 0.4, "learning_rate": 5e-4,
               "model_size": 128},
    }

    written = tuning.save(found, settings, path=tmp_path / "tuned.json")
    merged, note = tuning.apply({"tickers": ["AAA", "BBB"]}, path=written)

    assert merged["ts"]["vocabulary"] == 256
    assert merged["cs"]["vocabulary"] == 128
    assert merged["learning_rate"] == 5e-4         # lifted to the top, from both entries
    assert merged["model_size"] == 128
    assert "✅" in note                             # found on the SAME data: not stale
    bb_check(merged)                               # what comes back is a valid settings dict


def test_the_confirm_keeps_the_incumbent_when_the_winner_does_not_beat_it(
        tiny_batches, tiny_settings):
    """The transfer guard. A config that wins a 600-step sprint can lose the real run: CS
    did exactly that, its held-out error climbing while its codebook decayed.

    `confirm` scores the winner first, then the incumbent — so a scorer that hands out a
    poor score and then a good one is a winner that flattered to deceive.
    """
    scores = iter([
        {"score": 0.1, "direction": 0.0, "volatility": 0.1, "words_used": 9,
         "before_quant": 0.0, "why": ""},         # the winner,    at FULL budget
        {"score": 0.9, "direction": 0.4, "volatility": 0.5, "words_used": 9,
         "before_quant": 0.0, "why": ""},         # the incumbent, at FULL budget
    ])

    out = tuning.confirm("ts", {**tiny_settings["ts"], "learning_rate": 1e-3,
                                "model_size": 16},
                         tiny_batches, tiny_settings, scorer=lambda *a, **k: next(scores))

    assert out["kept"] == "incumbent"
    assert out["winner"] == 0.1 and out["incumbent"] == 0.9
    assert out["settings"]["vocabulary"] == tiny_settings["ts"]["vocabulary"]


def test_confirm_trains_at_the_full_budget_not_the_search_sprint(tiny_settings, monkeypatch):
    """The transfer guard's OWN correctness -- the test that would have caught the bug
    that once slipped past all 18 other tests in this file.

    `confirm()` exists because a config that wins a short search sprint can lose the real
    run (see `confirm`'s own docstring for the CS story: held-out error climbing 0.90 ->
    1.03 over nine thousand steps while its codebook decayed 187 -> 141 words). It guards
    against that by rebuilding `settings["search"]["steps"]` from the FULL, top-level
    `steps` before handing anything to `_run_one` -- never the search's own short one.

    A reviewer once replaced that rebuild with plain `full = settings`, silently reverting
    to the short budget -- and every test in this file still passed, because `tiny_settings`
    used to set the top-level `steps` and `search["steps"]` to the SAME number (both 20).
    A correct run and a broken one were numerically identical, so nothing could tell them
    apart. The fixture now gives them different numbers on purpose, and this test is the
    one that actually looks: it spies on `_run_one` and inspects the settings dict that
    reaches it, not the score that comes back out (a mock scorer can't tell a short run
    from a long one -- see the test above, which never noticed either).
    """
    seen_budgets = []

    def spy(entry, chosen, batches, settings, scorer, features, companies, trial=None):
        seen_budgets.append(settings["search"]["steps"])
        return {"score": 0.0, "direction": 0.0, "volatility": 0.0, "words_used": 9,
                "before_quant": 0.0, "why": ""}

    monkeypatch.setattr(tuning, "_run_one", spy)

    tuning.confirm("ts", {**tiny_settings["ts"], "learning_rate": 1e-3, "model_size": 16},
                   batches=None, settings=tiny_settings)

    # Both the winner and the incumbent must be trained at the FULL budget -- never the
    # search's short one.
    assert seen_budgets == [tiny_settings["steps"]] * 2, (
        f"confirm() handed _run_one a step budget of {seen_budgets}, not the full "
        f"budget {tiny_settings['steps']!r} twice over -- the transfer guard is not "
        "actually training at full budget, which is the entire point of confirm()."
    )


def test_a_model_built_before_apply_does_not_get_the_tuned_settings(tmp_path):
    """The Task 8 bug, reproduced directly. The notebook builds `ts`/`cs` once, early
    on -- then, much later, `bb.tuning.apply()` folds `tuned.json` into the settings.
    A model built BEFORE that point never hears about it: it was constructed from the
    settings dict as it existed at the time, and nothing after can reach back and
    change it. Only a model built from the SETTLED settings -- the ones `apply()` and
    `check()` hand back -- actually gets the tuned numbers.

    This is why the notebook fix rebuilds `ts`/`cs` (and `batches`) once tuning
    settles: skip that step, as the notebook once did, and every tuned setting is
    silently thrown away. This test is the guard against reverting to that shape.
    """
    import json

    from bubble_bi.settings import DEFAULTS, check

    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({
        "found_on": "2026-07-12", "trials": 12,
        "fingerprint": {"tickers": 1, "features": 26, "start": None, "search_steps": 600},
        "score": {}, "ts": {"vocabulary": 1024}, "cs": {},
    }))

    typed = {"tickers": ["AAA"]}
    features = len(tuning.names())

    # What the notebook has BEFORE the tuning section runs -- and the model it builds
    # from it, exactly like section 5 of the notebook does.
    settings = check(typed)
    stale = VQVAE(companies=1, features=features, width=settings["model_size"], **settings["ts"])

    # `apply()` runs: `tuned.json` changes `vocabulary` to 1024.
    merged, _ = tuning.apply(typed, path=path)
    settled = check(merged)                       # the notebook's `settings = bb.check(...)`
    fresh = VQVAE(companies=1, features=features, width=settled["model_size"], **settled["ts"])

    assert stale.codebook.words == DEFAULTS["ts"]["vocabulary"], (
        "the model built BEFORE apply() should still carry the untuned default -- "
        "it has no way to have heard about the tuning"
    )
    assert fresh.codebook.words == 1024, (
        "the model built AFTER apply(), from the settled settings, must carry the "
        "tuned vocabulary -- this is the whole point of rebuilding"
    )
    assert stale.codebook.words != fresh.codebook.words


def test_the_probe_target_is_TODAY_and_never_tomorrow():
    """TS and CS are autoencoders. The target is the LAST DAY of the window they were just
    handed — read straight out of the grid, so nothing from the future can reach it."""
    torch.manual_seed(0)
    model = VQVAE(companies=1, days=4, features=26, width=16, heads=2, vocabulary=8)
    grids = [torch.randn(1, 4, 26) for _ in range(64)]

    _, _, direction, _ = tuning.look(model, _loader(grids), DEFAULTS, limit=99)

    body = tuning.names().index("body")
    expected = np.array([float(g[0, -1, body]) for g in grids])   # last day, `body`
    assert np.allclose(direction[:, 0], expected, atol=1e-5)
