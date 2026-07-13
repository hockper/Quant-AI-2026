import math

import numpy as np
import pandas as pd
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


def test_score_tokenizer_never_builds_the_one_hot_matrix_it_already_has_ids_for(monkeypatch):
    """`score_tokenizer` already HAS `ids` straight from the codebook. Building
    `one_hot(ids, words)` just to hand it to `skill_and_noise()`, which would scan it
    (`_is_onehot`) and `argmax` it straight back into the ids it started from, is pure
    waste -- ~42MB at TS scale with `vocabulary=1024`, twice per trial. `score_tokenizer`
    must go through `skill_from_ids` instead, and never call `one_hot` at all.
    """
    torch.manual_seed(1)
    model = VQVAE(companies=1, days=4, features=26, width=16, heads=2, vocabulary=8)
    grids = [torch.randn(1, 4, 26) for _ in range(300)]

    def forbidden(*a, **k):
        raise AssertionError("score_tokenizer built a one-hot matrix it did not need")

    monkeypatch.setattr(tuning, "one_hot", forbidden)
    scored = tuning.score_tokenizer(model, _loader(grids), DEFAULTS)
    assert scored["words_used"] >= 2                        # took the real path, not collapse


def test_score_tokenizer_quick_mode_skips_the_dense_before_quant_probe():
    """`before_quant` runs a DENSE ridge fit on the continuous `summary` vector -- 65
    fits (the real one plus 64 shuffles), the expensive kind `_probe_onehot_ids` exists
    to let the TOKEN avoid. A mid-training pruning check only ever reads `["score"]`, so
    it has no use for `before_quant` at all. `quick=True` skips it; everything else
    (`score` itself, most of all) must come out identical either way.
    """
    torch.manual_seed(1)
    model = VQVAE(companies=1, days=4, features=26, width=16, heads=2, vocabulary=8)
    grids = [torch.randn(1, 4, 26) for _ in range(300)]

    full = tuning.score_tokenizer(model, _loader(grids), DEFAULTS)
    quick = tuning.score_tokenizer(model, _loader(grids), DEFAULTS, quick=True)

    assert math.isnan(quick["before_quant"])
    assert not math.isnan(full["before_quant"])
    assert quick["score"] == full["score"]
    assert quick["direction"] == full["direction"]
    assert quick["volatility"] == full["volatility"]


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
    import pandas as pd

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


@pytest.fixture
def healthy_settings(tmp_path):
    """A tiny config that actually SURVIVES -- unlike `tiny_settings` just above, which
    the whole file inherited without ever checking whether a real (non-stubbed)
    `search()`/`_run_one()` call could produce anything BUT a collapsed codebook.

    Verified: `tiny_settings` collapses to a single word across 20+ torch seeds, no
    matter what runs it -- it is a property of its `commitment`/`diversity`/`steps`,
    not bad luck. That means every non-stubbed `search()`/`_run_one()` test in this
    file, before this fixture existed, only ever exercised the "every trial collapsed"
    branch -- `search()`'s actual job, scoring several real survivors and ranking them,
    was never once demonstrated.

    Tuned by MEASURING -- the discipline this whole module preaches, not by guessing:

      commitment = 0.1     the SPACE's own floor (see `SPACE["balance"]`) -- the
                            standard 0.25 is already too strong for a codebook this
                            small and short-lived; too high pins the encoder to a
                            word it already has before the dictionary has spread out.
      diversity  = 1.0     the SPACE's own ceiling -- the anti-collapse term; too low
                            stops fighting the handful of words that win early.
      vocabulary = 8       small enough that `ALIVE * words` (the bar for "alive") is
                            trivial to clear even for a short run, keeping this FAST.
      search.steps = 160   long enough that even a slow draw from SPACE's
                            learning_rate range (log-uniform 3e-5 .. 3e-3) has time to
                            convince a few words to stay alive -- measured: shorter
                            budgets (80, 120) still collapse at the slow end of that
                            range; 160 is where it stopped.

    `steps` (the FULL, top-level budget `confirm()` uses) is deliberately more than
    `search.steps` -- the same trap `tiny_settings` guards against, see its own
    docstring -- so a `confirm()` that used the wrong budget by accident would not be
    numerically indistinguishable from a correct one.
    """
    return {
        **DEFAULTS,
        "tickers": ["AAA", "BBB", "CCC"],
        "data_dir": str(tmp_path),
        "model_size": 16,
        "learning_rate": 1e-3,
        "steps": 320,
        "ts": {**DEFAULTS["ts"], "days": 3, "batch": 8, "vocabulary": 8,
               "commitment": 0.1, "diversity": 1.0,
               "encoder_depth": 1, "decoder_depth": 1, "heads": 2, "steps": 20},
        "cs": {**DEFAULTS["cs"], "days": 3, "batch": 4, "vocabulary": 8,
               "commitment": 0.1, "diversity": 1.0,
               "encoder_depth": 1, "decoder_depth": 1, "heads": 2, "steps": 20},
        "search": {"run": True, "trials": 4, "steps": 160},
    }


@pytest.fixture
def healthy_batches(healthy_settings):
    """Same kind of synthetic panel as `tiny_batches` -- see `_synthetic_panel` -- built
    from `healthy_settings` instead. Kept as a SEPARATE fixture rather than parametrising
    `tiny_batches`, because the two exist to prove opposite things: one that the
    rejection path is handled correctly, one that the survive-and-rank path works at
    all. Collapsing them into one parametrised fixture would blur that distinction for
    no real saving.
    """
    return _synthetic_panel(healthy_settings)


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


def test_search_refuses_a_budget_that_cannot_cover_both_stages(tiny_batches, tiny_settings):
    """`search()` runs in TWO stages, `budget // 2` trials each. At `trials=1`,
    `budget // 2` is 0 -- no trial ever runs, `rows` stays empty, and
    `pd.DataFrame([]).sort_values("score")` used to raise an opaque `KeyError: 'score'`.
    `settings.check()` allows `trials=1` (it only checks that it is positive), so this
    has to be caught here, with a message a human can actually act on.
    """
    broken = {**tiny_settings, "search": {**tiny_settings["search"], "trials": 1}}
    with pytest.raises(ValueError, match="at least 2"):
        tuning.search("ts", tiny_batches, broken)


def test_disagreements_fires_when_the_best_score_is_not_the_best_direction():
    """The reviewer's exact repro. `search()` returns each entry's trials table sorted
    but NOT re-indexed (0..n-1 for TS, 0..n-1 again for CS), so a naive `pd.concat` of
    the two gives a combined frame with the SAME row label appearing twice. Comparing
    `.name` on that frame compares whichever row happens to share a label -- two
    completely different rows can compare equal, and the warning silently fails to
    fire even when it is true. This trials frame reproduces that exact collision (both
    entries indexed 0, 1) AND makes TS's top-scoring row genuinely NOT its
    top-direction row, so it demands `disagreements()` still catches it."""
    trials = pd.DataFrame({
        "entry":     ["TS", "TS", "CS", "CS"],
        "score":     [0.90, 0.80, 0.50, 0.40],
        "direction": [0.01, 0.05, 0.10, 0.02],   # TS: best score (row 0) != best direction (row 1)
    }, index=[0, 1, 0, 1])                        # <- the exact collision pd.concat causes

    lines = tuning.disagreements(trials)

    assert len(lines) == 1
    assert "TS" in lines[0]
    assert "0.01" in lines[0] and "0.05" in lines[0]   # both direction scores, named
    assert "CS" not in lines[0]                        # CS agreed: nothing to say about it


def test_disagreements_reports_nothing_when_score_and_direction_agree():
    """The other half of the proof: when the top-scoring trial IS the top-direction
    trial (for every entry), there is nothing to warn about, and this must say so by
    returning nothing -- not by returning a reassuring line that says "they agree"."""
    trials = pd.DataFrame({
        "entry":     ["TS", "TS", "CS", "CS"],
        "score":     [0.90, 0.80, 0.50, 0.40],
        "direction": [0.09, 0.05, 0.10, 0.02],   # top score IS top direction, both entries
    }, index=[0, 1, 0, 1])

    assert tuning.disagreements(trials) == []


def test_disagreements_names_the_entry_when_every_trial_collapsed():
    """The crash this finding is about: `score_tokenizer` marks a collapsed trial
    `score = -inf, direction = NaN`. If EVERY trial for an entry collapsed, `direction`
    is all-NaN, and the old `group["direction"].idxmax()` raised `ValueError:
    Encountered all NA values` -- an opaque pandas error that told a non-programmer
    nothing, and that fired BEFORE the notebook's "N trials is a SCREEN" banner. This
    is the test that would have caught it: `disagreements()` must not raise, and must
    say plainly that TS collapsed entirely, naming TS."""
    trials = pd.DataFrame({
        "entry":     ["TS", "TS", "TS"],
        "score":     [-math.inf, -math.inf, -math.inf],
        "direction": [float("nan"), float("nan"), float("nan")],
    })

    lines = tuning.disagreements(trials)

    assert len(lines) == 1
    assert "TS" in lines[0]
    assert "collapsed" in lines[0].lower()
    assert "commitment" in lines[0] and "diversity" in lines[0]


def test_disagreements_reports_the_healthy_entry_even_when_the_other_fully_collapsed():
    """The whole point of not crashing: one entry's total collapse must not silence a
    genuine disagreement in the OTHER entry. TS collapses completely here; CS has a
    real, disagreeing result. Both lines must come back -- CS's disagreement is not
    allowed to go missing just because TS blew up."""
    trials = pd.DataFrame({
        "entry":     ["TS", "TS", "CS", "CS"],
        "score":     [-math.inf, -math.inf, 0.50, 0.40],
        "direction": [float("nan"), float("nan"), 0.02, 0.10],
        # CS: best score (row 2, score 0.50) has direction 0.02; best direction (row 3,
        # direction 0.10) has the lower score 0.40 -- a genuine disagreement.
    })

    lines = tuning.disagreements(trials)

    assert len(lines) == 2
    ts_line = next(line for line in lines if "collapsed" in line.lower())
    cs_line = next(line for line in lines if line is not ts_line)
    assert "TS" in ts_line
    assert "CS" in cs_line
    assert "0.02" in cs_line and "0.10" in cs_line


def test_disagreements_never_reports_a_collapsed_trial_as_the_winner():
    """Some trials collapsed, some survived, for the SAME entry. `idxmax()` skipping
    NaN already gets this right today, but that must not be an accident of pandas
    behaviour -- a rejected (`score = -inf`) trial must be explicitly excluded before
    comparison, never eligible to be reported as a best-score or best-direction row."""
    trials = pd.DataFrame({
        "entry":     ["TS", "TS", "TS"],
        "score":     [-math.inf, 0.90, 0.30],
        "direction": [float("nan"), 0.01, 0.05],
        # Row 0 collapsed (-inf) but would be the numeric max of nothing -- it must
        # never win. Row 1 is the real best score; row 2 is the real best direction.
    })

    lines = tuning.disagreements(trials)

    assert len(lines) == 1
    assert "0.01" in lines[0] and "0.05" in lines[0]
    assert "inf" not in lines[0].lower()
    assert "nan" not in lines[0].lower()


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


def test_search_emits_the_entry_column_that_disagreements_actually_groups_by(
        tiny_batches, tiny_settings):
    """The real composition, exercised for once. Before this fix, `search()` never put
    an `entry` column on its own table at all -- only the NOTEBOOK did, via
    `.assign(entry="TS")` after the fact -- so every test of `disagreements()` (all of
    them above) fed it a table built by hand that already had the column, and the
    actual `search() → disagreements()` pipeline was never run together, not even once.
    """
    _, ts_trials = tuning.search("ts", tiny_batches, tiny_settings)
    _, cs_trials = tuning.search("cs", tiny_batches, tiny_settings)

    assert "entry" in ts_trials.columns and "entry" in cs_trials.columns
    assert set(ts_trials["entry"]) == {"ts"}
    assert set(cs_trials["entry"]) == {"cs"}

    trials = pd.concat([ts_trials, cs_trials], ignore_index=True)
    lines = tuning.disagreements(trials)          # must not raise -- this is the point
    assert isinstance(lines, list)


def test_search_carries_the_winning_balance_onto_every_sizes_stage_row(
        tiny_batches, tiny_settings, monkeypatch):
    """Stage B's own `trial.params` are ONLY the sizes knobs (model_size, vocabulary,
    days) -- the balance it trained WITH (learning_rate, commitment, diversity) lives in
    `fixed`, carried over from Stage A's winner, and used to never make it into the
    recorded row at all. That left every sizes-stage row showing NaN for the balance
    knobs, and `plots.tuning_importance` silently drops any column that is not a number
    for at least 3 trials -- so it looked, wrongly, like the balance never mattered.

    `_run_one` is stubbed to a fixed, healthy score: this test is about the ROW-MERGING
    logic in `search()` alone, not about whether this fixture's tiny synthetic model
    happens to collapse its codebook (it can, deterministically, now that training is
    seeded -- a collapsed Stage A leaves `fixed` empty for a real reason, which is a
    different thing entirely from the bug this test guards against).
    """
    def fake_run_one(entry, chosen, batches, settings, scorer, features, companies, trial=None):
        return {"score": 1.0, "direction": 0.1, "volatility": 0.5, "words_used": 9,
                "before_quant": 0.2, "why": ""}

    monkeypatch.setattr(tuning, "_run_one", fake_run_one)

    _, trials = tuning.search("ts", tiny_batches, tiny_settings)
    sizes_rows = trials[trials["stage"] == "sizes"]
    assert not sizes_rows.empty
    for knob in ("learning_rate", "commitment", "diversity"):
        assert sizes_rows[knob].notna().all(), (
            f"{knob} is NaN on a sizes-stage row -- Stage A's winning balance was not "
            "carried onto Stage B's own rows"
        )


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


def test_apply_names_every_typed_setting_that_overruled_a_tuned_one(tmp_path):
    """CRITICAL 1's exact repro: the shipped notebook's `SETTINGS` cell types four of the
    six knobs the search tunes (`ts.days`, `ts.vocabulary`, `model_size`,
    `learning_rate`), so a user who runs the search on Colab gets a `tuned.json` whose
    answer for every one of those four is silently thrown away the moment `apply()`
    folds it in -- and the note used to print a cheerful ✅ with no hint that had
    happened. Typing beating tuning is correct and stays correct (see the precedence
    tests above); staying SILENT about it is the bug. Every overruled setting must be
    named in the note, with both numbers, so nobody has to notice the trap on their own.
    """
    import json

    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({
        "found_on": "2026-07-12", "trials": 12,
        "fingerprint": {"tickers": 1, "features": 26, "start": None, "search_steps": 600},
        "score": {},
        "ts": {"vocabulary": 1024, "days": 15, "learning_rate": 4.2e-4, "model_size": 256,
               "commitment": 0.31},
        "cs": {},
    }))

    # Exactly the shape of the notebook's shipped SETTINGS: four of the searched knobs
    # typed by hand, `commitment` left alone to inherit the tuning.
    typed = {"tickers": ["AAA"], "ts": {"vocabulary": 512, "days": 4},
             "model_size": 128, "learning_rate": 1e-4}
    merged, note = tuning.apply(typed, path=path)

    # Precedence is unchanged: what you typed still wins.
    assert merged["ts"]["vocabulary"] == 512
    assert merged["ts"]["days"] == 4
    assert merged["model_size"] == 128
    assert merged["learning_rate"] == 1e-4
    assert merged["ts"]["commitment"] == 0.31      # untyped: the tuning still reaches this

    # But the note must say so, loudly, naming every overruled setting and BOTH values.
    assert "discarded" in note.lower()
    for needle in ("ts.vocabulary", "512", "1024",
                  "ts.days", "4", "15",
                  "model_size", "128", "256",
                  "learning_rate"):
        assert needle in note, f"{needle!r} missing from the overrule note:\n{note}"
    # `commitment` was never typed, so it must not be reported as overruled.
    assert "commitment" not in note.lower()


def test_apply_says_nothing_extra_when_nothing_was_overruled(tmp_path):
    """The other half of the proof: typing a setting that happens to match what the
    tuning ALSO found is not an overrule (nothing was discarded), and typing nothing at
    all in an entry block must not produce a false alarm either."""
    import json

    path = tmp_path / "tuned.json"
    path.write_text(json.dumps({
        "found_on": "2026-07-12", "trials": 12,
        "fingerprint": {"tickers": 1, "features": 26, "start": None, "search_steps": 600},
        "score": {}, "ts": {"vocabulary": 512}, "cs": {},
    }))

    merged, note = tuning.apply({"tickers": ["AAA"], "ts": {"vocabulary": 512}}, path=path)

    assert merged["ts"]["vocabulary"] == 512
    assert "discarded" not in note.lower()
    assert "you typed" not in note.lower()


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


def test_tuned_json_is_strict_json_even_when_confirm_kept_the_incumbent(tmp_path):
    """The old notebook cell recorded `"score": {"ts": ts_trials.iloc[0].to_dict()}` --
    always the best SEARCH trial, even on the runs where `confirm()` rejected it and
    kept the incumbent instead, so the saved score described a config that had just
    been thrown out. And `iloc[0]` is one STAGE's row, so the other stage's own knobs
    came back as `NaN` -- which `json.dumps` writes as a bare, non-standard `NaN`
    literal. Python's own `json.loads` reads that back without complaint (which is
    exactly why a naive round-trip test would never catch this), but it is not valid
    JSON by the spec, and no other parser has to accept it.

    This writes `tuned.json` the way the FIXED notebook does -- `confirm()`'s own
    `kept`/`winner`/`incumbent`, plain floats describing what was actually saved -- and
    demands the file parse under a STRICT reader that refuses `NaN`/`Infinity`.
    """
    import json

    from bubble_bi.settings import DEFAULTS

    settings = {**DEFAULTS, "tickers": ["AAA", "BBB"]}
    # Shaped exactly like the notebook's fixed cell: `confirm()` kept the INCUMBENT for
    # TS (the search's own winner lost the head-to-head), and the winner for CS.
    found = {
        "ts": {"vocabulary": 512, "commitment": 0.25, "learning_rate": 1e-4,
               "model_size": 128},
        "cs": {"vocabulary": 256, "commitment": 0.31, "learning_rate": 1e-4,
               "model_size": 128},
        "score": {"ts": {"kept": "incumbent", "winner": 0.10, "incumbent": 0.42},
                  "cs": {"kept": "winner", "winner": 0.55, "incumbent": 0.20}},
    }

    path = tuning.save(found, settings, path=tmp_path / "tuned.json")
    text = path.read_text()

    def _no_constants(token):
        raise ValueError(f"non-standard JSON constant found: {token!r}")

    # The strict check the standard library's own loader skips by default -- pass
    # `parse_constant` and `NaN`/`Infinity`/`-Infinity` become a hard error instead of
    # silently parsing.
    parsed = json.loads(text, parse_constant=_no_constants)   # raises if not strict JSON
    assert parsed["score"]["ts"]["kept"] == "incumbent"
    assert parsed["score"]["ts"]["incumbent"] == 0.42


def test_save_makes_a_double_collapse_representable_and_strict_json_safe(tmp_path):
    """IMPORTANT A's exact repro. `confirm()` trains the winner AND the incumbent, and if
    EVERY configuration destroyed the codebook on both sides, both come back
    `score = -math.inf` (a collapsed trial is REJECTED, not ranked -- see
    `score_tokenizer`'s own docstring). The old `save()` handed that straight to
    `json.dumps`, which writes the bare, non-standard `-Infinity` token: valid to Python's
    own reader (which is exactly why a naive round-trip test would never catch this), but
    not to the strict parser `test_tuned_json_is_strict_json_even_when_confirm_kept_the_incumbent`
    exists to defend against. Same defect the project thought it had closed, reached by a
    different, entirely plausible route: a routine failure (codebook collapse, which this
    project hits often) leaves an artifact no conformant parser can read.

    "Nothing survived" is a real, legitimate result to record, not a bug -- so `save()`
    must turn the non-finite score into JSON `null` deliberately, AND the file must still
    say so plainly: a reader must be able to tell "this entry's search found nothing, the
    incumbent was kept by default" apart from "this entry found a real winner". This
    checks both halves at once, and checks CS (a genuine, non-collapsed result) right next
    to it, so the fix cannot be "null out everything" -- only the collapsed entry may lose
    its numbers.
    """
    import json
    import math

    from bubble_bi.settings import DEFAULTS

    settings = {**DEFAULTS, "tickers": ["AAA", "BBB"]}
    found = {
        "ts": {"vocabulary": 512, "commitment": 1.7, "learning_rate": 1e-4,
               "model_size": 128},
        "cs": {"vocabulary": 256, "commitment": 0.31, "learning_rate": 1e-4,
               "model_size": 128},
        # TS: both sides collapsed -- confirm()'s exact shape when nothing survived.
        # CS: a genuine result, untouched, right alongside it.
        "score": {"ts": {"kept": "incumbent", "winner": -math.inf, "incumbent": -math.inf},
                  "cs": {"kept": "winner", "winner": 0.55, "incumbent": 0.20}},
    }

    path = tuning.save(found, settings, path=tmp_path / "tuned.json")
    text = path.read_text()

    def _no_constants(token):
        raise ValueError(f"non-standard JSON constant found: {token!r}")

    parsed = json.loads(text, parse_constant=_no_constants)   # raises if not strict JSON

    # The non-finite numbers are gone -- turned into JSON null, not left as a bare
    # -Infinity token no other conformant parser would accept.
    assert parsed["score"]["ts"]["winner"] is None
    assert parsed["score"]["ts"]["incumbent"] is None

    # And the situation is LEGIBLE, not just silently nulled: a reader (human or program)
    # can tell TS found nothing, distinctly from CS, which found a real winner.
    assert parsed["score"]["ts"]["collapsed"] is True
    assert parsed["score"]["cs"]["collapsed"] is False
    assert parsed["score"]["cs"]["winner"] == 0.55           # CS's real score is untouched
    assert parsed["score"]["cs"]["incumbent"] == 0.20


def test_search_ranks_surviving_trials_and_returns_the_highest_scorer(
        healthy_batches, healthy_settings):
    """The happy path `search()` was entirely missing. Every OTHER test in this file that
    calls a real (non-stubbed) `search()`/`_run_one()` uses `tiny_settings`, which
    collapses on every single trial (see `healthy_settings`'s own docstring) -- so none
    of them ever demonstrated `search()`'s actual job: score several trials for real, and
    correctly pick the best one. This is that test.

    `healthy_settings` is tuned to survive, so the balance stage genuinely produces more
    than one trial with a FINITE score -- which is what makes "search() ranks its
    survivors correctly" a checkable fact rather than an assumption: with only one
    survivor there would be nothing to rank against.
    """
    best, trials = tuning.search("ts", healthy_batches, healthy_settings)

    survivors = trials[np.isfinite(trials["score"])]
    assert len(survivors) >= 2, (
        f"expected at least two trials to survive with a finite score, got "
        f"{len(survivors)} of {len(trials)}:\n{trials[['stage', 'score', 'words_used']]}\n"
        "If this starts failing, `healthy_settings` needs RETUNING (see its own "
        "docstring for how it was measured) -- not deleting."
    )

    balance = survivors[survivors["stage"] == "balance"]
    assert len(balance) >= 2, (
        "need at least two survivors in the SAME stage to actually prove ranking "
        "picked the best one, rather than there only ever being one candidate"
    )

    # The winner `search()` settles on must be the highest-scoring survivor of the
    # balance stage -- not whichever trial happened to run first or last.
    top = balance.loc[balance["score"].idxmax()]
    assert best["commitment"] == top["commitment"]
    assert best["diversity"] == top["diversity"]
    assert best["learning_rate"] == top["learning_rate"]

    # And a codebook that actually survived, not a technicality: comfortably more than
    # the bare minimum (`max(2, ALIVE * vocabulary)`) of live words.
    assert (survivors["words_used"] > 2).all()


def test_run_one_seeds_training_so_the_same_config_scores_the_same_way_twice(
        healthy_batches, healthy_settings):
    """`_run_one` is called back to back -- twelve times over a search, twice more in
    `confirm` -- on ONE advancing global torch RNG. Without a reseed, the SECOND call
    with an identical config would start from wherever training the FIRST one left the
    RNG, a different weight initialisation that has nothing to do with the config being
    tested -- so 'the same config twice' would only agree by luck. Seeding from
    `settings["seed"]` at the top of `_run_one` makes it a fact instead: same config,
    same seed, same score.

    ⚠️ This USED TO run on `tiny_settings`, which collapses every trial to `-inf` no
    matter what -- so `first["score"] == second["score"]` passed even with the reseed
    line in `_run_one` deleted outright (`-inf == -inf`), and the test provided zero
    evidence for the property in its own name. `healthy_settings` gives both calls a
    FINITE score, which a missing reseed genuinely moves -- verified by deleting the
    `torch.manual_seed` line from `_run_one` and confirming this test fails (see the
    branch's own report for the exact numbers and the failure message).
    """
    config = {**healthy_settings["ts"], "learning_rate": 1e-3, "model_size": 16}
    features, companies = len(tuning.names()), len(healthy_settings["tickers"])

    first = tuning._run_one("ts", config, healthy_batches, healthy_settings,
                            tuning.score_tokenizer, features, companies)
    second = tuning._run_one("ts", config, healthy_batches, healthy_settings,
                             tuning.score_tokenizer, features, companies)

    assert math.isfinite(first["score"]) and math.isfinite(second["score"]), (
        "this fixture is supposed to survive -- got a collapsed (-inf) score, so this "
        "run proves nothing about reseeding. Retune `healthy_settings`, don't weaken "
        "this assertion."
    )
    assert first["score"] == second["score"], (
        "the same config and the same seed produced two different scores -- training "
        "is still drawing from wherever the global torch RNG happened to be left, not "
        "from a fresh, reproducible seed"
    )


def test_confirm_seeds_the_winner_and_the_incumbent_to_identical_starting_weights(
        tiny_batches, tiny_settings, monkeypatch):
    """The transfer guard's own correctness. `confirm()` trains the winner, THEN the
    incumbent, back to back -- so without a reseed in between, the incumbent would
    start from whatever random state the winner's training left behind: a different
    initialisation that has nothing to do with which config is actually better, on the
    ONE comparison that is allowed to overrule the entire search. This intercepts
    `training.train` before it ever takes a step, and demands the model hand to it is
    bit-for-bit identical both times when the two configs are the same shape.
    """
    from bubble_bi import training as training_module

    captured = []

    def fake_train(model, loaders, settings, **kwargs):
        captured.append({k: v.clone() for k, v in model.state_dict().items()})
        return None

    monkeypatch.setattr(training_module, "train", fake_train)

    config = {**tiny_settings["ts"], "learning_rate": tiny_settings["learning_rate"],
              "model_size": tiny_settings["model_size"]}

    tuning.confirm("ts", config, tiny_batches, tiny_settings)

    assert len(captured) == 2, "confirm() must train exactly two models: winner, incumbent"
    winner_weights, incumbent_weights = captured
    assert winner_weights.keys() == incumbent_weights.keys()
    for key in winner_weights:
        assert torch.equal(winner_weights[key], incumbent_weights[key]), (
            f"{key!r} differed at INIT between the winner and the incumbent -- confirm() "
            "is not reseeding between its two _run_one calls"
        )


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

    # ⚠️ THIS ASSERTION USED TO BE WRONG, and it was wrong in the way that matters: it
    # demanded the SHARED `settings["steps"]`, which is precisely the bug two real GPU runs
    # exposed. "Full budget" means THE BUDGET THIS ENTRY WILL ACTUALLY BE TRAINED AT --
    # `settings[entry]["steps"] or settings["steps"]`, the same rule `train()` has always
    # used -- because CS carries its own (it has ~30x less data and needs the passes). A
    # confirm run at the shared budget validates a run that never happens.
    real = tiny_settings["ts"]["steps"] or tiny_settings["steps"]
    assert seen_budgets == [real] * 2, (
        f"confirm() handed _run_one a step budget of {seen_budgets}, not {real} twice over "
        "-- the budget TS is actually trained at. It would be validating a config on a run "
        "nobody ever makes."
    )
    # And it must never be the search's short sprint. A guard that re-runs the sprint at
    # sprint length is not a guard.
    assert all(b > tiny_settings["search"]["steps"] for b in seen_budgets)


def test_the_pruning_check_does_not_pay_for_the_dense_before_quant_probe(
        tiny_batches, tiny_settings, monkeypatch):
    """`watch()` used to call the FULL scorer at every held-out check purely to feed
    `trial.report()` a number -- and threw away everything but `["score"]`, including
    `before_quant`, a dense ridge fit (65 fits) that is the exact cost the one-hot fast
    path exists to avoid paying anywhere it does not have to. This counts how many
    times that dense probe (`tuning.skill`) actually runs across a WHOLE `_run_one`
    call -- many held-out checks plus one final score -- and demands it is exactly ONE:
    the trial's real, final score, never a pruning check along the way.

    `training.train` is stubbed to fire `on_check` a few times without actually
    training -- this test is about how many times `watch()` pays for the dense probe
    per check, not about whether this fixture's tiny model happens to train its
    codebook into collapse (which would hide the waste behind the cheap "rejected"
    path instead of proving anything about the fix).
    """
    from bubble_bi import training as training_module

    def fake_train(model, loaders, settings, steps=None, quiet=False, on_check=None, **kw):
        for step in range(1, 6):                 # simulate 5 held-out checks
            on_check(step, {})

    monkeypatch.setattr(training_module, "train", fake_train)

    calls = {"n": 0}
    real_skill = tuning.skill

    def counting_skill(*a, **k):
        calls["n"] += 1
        return real_skill(*a, **k)

    monkeypatch.setattr(tuning, "skill", counting_skill)

    class FakeTrial:
        def report(self, value, step):
            pass

        def should_prune(self):
            return False

    config = {**tiny_settings["ts"], "learning_rate": 1e-3, "model_size": 16}
    tuning._run_one("ts", config, tiny_batches, tiny_settings, tuning.score_tokenizer,
                    len(tuning.names()), len(tiny_settings["tickers"]), trial=FakeTrial())

    assert calls["n"] == 1, (
        f"the dense before_quant probe ran {calls['n']} times over one trial -- it "
        "must run exactly once, for the FINAL score, never during a pruning check"
    )


def test_the_pruning_check_still_calls_an_injected_scorer_exactly_as_before(
        tiny_batches, tiny_settings):
    """⚠️ The `scorer=` injection point must not break. The quick path exists ONLY for
    this project's own default (`score_tokenizer`) -- an arbitrary custom `scorer`
    (several tests inject one) cannot be assumed to accept a `quick=` argument, so it
    must keep being called exactly as it always was: `scorer(model, loader, live)`,
    nothing added, nothing assumed.
    """
    seen = []

    def custom_scorer(model, loader, live):
        seen.append(True)
        return {"score": 0.0, "direction": 0.0, "volatility": 0.0, "words_used": 9,
                "before_quant": 0.0, "why": ""}

    class FakeTrial:
        def report(self, value, step):
            pass

        def should_prune(self):
            return False

    config = {**tiny_settings["ts"], "learning_rate": 1e-3, "model_size": 16}
    tuning._run_one("ts", config, tiny_batches, tiny_settings, custom_scorer,
                    len(tuning.names()), len(tiny_settings["tickers"]), trial=FakeTrial())

    assert seen, "the custom scorer was never called at all"


def test_a_model_built_before_apply_does_not_get_the_tuned_settings(tmp_path):
    """The Task 8 bug's LIBRARY half, reproduced directly -- ⚠️ this test never opens
    the notebook, and it is NOT a guard against the notebook reverting.

    What it DOES prove: a model built from settings taken BEFORE `apply()` folds in
    `tuned.json` carries the OLD (default) numbers, and a model built from the SETTLED
    settings that `apply()` + `check()` hand back afterwards carries the TUNED ones.
    That is the composition the notebook's rebuild-after-settle step relies on, and if
    it ever broke -- say `apply()` stopped changing anything, or `check()` dropped a
    tuned key -- this test would catch it.

    What it does NOT prove: that the notebook itself still calls its rebuild cells.
    Nothing in this suite executes the notebook, so a reviewer who deletes those cells
    (leaving the notebook training the STALE `ts`/`cs` built earlier, tuning silently
    inert) would sail straight past this test, and past everything else here too --
    there is no notebook-execution test in this suite. That gap is real; do not let
    this test's name talk you out of noticing it.
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


def test_corrupting_tomorrows_return_leaves_every_trial_score_bit_for_bit_identical(
        tiny_batches, tiny_settings, tmp_path):
    """The spec's own promised proof that the search is blind to the future.
    `arrays.y` IS tomorrow's return -- if the search's score depended on it even a
    little, scrambling it would move the score. It does not, because the probe's own
    target never comes from `arrays.y` at all: it comes from the LAST DAY of the grid
    itself (`look()`), read straight out of `arrays.x`. Two full searches, same seed,
    one on the real data and one on data where tomorrow's return has been shuffled
    against the wrong day -- their trial scores must match, bit for bit.

    Each run gets its OWN `data_dir`, so the second search cannot just resume the
    first's on-disk study (same entry, same stage names) and silently skip actually
    running on the corrupted data at all.
    """
    import dataclasses

    honest_settings = {**tiny_settings, "data_dir": str(tmp_path / "honest")}
    blind_settings = {**tiny_settings, "data_dir": str(tmp_path / "blind")}

    rng = np.random.default_rng(0)
    scrambled_y = tiny_batches.arrays.y[rng.permutation(len(tiny_batches.arrays.y))]
    blind_arrays = dataclasses.replace(tiny_batches.arrays, y=scrambled_y)
    blind_batches = dataclasses.replace(tiny_batches, arrays=blind_arrays)

    _, honest_trials = tuning.search("ts", tiny_batches, honest_settings)
    _, blind_trials = tuning.search("ts", blind_batches, blind_settings)

    pd.testing.assert_series_equal(
        honest_trials["score"].reset_index(drop=True),
        blind_trials["score"].reset_index(drop=True),
        check_names=False,
    )


# ─────────────────────────────────────────────── the pruner was silently dead
#
# Found on the first real GPU run. Optuna's MedianPruner prunes a trial by comparing it
# against the MEDIAN of every trial's intermediate values, via `np.nanpercentile`. We were
# reporting `-inf` at a check whose codebook had collapsed:
#
#     np.nanpercentile([-inf, nan, 0.9])  ->  RuntimeWarning: invalid value in add  ->  -inf
#     np.nanpercentile([-inf, -inf, 0.9]) ->  RuntimeWarning                        ->  nan
#
# The median it prunes against became -inf or NaN, every comparison against it is False,
# and NOTHING was ever pruned. Every hopeless trial ran to full budget, on a paid GPU.
#
# And the deeper mistake: an early collapse is NORMAL — this project's own notebook says
# "perplexity STARTS at 1.0", and dead-code revival pulls it back out. A collapsed codebook
# at step 60 is not a bad score, it is NOT YET A SCORE. Reporting it as a number was the bug.

def test_a_collapsed_check_reports_nothing_rather_than_minus_infinity(monkeypatch,
                                                                      tiny_batches,
                                                                      tiny_settings):
    """-inf poisons the pruner's median for EVERY OTHER TRIAL. Say nothing instead."""
    import math

    reported = []

    class Spy:
        def report(self, value, step):
            reported.append(value)

        def should_prune(self):
            return False

    # A scorer that always says "collapsed" — exactly what an early check sees.
    collapsed = {"score": -math.inf, "direction": float("nan"),
                 "volatility": float("nan"), "before_quant": float("nan"),
                 "words_used": 1, "why": "codebook collapsed: 1 of 16 words"}

    tuning._run_one("ts", {**tiny_settings["ts"], "learning_rate": 1e-3, "model_size": 16},
                    tiny_batches, tiny_settings, lambda *a, **k: dict(collapsed),
                    features=26, companies=3, trial=Spy())

    assert reported == [] or all(math.isfinite(v) for v in reported), (
        f"reported non-finite values to the pruner: {reported}. np.nanpercentile turns "
        "those into NaN and the pruner stops working for every trial in the study."
    )


# ──────────────────────────────────────── a probe that refuses to answer
#
# The first real GPU run reported, for CS:
#
#     direction (token)      +0.561
#     before_quant (source)  -4.164
#
# That is IMPOSSIBLE. `before_quant` probes the CONTINUOUS vector the token was quantised
# FROM, and quantising can only ever destroy information -- a token cannot carry more than
# its own source. The token "beat" the vector it came from by 4.7 R2.
#
# The cause is not the model, it is the probe. CS has ONE grid per DAY (~600 tune rows, so
# ~420 to fit on), and the continuous summary is up to 256 wide. That probe has as many
# free parameters as it has rows: on a target it knows NOTHING about it scores R2 = -2.38.
# It was measuring its own overfitting and we were printing it as if it meant something.
#
# No choice of ridge honestly rescues that -- it would just move the arbitrariness into a
# knob. The honest thing is to say "I cannot answer this" and say why.

def test_the_continuous_probe_refuses_to_answer_when_it_has_too_few_rows():
    """A number we cannot support is worse than no number: it gets quoted."""
    import math

    rng = np.random.default_rng(0)
    y = rng.normal(size=(600, 2))

    thin = rng.normal(size=(600, 256))       # CS: 420 fitting rows, 256 columns
    assert math.isnan(tuning.skill(thin, y)), (
        "answered with 256 columns and 420 rows to fit them on — that is overfitting, "
        "not information, and it produced an impossible -4.164 on the real run"
    )

    fat = rng.normal(size=(600, 16))         # plenty of rows per column: answer away
    assert math.isfinite(tuning.skill(fat, y))


def test_a_one_hot_token_is_still_answered_because_it_is_not_degenerate():
    """The guard must not silence the token probe. A 1024-word one-hot has 1024 COLUMNS but
    only `words_used` free parameters -- ~90 on the real CS run -- so it is well posed where
    a 256-wide dense vector is not. This is why CS's `direction` is trustworthy and its
    `before_quant` was not."""
    import math

    rng = np.random.default_rng(1)
    ids = rng.integers(0, 90, size=600)
    y = rng.normal(size=(600, 2))
    answer, _noise = tuning.skill_from_ids(ids, words=1024, y=y)
    assert math.isfinite(answer)


# ────────────────────────────────── the transfer guard was running BACKWARDS
#
# Found by comparing two real GPU runs (steps=300, then steps=600). `confirm()` exists to
# ask: "this config won a SHORT 600-step sprint -- does it still win the REAL, LONGER run?"
#
# It was handing `_run_one` the shared `settings["steps"]`. Two things wrong with that:
#
#   1. It IGNORES the entry's own budget. CS is really trained for `cs["steps"]` = 2000
#      steps -- it has 30x less data than TS and needs them -- while `train()` has always
#      honoured that. So `confirm()` was validating CS at a SEVENTH of the budget CS is
#      actually trained at, which is exactly the failure it was built to catch.
#
#   2. Nothing stopped it being SMALLER than the sprint. The notebook ships `steps = 300`
#      against `search["steps"] = 600`, so the "full budget" confirm trained for HALF as
#      long as the trials it was meant to validate. A guard that re-runs the sprint SHORTER
#      is not a guard.

def test_confirm_trains_each_entry_at_the_budget_it_will_REALLY_be_trained_at(monkeypatch,
                                                                              tiny_batches,
                                                                              tiny_settings):
    """CS has its own step budget because it has 30x less data. `train()` honours it.
    `confirm()` must honour the SAME rule, or it is not testing the run that will happen."""
    seen = []

    def spy(entry, chosen, batches, settings, scorer, features, companies, trial=None):
        seen.append(settings["search"]["steps"])
        return {"score": 0.5, "direction": 0.1, "volatility": 0.4, "words_used": 9,
                "before_quant": 0.0, "why": ""}

    monkeypatch.setattr(tuning, "_run_one", spy)

    settings = {**tiny_settings,
                "steps": 40,                                   # the shared budget
                "cs": {**tiny_settings["cs"], "steps": 900},   # ...but CS has its OWN
                "search": {**tiny_settings["search"], "steps": 10}}

    tuning.confirm("cs", {**settings["cs"], "learning_rate": 1e-3, "model_size": 16},
                   tiny_batches, settings)

    assert seen == [900, 900], (
        f"confirm() trained CS at {seen}, not at cs['steps'] = 900 — the budget CS is "
        "ACTUALLY trained at. It was validating a config on a run that never happens."
    )


def test_a_confirm_shorter_than_the_sprint_is_rejected():
    """A guard that re-runs the sprint SHORTER than the sprint is not a guard, and shipping
    one that silently reverses its own meaning is worse than shipping none."""
    import pytest

    from bubble_bi.settings import check

    with pytest.raises(ValueError, match="(?i)shorter than the sprint"):
        check({"tickers": ["AAPL"],
               "steps": 300,                                     # the real run
               "search": {"run": True, "trials": 12, "steps": 600}})   # the sprint. LONGER.


def test_a_short_run_is_still_fine_when_the_search_is_OFF():
    """steps=300 is a perfectly good laptop read-through. It only becomes a lie when the
    search is on, because only then does anything claim to validate against it."""
    from bubble_bi.settings import check

    check({"tickers": ["AAPL"], "steps": 300,
           "search": {"run": False, "trials": 12, "steps": 600}})     # must not raise


def test_apply_never_treats_its_OWN_PREVIOUS_OUTPUT_as_something_you_typed(tmp_path):
    """⚠️ THE BUG THAT THREW AWAY A 60-TRIAL GPU RUN.

    The notebook did `SETTINGS, note = bb.tuning.apply(SETTINGS)` — REASSIGNING the very
    dict it hands back in. Precedence is DEFAULTS < tuned.json < what you TYPED, so on the
    next execution the *previous* run's tuned values are sitting in `SETTINGS` and `apply()`
    reads them as things the user typed. Typed wins. The new tuning is silently discarded.

    Observed for real: a 60-trial search found ts.days=5, model_size=128, wrote them to
    tuned.json — and the notebook went on to train days=10, width=64, the answer from the
    PREVIOUS run, because the previous apply() had baked them into SETTINGS.

    A search whose own output overrides its next output is a search that can only ever be
    run once.
    """
    import json

    path = tmp_path / "tuned.json"

    def search_writes(days, size):
        path.write_text(json.dumps({
            "found_on": "2026-07-13", "trials": 60,
            "fingerprint": {"tickers": 1, "features": 26, "start": None,
                            "search_steps": 600},
            "score": {}, "ts": {"days": days, "model_size": size}, "cs": {}}))

    typed = {"tickers": ["AAPL"]}          # the user typed NOTHING about days or model_size

    search_writes(10, 64)                  # an earlier run
    first, _ = tuning.apply(typed, path=path)
    assert first["ts"]["days"] == 10

    search_writes(5, 128)                  # a NEW, better run
    second, _ = tuning.apply(typed, path=path)      # `typed` is untouched — as it must be

    assert second["ts"]["days"] == 5, "the new tuning was overridden by the old one"
    assert second["model_size"] == 128

    # And the reason it works: apply() must not MUTATE what it was given. If it did, the
    # notebook's `SETTINGS` would quietly accumulate every run's answers as "typed".
    assert "ts" not in typed, "apply() mutated the caller's settings — that IS the bug"


def test_a_config_already_tried_is_never_retrained(tiny_batches, tiny_settings, monkeypatch):
    """⚠️ 40% OF A 60-TRIAL GPU RUN WAS SPENT RE-TRAINING CONFIGS IT HAD ALREADY TRIED.

    Real numbers: TS produced ELEVEN bit-identical rows (score 0.537, 232 words) and CS
    produced THIRTEEN (1.846, 62 words). TPE converges on a good region and, in a discrete
    `sizes` space, simply re-suggests the same corner of it. Optuna does not stop it.

    Seeding the training (so the two sides of a confirm start alike) made this WORSE, not
    better: the duplicates became bit-identical rather than merely near-identical, so the
    same GPU minutes bought a number we already had, to the last decimal.

    We do not leave this to chance in the test: the sampler is pinned so EVERY trial asks
    for the same configuration. Without de-duplication that is N trainings for one answer.
    """
    trained = []
    real_run_one = tuning._run_one

    def counting(entry, chosen, batches, settings, scorer, features, companies, trial=None):
        trained.append(1)
        return real_run_one(entry, chosen, batches, settings, scorer,
                            features, companies, trial)

    # Pin the sampler: every trial asks for the SAME thing, so every trial after the first
    # is a duplicate. This is the pathological case the real run kept walking into.
    def always_the_same(trial, name, rule):
        kind = rule[0]
        if kind == "pick":
            return rule[1][0]
        return float(rule[1])

    monkeypatch.setattr(tuning, "_run_one", counting)
    monkeypatch.setattr(tuning, "_ask", always_the_same)

    settings = {**tiny_settings, "search": {**tiny_settings["search"], "trials": 8}}
    _best, trials = tuning.search("ts", tiny_batches, settings)

    # Two stages, one distinct configuration each -> at most two trainings, ever.
    assert len(trained) <= 2, (
        f"trained {len(trained)} times for 2 distinct configurations — "
        f"{len(trained) - 2} of those bought a score we already had"
    )
    # ...and the duplicates must still SHOW UP in the table, with the score they earned.
    assert len(trials) == 8, "de-duplicating must not lose the trials from the record"
    assert trials["score"].nunique() <= 2
