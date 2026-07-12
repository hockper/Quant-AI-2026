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
