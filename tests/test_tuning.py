import math

import numpy as np
import pytest
import torch

from bubble_bi import tuning
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

    This test fails on the naive implementation (skill = plain R2).
    """
    rng = np.random.default_rng(1)
    target = rng.normal(size=(600, 2))
    narrow = tuning.one_hot(rng.integers(0, 128, size=600), words=128)
    wide = tuning.one_hot(rng.integers(0, 1024, size=600), words=1024)

    assert abs(tuning.skill(wide, target) - tuning.skill(narrow, target)) < 0.1


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
