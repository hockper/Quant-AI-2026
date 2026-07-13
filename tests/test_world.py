import pytest
import torch

from bubble_bi.models import VQVAE, Block, Fusion, Rotary, SwiGLU, Tokenizer, WorldModel

FEATURES, COMPANIES = 6, 5


def _pair(width=32, heads=2):
    ts = VQVAE(companies=1, days=4, features=FEATURES, vocabulary=32, width=width,
               heads=heads, dropout=0.0)
    cs = VQVAE(companies=COMPANIES, days=3, features=FEATURES, vocabulary=32,
               width=width, heads=heads, dropout=0.0)
    return ts, cs


# ------------------------------------------------------------------ cross-attention

def test_the_output_is_as_long_as_the_query_not_the_keys():
    """The rule the whole design rests on. Many keys in, ONE vector out."""
    fusion = Fusion(width=32, depth=2, heads=2, dropout=0.0)
    company = torch.randn(7, 32)               # ONE query per company-day
    for keys in (3, 30, 150):                  # however much market we offer...
        market = torch.randn(7, keys, 32)
        fused, weights = fusion(company, market)
        assert fused.shape == (7, 32)          # ...we always come out with one vector
        assert weights.shape == (7, keys)      # and one weight per key


def test_the_attention_weights_are_a_real_choice():
    fusion = Fusion(width=32, depth=1, heads=2, dropout=0.0).eval()
    _, weights = fusion(torch.randn(4, 32), torch.randn(4, 6, 32))
    assert torch.allclose(weights.sum(-1), torch.ones(4), atol=1e-5)   # they are shares
    assert (weights >= 0).all()


def test_the_residual_keeps_two_companies_apart():
    """Without it, the fused vector is a blend of MARKET vectors only -- so two
    companies reading the same market the same way would get the SAME token, and their
    own identity would vanish."""
    fusion = Fusion(width=32, depth=1, heads=2, dropout=0.0).eval()
    market = torch.randn(1, 5, 32).expand(2, 5, 32)     # the same market for both
    different = torch.randn(2, 32)                       # ...but different companies

    with torch.no_grad():
        fused, _ = fusion(different, market)
    assert not torch.allclose(fused[0], fused[1], atol=1e-3)


def test_a_single_market_vector_would_be_a_no_op():
    # Why CS must never hand over just one summary: softmax over one key is 1.0, so
    # every company would receive an identical market vector.
    fusion = Fusion(width=32, depth=1, heads=2, dropout=0.0).eval()
    _, weights = fusion(torch.randn(4, 32), torch.randn(4, 1, 32))
    assert torch.allclose(weights, torch.ones(4, 1), atol=1e-6)   # no choice at all


# --------------------------------------------------------------- what CS offers

@pytest.mark.parametrize("how,keys", [("days", 3), ("companies", COMPANIES),
                                      ("cells", COMPANIES * 3)])
def test_cs_can_offer_the_market_at_three_granularities(how, keys):
    _, cs = _pair()
    grid = torch.randn(4, COMPANIES, 3, FEATURES)
    assert cs.context(grid, how=how).shape == (4, keys, 32)


def test_an_unknown_granularity_is_rejected_with_a_useful_message():
    _, cs = _pair()
    with pytest.raises(ValueError, match="must be 'days', 'companies' or 'cells'"):
        cs.context(torch.randn(2, COMPANIES, 3, FEATURES), how="whatever")


def test_a_company_that_did_not_trade_is_left_out_of_the_market():
    _, cs = _pair()
    grid = torch.randn(4, COMPANIES, 3, FEATURES)
    present = torch.ones(4, COMPANIES, dtype=torch.bool)
    present[:, -1] = False

    with torch.no_grad():
        clean = cs.context(grid, present, how="days")
        poisoned = grid.clone()
        poisoned[:, -1] = 999.0                        # garbage in the absent company
        assert torch.allclose(clean, cs.context(poisoned, present, how="days"), atol=1e-4)


# ------------------------------------------------------------------ the tokenizer
#
# ⚠️ These tests describe TWO anchored codebooks (TS + CS), not the old fused-codebook
# design. The fused codebook was the ONE we watched collapse -- 10 words of 512, while TS
# (which rebuilds its own grid) sat at 157. It is deleted, not patched, and so are the old
# tests that asserted its existence (`tok.codebook`), a single `token`/`vector` per
# company-day, and encoders frozen into scaffolding -- under joint training nothing is
# frozen any more, that is the entire point of training everything against one loss.
#
# `_tokenizer_pair()` is a SEPARATE fixture from the top-level `_pair()` above (which the
# "what CS offers" tests still rely on, at its own companies/features/vocabulary): giving
# it a different name avoids a silent module-level shadowing bug, where the second
# `def _pair():` would win for every caller regardless of where in the file it sits.


def _tokenizer_pair():
    ts = VQVAE(companies=1, days=2, features=26, width=32, heads=2, vocabulary=16)
    cs = VQVAE(companies=4, days=1, features=26, width=32, heads=2, vocabulary=16)
    return ts, cs


def test_a_mismatched_model_size_is_rejected_with_a_useful_message():
    """The fusion is built from the encoders' ACTUAL width; `model_size` is a number typed
    in by hand. If they ever disagreed, the failure used to be a bare tensor-shape error
    deep inside the cross-attention. Catch it at the door instead, with a message that
    names the setting to fix."""
    ts, cs = _tokenizer_pair()
    with pytest.raises(ValueError, match="TS/CS encoders were built 32 wide"):
        Tokenizer(ts, cs, model_size=16, heads=2, dropout=0.0)


def test_the_tokenizer_has_no_codebook_of_its_own():
    """The fused codebook is the ONE we watched collapse -- 10 words of 512, while TS (which
    rebuilds its own grid) sat at 157. We delete it rather than keep patching it. TS and CS
    keep theirs, and theirs are ANCHORED."""
    ts, cs = _tokenizer_pair()
    tok = Tokenizer(ts, cs, model_size=32, attend_to="companies")

    assert not hasattr(tok, "codebook") or tok.codebook is None
    assert tok.ts.codebook is ts.codebook
    assert tok.cs.codebook is cs.codebook


def test_it_speaks_TWO_words_a_day_and_anchors_both():
    ts, cs = _tokenizer_pair()
    tok = Tokenizer(ts, cs, model_size=32, attend_to="companies")
    out = tok(torch.randn(5, 1, 2, 26), torch.randn(5, 4, 1, 26))

    assert out["ts_token"].shape == (5,)
    assert out["cs_token"].shape == (5,)
    assert torch.isfinite(out["recon_loss"])          # rebuilds BOTH grids
    assert out["attention"].shape[1] == 4             # one key per company


def test_the_ts_token_actually_READS_the_market():
    """The cross-attention must sit BEFORE the TS codebook. After a 9-bit quantisation the
    detail it needs is already destroyed, so this is the only place it can happen -- and if
    changing the market does not change the TS vector, the fusion is decorative."""
    torch.manual_seed(0)
    ts, cs = _tokenizer_pair()
    tok = Tokenizer(ts, cs, model_size=32, attend_to="companies").eval()

    stock = torch.randn(4, 1, 2, 26)
    with torch.no_grad():
        calm = tok(stock, torch.zeros(4, 4, 1, 26))["ts_vector"]
        panic = tok(stock, torch.randn(4, 4, 1, 26) * 5)["ts_vector"]

    assert not torch.allclose(calm, panic, atol=1e-4), (
        "the same stock, in two completely different markets, produced the same TS vector — "
        "the cross-attention is doing nothing"
    )


# ------------------------------------------------------------------ the predictor

def test_the_predictor_cannot_see_tomorrow():
    """THE test.

    Everything upstream is causal -- the features, the windows, the splits. If the GPT
    could read the token it is being asked to predict, all of that would be for
    nothing, and the model would look brilliant while knowing nothing.

    So: change the FUTURE of a sentence and check that what the model thought about the
    PAST did not move. `understand()` works on plain [batch, T, width] token vectors --
    it does not care whether that batch axis is companies, sentences, or both, so this
    test needs nothing beyond a real `WorldModel` to call it on.
    """
    torch.manual_seed(0)
    ts, cs = _pair()
    tok = Tokenizer(ts, cs, model_size=32, attend_to="companies", heads=2, dropout=0.0)
    world = WorldModel(tok, sentence=8, depth=2, heads=2, dropout=0.0).eval()

    vectors = torch.randn(2, 8, 32)
    with torch.no_grad():
        before = world.understand(vectors)
        tampered = vectors.clone()
        tampered[:, 5:] = torch.randn(2, 3, 32) * 10       # rewrite the future
        after = world.understand(tampered)

    assert torch.allclose(before[:, :5], after[:, :5], atol=1e-5)   # the past is intact
    assert not torch.allclose(before[:, 5:], after[:, 5:], atol=1e-3)  # the future moved


def test_grouped_query_attention_shares_key_heads():
    block = Block(width=32, heads=4, kv_heads=2)
    assert block.attend.share == 2                    # 2 query heads per key/value head
    assert block(torch.randn(2, 6, 32)).shape == (2, 6, 32)


def test_heads_must_divide_evenly_into_kv_heads():
    with pytest.raises(ValueError, match="must divide evenly"):
        Block(width=32, heads=4, kv_heads=3)


def test_rotary_encodes_distance_not_position():
    # Two days three apart should look the same wherever they sit in the window --
    # that is the point of rotating rather than adding a position.
    rotary = Rotary(head_width=8)
    cos, sin = rotary(10, torch.device("cpu"))
    assert cos.shape == (1, 1, 10, 8)
    assert torch.allclose(cos[0, 0, 0], torch.ones(8))     # position 0 = no rotation


def test_swiglu_keeps_the_width():
    assert SwiGLU(32)(torch.randn(2, 5, 32)).shape == (2, 5, 32)


# ------------------------------------------------------------------ the whole thing
#
# A batch is one time WINDOW across ALL companies at once, not one company's sentence.
# The CS grid is identical for every company on a day, so `_tiny_world` gives it NO
# company axis of its own outside the grid -- `cs_grid` is [B, T, companies, cs_days, F],
# one copy per day, which every one of that day's companies then reads.

def _tiny_world(sentence: int = 6, companies: int = 4, batch: int = 2, **loss_weights):
    """A whole joint model over a tiny synthetic sentence. CPU, milliseconds."""
    from bubble_bi.models.world import Tokenizer, WorldModel

    torch.manual_seed(0)
    ts = VQVAE(companies=1, days=2, features=26, width=32, heads=2, vocabulary=16)
    cs = VQVAE(companies=companies, days=1, features=26, width=32, heads=2, vocabulary=16)
    tok = Tokenizer(ts, cs, model_size=32, attend_to="companies")
    world = WorldModel(tok, sentence=sentence, depth=1, heads=2, **loss_weights)

    data = {
        # TS: one grid per COMPANY per day.
        "ts_grid": torch.randn(batch, sentence, companies, 1, 2, 26),
        # CS: ONE grid per day -- no separate axis per company, the `companies` dimension
        # here is INSIDE the market grid itself (it is what CS was built to read).
        "cs_grid": torch.randn(batch, sentence, companies, 1, 26),
        "cs_present": torch.ones(batch, sentence, companies, dtype=torch.bool),
        "candle": torch.randn(batch, sentence, companies, 4),
    }
    return world, data


def test_the_naming_loss_CANNOT_touch_the_vocabulary():
    """⚠️ THE TEST THIS WHOLE DESIGN EXISTS FOR.

    The model invents its own words and is then graded on guessing them:

        naming = CrossEntropy( GPT(z1..zt),  id(z_{t+1}) )
                               └ the model's ┘ └ ALSO the model's ┘

    Two ways to drive that to zero: learn real dynamics (hard), or make every day the
    same word (trivial). Gradient descent takes the second, and it is a STABLE fixed
    point -- once the vocabulary is dead nothing pulls it back. We measured it: 92%
    accuracy at perplexity 2.2.

    So the naming head reads a DETACHED copy of the token vectors. It can make the GPT
    better at predicting the language; it can never make the language easier. This test
    asserts the gradient path is severed -- if it ever reconnects, the codebook dies and
    the loss curve will look FINE while it happens.
    """
    world, batch = _tiny_world()
    world.zero_grad()
    world(batch)["naming_loss"].backward()

    guilty = [name for name, p in world.tokenizer.named_parameters()
              if p.grad is not None and p.grad.abs().sum() > 0]
    assert not guilty, (
        f"the naming loss reached the tokenizer through {guilty[:3]} — it can now buy a "
        "cheap win by collapsing the vocabulary, and it will"
    )


def test_the_anchor_DOES_reach_the_vocabulary():
    """The counterpart. Reconstruction must reach the encoders and codebooks -- it is the
    only thing holding the words apart. A severed naming channel is worthless if the anchor
    is severed too."""
    world, batch = _tiny_world()
    world.zero_grad()
    world(batch)["recon_loss"].backward()

    reached = [name for name, p in world.tokenizer.named_parameters()
               if p.grad is not None and p.grad.abs().sum() > 0]
    assert reached, "reconstruction reaches nothing — the codebooks have no anchor at all"


def test_the_candle_loss_DOES_reach_the_vocabulary():
    """This is the whole point of joint training: the FORECAST shapes the token. If the
    prediction loss cannot reach the tokenizer, we have simply rebuilt the two-stage model
    with extra steps."""
    world, batch = _tiny_world()
    world.zero_grad()
    world(batch)["drawing_loss"].backward()

    reached = [n for n, p in world.tokenizer.named_parameters()
               if p.grad is not None and p.grad.abs().sum() > 0]
    assert reached, "the forecast cannot shape the token — joint training is doing nothing"


def test_it_reports_persistence_and_shrugging_beside_its_own_scores():
    """A number without its floor gets quoted. Persistence is 'tomorrow's word is today's
    word'; shrugging is 'draw the average candle'."""
    world, batch = _tiny_world()
    out = world(batch)
    assert torch.isfinite(out["persistence"])
    assert torch.isfinite(out["shrugging"])


def test_the_world_model_runs_over_every_company_at_once():
    """The shape contract the whole design rests on: ONE window, ALL companies. Each
    company must get its OWN sentence (and so its own accuracy/candle contribution), not
    be pooled into a single anonymous sequence."""
    companies, sentence, batch = 5, 6, 3
    world, data = _tiny_world(sentence=sentence, companies=companies, batch=batch)
    out = world(data)

    assert torch.isfinite(out["loss"])
    assert 0.0 <= float(out["accuracy"]) <= 1.0
    assert 0.0 <= float(out["persistence"]) <= 1.0
    assert out["ts_perplexity"] > 0 and out["cs_perplexity"] > 0
    # One attention weight per (batch, day, company, market-key).
    assert out["attention"].shape == (batch, sentence, companies, companies)


def test_the_market_is_encoded_once_per_day_not_once_per_company_day():
    """⚠️ THE SAVING THE WHOLE TWO-TOKEN DESIGN RESTS ON.

    The CS grid is identical for every company on a day. If `WorldModel.forward` ever
    starts encoding it once per COMPANY-day instead of once per day, training gets ~N
    times slower and nothing about the loss curve would say so -- the model would just
    quietly get much harder to afford. So we count: the CS encoder must be called on
    exactly `B*T` rows, never `B*T*N`.
    """
    companies, sentence, batch = 4, 6, 2
    world, data = _tiny_world(sentence=sentence, companies=companies, batch=batch)

    seen_batches = []
    real_read_grid = world.tokenizer.cs.read_grid.__func__

    def counting_read_grid(self, grid, present=None):
        seen_batches.append(grid.shape[0])
        return real_read_grid(self, grid, present)

    world.tokenizer.cs.read_grid = counting_read_grid.__get__(world.tokenizer.cs)
    world(data)

    assert seen_batches == [batch * sentence], (
        f"the CS encoder saw batches of {seen_batches}, not a single call of "
        f"{batch * sentence} (= B*T) rows -- it is re-encoding the market per company, "
        "which is the ~30x slowdown this whole design exists to avoid"
    )


def test_the_loss_weights_are_obeyed():
    """`predict`/`naming`/`recon` must actually reach the total, not just be accepted."""
    world, batch = _tiny_world(predict=2.0, naming=0.3, recon=0.5)
    out = world(batch)
    total = (2.0 * out["drawing_loss"] + 0.3 * out["naming_loss"]
             + 0.5 * out["recon_loss"] + out["commitment_loss"] + out["diversity_loss"])
    assert torch.allclose(out["loss"], total)


# The two tests that used to live here (`test_the_fusion_codebook_carries_the_diversity_
# penalty`, `test_the_fusion_codebook_gets_its_own_knobs`) asserted `tok.codebook` /
# `tokenizer.codebook` directly -- the fused codebook itself. There is no such attribute
# any more (see `test_the_tokenizer_has_no_codebook_of_its_own` above): TS and CS carry
# their OWN commitment/diversity/decay knobs, already covered by
# `test_the_codebook_knobs_actually_reach_the_codebook` in tests/test_settings.py.
#
# Also gone: `test_the_prediction_loss_reaches_back_into_the_fusion` and
# `test_the_candle_loss_reaches_the_fusion_so_an_empty_token_is_punished`. The first
# asserted that NAMING reaches the fusion -- exactly the collapse channel this file now
# severs on purpose (see `test_the_naming_loss_CANNOT_touch_the_vocabulary` above). The
# second is superseded by `test_the_candle_loss_DOES_reach_the_vocabulary`, which checks
# the same thing against the tokenizer as a whole rather than the fusion alone.
# `test_it_can_actually_learn_to_predict_the_next_token`,
# `test_the_model_is_asked_to_draw_tomorrows_candle` and
# `test_without_a_candle_the_model_still_runs_but_says_so` all built batches out of
# cached `z_ts`/`market` tensors and read `out["tokens"]`/`out["drawn"]` -- a single
# fused token stream that no longer exists. `candle` is no longer optional either: it is
# what every sentence carries now, not an extra a caller could omit.
