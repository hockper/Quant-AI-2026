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


def _fusion_settings(vocabulary=32, depth=1, attend_to="days", model_size=32,
                     commitment=0.25, diversity=0.1, decay=0.99):
    """A minimal settings dict -- just enough for `Tokenizer`, which reads `fusion`
    and `model_size` straight out of settings rather than taking them as separate
    keyword arguments (see world.py). `model_size` must match `_pair()`'s `width`,
    since that is what the fused vector -- and so the codebook -- actually is."""
    return {
        "model_size": model_size,
        "fusion": {
            "vocabulary": vocabulary, "depth": depth, "attend_to": attend_to,
            "commitment": commitment, "diversity": diversity, "decay": decay,
        },
    }


def _sentence(batch=2, days=8, width=32, keys=3):
    """A sentence, straight from the cache — which is how the world model is fed."""
    return {
        "z_ts": torch.randn(batch, days, width),          # what the company was doing
        "market": torch.randn(batch, days, keys, width),  # what the market was offering
        "candle": torch.randn(batch, days, 4),            # the candle each day really was
    }


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

def test_the_encoders_are_frozen_and_stay_frozen():
    ts, cs = _pair()
    tok = Tokenizer(ts, cs, _fusion_settings(), heads=2, dropout=0.0)

    assert not any(p.requires_grad for p in tok.ts.parameters())
    assert not any(p.requires_grad for p in tok.cs.parameters())
    assert any(p.requires_grad for p in tok.fusion.parameters())

    before = ts.read.weight.clone()
    out = tok(torch.randn(4, 1, 4, FEATURES), torch.randn(4, COMPANIES, 3, FEATURES))
    (out["commitment_loss"] + out["vector"].sum()).backward()
    assert ts.read.weight.grad is None                  # no gradient reached them
    assert torch.equal(ts.read.weight, before)          # and they did not move


def test_one_token_per_company_day():
    ts, cs = _pair()
    tok = Tokenizer(ts, cs, _fusion_settings(), heads=2, dropout=0.0)
    out = tok(torch.randn(9, 1, 4, FEATURES), torch.randn(9, COMPANIES, 3, FEATURES))
    assert out["token"].shape == (9,)
    assert out["token"].max() < 32


# ------------------------------------------------------------------ the predictor

def test_the_predictor_cannot_see_tomorrow():
    """THE test.

    Everything upstream is causal -- the features, the windows, the splits. If the GPT
    could read the token it is being asked to predict, all of that would be for
    nothing, and the model would look brilliant while knowing nothing.

    So: change the FUTURE of a sentence and check that what the model thought about the
    PAST did not move.
    """
    torch.manual_seed(0)
    ts, cs = _pair()
    world = WorldModel(Tokenizer(ts, cs, _fusion_settings(), heads=2, dropout=0.0),
                       sentence=8, depth=2, heads=2, dropout=0.0).eval()

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

def test_the_world_model_turns_a_sentence_into_tokens_and_guesses_the_next():
    ts, cs = _pair()
    world = WorldModel(Tokenizer(ts, cs, _fusion_settings(), heads=2, dropout=0.0),
                       sentence=8, depth=2, heads=2, dropout=0.0)
    out = world(_sentence(batch=3, days=8))

    assert out["tokens"].shape == (3, 8)              # one token per day
    assert out["attention"].shape == (3, 8, 3)        # ...and what each one read
    assert out["drawn"].shape == (3, 7, 4)            # tomorrow's candle, for each day
    assert torch.isfinite(out["loss"])
    assert 0.0 <= float(out["accuracy"]) <= 1.0


def test_the_prediction_loss_reaches_back_into_the_fusion():
    """The whole reason fusion and predictor train together: the token must be shaped
    by whether it is PREDICTABLE, not by whether it can be redrawn."""
    ts, cs = _pair()
    world = WorldModel(Tokenizer(ts, cs, _fusion_settings(), heads=2, dropout=0.0),
                       sentence=8, depth=2, heads=2, dropout=0.0)
    out = world(_sentence())
    out["naming_loss"].backward()

    reached = [p.grad is not None and p.grad.abs().sum() > 0
               for p in world.tokenizer.fusion.parameters() if p.requires_grad]
    assert any(reached), "the predictor's loss never reached the cross-attention"


def test_it_can_actually_learn_to_predict_the_next_token():
    # A sentence that repeats itself. The model must do better than chance (1/32).
    torch.manual_seed(0)
    ts, cs = _pair()
    world = WorldModel(Tokenizer(ts, cs, _fusion_settings(), heads=2, dropout=0.0),
                       sentence=12, depth=2, heads=2, dropout=0.0).train()

    batch = _sentence(batch=4, days=12)
    opt = torch.optim.AdamW(
        [p for p in world.parameters() if p.requires_grad], lr=3e-3)

    first = float(world(batch)["naming_loss"])
    for _ in range(200):
        out = world(batch)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    last = float(world(batch)["naming_loss"])

    assert last < first * 0.7, f"barely learned: {first:.2f} -> {last:.2f}"


# ------------------------------------------------- the candle head, and the cheat

def test_the_model_is_asked_to_draw_tomorrows_candle():
    ts, cs = _pair()
    world = WorldModel(Tokenizer(ts, cs, _fusion_settings(), heads=2, dropout=0.0),
                       sentence=8, depth=2, heads=2, dropout=0.0)
    out = world(_sentence(batch=3, days=8))

    assert out["drawn"].shape == (3, 7, 4)          # gap, body, and the two wicks
    assert "drawing_loss" in out and "shrugging" in out
    # `shrugging` is what you score by drawing the AVERAGE candle. It is what makes
    # `drawing_loss` mean something on its own.
    assert float(out["shrugging"]) > 0


def test_the_candle_loss_reaches_the_fusion_so_an_empty_token_is_punished():
    """The whole point of the candle head. A collapsed token carries no information, so
    it cannot draw a candle -- but only if the drawing loss actually reaches back into
    the fusion that produced the token."""
    ts, cs = _pair()
    world = WorldModel(Tokenizer(ts, cs, _fusion_settings(), heads=2, dropout=0.0),
                       sentence=8, depth=2, heads=2, dropout=0.0)
    world(_sentence())["drawing_loss"].backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in world.tokenizer.fusion.parameters())


def test_the_loss_weights_are_obeyed():
    ts, cs = _pair()
    world = WorldModel(Tokenizer(ts, cs, _fusion_settings(), heads=2, dropout=0.0),
                       sentence=8, depth=2, heads=2, dropout=0.0,
                       naming=0.1, candle=1.0)
    out = world(_sentence())
    total = (0.1 * out["naming_loss"] + 1.0 * out["drawing_loss"]
             + out["commitment_loss"] + out["diversity_loss"])
    assert torch.allclose(out["loss"], total)


def test_without_a_candle_the_model_still_runs_but_says_so():
    # No candle in the batch -> no drawing loss. The model must not silently invent one.
    ts, cs = _pair()
    world = WorldModel(Tokenizer(ts, cs, _fusion_settings(), heads=2, dropout=0.0),
                       sentence=8, depth=2, heads=2, dropout=0.0)
    batch = _sentence()
    del batch["candle"]
    out = world(batch)
    assert "drawing_loss" not in out
    assert torch.isfinite(out["loss"])


def test_the_fusion_codebook_carries_the_diversity_penalty():
    # The anti-collapse term must be on the FUSION's codebook, not just TS and CS.
    ts, cs = _pair()
    tok = Tokenizer(ts, cs, _fusion_settings(commitment=1.0, diversity=0.5),
                    heads=2, dropout=0.0)
    assert tok.codebook.commitment == 1.0
    assert tok.codebook.diversity == 0.5

    world = WorldModel(tok, sentence=8, depth=2, heads=2, dropout=0.0)
    out = world(_sentence())
    assert float(out["diversity_loss"]) != 0.0

    out["diversity_loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in tok.fusion.parameters())


def test_the_fusion_codebook_gets_its_own_knobs():
    """The fusion codebook is the one that COLLAPSES to ~12 words. It had been running on
    class defaults, unexamined, because nothing passed it anything."""
    from bubble_bi.models.world import Tokenizer
    from bubble_bi.settings import DEFAULTS

    settings = {
        **{k: v for k, v in DEFAULTS.items() if k != "tickers"},
        "tickers": ["AAA"],
        "fusion": {**DEFAULTS["fusion"], "commitment": 0.8, "diversity": 0.6, "decay": 0.5},
    }
    ts = VQVAE(companies=1, days=4, features=6, width=16, heads=2)
    cs = VQVAE(companies=3, days=4, features=6, width=16, heads=2)
    tokenizer = Tokenizer(ts, cs, settings)

    assert tokenizer.codebook.commitment == 0.8
    assert tokenizer.codebook.diversity == 0.6
    assert tokenizer.codebook.decay == 0.5
