import pytest
import torch

from bubble_bi.models import VQVAE, Codebook

torch.manual_seed(0)


def _grid(batch=8, companies=3, days=4, features=6):
    return torch.randn(batch, companies, days, features)


# ------------------------------------------------------------------ the codebook

def test_every_vector_gets_snapped_to_a_real_word():
    # eval mode: in training the dictionary drifts DURING the call, so comparing
    # against it afterwards would be comparing against a different dictionary.
    book = Codebook(words=16, width=8).eval()
    out = book(torch.randn(32, 8))
    assert out["ids"].shape == (32,)
    assert out["ids"].min() >= 0 and out["ids"].max() < 16
    # the snapped vector must actually BE one of the words (the straight-through trick
    # leaves the forward value untouched)
    assert torch.allclose(out["snapped"], book.dictionary[out["ids"]], atol=1e-5)


def test_the_dictionary_only_moves_while_training():
    book = Codebook(words=16, width=8).eval()
    before = book.dictionary.clone()
    book(torch.randn(32, 8))
    assert torch.equal(book.dictionary, before)      # frozen at inference

    book.train()
    book(torch.randn(32, 8))
    assert not torch.equal(book.dictionary, before)  # learning while training


def test_gradient_survives_the_snap():
    # Snapping to the nearest word is a step function -- it has no gradient. The
    # straight-through trick is what lets the encoder learn at all. If this breaks,
    # the encoder silently never trains.
    book = Codebook(words=16, width=8)
    z = torch.randn(4, 8, requires_grad=True)
    book(z)["snapped"].sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_the_moving_average_update_does_not_corrupt_the_gradient():
    # The dictionary is updated IN PLACE while training. Without cloning it first,
    # PyTorch raises (or worse, silently computes the wrong gradient).
    book = Codebook(words=16, width=8).train()
    z = torch.randn(32, 8, requires_grad=True)
    out = book(z)
    (out["commitment_loss"] + out["snapped"].sum()).backward()   # must not raise
    assert torch.isfinite(z.grad).all()


def test_the_dictionary_learns_where_the_data_actually_is():
    # Feed two tight clusters. The words that win should move ONTO those clusters.
    book = Codebook(words=8, width=2, decay=0.9).train()
    left = torch.tensor([[-5.0, -5.0]]).repeat(64, 1)
    right = torch.tensor([[5.0, 5.0]]).repeat(64, 1)
    data = torch.cat([left, right])
    for _ in range(200):
        book(data + 0.01 * torch.randn_like(data))

    chosen = book.dictionary[book(data)["ids"].unique()]
    assert torch.cdist(chosen, torch.tensor([[-5.0, -5.0], [5.0, 5.0]])).min(0).values.max() < 0.5


def test_perplexity_reports_a_collapsed_dictionary():
    book = Codebook(words=64, width=4)
    same = torch.zeros(100, 4)                  # every input identical -> one word
    assert book(same)["perplexity"].item() == pytest.approx(1.0, abs=0.01)


def test_dead_words_are_revived_onto_real_data():
    book = Codebook(words=32, width=4).train()
    book.usage.zero_()                          # nobody is using anything
    revived = book.revive_dead_words(torch.randn(50, 4))
    assert revived == 32
    assert (book.usage > 0).all()


# ---------------------------------------------------------------------- the VQVAE

def test_one_class_serves_as_both_ts_and_cs():
    ts = VQVAE(companies=1, days=4, features=6, vocabulary=32, width=16, heads=2)
    cs = VQVAE(companies=30, days=5, features=6, vocabulary=32, width=16, heads=2)

    ts_out = ts({"grid": _grid(8, 1, 4, 6)})
    cs_out = cs({"grid": _grid(8, 30, 5, 6)})

    for out in (ts_out, cs_out):
        assert out["ids"].shape == (8,)          # ONE word per example, from both
        assert torch.isfinite(out["loss"])


def test_the_rebuilt_grid_has_the_shape_of_the_original():
    model = VQVAE(companies=3, days=4, features=6, vocabulary=32, width=16, heads=2)
    grid = _grid(8, 3, 4, 6)
    assert model.rebuild(model.summarise(grid)).shape == grid.shape


def test_a_wrongly_shaped_grid_is_rejected_with_a_useful_message():
    model = VQVAE(companies=3, days=4, features=6, vocabulary=32, width=16, heads=2)
    with pytest.raises(ValueError, match=r"expects grids of"):
        model({"grid": _grid(8, 5, 4, 6)})       # 5 companies, not 3


def test_missing_companies_are_ignored_not_treated_as_real_data():
    model = VQVAE(companies=4, days=3, features=5, vocabulary=32, width=16, heads=2).eval()
    grid = _grid(6, 4, 3, 5)
    present = torch.ones(6, 4, dtype=torch.bool)
    present[:, 3] = False                        # the 4th company did not trade

    with torch.no_grad():
        base = model.summarise(grid, present)
        noisy = grid.clone()
        noisy[:, 3] = 999.0                      # garbage in the absent company
        after = model.summarise(noisy, present)
    # If the absent company were being read, this garbage would change the summary.
    assert torch.allclose(base, after, atol=1e-5)


def test_the_loss_ignores_companies_that_were_not_there():
    model = VQVAE(companies=3, days=2, features=4, vocabulary=16, width=16, heads=2).eval()
    grid = _grid(5, 3, 2, 4)
    present = torch.ones(5, 3, dtype=torch.bool)
    present[:, 2] = False

    with torch.no_grad():
        a = model({"grid": grid, "present": present})["rebuild_loss"]
        broken = grid.clone()
        broken[:, 2] = 50.0                      # impossible to rebuild -- but absent
        b = model({"grid": broken, "present": present})["rebuild_loss"]
    assert torch.allclose(a, b, atol=1e-5)


def test_it_can_actually_learn_to_rebuild():
    # The real test: on a handful of repeating grids, the rebuild error must fall a
    # long way. If it does not, the encoder/codebook/decoder chain is not connected.
    torch.manual_seed(1)
    model = VQVAE(companies=2, days=3, features=4, vocabulary=16, width=32,
                  encoder_depth=2, decoder_depth=2, heads=2, dropout=0.0).train()
    data = torch.randn(16, 2, 3, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    first = model({"grid": data})["rebuild_loss"].item()
    for step in range(300):
        out = model({"grid": data})
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        if step % 50 == 0:
            model.codebook.revive_dead_words(out["summary"].detach())
    last = model({"grid": data})["rebuild_loss"].item()

    assert last < first * 0.5, f"rebuild error barely moved: {first:.3f} -> {last:.3f}"


def test_describe_tells_the_notebook_what_it_built():
    ts = VQVAE(companies=1, days=4, features=22, vocabulary=512, width=128)
    cs = VQVAE(companies=30, days=5, features=22, vocabulary=512, width=128)
    assert "1 company" in ts.describe()
    assert "all 30 companies" in cs.describe()
    assert "1 word out of 512" in ts.describe()
