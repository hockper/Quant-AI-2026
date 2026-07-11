import torch

from bubble_bi.models.vq import VectorQuantizerEMA


def test_nearest_code_selection():
    vq = VectorQuantizerEMA(num_codes=3, dim=2)
    vq.embed.copy_(torch.tensor([[0.0, 0.0], [10.0, 10.0], [-5.0, -5.0]]))
    z = torch.tensor([[0.1, 0.1], [9.0, 9.0]])
    out = vq(z)
    assert out["ids"].tolist() == [0, 1]


def test_straight_through_gradient_reaches_encoder():
    vq = VectorQuantizerEMA(num_codes=4, dim=3)
    z = torch.randn(5, 3, requires_grad=True)
    out = vq(z)
    out["z_q"].sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_perplexity_extremes():
    vq = VectorQuantizerEMA(num_codes=4, dim=2)
    vq.embed.copy_(torch.tensor([[0.0, 0], [1, 0], [2, 0], [3, 0]]))
    # one distinct sample per code -> perplexity == K
    z_uniform = vq.embed.clone()
    assert torch.isclose(vq(z_uniform)["perplexity"], torch.tensor(4.0), atol=1e-3)
    # all identical -> perplexity == 1
    z_same = torch.zeros(8, 2)
    assert torch.isclose(vq(z_same)["perplexity"], torch.tensor(1.0), atol=1e-3)


def test_ema_update_moves_codebook_in_training():
    torch.manual_seed(0)
    vq = VectorQuantizerEMA(num_codes=2, dim=2, decay=0.5)
    vq.train()
    before = vq.embed.clone()
    for _ in range(5):
        vq(torch.randn(16, 2))
    assert not torch.allclose(before, vq.embed)


def test_orthogonality_loss_is_finite_scalar():
    vq = VectorQuantizerEMA(num_codes=8, dim=4)
    loss = vq.orthogonality_loss()
    assert loss.ndim == 0 and torch.isfinite(loss)
