import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.cs_vqvae import CSVQVAE


def _cfg(**kw):
    base = dict(cs_p=3, d_model=16, cs_codebook_size=16, enc_layers=1, dec_layers=1,
                heads=2, ff=32, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


def _batch(B=3, N=5, L=3, D=6):
    block = torch.randn(B, N, L, D)
    valid = torch.ones(B, N, dtype=torch.bool)
    valid[0, 4] = False
    return {"block": block, "valid": valid}


def test_forward_shapes_and_loss():
    model = CSVQVAE(_cfg(), d_in=6, n_stocks=5)
    out = model(_batch())
    assert out["recon"].shape == (3, 5, 3, 6)
    assert out["ids"].shape == (3,)               # one market token per day
    assert torch.isfinite(out["loss"])


def test_invalid_stock_does_not_change_loss():
    torch.manual_seed(0)
    model = CSVQVAE(_cfg(), d_in=6, n_stocks=5).eval()
    b1 = _batch()
    b2 = {"block": b1["block"].clone(), "valid": b1["valid"].clone()}
    b2["block"][0, 4] = 123.0
    with torch.no_grad():
        assert torch.allclose(model(b1)["recon_loss"], model(b2)["recon_loss"], atol=1e-4)


def test_overfits_tiny_batch():
    torch.manual_seed(0)
    model = CSVQVAE(_cfg(), d_in=6, n_stocks=5).train()
    batch = _batch(B=6)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    # VQ cold-start + a hard one-token→whole-field bottleneck: needs a few thousand
    # steps to get past the initial single-code collapse.
    for _ in range(3000):
        opt.zero_grad()
        out = model(batch)
        out["loss"].backward()
        opt.step()
        if first is None:
            first = out["recon_loss"].item()
    assert out["recon_loss"].item() < 0.5 * first
