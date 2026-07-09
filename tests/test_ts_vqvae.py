import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.ts_vqvae import TSVQVAE


def _tiny_cfg():
    return ModelConfig(p=4, d_model=16, codebook_size=16, enc_layers=1,
                       dec_layers=1, heads=2, ff=32, dropout=0.0)


def test_forward_shapes_and_finite_loss():
    cfg = _tiny_cfg()
    model = TSVQVAE(cfg, d_in=5)
    x = torch.randn(8, cfg.p, 5)
    out = model(x)
    assert out["recon"].shape == (8, cfg.p, 5)
    assert out["ids"].shape == (8,)
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])


def test_encode_returns_token_ids():
    cfg = _tiny_cfg()
    model = TSVQVAE(cfg, d_in=5)
    ids = model.encode(torch.randn(3, cfg.p, 5))
    assert ids.shape == (3,)
    assert ids.dtype == torch.long
    assert (ids >= 0).all() and (ids < cfg.codebook_size).all()


def test_overfits_a_tiny_batch():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = TSVQVAE(cfg, d_in=5)
    model.train()
    x = torch.randn(8, cfg.p, 5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    # VQ has a cold-start: samples collapse to one code for a few hundred steps
    # before the codebook spreads, so give it enough steps to prove it can learn.
    for _ in range(2000):
        opt.zero_grad()
        out = model(x)
        out["loss"].backward()
        opt.step()
        if first is None:
            first = out["recon_loss"].item()
    assert out["recon_loss"].item() < 0.05 * first
