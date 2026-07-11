import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.fusion_vqvae import FusionVQVAE


def _cfg(**kw):
    base = dict(p=4, cs_p=3, d_model=16, fusion_codebook_size=16, enc_layers=1,
                dec_layers=1, fusion_layers=1, heads=2, ff=32, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


def _batch(B=3, N=5, L=4, D=6):
    block = torch.randn(B, N, L, D)
    valid = torch.ones(B, N, dtype=torch.bool)
    valid[0, 4] = False
    return {"block": block, "valid": valid}


def test_forward_shapes():
    model = FusionVQVAE(_cfg(), d_in=6, n_stocks=5)
    out = model(_batch())
    assert out["recon"].shape == (3, 5, 4, 6)      # p=4 window, whole market
    assert out["ids"].shape == (3, 5)              # one token per (stock, day)
    assert torch.isfinite(out["loss"])


def test_encode_returns_token_grid():
    model = FusionVQVAE(_cfg(), d_in=6, n_stocks=5)
    ids = model.encode(_batch())
    assert ids.shape == (3, 5) and ids.dtype == torch.long
    assert (ids >= 0).all() and (ids < 16).all()


def test_use_fusion_false_bypasses_cs():
    model = FusionVQVAE(_cfg(use_fusion=False), d_in=6, n_stocks=5)
    assert not hasattr(model, "cs_enc")
    assert not hasattr(model, "fusion")
    out = model(_batch())
    assert torch.isfinite(out["loss"])


def test_load_frozen_freezes_encoders(tmp_path):
    torch.manual_seed(0)
    model = FusionVQVAE(_cfg(), d_in=6, n_stocks=5)
    ts_ck = tmp_path / "ts.pt"
    cs_ck = tmp_path / "cs.pt"
    torch.save({"model": {f"enc.{k}": v for k, v in model.ts_enc.state_dict().items()}}, ts_ck)
    torch.save({"model": {f"enc.{k}": v for k, v in model.cs_enc.state_dict().items()}}, cs_ck)
    model.load_frozen(str(ts_ck), str(cs_ck))
    assert all(not p.requires_grad for p in model.ts_enc.parameters())
    assert all(not p.requires_grad for p in model.cs_enc.parameters())

    before = model.ts_enc.embed.weight.clone()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    out = model(_batch())
    out["loss"].backward()
    opt.step()
    assert torch.allclose(before, model.ts_enc.embed.weight)   # frozen, unchanged
