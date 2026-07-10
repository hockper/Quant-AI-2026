import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.dual_vqvae import DualVQVAE


def _cfg(**kw):
    base = dict(p=4, d_model=16, codebook_size=16, cs_codebook_size=16, enc_layers=1,
                dec_layers=1, fusion_layers=1, heads=2, ff=32, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


def _batch(B=3, N=5, p=4, D=6):
    windows = torch.randn(B, N, p, D)
    valid = torch.ones(B, N, dtype=torch.bool)
    valid[0, 4] = False
    return {"windows": windows, "valid": valid}


def test_dual_forward_has_both_losses():
    model = DualVQVAE(_cfg(), d_in=6, n_stocks=5)
    out = model(_batch())
    assert torch.isfinite(out["loss"])
    assert "ts_recon" in out and "cs_recon" in out
    assert "ts_perplexity" in out and "cs_perplexity" in out


def test_active_modules_ts_only_builds_no_cs():
    model = DualVQVAE(_cfg(active_modules=["ts"]), d_in=6, n_stocks=5)
    assert not hasattr(model, "cs_enc")
    assert model.use_fusion is False
    out = model(_batch())
    assert "ts_recon" in out and "cs_recon" not in out


def test_use_fusion_false_skips_fusion():
    model = DualVQVAE(_cfg(use_fusion=False), d_in=6, n_stocks=5)
    assert not hasattr(model, "fusion")
    out = model(_batch())
    assert torch.isfinite(out["loss"])


def test_invalid_stock_does_not_change_losses():
    torch.manual_seed(0)
    model = DualVQVAE(_cfg(), d_in=6, n_stocks=5).eval()
    b1 = _batch()
    b2 = {"windows": b1["windows"].clone(), "valid": b1["valid"].clone()}
    b2["windows"][0, 4] = 123.0              # stock 4 in row 0 is invalid
    with torch.no_grad():
        o1 = model(b1)
        o2 = model(b2)
    assert torch.allclose(o1["ts_recon"], o2["ts_recon"], atol=1e-4)
    assert torch.allclose(o1["cs_recon"], o2["cs_recon"], atol=1e-4)


def test_encode_returns_token_shapes():
    model = DualVQVAE(_cfg(), d_in=6, n_stocks=5)
    ts_tok, cs_tok = model.encode(_batch(B=3, N=5))
    assert ts_tok.shape == (3, 5) and ts_tok.dtype == torch.long
    assert cs_tok.shape == (3,) and cs_tok.dtype == torch.long


def test_reinit_dead_codes_runs_for_both():
    model = DualVQVAE(_cfg(), d_in=6, n_stocks=5)
    out = model(_batch())
    model.reinit_dead_codes(out)             # must not raise
