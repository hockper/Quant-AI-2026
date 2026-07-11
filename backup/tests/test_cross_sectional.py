import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.cross_sectional import CSFieldEncoder, CSFieldDecoder


def _cfg():
    return ModelConfig(cs_p=3, d_model=16, enc_layers=1, dec_layers=1, heads=2,
                       ff=32, dropout=0.0)


def test_cs_field_shapes():
    cfg = _cfg()
    enc = CSFieldEncoder(cfg, d_in=5, n_stocks=6)
    dec = CSFieldDecoder(cfg, d_out=5, n_stocks=6)
    x = torch.randn(2, 6, cfg.cs_p, 5)
    valid = torch.ones(2, 6, dtype=torch.bool)
    z = enc(x, valid)
    assert z.shape == (2, 16)
    out = dec(z)
    assert out.shape == (2, 6, cfg.cs_p, 5)


def test_cs_field_encoder_ignores_invalid_stocks():
    torch.manual_seed(0)
    cfg = _cfg()
    enc = CSFieldEncoder(cfg, d_in=5, n_stocks=4).eval()
    valid = torch.tensor([[True, True, False, True]])
    x1 = torch.randn(1, 4, cfg.cs_p, 5)
    x2 = x1.clone()
    x2[0, 2] = 999.0                              # garbage in the invalid stock across all cs_p days
    with torch.no_grad():
        z1 = enc(x1, valid)
        z2 = enc(x2, valid)
    assert torch.allclose(z1, z2, atol=1e-5)
