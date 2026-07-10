import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.cross_sectional import CSEncoder, CSDecoder


def _cfg():
    return ModelConfig(p=4, d_model=16, enc_layers=1, dec_layers=1, heads=2,
                       ff=32, dropout=0.0)


def test_cs_encoder_decoder_shapes():
    cfg = _cfg()
    enc = CSEncoder(cfg, d_in=5, n_stocks=6)
    dec = CSDecoder(cfg, d_out=5, n_stocks=6)
    x = torch.randn(3, 6, 5)
    valid = torch.ones(3, 6, dtype=torch.bool)
    z = enc(x, valid)
    assert z.shape == (3, 16)
    out = dec(z)
    assert out.shape == (3, 6, 5)


def test_cs_encoder_ignores_invalid_stocks():
    torch.manual_seed(0)
    cfg = _cfg()
    enc = CSEncoder(cfg, d_in=5, n_stocks=4).eval()
    valid = torch.tensor([[True, True, False, True]])
    x1 = torch.randn(1, 4, 5)
    x2 = x1.clone()
    x2[0, 2] = 999.0                         # garbage in the invalid stock slot
    with torch.no_grad():
        z1 = enc(x1, valid)
        z2 = enc(x2, valid)
    assert torch.allclose(z1, z2, atol=1e-5)
