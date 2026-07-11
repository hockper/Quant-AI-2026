import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.joint_decoder import JointDecoder


def _cfg():
    return ModelConfig(p=4, d_model=16, dec_layers=1, heads=2, ff=32, dropout=0.0)


def test_joint_decoder_shapes():
    dec = JointDecoder(_cfg(), d_out=5, n_stocks=6)
    z_q = torch.randn(3, 6, 16)
    valid = torch.ones(3, 6, dtype=torch.bool)
    out = dec(z_q, valid)
    assert out.shape == (3, 6, 4, 5)


def test_joint_decoder_masks_invalid_stock_keys():
    torch.manual_seed(0)
    dec = JointDecoder(_cfg(), d_out=5, n_stocks=4).eval()
    valid = torch.tensor([[True, True, False, True]])
    z1 = torch.randn(1, 4, 16)
    z2 = z1.clone()
    z2[0, 2] = 50.0                               # perturb the invalid stock's token
    with torch.no_grad():
        o1 = dec(z1, valid)
        o2 = dec(z2, valid)
    assert torch.allclose(o1[:, [0, 1, 3]], o2[:, [0, 1, 3]], atol=1e-5)
