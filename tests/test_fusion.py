import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.fusion import CrossAttentionFusion


def _cfg():
    return ModelConfig(d_model=16, heads=2, fusion_layers=2, dropout=0.0)


def test_fusion_shapes():
    fus = CrossAttentionFusion(_cfg())
    z_ts = torch.randn(3, 6, 16)
    z_cs = torch.randn(3, 16)
    valid = torch.ones(3, 6, dtype=torch.bool)
    ft, fc = fus(z_ts, z_cs, valid)
    assert ft.shape == (3, 6, 16)
    assert fc.shape == (3, 16)


def test_fused_cs_ignores_masked_stocks():
    torch.manual_seed(0)
    fus = CrossAttentionFusion(_cfg()).eval()
    valid = torch.tensor([[True, True, False, True]])
    z_ts1 = torch.randn(1, 4, 16)
    z_ts2 = z_ts1.clone()
    z_ts2[0, 2] = 50.0                       # change the masked stock's latent (a key)
    z_cs = torch.randn(1, 16)
    with torch.no_grad():
        _, fc1 = fus(z_ts1, z_cs, valid)
        _, fc2 = fus(z_ts2, z_cs, valid)
    assert torch.allclose(fc1, fc2, atol=1e-5)
