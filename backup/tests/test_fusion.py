import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.fusion import MarketToStockFusion


def _cfg():
    return ModelConfig(d_model=16, heads=2, fusion_layers=2, dropout=0.0)


def test_fusion_shapes():
    fus = MarketToStockFusion(_cfg())
    z_ts = torch.randn(3, 6, 16)
    z_cs = torch.randn(3, 16)
    out = fus(z_ts, z_cs)
    assert out.shape == (3, 6, 16)


def test_residual_keeps_stocks_distinct():
    # With a single shared market key, the attention output is identical for every
    # stock; the residual z_ts is what keeps distinct stocks distinct.
    torch.manual_seed(0)
    fus = MarketToStockFusion(_cfg()).eval()
    z_ts = torch.randn(1, 4, 16)
    z_cs = torch.randn(1, 16)
    with torch.no_grad():
        out = fus(z_ts, z_cs)
    assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-4)
