import torch

from bubble_bi.models.llama import RMSNorm, RotaryEmbedding, SwiGLU


def test_rmsnorm_unit_rms():
    x = torch.randn(4, 8) * 5.0
    y = RMSNorm(8)(x)                                  # weight=1 initially
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones(4), atol=1e-4)


def test_rope_preserves_norm_and_varies_by_position():
    rope = RotaryEmbedding(head_dim=8, max_len=16)
    x = torch.randn(2, 3, 16, 8)                       # [B,H,T,hd]
    y = rope(x)
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-4)   # rotation preserves norm
    xc = x.clone()
    xc[:, :, 5] = xc[:, :, 0]
    yc = rope(xc)
    assert not torch.allclose(yc[:, :, 0], yc[:, :, 5], atol=1e-4)


def test_swiglu_shape_and_bias_free():
    m = SwiGLU(8, 16)
    assert m(torch.randn(3, 8)).shape == (3, 8)
    assert all("bias" not in n for n, _ in m.named_parameters())
