import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.llama import LlamaAttention, LlamaBlock


def _cfg(**kw):
    base = dict(d_model=16, heads=4, n_kv_heads=0, ff=32, dropout=0.0,
                pred_window=8, rope_theta=10000.0)
    base.update(kw)
    return ModelConfig(**base)


def test_attention_shape_and_bias_free():
    attn = LlamaAttention(_cfg())
    x = torch.randn(2, 8, 16)
    assert attn(x).shape == (2, 8, 16)
    assert all("bias" not in n for n, _ in attn.named_parameters())


def test_attention_is_causal():
    torch.manual_seed(0)
    attn = LlamaAttention(_cfg()).eval()
    x = torch.randn(1, 8, 16)
    x2 = x.clone()
    x2[0, 7] = 99.0                                    # perturb the LAST position
    with torch.no_grad():
        y = attn(x)
        y2 = attn(x2)
    assert torch.allclose(y[:, :7], y2[:, :7], atol=1e-5)


def test_gqa_runs_and_matches_shape():
    attn = LlamaAttention(_cfg(n_kv_heads=2))          # 4 query heads, 2 kv heads
    assert attn(torch.randn(2, 8, 16)).shape == (2, 8, 16)


def test_block_shape():
    blk = LlamaBlock(_cfg())
    assert blk(torch.randn(2, 8, 16)).shape == (2, 8, 16)
