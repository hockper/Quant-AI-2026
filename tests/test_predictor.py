import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.predictor import NextTokenPredictor


def _cfg(**kw):
    base = dict(d_model=16, heads=4, n_kv_heads=0, ff=32, dropout=0.0,
                pred_window=8, pred_layers=2, rope_theta=10000.0)
    base.update(kw)
    return ModelConfig(**base)


def _batch(B=3, W=8, vocab=16):
    tokens = torch.randint(0, vocab, (B, W))
    targets = torch.randint(0, vocab, (B, W))
    return {"tokens": tokens, "targets": targets}


def test_forward_shapes_and_keys():
    model = NextTokenPredictor(_cfg(), vocab=16)
    out = model(_batch())
    assert out["logits"].shape == (3, 8, 16)
    assert torch.isfinite(out["loss"])
    assert 0.0 <= float(out["accuracy"]) <= 1.0


def test_hidden_state_shape():
    model = NextTokenPredictor(_cfg(), vocab=16)
    h = model.hidden_state(torch.randint(0, 16, (2, 8)))
    assert h.shape == (2, 8, 16)


def test_causal_future_does_not_change_earlier_logits():
    torch.manual_seed(0)
    model = NextTokenPredictor(_cfg(), vocab=16).eval()
    toks = torch.randint(0, 16, (1, 8))
    toks2 = toks.clone()
    toks2[0, 7] = (toks[0, 7] + 1) % 16                # change last token
    with torch.no_grad():
        l1 = model({"tokens": toks, "targets": toks})["logits"]
        l2 = model({"tokens": toks2, "targets": toks2})["logits"]
    assert torch.allclose(l1[:, :7], l2[:, :7], atol=1e-5)


def test_overfits_tiny_batch():
    torch.manual_seed(0)
    model = NextTokenPredictor(_cfg(), vocab=16).train()
    batch = _batch(B=4, W=8, vocab=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        out = model(batch)
        out["loss"].backward()
        opt.step()
    assert float(out["accuracy"]) > 0.95
