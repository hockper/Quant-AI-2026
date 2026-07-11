from bubble_bi.config import ModelConfig, load_config


def test_predictor_config_defaults():
    m = ModelConfig()
    assert m.pred_window == 64
    assert m.pred_layers == 4
    assert m.n_kv_heads == 0
    assert m.rope_theta == 10000.0


def test_predictor_config_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  tickers: [AAPL]\nmodel:\n  pred_window: 32\n  n_kv_heads: 2\n")
    cfg = load_config(str(p))
    assert cfg.model.pred_window == 32
    assert cfg.model.n_kv_heads == 2
