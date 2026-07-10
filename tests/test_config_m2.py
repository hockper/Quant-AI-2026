from bubble_bi.config import load_config


def test_model_config_m2_fields(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "data:\n  tickers: [AAPL]\n"
        "model:\n  cs_codebook_size: 128\n  fusion_layers: 1\n"
        "  use_fusion: false\n  active_modules: [cs]\n"
    )
    cfg = load_config(str(p))
    assert cfg.model.cs_codebook_size == 128
    assert cfg.model.fusion_layers == 1
    assert cfg.model.use_fusion is False
    assert cfg.model.active_modules == ["cs"]
    # defaults preserved
    assert cfg.model.codebook_size == 512


def test_model_config_m2_defaults():
    from bubble_bi.config import ModelConfig
    m = ModelConfig()
    assert m.cs_codebook_size == 512
    assert m.fusion_layers == 2
    assert m.use_fusion is True
    assert m.active_modules == ["ts", "cs"]
