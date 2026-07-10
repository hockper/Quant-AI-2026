from bubble_bi.config import load_config, ModelConfig


def test_model_config_new_fields(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "data:\n  tickers: [AAPL]\n"
        "model:\n  cs_p: 8\n  fusion_codebook_size: 256\n  cs_codebook_size: 128\n"
    )
    cfg = load_config(str(p))
    assert cfg.model.cs_p == 8
    assert cfg.model.fusion_codebook_size == 256
    assert cfg.model.cs_codebook_size == 128
    assert cfg.model.codebook_size == 512      # default preserved


def test_model_config_defaults():
    m = ModelConfig()
    assert m.cs_p == 5
    assert m.fusion_codebook_size == 512
    assert not hasattr(m, "active_modules")
