from bubble_bi.config import load_config, ModelConfig, TrainConfig


def test_config_has_model_and_train_defaults_and_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "data:\n  tickers: [AAPL]\n  train_frac: 0.6\n"
        "model:\n  p: 8\n  codebook_size: 256\n"
        "train:\n  lr: 0.0005\n  max_steps: 10\n"
    )
    cfg = load_config(str(p))
    assert isinstance(cfg.model, ModelConfig) and isinstance(cfg.train, TrainConfig)
    assert cfg.model.p == 8
    assert cfg.model.codebook_size == 256
    assert cfg.model.d_model == 128           # default preserved
    assert cfg.train.lr == 0.0005
    assert cfg.train.max_steps == 10
    assert cfg.data.train_frac == 0.6
    assert cfg.data.val_frac == 0.15          # default preserved
