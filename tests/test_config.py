from bubble_bi.config import load_config, Config


def test_load_config_reads_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "data:\n"
        "  tickers: [AAPL, MSFT]\n"
        "  min_history: 100\n"
        "splits:\n"
        "  train_days: 300\n"
        "seed: 7\n"
    )
    cfg = load_config(str(p))
    assert isinstance(cfg, Config)
    assert cfg.data.tickers == ["AAPL", "MSFT"]
    assert cfg.data.min_history == 100
    assert cfg.splits.train_days == 300
    assert cfg.features.rsi_window == 14  # default preserved
    assert cfg.seed == 7
