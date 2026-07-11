from bubble_bi.config import TrainConfig, load_config


def test_train_config_log_every_default():
    assert TrainConfig().log_every == 10


def test_load_config_parses_log_every(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  tickers: [AAPL]\ntrain:\n  log_every: 5\n")
    assert load_config(str(p)).train.log_every == 5
