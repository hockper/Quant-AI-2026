import pytest

from bubble_bi.colab import load_colab_config


def _write_cfg(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  tickers: [AAPL]\ntrain:\n  batch_size: 64\n  device: cpu\n")
    return str(p)


def test_points_at_drive_and_auto_device(tmp_path):
    cfg = load_colab_config(_write_cfg(tmp_path), "/drive/bubble_bi")
    assert cfg.data.raw_dir == "/drive/bubble_bi/artifacts/raw"
    assert cfg.data.cache_dir == "/drive/bubble_bi/artifacts/cache"
    assert cfg.train.device == "auto"


def test_overrides_apply_to_the_right_section(tmp_path):
    cfg = load_colab_config(_write_cfg(tmp_path), "/drive/bubble_bi",
                            batch_size=256, max_steps=5000, d_model=256)
    assert cfg.train.batch_size == 256
    assert cfg.train.max_steps == 5000
    assert cfg.model.d_model == 256


def test_unknown_override_raises(tmp_path):
    with pytest.raises(AttributeError):
        load_colab_config(_write_cfg(tmp_path), "/drive/bubble_bi", nonsense=1)
