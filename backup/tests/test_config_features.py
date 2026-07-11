from bubble_bi.config import FeatureConfig, load_config


def test_feature_config_defaults():
    f = FeatureConfig()
    assert f.frac_d == 0.45
    assert f.frac_thresh == 1e-3
    assert f.frac_max_lags == 200
    assert f.atr_window == 14
    assert f.hurst_window == 100
    assert f.entropy_window == 60
    assert f.entropy_bins == 10
    assert f.amihud_window == 21
    assert f.roll_window == 21
    assert f.cs_window == 21
    assert f.vol_window == 20          # pre-existing, reused


def test_feature_config_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  tickers: [AAPL]\nfeatures:\n  frac_d: 0.3\n  hurst_window: 64\n")
    cfg = load_config(str(p))
    assert cfg.features.frac_d == 0.3
    assert cfg.features.hurst_window == 64
