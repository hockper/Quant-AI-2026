import numpy as np
import pandas as pd

from bubble_bi.config import DataConfig, FeatureConfig
from bubble_bi.data.panel import build_panel, save_panel, load_panel


def _ohlcv(n, start="2015-01-01", seed=0):
    dates = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    c = pd.Series(100 + np.cumsum(rng.normal(size=n)), index=dates)
    return pd.DataFrame(
        {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1e6},
        index=dates,
    )


def _cfgs():
    return DataConfig(tickers=["A", "B"], min_history=50), FeatureConfig()


def test_target_is_next_day_return_and_last_is_masked():
    data_cfg, feat_cfg = _cfgs()
    per = {"A": _ohlcv(120, seed=1), "B": _ohlcv(120, seed=2)}
    panel = build_panel(per, data_cfg, feat_cfg)
    close_a = per["A"]["close"].reindex(panel.dates).to_numpy()
    ai = panel.tickers.index("A")
    expected = close_a[1:] / close_a[:-1] - 1
    got = panel.target[:-1, ai]
    finite = np.isfinite(got) & np.isfinite(expected)
    assert np.allclose(got[finite], expected[finite], atol=1e-10)
    assert not panel.mask[-1, ai]  # last day has no future target


def test_thin_history_ticker_is_dropped():
    data_cfg, feat_cfg = _cfgs()
    per = {"A": _ohlcv(120, seed=1), "B": _ohlcv(20, seed=2)}
    panel = build_panel(per, data_cfg, feat_cfg)
    assert panel.tickers == ["A"]


def test_shape_consistency_and_roundtrip(tmp_path):
    data_cfg, feat_cfg = _cfgs()
    per = {"A": _ohlcv(120, seed=1), "B": _ohlcv(120, seed=2)}
    panel = build_panel(per, data_cfg, feat_cfg)
    T, N, D = panel.features.shape
    assert (T, N) == panel.target.shape == panel.mask.shape
    assert N == len(panel.tickers) and D == len(panel.feature_names)
    p = tmp_path / "panel.npz"
    save_panel(panel, str(p))
    again = load_panel(str(p))
    assert again.tickers == panel.tickers
    assert np.array_equal(again.mask, panel.mask)
