import numpy as np
import pandas as pd

from bubble_bi.config import FeatureConfig
from bubble_bi.data.features import compute_features, FEATURE_NAMES


def _synthetic_ohlcv(n=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n)))
    close = pd.Series(price, index=dates)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1e6, 5e6, size=n).astype(float),
        },
        index=dates,
    )


def test_feature_columns_match_names():
    cfg = FeatureConfig()
    df = _synthetic_ohlcv()
    feats = compute_features(df, cfg)
    assert list(feats.columns) == FEATURE_NAMES(cfg)
    assert len(feats) == len(df)


def test_features_are_causal_truncating_future_does_not_change_past():
    cfg = FeatureConfig()
    df = _synthetic_ohlcv(n=300)
    cutoff = 200
    full = compute_features(df, cfg)
    truncated = compute_features(df.iloc[: cutoff + 1], cfg)
    a = full.iloc[: cutoff + 1].to_numpy()
    b = truncated.to_numpy()
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.allclose(a[~both_nan], b[~both_nan], atol=1e-10)


def test_features_have_warmup_nans_then_finite():
    cfg = FeatureConfig()
    feats = compute_features(_synthetic_ohlcv(), cfg)
    assert feats.iloc[0].isna().any()
    assert np.isfinite(feats.iloc[-1].to_numpy()).all()
