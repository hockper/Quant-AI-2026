import numpy as np
import pandas as pd
import pytest

from bubble_bi.data.features.memory import build


def _df_from_close(close: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(close), freq="D")
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(len(close), 1_000.0),
        },
        index=idx,
    )


def _prices_from_ar1(rho, n=1200, seed=0, sigma=0.01):
    """Zero-drift AR(1) return process -> price. rho>0 persistent, rho<0 anti-persistent."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = rho * r[i - 1] + eps[i]
    return 100 * np.exp(np.cumsum(r))


def test_build_returns_exactly_the_three_named_features_sharing_the_index():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    df = _df_from_close(close)

    out = build(df, {})

    assert set(out) == {"hurst", "close_frac", "entropy"}
    for name, series in out.items():
        assert isinstance(series, pd.Series), name
        assert series.index.equals(df.index), name


def test_features_are_causal_truncating_the_future_does_not_change_the_past():
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    df_full = _df_from_close(close)
    df_trunc = _df_from_close(close[:251])

    full = build(df_full, {})
    trunc = build(df_trunc, {})

    for name in full:
        a = full[name].iloc[:251].to_numpy()
        b = trunc[name].to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        assert np.allclose(a[~both_nan], b[~both_nan], atol=1e-10), name


def test_hurst_ordering_persistent_gt_random_walk_gt_antipersistent():
    # 200-day window: this is the scale at which the ordering was measured.
    import bubble_bi.data.features.memory as memory

    original_window = memory.HURST_WINDOW
    try:
        memory.HURST_WINDOW = 200
        h_persistent = build(_df_from_close(_prices_from_ar1(0.4, seed=1)), {})["hurst"].dropna().mean()
        h_rw = build(_df_from_close(_prices_from_ar1(0.0, seed=1)), {})["hurst"].dropna().mean()
        h_anti = build(_df_from_close(_prices_from_ar1(-0.4, seed=1)), {})["hurst"].dropna().mean()
    finally:
        memory.HURST_WINDOW = original_window

    assert h_persistent > h_rw > h_anti


def test_hurst_near_half_for_a_random_walk():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600)))
    df = _df_from_close(close)

    h = build(df, {})["hurst"].dropna()

    assert abs(h.mean() - 0.5) < 0.15


def test_entropy_is_zero_for_constant_returns_and_positive_for_noisy_ones():
    constant_close = 100 * np.exp(np.arange(200) * 0.001)
    e_constant = build(_df_from_close(constant_close), {})["entropy"].dropna()
    assert np.allclose(e_constant.to_numpy(), 0.0, atol=1e-9)

    rng = np.random.default_rng(2)
    noisy_close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
    e_noisy = build(_df_from_close(noisy_close), {})["entropy"].dropna()
    assert (e_noisy > 0.5).mean() > 0.9


def test_hurst_raises_for_a_window_too_small_for_two_scales():
    import bubble_bi.data.features.memory as memory

    original_window = memory.HURST_WINDOW
    try:
        memory.HURST_WINDOW = 15  # candidate scales {1,3,7,15}; only 15 clears >=8
        close = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 200)))
        with pytest.raises(ValueError):
            build(_df_from_close(close), {})
    finally:
        memory.HURST_WINDOW = original_window
