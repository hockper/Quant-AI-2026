import numpy as np
import pandas as pd

from bubble_bi.data.fracdiff import frac_diff, fracdiff_weights


def test_weights_follow_the_recurrence():
    d = 0.45
    w = fracdiff_weights(d, thresh=1e-3, max_lags=200)
    assert w[0] == 1.0
    assert np.isclose(w[1], -d)                       # w1 = -w0*(d-1+1)/1 = -d
    for k in range(2, len(w)):
        assert np.isclose(w[k], -w[k - 1] * (d - k + 1) / k)
    assert abs(w[-1]) >= 1e-3                          # truncated at the threshold


def test_weights_respect_max_lags():
    w = fracdiff_weights(0.45, thresh=1e-12, max_lags=20)
    assert len(w) == 20


def test_d0_is_identity_and_d1_is_first_difference():
    x = pd.Series(np.arange(20, dtype=float) ** 1.3)
    y0 = frac_diff(x, d=0.0)
    assert np.allclose(y0.dropna().to_numpy(), x.iloc[len(x) - len(y0.dropna()):].to_numpy())

    y1 = frac_diff(x, d=1.0)
    expected = x.diff()
    both = y1.notna() & expected.notna()
    assert np.allclose(y1[both].to_numpy(), expected[both].to_numpy())


def test_frac_diff_is_causal():
    rng = np.random.default_rng(0)
    x = pd.Series(np.cumsum(rng.normal(size=300)))
    full = frac_diff(x, d=0.45)
    trunc = frac_diff(x.iloc[:201], d=0.45)
    a, b = full.iloc[:201].to_numpy(), trunc.to_numpy()
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.allclose(a[~both_nan], b[~both_nan], atol=1e-10)


def test_warmup_is_nan_then_finite():
    x = pd.Series(np.cumsum(np.random.default_rng(1).normal(size=300)))
    w = fracdiff_weights(0.45)
    y = frac_diff(x, d=0.45)
    assert y.iloc[: len(w) - 1].isna().all()
    assert np.isfinite(y.iloc[-1])
