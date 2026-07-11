# Feature Expansion Implementation Plan (D: 10 → 22)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 12 causal daily features (range-based volatility, stochastic memory, microstructure friction, flow) taking the panel from `D=10` to `D=22`.

**Architecture:** A reusable frac-diff primitive (`fracdiff.py`) plus three focused feature modules (`features_volatility.py`, `features_memory.py`, `features_micro.py`), all pure `DataFrame → Series` functions. `features.py` stays the orchestrator with an unchanged public interface, so every downstream consumer picks up the new width automatically via `d_in = panel.features.shape[2]`.

**Tech Stack:** numpy (`sliding_window_view`, `convolve`), pandas, existing `bubble_bi` package.

## Global Constraints

- Run everything as `.venv/bin/python` from repo root `/home/hockper/Documents/Code/Bubble Bi`.
- **Every feature must be causal** — backward-looking windows only. The existing `test_features_are_causal_truncating_future_does_not_change_past` iterates all output columns and is the enforcement mechanism; it MUST stay green.
- **Skipped (do not implement):** VPIN, Kyle's λ, Higuchi fractal dimension, any CFI/clustering stage.
- **Guards are mandatory:** Roll spread returns **0 when serial covariance ≥ 0**; Corwin-Schultz spreads are **clamped to ≥ 0**; Garman-Klass variance is **clipped to ≥ 0** before `sqrt`.
- **Performance:** use `numpy.lib.stride_tricks.sliding_window_view` / `convolve`; **no `pandas.rolling.apply`**.
- Final column order (D=22): existing 10 → volatility (4) → memory (3) → microstructure (3) → flow (2).
- `D=22` invalidates `panel.npz`, `tokens.npz`, and all four checkpoints — a full retrain is expected (on Colab GPU).
- TDD; frequent commits.

---

### Task 0: Config fields

**Files:**
- Modify: `bubble_bi/config.py`
- Test: `tests/test_config_features.py`

**Interfaces:**
- Produces: `FeatureConfig` gains `frac_d=0.45`, `frac_thresh=1e-3`, `frac_max_lags=200`, `atr_window=14`, `hurst_window=100`, `entropy_window=60`, `entropy_bins=10`, `amihud_window=21`, `roll_window=21`, `cs_window=21`. (`vol_window=20` already exists and is reused.)

- [ ] **Step 1: Write the failing test**

`tests/test_config_features.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config_features.py -v`
Expected: FAIL (`AttributeError: ... 'frac_d'`).

- [ ] **Step 3: Implement**

In `bubble_bi/config.py`, add to `FeatureConfig` (after `volume_window`):
```python
    frac_d: float = 0.45
    frac_thresh: float = 1e-3
    frac_max_lags: int = 200
    atr_window: int = 14
    hurst_window: int = 100
    entropy_window: int = 60
    entropy_bins: int = 10
    amihud_window: int = 21
    roll_window: int = 21
    cs_window: int = 21
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config_features.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/config.py tests/test_config_features.py
git commit -m "feat: FeatureConfig fields for the expanded feature set"
```

---

### Task 1: Frac-diff primitive

**Files:**
- Create: `bubble_bi/data/fracdiff.py`
- Test: `tests/test_fracdiff.py`

**Interfaces:**
- Produces, in `bubble_bi/data/fracdiff.py`:
  - `fracdiff_weights(d: float, thresh: float = 1e-3, max_lags: int = 200) -> np.ndarray`
  - `frac_diff(series: pd.Series, d: float, thresh: float = 1e-3, max_lags: int = 200) -> pd.Series`
    (causal: `y_t = Σ_k w_k · x_{t−k}`; the first `len(w)−1` values are NaN)

- [ ] **Step 1: Write the failing test**

`tests/test_fracdiff.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fracdiff.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write implementation**

`bubble_bi/data/fracdiff.py`:
```python
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def fracdiff_weights(d: float, thresh: float = 1e-3, max_lags: int = 200) -> np.ndarray:
    """Binomial weights for fractional differentiation (Lopez de Prado FFD).

    w_0 = 1 ;  w_k = -w_{k-1} * (d - k + 1) / k
    Truncated once |w_k| < thresh, and hard-capped at max_lags.
    """
    w = [1.0]
    for k in range(1, max_lags):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
    return np.asarray(w, dtype=float)


def frac_diff(series: pd.Series, d: float, thresh: float = 1e-3,
              max_lags: int = 200) -> pd.Series:
    """Causal fixed-width fractional differentiation: y_t = sum_k w_k * x_{t-k}."""
    w = fracdiff_weights(d, thresh, max_lags)
    L = len(w)
    x = series.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= L:
        # windows[i] = x[i : i+L]; the value at t = i+L-1 needs x[t], x[t-1], ..., x[t-L+1]
        windows = sliding_window_view(x, L)
        out[L - 1:] = windows @ w[::-1]
    return pd.Series(out, index=series.index)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fracdiff.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/data/fracdiff.py tests/test_fracdiff.py
git commit -m "feat: fractional differentiation primitive (causal FFD)"
```

---

### Task 2: Volatility estimators

**Files:**
- Create: `bubble_bi/data/features_volatility.py`
- Test: `tests/test_features_volatility.py`

**Interfaces:**
- Consumes: a per-ticker OHLCV `pd.DataFrame` (columns `open, high, low, close, volume`).
- Produces, in `bubble_bi/data/features_volatility.py`:
  - `parkinson(df, window) -> pd.Series`
  - `garman_klass(df, window) -> pd.Series`
  - `rogers_satchell(df) -> pd.Series`
  - `yang_zhang(df, window) -> pd.Series`
  - `atr(df, window) -> pd.Series` (Wilder)

- [ ] **Step 1: Write the failing test**

`tests/test_features_volatility.py`:
```python
import numpy as np
import pandas as pd

from bubble_bi.data.features_volatility import (atr, garman_klass, parkinson,
                                                rogers_satchell, yang_zhang)


def _ohlcv(n=200, seed=0):
    rng = np.random.default_rng(seed)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    return pd.DataFrame({
        "open": c.shift(1).fillna(c.iloc[0]),
        "high": c * 1.01,
        "low": c * 0.99,
        "close": c,
        "volume": rng.integers(1e6, 5e6, n).astype(float),
    })


def test_estimators_are_nonnegative_and_finite_after_warmup():
    df = _ohlcv()
    for f in (parkinson(df, 20), garman_klass(df, 20), yang_zhang(df, 20), atr(df, 14)):
        tail = f.iloc[50:]
        assert np.isfinite(tail).all()
        assert (tail >= 0).all()


def test_parkinson_matches_closed_form_for_constant_range():
    # high/low ratio is constant -> parkinson = |ln r| / (2*sqrt(ln 2))
    n = 60
    c = pd.Series(np.full(n, 100.0))
    r = 1.02
    df = pd.DataFrame({"open": c, "high": c * r, "low": c, "close": c,
                       "volume": np.ones(n)})
    expected = abs(np.log(r)) / (2 * np.sqrt(np.log(2)))
    assert np.isclose(parkinson(df, 20).iloc[-1], expected)


def test_rogers_satchell_is_zero_when_open_equals_close_equals_high_equals_low():
    c = pd.Series(np.full(10, 50.0))
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": np.ones(10)})
    assert np.allclose(rogers_satchell(df).to_numpy(), 0.0)


def test_atr_is_causal():
    df = _ohlcv(n=200)
    full = atr(df, 14)
    trunc = atr(df.iloc[:121], 14)
    a, b = full.iloc[:121].to_numpy(), trunc.to_numpy()
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.allclose(a[~both_nan], b[~both_nan], atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_features_volatility.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write implementation**

`bubble_bi/data/features_volatility.py`:
```python
from __future__ import annotations

import numpy as np
import pandas as pd

_LN2 = np.log(2.0)


def parkinson(df: pd.DataFrame, window: int) -> pd.Series:
    hl2 = np.log(df["high"] / df["low"]) ** 2
    var = hl2.rolling(window).mean() / (4.0 * _LN2)
    return np.sqrt(var.clip(lower=0.0))


def garman_klass(df: pd.DataFrame, window: int) -> pd.Series:
    hl2 = np.log(df["high"] / df["low"]) ** 2
    co2 = np.log(df["close"] / df["open"]) ** 2
    var = (0.5 * hl2 - (2.0 * _LN2 - 1.0) * co2).rolling(window).mean()
    return np.sqrt(var.clip(lower=0.0))          # the -(2ln2-1) term can go negative


def rogers_satchell(df: pd.DataFrame) -> pd.Series:
    h, l, c, o = df["high"], df["low"], df["close"], df["open"]
    return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)


def yang_zhang(df: pd.DataFrame, window: int) -> pd.Series:
    o, c = df["open"], df["close"]
    log_overnight = np.log(o / c.shift(1))       # close -> next open (the gap)
    log_open_close = np.log(c / o)               # intraday
    sigma_o = log_overnight.rolling(window).var()
    sigma_c = log_open_close.rolling(window).var()
    sigma_rs = rogers_satchell(df).rolling(window).mean()
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    var = sigma_o + k * sigma_c + (1.0 - k) * sigma_rs
    return np.sqrt(var.clip(lower=0.0))


def atr(df: pd.DataFrame, window: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_features_volatility.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/data/features_volatility.py tests/test_features_volatility.py
git commit -m "feat: range-based volatility estimators (Parkinson, GK, Yang-Zhang, ATR)"
```

---

### Task 3: Memory features (Hurst, entropy)

**Files:**
- Create: `bubble_bi/data/features_memory.py`
- Test: `tests/test_features_memory.py`

**Interfaces:**
- Consumes: a `close` `pd.Series`.
- Produces, in `bubble_bi/data/features_memory.py`:
  - `hurst(close: pd.Series, window: int) -> pd.Series` — rolling rescaled-range exponent.
  - `entropy(close: pd.Series, window: int, bins: int) -> pd.Series` — rolling Shannon entropy of log-returns.

- [ ] **Step 1: Write the failing test**

`tests/test_features_memory.py`:
```python
import numpy as np
import pandas as pd

from bubble_bi.data.features_memory import entropy, hurst


def test_hurst_near_half_for_random_walk():
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600))))
    h = hurst(close, window=200).dropna()
    assert abs(h.mean() - 0.5) < 0.15          # random walk -> H ~ 0.5


def test_hurst_above_half_for_strong_trend():
    n = 600
    rng = np.random.default_rng(1)
    drift = np.linspace(0, 1.2, n)                       # persistent trend
    close = pd.Series(100 * np.exp(drift + 0.001 * rng.normal(size=n)))
    h = hurst(close, window=200).dropna()
    assert h.mean() > 0.6


def test_entropy_zero_for_constant_returns():
    close = pd.Series(100 * np.exp(np.arange(200) * 0.001))   # constant log-return
    e = entropy(close, window=60, bins=10).dropna()
    assert np.allclose(e.to_numpy(), 0.0, atol=1e-9)


def test_entropy_positive_for_noisy_returns():
    rng = np.random.default_rng(2)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))))
    e = entropy(close, window=60, bins=10).dropna()
    assert (e > 0.5).mean() > 0.9


def test_memory_features_are_causal():
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))))
    for fn in (lambda s: hurst(s, 100), lambda s: entropy(s, 60, 10)):
        full = fn(close)
        trunc = fn(close.iloc[:251])
        a, b = full.iloc[:251].to_numpy(), trunc.to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        assert np.allclose(a[~both_nan], b[~both_nan], atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_features_memory.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write implementation**

`bubble_bi/data/features_memory.py`:
```python
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _rs_for_scale(views: np.ndarray, s: int) -> np.ndarray:
    """Mean rescaled-range (R/S) at sub-scale s, for every window. -> [n_windows]"""
    n_win, W = views.shape
    n_chunks = W // s
    seg = views[:, :n_chunks * s].reshape(n_win, n_chunks, s)
    mean = seg.mean(axis=2, keepdims=True)
    dev = np.cumsum(seg - mean, axis=2)
    R = dev.max(axis=2) - dev.min(axis=2)          # [n_win, n_chunks]
    S = seg.std(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(S > 0, R / S, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(rs, axis=1)              # [n_win]


def hurst(close: pd.Series, window: int) -> pd.Series:
    """Rolling Hurst exponent via rescaled-range: slope of log(R/S) vs log(scale)."""
    r = np.log(close.to_numpy(dtype=float))
    r = np.diff(r, prepend=np.nan)                 # log-returns, NaN at index 0
    out = np.full(len(r), np.nan)
    scales = sorted({s for s in (window // 8, window // 4, window // 2, window) if s >= 8})
    if len(r) < window or len(scales) < 2:
        return pd.Series(out, index=close.index)

    views = sliding_window_view(r, window)         # [n_win, window]
    rs = np.vstack([_rs_for_scale(views, s) for s in scales])       # [n_scales, n_win]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rs = np.log(rs)
    log_n = np.log(np.asarray(scales, dtype=float))[:, None]        # [n_scales, 1]

    # least-squares slope per window, vectorised
    valid = np.isfinite(log_rs).all(axis=0)
    ln_c = log_n - log_n.mean()
    lr_c = log_rs - log_rs.mean(axis=0, keepdims=True)
    with np.errstate(invalid="ignore"):
        slope = (ln_c * lr_c).sum(axis=0) / (ln_c ** 2).sum()
    slope = np.where(valid, slope, np.nan)
    out[window - 1:] = slope
    return pd.Series(out, index=close.index)


def entropy(close: pd.Series, window: int, bins: int) -> pd.Series:
    """Rolling Shannon entropy of log-returns (histogram over the window's own range)."""
    r = np.log(close.to_numpy(dtype=float))
    r = np.diff(r, prepend=np.nan)
    out = np.full(len(r), np.nan)
    if len(r) < window:
        return pd.Series(out, index=close.index)

    views = sliding_window_view(r, window)
    for i in range(views.shape[0]):
        seg = views[i]
        if not np.isfinite(seg).all():
            continue
        lo, hi = seg.min(), seg.max()
        if hi <= lo:                               # constant returns -> no surprise
            out[i + window - 1] = 0.0
            continue
        counts, _ = np.histogram(seg, bins=bins, range=(lo, hi))
        p = counts / counts.sum()
        p = p[p > 0]
        out[i + window - 1] = float(-(p * np.log(p)).sum())
    return pd.Series(out, index=close.index)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_features_memory.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/data/features_memory.py tests/test_features_memory.py
git commit -m "feat: memory features (rolling Hurst R/S, Shannon entropy)"
```

---

### Task 4: Microstructure features

**Files:**
- Create: `bubble_bi/data/features_micro.py`
- Test: `tests/test_features_micro.py`

**Interfaces:**
- Consumes: a per-ticker OHLCV `pd.DataFrame`.
- Produces, in `bubble_bi/data/features_micro.py`:
  - `obv(df) -> pd.Series` (cumulative signed volume)
  - `amihud(df, window) -> pd.Series` (log-compressed illiquidity)
  - `roll_spread(df, window) -> pd.Series` (**0 when serial covariance ≥ 0**)
  - `corwin_schultz(df, window) -> pd.Series` (**clamped to ≥ 0**)

- [ ] **Step 1: Write the failing test**

`tests/test_features_micro.py`:
```python
import numpy as np
import pandas as pd

from bubble_bi.data.features_micro import amihud, corwin_schultz, obv, roll_spread


def _df(close, high=None, low=None, volume=None):
    n = len(close)
    close = pd.Series(close, dtype=float)
    high = pd.Series(high if high is not None else close * 1.01, dtype=float)
    low = pd.Series(low if low is not None else close * 0.99, dtype=float)
    volume = pd.Series(volume if volume is not None else np.full(n, 1e6), dtype=float)
    return pd.DataFrame({"open": close, "high": high, "low": low,
                         "close": close, "volume": volume})


def test_obv_accumulates_signed_volume():
    df = _df([10, 11, 10, 12], volume=[100, 200, 300, 400])
    # signs: nan->0, +, -, +   => 0, +200, -300, +400 cumulated
    assert obv(df).tolist() == [0.0, 200.0, -100.0, 300.0]


def test_roll_spread_is_zero_when_serial_cov_is_positive():
    # a strong trend has POSITIVE serial covariance -> Roll is undefined -> 0
    df = _df(np.linspace(100, 140, 120))
    s = roll_spread(df, 21).dropna()
    assert (s == 0).all()


def test_roll_spread_positive_for_bid_ask_bounce():
    # alternating price -> negative serial covariance -> a real spread estimate
    base = np.full(120, 100.0)
    base[1::2] += 0.5                                   # bounce
    s = roll_spread(_df(base), 21).dropna()
    assert (s > 0).mean() > 0.9


def test_corwin_schultz_never_negative():
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    df = _df(c, high=c * (1 + rng.uniform(0, 0.02, 200)),
             low=c * (1 - rng.uniform(0, 0.02, 200)))
    s = corwin_schultz(df, 21).dropna()
    assert (s >= 0).all()


def test_amihud_decreases_with_volume():
    c = np.array([100.0, 101.0, 100.0, 101.0] * 30)
    low_vol = amihud(_df(c, volume=np.full(len(c), 1e5)), 21).dropna()
    high_vol = amihud(_df(c, volume=np.full(len(c), 1e8)), 21).dropna()
    assert low_vol.iloc[-1] > high_vol.iloc[-1]


def test_micro_features_are_causal():
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    df = _df(c)
    for fn in (lambda d: amihud(d, 21), lambda d: roll_spread(d, 21),
               lambda d: corwin_schultz(d, 21), obv):
        full = fn(df)
        trunc = fn(df.iloc[:201])
        a, b = full.iloc[:201].to_numpy(), trunc.to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        assert np.allclose(a[~both_nan], b[~both_nan], atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_features_micro.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write implementation**

`bubble_bi/data/features_micro.py`:
```python
from __future__ import annotations

import numpy as np
import pandas as pd

_DEN = 3.0 - 2.0 * np.sqrt(2.0)      # Corwin-Schultz denominator


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume: cumulative signed volume (non-stationary by construction)."""
    sign = np.sign(df["close"].diff()).fillna(0.0)
    return (sign * df["volume"]).cumsum()


def amihud(df: pd.DataFrame, window: int) -> pd.Series:
    """Amihud illiquidity: |return| per dollar traded. log1p-compressed (extreme skew)."""
    r = np.log(df["close"]).diff().abs()
    dollar = (df["close"] * df["volume"]).replace(0.0, np.nan)
    illiq = (r / dollar).rolling(window).mean()
    return np.log1p(1e6 * illiq)


def roll_spread(df: pd.DataFrame, window: int) -> pd.Series:
    """Roll (1984) implied spread: 2*sqrt(-Cov(dP_t, dP_{t-1})).

    The estimator is UNDEFINED when the serial covariance is positive (common on
    trending daily data); we return 0 there, which is the standard convention.
    """
    dp = df["close"].diff()
    cov = dp.rolling(window).cov(dp.shift(1))
    return 2.0 * np.sqrt((-cov).clip(lower=0.0))


def corwin_schultz(df: pd.DataFrame, window: int) -> pd.Series:
    """Corwin-Schultz (2012) bid-ask spread from high/low only. Negatives clamped to 0."""
    h, l = df["high"], df["low"]
    hl2 = np.log(h / l) ** 2
    beta = hl2 + hl2.shift(1)
    h2 = pd.concat([h, h.shift(1)], axis=1).max(axis=1)
    l2 = pd.concat([l, l.shift(1)], axis=1).min(axis=1)
    gamma = np.log(h2 / l2) ** 2

    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _DEN - np.sqrt(gamma / _DEN)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return spread.clip(lower=0.0).rolling(window).mean()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_features_micro.py -v`
Expected: PASS (6 tests). The Roll guard test is the important one — it pins the `Cov ≥ 0 → 0` convention.

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/data/features_micro.py tests/test_features_micro.py
git commit -m "feat: microstructure features (Amihud, Roll, Corwin-Schultz, OBV)"
```

---

### Task 5: Wire everything into the orchestrator (D=22)

**Files:**
- Modify: `bubble_bi/data/features.py`
- Modify: `tests/test_features.py`

**Interfaces:**
- Consumes: `frac_diff` (Task 1); `parkinson`/`garman_klass`/`yang_zhang`/`atr` (Task 2); `hurst`/`entropy` (Task 3); `amihud`/`roll_spread`/`corwin_schultz`/`obv` (Task 4); `FeatureConfig` (Task 0).
- Produces: `FEATURE_NAMES(cfg)` returns **22** names; `compute_features(df, cfg)` returns exactly those columns, in order. Public interface unchanged — `build_panel` and every model pick up `D=22` automatically.

- [ ] **Step 1: Add the failing tests to `tests/test_features.py`**

Append:
```python
def test_feature_count_is_22():
    cfg = FeatureConfig()
    names = FEATURE_NAMES(cfg)
    assert len(names) == 22
    assert names[:10] == ["log_return", "sma_ratio_5", "sma_ratio_10", "sma_ratio_20",
                          "rsi", "macd", "macd_signal", "macd_hist", "realized_vol",
                          "volume_z"]
    assert names[10:] == ["parkinson", "garman_klass", "yang_zhang", "atr_frac",
                          "hurst", "close_frac", "entropy",
                          "amihud", "roll_spread", "corwin_schultz",
                          "volume_frac", "obv_frac"]


def test_all_22_columns_present_and_finite_at_the_end():
    cfg = FeatureConfig()
    feats = compute_features(_synthetic_ohlcv(n=600), cfg)
    assert list(feats.columns) == FEATURE_NAMES(cfg)
    assert np.isfinite(feats.iloc[-1].to_numpy()).all()
```

(The existing `test_features_are_causal_truncating_future_does_not_change_past` already iterates every column — it now covers all 12 new features. Bump its synthetic series to `n=600` so the Hurst/frac-diff warm-ups fit:)
```python
def test_features_are_causal_truncating_future_does_not_change_past():
    cfg = FeatureConfig()
    df = _synthetic_ohlcv(n=600)
    cutoff = 400
    full = compute_features(df, cfg)
    truncated = compute_features(df.iloc[: cutoff + 1], cfg)
    a = full.iloc[: cutoff + 1].to_numpy()
    b = truncated.to_numpy()
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.allclose(a[~both_nan], b[~both_nan], atol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_features.py -v`
Expected: FAIL — `len(names) == 22` fails (still 10).

- [ ] **Step 3: Rewrite `bubble_bi/data/features.py`**

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from bubble_bi.config import FeatureConfig
from bubble_bi.data.fracdiff import frac_diff
from bubble_bi.data.features_memory import entropy, hurst
from bubble_bi.data.features_micro import amihud, corwin_schultz, obv, roll_spread
from bubble_bi.data.features_volatility import atr, garman_klass, parkinson, yang_zhang


def FEATURE_NAMES(cfg: FeatureConfig) -> list[str]:
    names = ["log_return"]
    names += [f"sma_ratio_{w}" for w in cfg.ma_windows]
    names += ["rsi", "macd", "macd_signal", "macd_hist", "realized_vol", "volume_z"]
    names += ["parkinson", "garman_klass", "yang_zhang", "atr_frac"]          # volatility
    names += ["hurst", "close_frac", "entropy"]                               # memory
    names += ["amihud", "roll_spread", "corwin_schultz"]                      # microstructure
    names += ["volume_frac", "obv_frac"]                                      # flow
    return names


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    log_ret = np.log(close).diff()

    out = pd.DataFrame(index=df.index)

    # --- original 10 -------------------------------------------------------
    out["log_return"] = log_ret
    for w in cfg.ma_windows:
        out[f"sma_ratio_{w}"] = close / close.rolling(w).mean() - 1.0
    out["rsi"] = _rsi(close, cfg.rsi_window)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal
    out["realized_vol"] = log_ret.rolling(cfg.vol_window).std()
    vmean = volume.rolling(cfg.volume_window).mean()
    vstd = volume.rolling(cfg.volume_window).std()
    out["volume_z"] = (volume - vmean) / vstd

    fd = dict(d=cfg.frac_d, thresh=cfg.frac_thresh, max_lags=cfg.frac_max_lags)

    # --- volatility --------------------------------------------------------
    out["parkinson"] = parkinson(df, cfg.vol_window)
    out["garman_klass"] = garman_klass(df, cfg.vol_window)
    out["yang_zhang"] = yang_zhang(df, cfg.vol_window)
    out["atr_frac"] = frac_diff(atr(df, cfg.atr_window), **fd)

    # --- memory ------------------------------------------------------------
    out["hurst"] = hurst(close, cfg.hurst_window)
    out["close_frac"] = frac_diff(np.log(close), **fd)
    out["entropy"] = entropy(close, cfg.entropy_window, cfg.entropy_bins)

    # --- microstructure ----------------------------------------------------
    out["amihud"] = amihud(df, cfg.amihud_window)
    out["roll_spread"] = roll_spread(df, cfg.roll_window)
    out["corwin_schultz"] = corwin_schultz(df, cfg.cs_window)

    # --- flow --------------------------------------------------------------
    out["volume_frac"] = frac_diff(np.log1p(volume), **fd)
    out["obv_frac"] = frac_diff(obv(df), **fd)

    return out[FEATURE_NAMES(cfg)]
```

- [ ] **Step 4: Run the feature tests, then the full suite**

Run: `.venv/bin/python -m pytest tests/test_features.py -v`
Expected: PASS — including the **causality test now covering all 22 columns**.

Run: `.venv/bin/python -m pytest -q`
Expected: all green. (Panel/model tests use `d_in = features.shape[2]`, so `D=22` flows through automatically. If a test hard-codes a small synthetic series, lengthen it so the new warm-ups fit.)

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/data/features.py tests/test_features.py
git commit -m "feat: wire 12 new features into the orchestrator (D=22)"
```

---

### Task 6: Rebuild the panel + new baseline (manual verification)

**Files:** none created.

- [ ] **Step 1: Invalidate the stale caches**

The old `panel.npz` / `tokens.npz` have `D=10` and every checkpoint was trained on that width.
```bash
rm -f artifacts/cache/panel.npz artifacts/cache/tokens.npz
```

- [ ] **Step 2: Rebuild the panel and time it**

Run: `time .venv/bin/python -m bubble_bi.cli build-panel --config configs/m0.yaml`
Expected: prints `panel: (≈4153, 30, 22)`. Should complete in well under a minute (vectorised). Note the new shape.

- [ ] **Step 3: Check how much history the warm-up costs**

Run:
```bash
.venv/bin/python -c "
from bubble_bi.data.panel import load_panel
p = load_panel('artifacts/cache/panel.npz')
print('shape', p.features.shape, '| valid', f'{p.mask.mean():.1%}')
import numpy as np
first = int(np.argmax(p.mask.any(axis=1)))
print('first fully-usable day index:', first, '->', p.dates[first].date())
"
```
Expected: `D=22`; valid fraction lower than the old 99.0% (the Hurst/frac-diff warm-up). Record both numbers.

- [ ] **Step 4: Recompute the ridge floor**

Run: `.venv/bin/python -m bubble_bi.cli baseline --config configs/m0.yaml`
Expected: a new RankIC / RankICIR with 22 regressors. **This replaces 0.0062 as the floor** the neural stack must beat. Record it.

- [ ] **Step 5: Record the results**

Append the new panel shape, valid fraction, and the new RankIC floor to the design doc
(`docs/superpowers/specs/2026-07-11-feature-expansion-design.md`) under a "Results" note, and
note that all four checkpoints are now stale and must be retrained on Colab. Commit:
```bash
git add -A && git commit -m "docs: record feature-expansion results (D=22, new ridge floor)"
```

---

## Self-Review

**Spec coverage:**
- Config fields (frac_d/thresh/max_lags, atr/hurst/entropy/amihud/roll/cs windows) → Task 0 ✅
- Frac-diff primitive (weights recurrence, threshold + max_lags cap, causal apply) → Task 1 ✅
- Volatility: Parkinson, Garman-Klass (variance clipped ≥ 0), Yang-Zhang (+ Rogers-Satchell), ATR → Task 2; `atr_frac` composed in Task 5 ✅
- Memory: Hurst (rolling R/S, vectorised), entropy (rolling binned) → Task 3; `close_frac` composed in Task 5 ✅
- Microstructure: Amihud (log1p), Roll (**0 when cov ≥ 0**), Corwin-Schultz (**clamped**) → Task 4 ✅
- Flow: `volume_frac`, `obv_frac` (OBV from Task 4 + frac-diff) → Task 5 ✅
- Orchestrator with unchanged public interface, D=22, fixed column order → Task 5 ✅
- Causality enforced by the existing truncation test across all 22 columns → Task 5 ✅
- Warm-up / valid-fraction impact + new ridge floor + stale checkpoints → Task 6 ✅
- Skipped (VPIN, Kyle's λ, Higuchi, CFI) → not planned, as specified ✅
- Performance (sliding_window_view / convolve, no rolling.apply) → Tasks 1, 3 ✅

**Placeholder scan:** none. Task 6 is explicit manual verification (it rebuilds real artifacts).

**Type consistency:** `frac_diff(series, d, thresh, max_lags)` (Task 1) is called in Task 5 with `**fd = dict(d=..., thresh=..., max_lags=...)` — keyword names match exactly. `parkinson/garman_klass/yang_zhang(df, window)` and `atr(df, window)` (Task 2), `hurst(close, window)` / `entropy(close, window, bins)` (Task 3), and `amihud/roll_spread/corwin_schultz(df, window)` / `obv(df)` (Task 4) are all called with those exact signatures in Task 5. `FEATURE_NAMES(cfg)` / `compute_features(df, cfg)` keep their existing signatures, so `build_panel` (M0) needs no change.

**Note on the frac-diff window:** at `d=0.45`, `thresh=1e-3`, the weights fall below the threshold around **~65–70 lags** — comfortably under the 200-lag cap. So the binding warm-up is actually `hurst_window=100`, not the frac-diff cap. Task 6 Step 3 measures the real number rather than assuming it.
