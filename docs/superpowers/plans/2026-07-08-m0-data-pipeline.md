# M0 — Data Pipeline + Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leak-free, cached, walk-forward US-equity data pipeline (yfinance → feature panel → splits) and a ridge-regression next-day-return baseline that establishes a RankIC floor.

**Architecture:** A small Python package `bubble_bi`. Data flows: yfinance → per-ticker parquet cache → causal technical features → an aligned `[T, N, D]` panel with a strictly-forward-shifted target → walk-forward split windows → a scikit-learn Ridge baseline evaluated with RankIC/RankICIR. Every stage is a focused module with one responsibility; a data-source abstraction keeps ingestion testable without network access.

**Tech Stack:** Python 3.14, numpy, pandas, scikit-learn, pyarrow (parquet), yfinance, PyYAML, pytest. (No PyTorch in M0 — it arrives in M1.)

## Global Constraints

- Python interpreter: `python3` (3.14.4 is the only one installed); all work happens inside a project venv at `.venv`.
- **No lookahead, ever:** a feature at day `t` may use data only up to and including day `t`; the target at day `t` is `close[t+1]/close[t] - 1`. This is enforced by tests, not convention.
- **No network in tests:** ingestion is tested through an injected fake `PriceSource`; only the real CLI touches yfinance.
- Package importable/runnable from the repo root as `python -m bubble_bi.cli <command>`.
- All generated data lives under `artifacts/` (git-ignored); never commit downloaded market data.
- TDD throughout: failing test first, minimal code, passing test, commit. Frequent commits.
- Reproducibility: any randomness uses `Config.seed`.

---

### Task 0: Project scaffolding, venv, and config loader

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `bubble_bi/__init__.py`
- Create: `bubble_bi/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`
- Create: `configs/m0.yaml`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `DataConfig`, `FeatureConfig`, `SplitConfig`, `Config` dataclasses and `load_config(path: str) -> Config` in `bubble_bi/config.py`.

- [ ] **Step 1: Initialize git and the venv, bootstrap pip**

Run from repo root (`/home/hockper/Documents/Code/Bubble Bi`):
```bash
git init
python3 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install --upgrade pip
```
Expected: `.venv/` created; `pip --version` works. If `ensurepip` fails, run `sudo apt-get install -y python3-venv python3-pip` then recreate the venv.

- [ ] **Step 2: Write `.gitignore` and `requirements.txt`**

`.gitignore`:
```
.venv/
artifacts/
__pycache__/
*.pyc
.pytest_cache/
```

`requirements.txt`:
```
numpy
pandas
scikit-learn
pyarrow
yfinance
PyYAML
pytest
```

- [ ] **Step 3: Install dependencies**

Run: `.venv/bin/python -m pip install -r requirements.txt`
Expected: all install successfully. If a package has no cp314 wheel, note it and pin an available version; M0's packages (numpy/pandas/sklearn/pyarrow/yfinance) publish 3.14 wheels.

- [ ] **Step 4: Write the failing test**

`tests/test_config.py`:
```python
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
```

- [ ] **Step 5: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bubble_bi'`.

- [ ] **Step 6: Write `bubble_bi/__init__.py`, `tests/__init__.py`, and `bubble_bi/config.py`**

`bubble_bi/__init__.py`: empty file.
`tests/__init__.py`: empty file.

`bubble_bi/config.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import yaml


@dataclass
class DataConfig:
    tickers: list[str]
    start: str | None = None
    end: str | None = None
    raw_dir: str = "artifacts/raw"
    cache_dir: str = "artifacts/cache"
    min_history: int = 252


@dataclass
class FeatureConfig:
    ma_windows: list[int] = field(default_factory=lambda: [5, 10, 20])
    rsi_window: int = 14
    vol_window: int = 20
    volume_window: int = 20


@dataclass
class SplitConfig:
    train_days: int = 756
    val_days: int = 126
    test_days: int = 126
    step_days: int = 126


@dataclass
class Config:
    data: DataConfig
    features: FeatureConfig = field(default_factory=FeatureConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    seed: int = 42


def _build(cls: type, data: dict[str, Any]) -> Any:
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        if is_dataclass(f.type) if isinstance(f.type, type) else False:
            kwargs[f.name] = _build(f.type, data[f.name])
        else:
            kwargs[f.name] = data[f.name]
    return cls(**kwargs)


def load_config(path: str) -> Config:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    data = DataConfig(**raw.get("data", {"tickers": []}))
    features = FeatureConfig(**raw.get("features", {}))
    splits = SplitConfig(**raw.get("splits", {}))
    return Config(data=data, features=features, splits=splits, seed=raw.get("seed", 42))
```

- [ ] **Step 7: Write `configs/m0.yaml`**

```yaml
data:
  tickers: [AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, V, JNJ, WMT,
            PG, HD, BAC, XOM, CVX, KO, PEP, DIS, CSCO, INTC,
            VZ, T, MRK, PFE, ABT, NKE, MCD, CAT, BA, IBM]
  start: "2010-01-01"
  min_history: 252
features:
  ma_windows: [5, 10, 20]
  rsi_window: 14
  vol_window: 20
  volume_window: 20
splits:
  train_days: 756
  val_days: 126
  test_days: 126
  step_days: 126
seed: 42
```

- [ ] **Step 8: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add .gitignore requirements.txt bubble_bi/ tests/ configs/
git commit -m "chore: scaffold bubble_bi package and config loader"
```

---

### Task 1: Ticker universe

**Files:**
- Create: `bubble_bi/data/__init__.py`
- Create: `bubble_bi/data/universe.py`
- Create: `tests/test_universe.py`

**Interfaces:**
- Consumes: `DataConfig` from Task 0.
- Produces: `load_universe(cfg: DataConfig) -> list[str]` in `bubble_bi/data/universe.py`.

- [ ] **Step 1: Write the failing test**

`tests/test_universe.py`:
```python
from bubble_bi.config import DataConfig
from bubble_bi.data.universe import load_universe


def test_load_universe_returns_configured_tickers():
    cfg = DataConfig(tickers=["AAPL", "MSFT", "AAPL"])
    assert load_universe(cfg) == ["AAPL", "MSFT"]  # dedup, order preserved


def test_load_universe_rejects_empty():
    import pytest
    with pytest.raises(ValueError):
        load_universe(DataConfig(tickers=[]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_universe.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write implementation**

`bubble_bi/data/__init__.py`: empty file.

`bubble_bi/data/universe.py`:
```python
from __future__ import annotations

from bubble_bi.config import DataConfig


def load_universe(cfg: DataConfig) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in cfg.tickers:
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    if not out:
        raise ValueError("universe is empty; set data.tickers in the config")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_universe.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/data/ tests/test_universe.py
git commit -m "feat: ticker universe loader with dedup and validation"
```

---

### Task 2: Metrics (RankIC / RankICIR / IC)

**Files:**
- Create: `bubble_bi/eval/__init__.py`
- Create: `bubble_bi/eval/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Consumes: numpy arrays only.
- Produces, in `bubble_bi/eval/metrics.py`:
  - `daily_rank_ic(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray` — shape `[T]`, NaN on days with <2 valid names.
  - `rank_ic(pred, target, mask) -> float` — mean of daily rank IC (NaNs ignored).
  - `rank_icir(pred, target, mask) -> float` — `mean / std` of daily rank IC.
  - `information_coefficient(pred, target, mask) -> float` — mean daily Pearson IC.
  - All accept `pred`/`target`/`mask` of shape `[T, N]`.

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:
```python
import numpy as np

from bubble_bi.eval.metrics import daily_rank_ic, rank_ic, rank_icir


def _mask_all(shape):
    return np.ones(shape, dtype=bool)


def test_perfect_ranking_gives_rank_ic_one():
    target = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    pred = target.copy()
    assert rank_ic(pred, target, _mask_all(target.shape)) == 1.0


def test_reversed_ranking_gives_rank_ic_minus_one():
    target = np.array([[1.0, 2.0, 3.0, 4.0]])
    pred = -target
    assert rank_ic(pred, target, _mask_all(target.shape)) == -1.0


def test_masked_names_are_ignored():
    target = np.array([[1.0, 2.0, 3.0, np.nan]])
    pred = np.array([[1.0, 2.0, 3.0, 999.0]])
    mask = np.array([[True, True, True, False]])
    assert rank_ic(pred, target, mask) == 1.0


def test_day_with_one_valid_name_is_nan():
    target = np.array([[1.0, np.nan, np.nan]])
    pred = np.array([[1.0, 2.0, 3.0]])
    mask = np.array([[True, False, False]])
    assert np.isnan(daily_rank_ic(pred, target, mask)[0])


def test_rank_icir_is_finite_for_varied_days():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(50, 10))
    pred = target + rng.normal(scale=0.5, size=(50, 10))
    val = rank_icir(pred, target, _mask_all(target.shape))
    assert np.isfinite(val)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write implementation**

`bubble_bi/eval/__init__.py`: empty file.

`bubble_bi/eval/metrics.py`:
```python
from __future__ import annotations

import numpy as np


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = x.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def daily_rank_ic(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    T = pred.shape[0]
    out = np.full(T, np.nan)
    for t in range(T):
        m = mask[t] & np.isfinite(pred[t]) & np.isfinite(target[t])
        if m.sum() < 2:
            continue
        out[t] = _corr(_rankdata(pred[t][m]), _rankdata(target[t][m]))
    return out


def rank_ic(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    daily = daily_rank_ic(pred, target, mask)
    return float(np.nanmean(daily))


def rank_icir(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    daily = daily_rank_ic(pred, target, mask)
    sd = np.nanstd(daily)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return float(np.nanmean(daily) / sd)


def information_coefficient(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    T = pred.shape[0]
    vals = np.full(T, np.nan)
    for t in range(T):
        m = mask[t] & np.isfinite(pred[t]) & np.isfinite(target[t])
        if m.sum() < 2:
            continue
        vals[t] = _corr(pred[t][m], target[t][m])
    return float(np.nanmean(vals))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/eval/ tests/test_metrics.py
git commit -m "feat: rank IC / rank ICIR / IC metrics with masking"
```

---

### Task 3: Causal technical features

**Files:**
- Create: `bubble_bi/data/features.py`
- Create: `tests/test_features.py`

**Interfaces:**
- Consumes: `FeatureConfig` from Task 0; a per-ticker OHLCV `pd.DataFrame` indexed by a `DatetimeIndex` with columns `open, high, low, close, volume`.
- Produces: `compute_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame` returning a DataFrame indexed by the same dates with one column per feature (NaN during warmup). `FEATURE_NAMES(cfg: FeatureConfig) -> list[str]` returns the column order.

- [ ] **Step 1: Write the failing test**

`tests/test_features.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write implementation**

`bubble_bi/data/features.py`:
```python
from __future__ import annotations

import numpy as np
import pandas as pd

from bubble_bi.config import FeatureConfig


def FEATURE_NAMES(cfg: FeatureConfig) -> list[str]:
    names = ["log_return"]
    names += [f"sma_ratio_{w}" for w in cfg.ma_windows]
    names += ["rsi", "macd", "macd_signal", "macd_hist", "realized_vol", "volume_z"]
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

    return out[FEATURE_NAMES(cfg)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_features.py -v`
Expected: PASS (3 tests). The causality test is the critical guard.

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/data/features.py tests/test_features.py
git commit -m "feat: causal technical features (verified lookahead-free)"
```

---

### Task 4: Ingestion with an injectable price source

**Files:**
- Create: `bubble_bi/data/ingest.py`
- Create: `tests/test_ingest.py`

**Interfaces:**
- Consumes: nothing from prior tasks (uses stdlib + pandas + pyarrow).
- Produces, in `bubble_bi/data/ingest.py`:
  - `class PriceSource(Protocol)` with `fetch(self, ticker: str, start: str | None, end: str | None) -> pd.DataFrame` returning an OHLCV frame indexed by a `DatetimeIndex` with columns `open, high, low, close, volume`.
  - `class YFinanceSource` implementing `PriceSource` via `yfinance.download(..., auto_adjust=True)`.
  - `ingest(tickers: list[str], source: PriceSource, raw_dir: str, start=None, end=None) -> dict[str, str]` — writes/updates `raw_dir/<TICKER>.parquet` incrementally and returns the paths. Idempotent: re-running fetches only dates after the last cached date.

- [ ] **Step 1: Write the failing test**

`tests/test_ingest.py`:
```python
import numpy as np
import pandas as pd

from bubble_bi.data.ingest import ingest


class FakeSource:
    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def fetch(self, ticker, start, end):
        self.calls.append((ticker, start, end))
        df = self.frames[ticker]
        if start is not None:
            df = df[df.index > pd.Timestamp(start)]
        return df


def _frame(n=10, seed=0):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    c = pd.Series(100 + np.cumsum(rng.normal(size=n)), index=dates)
    return pd.DataFrame(
        {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1e6},
        index=dates,
    )


def test_ingest_writes_parquet(tmp_path):
    src = FakeSource({"AAPL": _frame()})
    paths = ingest(["AAPL"], src, str(tmp_path))
    got = pd.read_parquet(paths["AAPL"])
    assert list(got.columns) == ["open", "high", "low", "close", "volume"]
    assert len(got) == 10


def test_ingest_is_incremental(tmp_path):
    full = _frame(n=15)
    src = FakeSource({"AAPL": full.iloc[:10]})
    ingest(["AAPL"], src, str(tmp_path))
    # add later data and re-run; should only fetch the tail
    src.frames["AAPL"] = full
    ingest(["AAPL"], src, str(tmp_path))
    got = pd.read_parquet(next(iter((tmp_path).glob("AAPL.parquet"))))
    assert len(got) == 15
    # second call requested a start date (incremental), not a full refetch
    assert src.calls[-1][1] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write implementation**

`bubble_bi/data/ingest.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


class PriceSource(Protocol):
    def fetch(self, ticker: str, start: str | None, end: str | None) -> pd.DataFrame: ...


class YFinanceSource:
    def fetch(self, ticker: str, start: str | None, end: str | None) -> pd.DataFrame:
        import yfinance as yf

        df = yf.download(
            ticker, start=start, end=end, auto_adjust=True, progress=False
        )
        if df.empty:
            return pd.DataFrame(columns=COLUMNS)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df


def ingest(
    tickers: list[str],
    source: PriceSource,
    raw_dir: str,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, str]:
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for ticker in tickers:
        path = Path(raw_dir) / f"{ticker}.parquet"
        existing = pd.read_parquet(path) if path.exists() else None
        fetch_start = start
        if existing is not None and len(existing):
            fetch_start = str(existing.index.max().date())
        new = source.fetch(ticker, fetch_start, end)
        if existing is not None and len(existing):
            combined = pd.concat([existing, new])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = new.sort_index()
        combined[COLUMNS].to_parquet(path)
        paths[ticker] = str(path)
    return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ingest.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/data/ingest.py tests/test_ingest.py
git commit -m "feat: incremental parquet ingestion with injectable price source"
```

---

### Task 5: Panel assembly (align + target + mask)

**Files:**
- Create: `bubble_bi/data/panel.py`
- Create: `tests/test_panel.py`

**Interfaces:**
- Consumes: `compute_features`/`FEATURE_NAMES` (Task 3), `FeatureConfig`/`DataConfig` (Task 0), `COLUMNS` (Task 4).
- Produces, in `bubble_bi/data/panel.py`:
  - `@dataclass Panel` with `dates: pd.DatetimeIndex`, `tickers: list[str]`, `features: np.ndarray [T,N,D]`, `target: np.ndarray [T,N]`, `mask: np.ndarray [T,N] bool`, `feature_names: list[str]`.
  - `build_panel(per_ticker: dict[str, pd.DataFrame], data_cfg: DataConfig, feat_cfg: FeatureConfig) -> Panel`.
  - `save_panel(panel: Panel, path: str) -> None` / `load_panel(path: str) -> Panel` (npz).

- [ ] **Step 1: Write the failing test**

`tests/test_panel.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_panel.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write implementation**

`bubble_bi/data/panel.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bubble_bi.config import DataConfig, FeatureConfig
from bubble_bi.data.features import FEATURE_NAMES, compute_features


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    tickers: list[str]
    features: np.ndarray  # [T, N, D]
    target: np.ndarray  # [T, N]
    mask: np.ndarray  # [T, N] bool
    feature_names: list[str]


def build_panel(
    per_ticker: dict[str, pd.DataFrame],
    data_cfg: DataConfig,
    feat_cfg: FeatureConfig,
) -> Panel:
    feature_names = FEATURE_NAMES(feat_cfg)
    kept = {t: df for t, df in per_ticker.items() if len(df) >= data_cfg.min_history}
    tickers = sorted(kept)
    if not tickers:
        raise ValueError("no ticker meets min_history")

    all_dates = sorted(set().union(*[df.index for df in kept.values()]))
    dates = pd.DatetimeIndex(all_dates)
    T, N, D = len(dates), len(tickers), len(feature_names)

    features = np.full((T, N, D), np.nan, dtype=np.float32)
    target = np.full((T, N), np.nan, dtype=np.float32)

    for j, t in enumerate(tickers):
        df = kept[t].reindex(dates)
        feats = compute_features(df, feat_cfg)
        features[:, j, :] = feats.to_numpy(dtype=np.float32)
        close = df["close"].astype(float)
        target[:, j] = (close.shift(-1) / close - 1.0).to_numpy(dtype=np.float32)

    mask = np.isfinite(target) & np.isfinite(features).all(axis=2)
    return Panel(dates, tickers, features, target, mask, feature_names)


def save_panel(panel: Panel, path: str) -> None:
    np.savez_compressed(
        path,
        dates=panel.dates.astype("datetime64[ns]").to_numpy(),
        tickers=np.array(panel.tickers, dtype=object),
        features=panel.features,
        target=panel.target,
        mask=panel.mask,
        feature_names=np.array(panel.feature_names, dtype=object),
    )


def load_panel(path: str) -> Panel:
    z = np.load(path, allow_pickle=True)
    return Panel(
        dates=pd.DatetimeIndex(z["dates"]),
        tickers=list(z["tickers"]),
        features=z["features"],
        target=z["target"],
        mask=z["mask"],
        feature_names=list(z["feature_names"]),
    )
```

Note: `.npz` needs a filename; if `path` lacks `.npz`, numpy appends it. Tests pass an explicit `.npz`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_panel.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/data/panel.py tests/test_panel.py
git commit -m "feat: aligned [T,N,D] panel with forward-shifted target and mask"
```

---

### Task 6: Walk-forward splits

**Files:**
- Create: `bubble_bi/data/splits.py`
- Create: `tests/test_splits.py`

**Interfaces:**
- Consumes: `SplitConfig` from Task 0.
- Produces, in `bubble_bi/data/splits.py`:
  - `@dataclass WalkForwardSplit` with `train: tuple[int, int]`, `val: tuple[int, int]`, `test: tuple[int, int]` (each a `[start, end)` half-open index range into the panel's date axis).
  - `walk_forward_splits(n_dates: int, cfg: SplitConfig) -> list[WalkForwardSplit]`.

- [ ] **Step 1: Write the failing test**

`tests/test_splits.py`:
```python
from bubble_bi.config import SplitConfig
from bubble_bi.data.splits import walk_forward_splits


def test_splits_are_ordered_and_non_overlapping():
    cfg = SplitConfig(train_days=100, val_days=20, test_days=20, step_days=20)
    splits = walk_forward_splits(200, cfg)
    assert len(splits) >= 1
    for s in splits:
        assert s.train[0] < s.train[1] == s.val[0] < s.val[1] == s.test[0] < s.test[1]
        assert s.test[1] <= 200


def test_windows_advance_by_step():
    cfg = SplitConfig(train_days=100, val_days=20, test_days=20, step_days=20)
    splits = walk_forward_splits(260, cfg)
    assert splits[1].train[0] - splits[0].train[0] == 20


def test_no_split_when_insufficient_history():
    cfg = SplitConfig(train_days=100, val_days=20, test_days=20, step_days=20)
    assert walk_forward_splits(50, cfg) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_splits.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write implementation**

`bubble_bi/data/splits.py`:
```python
from __future__ import annotations

from dataclasses import dataclass

from bubble_bi.config import SplitConfig


@dataclass
class WalkForwardSplit:
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]


def walk_forward_splits(n_dates: int, cfg: SplitConfig) -> list[WalkForwardSplit]:
    window = cfg.train_days + cfg.val_days + cfg.test_days
    splits: list[WalkForwardSplit] = []
    start = 0
    while start + window <= n_dates:
        tr_end = start + cfg.train_days
        va_end = tr_end + cfg.val_days
        te_end = va_end + cfg.test_days
        splits.append(
            WalkForwardSplit(
                train=(start, tr_end), val=(tr_end, va_end), test=(va_end, te_end)
            )
        )
        start += cfg.step_days
    return splits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_splits.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/data/splits.py tests/test_splits.py
git commit -m "feat: rolling walk-forward split generator"
```

---

### Task 7: Ridge baseline

**Files:**
- Create: `bubble_bi/baselines/__init__.py`
- Create: `bubble_bi/baselines/ridge.py`
- Create: `tests/test_baseline.py`

**Interfaces:**
- Consumes: `Panel` (Task 5), `WalkForwardSplit`/`walk_forward_splits` (Task 6), `SplitConfig` (Task 0), `rank_ic`/`rank_icir` (Task 2).
- Produces, in `bubble_bi/baselines/ridge.py`:
  - `predict_test_ridge(panel: Panel, split: WalkForwardSplit, alpha: float = 1.0) -> np.ndarray` — predictions of shape `[test_len, N]`, NaN where the test entry is masked or has non-finite features.
  - `evaluate_baseline(panel: Panel, splits: list[WalkForwardSplit], alpha: float = 1.0) -> dict` — returns `{"rank_ic": float, "rank_icir": float, "n_splits": int}` aggregated over the concatenated test windows.

- [ ] **Step 1: Write the failing test**

`tests/test_baseline.py`:
```python
import numpy as np
import pandas as pd

from bubble_bi.config import DataConfig, FeatureConfig, SplitConfig
from bubble_bi.data.panel import build_panel
from bubble_bi.data.splits import walk_forward_splits
from bubble_bi.baselines.ridge import predict_test_ridge, evaluate_baseline


def _panel_with_signal(n=400, N=12, seed=0):
    rng = np.random.default_rng(seed)
    per = {}
    for k in range(N):
        dates = pd.bdate_range("2015-01-01", periods=n)
        c = pd.Series(100 + np.cumsum(rng.normal(size=n)), index=dates)
        per[f"T{k:02d}"] = pd.DataFrame(
            {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1e6},
            index=dates,
        )
    return build_panel(per, DataConfig(tickers=list(per), min_history=50), FeatureConfig())


def test_prediction_shape_matches_test_window():
    panel = _panel_with_signal()
    cfg = SplitConfig(train_days=200, val_days=40, test_days=40, step_days=40)
    split = walk_forward_splits(len(panel.dates), cfg)[0]
    preds = predict_test_ridge(panel, split, alpha=1.0)
    assert preds.shape == (40, len(panel.tickers))


def test_evaluate_baseline_returns_finite_metrics():
    panel = _panel_with_signal()
    cfg = SplitConfig(train_days=200, val_days=40, test_days=40, step_days=40)
    splits = walk_forward_splits(len(panel.dates), cfg)
    result = evaluate_baseline(panel, splits, alpha=1.0)
    assert result["n_splits"] == len(splits)
    assert np.isfinite(result["rank_ic"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write implementation**

`bubble_bi/baselines/__init__.py`: empty file.

`bubble_bi/baselines/ridge.py`:
```python
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from bubble_bi.data.panel import Panel
from bubble_bi.data.splits import WalkForwardSplit
from bubble_bi.eval.metrics import rank_ic, rank_icir


def _flatten(panel: Panel, lo: int, hi: int):
    T, N, D = panel.features.shape
    X = panel.features[lo:hi].reshape(-1, D)
    y = panel.target[lo:hi].reshape(-1)
    m = panel.mask[lo:hi].reshape(-1) & np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X, y, m


def predict_test_ridge(panel: Panel, split: WalkForwardSplit, alpha: float = 1.0) -> np.ndarray:
    lo_tr, hi_tr = split.train
    Xtr, ytr, mtr = _flatten(panel, lo_tr, hi_tr)
    scaler = StandardScaler().fit(Xtr[mtr])
    model = Ridge(alpha=alpha).fit(scaler.transform(Xtr[mtr]), ytr[mtr])

    lo_te, hi_te = split.test
    N = panel.features.shape[1]
    D = panel.features.shape[2]
    Xte = panel.features[lo_te:hi_te].reshape(-1, D)
    valid = np.isfinite(Xte).all(axis=1)
    preds = np.full(Xte.shape[0], np.nan)
    if valid.any():
        preds[valid] = model.predict(scaler.transform(Xte[valid]))
    return preds.reshape(hi_te - lo_te, N)


def evaluate_baseline(panel: Panel, splits: list[WalkForwardSplit], alpha: float = 1.0) -> dict:
    preds_all, target_all, mask_all = [], [], []
    for split in splits:
        lo_te, hi_te = split.test
        preds_all.append(predict_test_ridge(panel, split, alpha))
        target_all.append(panel.target[lo_te:hi_te])
        mask_all.append(panel.mask[lo_te:hi_te])
    if not splits:
        return {"rank_ic": float("nan"), "rank_icir": float("nan"), "n_splits": 0}
    pred = np.concatenate(preds_all, axis=0)
    target = np.concatenate(target_all, axis=0)
    mask = np.concatenate(mask_all, axis=0)
    return {
        "rank_ic": rank_ic(pred, target, mask),
        "rank_icir": rank_icir(pred, target, mask),
        "n_splits": len(splits),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baseline.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/baselines/ tests/test_baseline.py
git commit -m "feat: walk-forward ridge baseline with rank IC evaluation"
```

---

### Task 8: CLI wiring + end-to-end integration test

**Files:**
- Create: `bubble_bi/cli.py`
- Create: `tests/test_cli_integration.py`

**Interfaces:**
- Consumes: everything above.
- Produces, in `bubble_bi/cli.py`:
  - `build_panel_from_raw(cfg: Config) -> Panel` — reads `raw_dir/<TICKER>.parquet`, builds and caches the panel to `cache_dir/panel.npz`.
  - `run_baseline(cfg: Config) -> dict` — loads/builds the panel, generates walk-forward splits, evaluates the ridge baseline, prints and returns the metrics.
  - `main(argv: list[str] | None = None) -> int` — argparse with subcommands `ingest`, `build-panel`, `baseline`, each taking `--config`.

- [ ] **Step 1: Write the failing test**

`tests/test_cli_integration.py`:
```python
import numpy as np
import pandas as pd

from bubble_bi.config import Config, DataConfig, FeatureConfig, SplitConfig
from bubble_bi.cli import build_panel_from_raw, run_baseline


def _write_raw(raw_dir, n=400, N=12, seed=0):
    rng = np.random.default_rng(seed)
    for k in range(N):
        dates = pd.bdate_range("2015-01-01", periods=n)
        c = pd.Series(100 + np.cumsum(rng.normal(size=n)), index=dates)
        df = pd.DataFrame(
            {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1e6},
            index=dates,
        )
        df.index.name = "date"
        df.to_parquet(f"{raw_dir}/T{k:02d}.parquet")


def test_end_to_end_pipeline_produces_finite_rank_ic(tmp_path):
    raw = tmp_path / "raw"
    cache = tmp_path / "cache"
    raw.mkdir()
    cache.mkdir()
    _write_raw(str(raw))
    cfg = Config(
        data=DataConfig(
            tickers=[f"T{k:02d}" for k in range(12)],
            raw_dir=str(raw),
            cache_dir=str(cache),
            min_history=50,
        ),
        features=FeatureConfig(),
        splits=SplitConfig(train_days=200, val_days=40, test_days=40, step_days=40),
    )
    panel = build_panel_from_raw(cfg)
    assert (cache / "panel.npz").exists()
    assert panel.features.shape[1] == 12
    result = run_baseline(cfg)
    assert result["n_splits"] >= 1
    assert np.isfinite(result["rank_ic"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli_integration.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError`.

- [ ] **Step 3: Write implementation**

`bubble_bi/cli.py`:
```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bubble_bi.config import Config, load_config
from bubble_bi.data.ingest import YFinanceSource, ingest
from bubble_bi.data.panel import Panel, build_panel, load_panel, save_panel
from bubble_bi.data.splits import walk_forward_splits
from bubble_bi.data.universe import load_universe
from bubble_bi.baselines.ridge import evaluate_baseline


def run_ingest(cfg: Config) -> dict[str, str]:
    tickers = load_universe(cfg.data)
    return ingest(tickers, YFinanceSource(), cfg.data.raw_dir, cfg.data.start, cfg.data.end)


def build_panel_from_raw(cfg: Config) -> Panel:
    tickers = load_universe(cfg.data)
    per: dict[str, pd.DataFrame] = {}
    for t in tickers:
        path = Path(cfg.data.raw_dir) / f"{t}.parquet"
        if path.exists():
            per[t] = pd.read_parquet(path)
    if not per:
        raise FileNotFoundError(f"no parquet files in {cfg.data.raw_dir}; run ingest first")
    panel = build_panel(per, cfg.data, cfg.features)
    Path(cfg.data.cache_dir).mkdir(parents=True, exist_ok=True)
    save_panel(panel, str(Path(cfg.data.cache_dir) / "panel.npz"))
    return panel


def run_baseline(cfg: Config) -> dict:
    cache = Path(cfg.data.cache_dir) / "panel.npz"
    panel = load_panel(str(cache)) if cache.exists() else build_panel_from_raw(cfg)
    splits = walk_forward_splits(len(panel.dates), cfg.splits)
    result = evaluate_baseline(panel, splits)
    print(f"walk-forward splits: {result['n_splits']}")
    print(f"RankIC:    {result['rank_ic']:.4f}")
    print(f"RankICIR:  {result['rank_icir']:.4f}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bubble_bi")
    parser.add_argument("command", choices=["ingest", "build-panel", "baseline"])
    parser.add_argument("--config", default="configs/m0.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "ingest":
        paths = run_ingest(cfg)
        print(f"ingested {len(paths)} tickers into {cfg.data.raw_dir}")
    elif args.command == "build-panel":
        panel = build_panel_from_raw(cfg)
        print(f"panel: {panel.features.shape} dates={len(panel.dates)}")
    elif args.command == "baseline":
        run_baseline(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test + full suite to verify green**

Run: `.venv/bin/python -m pytest -v`
Expected: ALL tests pass across all files.

- [ ] **Step 5: Commit**

```bash
git add bubble_bi/cli.py tests/test_cli_integration.py
git commit -m "feat: CLI (ingest / build-panel / baseline) + end-to-end test"
```

---

### Task 9: Real-data smoke run (manual verification, no new tests)

**Files:** none created; this exercises the CLI against live yfinance.

- [ ] **Step 1: Ingest real data**

Run: `.venv/bin/python -m bubble_bi.cli ingest --config configs/m0.yaml`
Expected: `artifacts/raw/*.parquet` for the 30 configured tickers; re-running is fast (incremental).

- [ ] **Step 2: Build the panel**

Run: `.venv/bin/python -m bubble_bi.cli build-panel --config configs/m0.yaml`
Expected: prints a panel shape like `(N_dates, ~30, 9)` and writes `artifacts/cache/panel.npz`.

- [ ] **Step 3: Run the baseline**

Run: `.venv/bin/python -m bubble_bi.cli baseline --config configs/m0.yaml`
Expected: prints the number of walk-forward splits and a RankIC / RankICIR. A small positive RankIC (roughly 0.01–0.05) is a plausible floor; the point is a finite, reproducible number, not a strong result.

- [ ] **Step 4: Record the baseline number**

Append the observed RankIC/RankICIR and the config used to the M0 section of the design spec (`docs/superpowers/specs/2026-07-08-storm-trading-design.md`) so M1 has a documented floor to beat. Commit:
```bash
git add -A && git commit -m "docs: record M0 ridge baseline RankIC floor"
```

---

## Self-Review

**Spec coverage (M0 items from the design spec):**
- ingest → ✅ Task 4 (+ real run Task 9)
- panel → ✅ Task 5
- features (causal) → ✅ Task 3
- splits (walk-forward) → ✅ Task 6
- DataLoader → deferred to M1 (the torch windowed `Dataset` belongs with the VQ-VAE that consumes it; M0's baseline needs only the flattened panel — noted here so it is a conscious deferral, not a gap).
- ridge baseline / RankIC floor → ✅ Task 7 + Task 9
- no-lookahead enforcement → ✅ Task 3 causality test + Task 5 target test
- config-driven + caching → ✅ Task 0 config, Task 5 npz cache, Task 8 panel cache
- reproducibility (seed) → ✅ `Config.seed` (baseline is deterministic; used by models in M1)

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows complete code.

**Type consistency:** `Panel`, `WalkForwardSplit`, `FEATURE_NAMES`, `compute_features`, `rank_ic`, `predict_test_ridge`, `evaluate_baseline`, `build_panel_from_raw`, `run_baseline` are defined once and referenced with matching signatures across Tasks 2–8. `PriceSource.fetch` signature matches the `FakeSource` in tests and `YFinanceSource`.

**Deferred-with-note (not gaps):** cross-sectional per-day feature standardization (spec mentions it) is intentionally left to the M1 model dataset; the baseline uses a train-fit `StandardScaler`, which is leak-free. Point-in-time universe membership / survivorship remains a documented later enhancement.

---

## M0 Results (recorded 2026-07-09)

Config: `configs/m0.yaml` — 30 DJ30-like tickers, daily bars from 2010-01-01, features = 10 causal indicators, walk-forward train=756 / val=126 / test=126 / step=126.

| Metric | Value |
|---|---|
| Panel shape | `(4153 dates, 30 stocks, 10 features)` |
| Walk-forward splits | 25 |
| **RankIC** (pooled ridge, aggregated over test windows) | **0.0062** |
| **RankICIR** | **0.0230** |

This is the leak-free floor M1's Dual VQ-VAE + Transformer must beat. All 22 unit/integration tests pass; the baseline is deterministic and reproducible via the CLI.
