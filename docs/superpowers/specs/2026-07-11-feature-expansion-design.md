# Feature Expansion — Topological & Microstructure Indicators (Design)

> Expands the causal daily feature set from **D=10 → D=22** with volatility
> estimators, stochastic-memory probes, and microstructure friction measures.

## Context

The panel currently carries 10 causal indicators (`log_return`, `sma_ratio_{5,10,20}`,
`rsi`, `macd`, `macd_signal`, `macd_hist`, `realized_vol`, `volume_z`) — all
close-to-close / trend-following transforms. We add 12 features spanning four
dimensions the current set misses: **instability** (range-based volatility),
**memory** (long-range dependence), **friction** (implicit transaction costs), and
**flow** (order-flow kinematics).

## Decisions (from analysis + brainstorming)

| Decision | Rationale |
|---|---|
| **Skip VPIN** | True VPIN needs volume-time buckets + intraday bulk-volume classification. Daily OHLCV cannot support it; a "daily VPIN" would be a misnomer. |
| **Skip Kyle's λ** | Its daily proxy (`\|return\|/volume`) **is** Amihud. Without signed intraday flow it adds no distinct signal. |
| **Drop Higuchi fractal dimension** | For fBm `D ≈ 2 − H` — on a rolling window it is close to a monotone transform of Hurst. Little marginal information. |
| **No CFI / clustering / orthogonalization stage** | The VQ-VAE codebook already performs this compression **nonlinearly**; redundant inputs are absorbed into shared codes. Redundancy (e.g. Garman-Klass vs Yang-Zhang) is therefore acceptable and is *not* pruned. |

## The 12 new features

All computed **per ticker from daily OHLCV**, all strictly **causal** (past/current
data only).

### Group 1 — Instability (range-based volatility)

| Name | Definition |
|---|---|
| `parkinson` | `sqrt( mean_w[ (1/(4·ln2)) · ln(H/L)² ] )` |
| `garman_klass` | `sqrt( mean_w[ 0.5·ln(H/L)² − (2·ln2 − 1)·ln(C/O)² ] )` (clip negative variance to 0) |
| `yang_zhang` | `sqrt( σ²_o + k·σ²_c + (1−k)·σ²_RS )` where `σ²_o = var_w(ln(O_t/C_{t−1}))`, `σ²_c = var_w(ln(C_t/O_t))`, `RS = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)`, `σ²_RS = mean_w(RS)`, `k = 0.34/(1.34 + (w+1)/(w−1))` |
| `atr_frac` | Wilder ATR over `atr_window` (`TR = max(H−L, \|H−C_{t−1}\|, \|L−C_{t−1}\|)`) → **frac-diff** |

Window = `vol_window` (existing config field, default 20).

### Group 2 — Stochastic memory

| Name | Definition |
|---|---|
| `hurst` | Rolling rescaled-range (R/S) over `hurst_window`: within each window, R/S is computed at several sub-scales and `H` is the slope of `log(R/S)` vs `log(scale)` |
| `close_frac` | **Frac-diff** of `log(close)` — stationary but memory-preserving |
| `entropy` | Rolling Shannon entropy of log-returns: histogram with `entropy_bins` over the window's own range, `−Σ p·log p` |

### Group 3 — Microstructure friction

| Name | Definition |
|---|---|
| `amihud` | `log1p( 1e6 · mean_w( \|r_t\| / (close_t · volume_t) ) )` — the log compresses its extreme right skew |
| `roll_spread` | `2·sqrt(−Cov_w(Δp_t, Δp_{t−1}))` if `Cov < 0`, **else 0**. The guard is essential: on daily equity data the serial covariance is frequently **positive**, where the Roll estimator is undefined |
| `corwin_schultz` | `β = ln(H_t/L_t)² + ln(H_{t−1}/L_{t−1})²`; `γ = ln(max(H_t,H_{t−1}) / min(L_t,L_{t−1}))²`; `α = (√(2β) − √β)/(3 − 2√2) − √(γ/(3 − 2√2))`; `S = 2(e^α − 1)/(1 + e^α)`, **negatives clamped to 0**, then averaged over `cs_window` |

### Group 4 — Flow kinematics

| Name | Definition |
|---|---|
| `volume_frac` | **Frac-diff** of `log1p(volume)` |
| `obv_frac` | `OBV = cumsum(sign(Δclose) · volume)` (non-stationary by construction) → **frac-diff** |

## The frac-diff primitive (`bubble_bi/data/fracdiff.py`)

López de Prado fixed-width-window fractional differentiation:

```
w_0 = 1 ;  w_k = −w_{k−1} · (d − k + 1) / k
keep lags while |w_k| >= frac_thresh, hard-capped at frac_max_lags
y_t = Σ_{k=0}^{L−1} w_k · x_{t−k}          # causal: past lags only
first L−1 values are NaN (warm-up)
```

**Window-length trade-off (a real number, not a detail):** the weights decay like
`k^-(d+1)`, so the threshold sets the window. At `d = 0.45`, a threshold of `1e-4`
requires **~575 lags** (~14% of our 4153 days lost to warm-up). We default to
`frac_thresh = 1e-3`, where the weights cross the threshold at **~65–70 lags** —
deep memory retained, cheap warm-up. `frac_max_lags = 200` is a safety cap that
does *not* bind at these defaults (it only matters if `d` or the threshold is
lowered). A looser threshold retains less memory — hence both are config-exposed.

**So the binding warm-up is `hurst_window = 100`, not the frac-diff window.**

Produces: `fracdiff_weights(d, thresh, max_lags) -> np.ndarray` and
`frac_diff(series, d, thresh, max_lags) -> pd.Series`.

## Code organization

`features.py` is currently ~50 lines; 12 more features plus helpers would make it
unwieldy. Split by responsibility, each a pure, independently testable module:

```
bubble_bi/data/
  fracdiff.py             # FFD primitive (weights + causal apply)
  features_volatility.py  # parkinson, garman_klass, yang_zhang, atr
  features_memory.py      # hurst, entropy
  features_micro.py       # amihud, roll_spread, corwin_schultz, obv
  features.py             # ORCHESTRATOR: FEATURE_NAMES + compute_features
```

`features.py` keeps its **exact public interface** (`compute_features(df, cfg)`,
`FEATURE_NAMES(cfg)`), so **nothing downstream changes** — `build_panel` and every
model pick up the new width automatically via `d_in = panel.features.shape[2]`.

Final column order (D=22): the existing 10, then volatility (4), memory (3),
microstructure (3), flow (2).

## Config (`FeatureConfig` additions)

| Field | Default |
|---|---|
| `frac_d` | 0.45 |
| `frac_thresh` | 1e-3 |
| `frac_max_lags` | 200 |
| `atr_window` | 14 |
| `hurst_window` | 100 |
| `entropy_window` | 60 |
| `entropy_bins` | 10 |
| `amihud_window` | 21 |
| `roll_window` | 21 |
| `cs_window` | 21 |

`vol_window` (=20) already exists and is reused by the range estimators.

## Performance

30 tickers × 4153 days. Use `numpy.lib.stride_tricks.sliding_window_view` for
Hurst / entropy / Roll covariance, `np.convolve` for frac-diff, and pandas
`rolling` for simple moments. **No `rolling.apply`** — it would take minutes.
Target: `build-panel` stays well under a minute.

## Consequences

- **Warm-up grows to ~100 days** (driven by `hurst_window`; the frac-diff window is
  ~65–70 lags at the defaults), so `Panel.mask` drops more early rows (~2–3% of
  history). Handled automatically — the mask already gates every consumer. The
  build step measures the real valid fraction rather than assuming it.
- **D: 10 → 22 invalidates all four checkpoints** (`d_in` mismatch on load).
  `panel.npz` and `tokens.npz` must be rebuilt and **every stage retrained**
  (TS → CS → fusion → predictor). This is expected and is what the Colab GPU
  migration is for.
- The M0 ridge baseline will also change (more regressors) — a new RankIC floor.

## Error handling & pitfalls

- **Roll spread undefined** when serial covariance ≥ 0 → return 0 (tested).
- **Corwin-Schultz negative spreads** → clamp to 0 (tested).
- **Garman-Klass negative variance** (possible from the `−(2ln2−1)` term) → clip
  to 0 before `sqrt`.
- **Amihud extreme skew** → `log1p` compression; zero-volume days give `inf` →
  guarded and left NaN (masked out).
- **Hurst on flat windows** (zero variance) → NaN, masked out.
- **Frac-diff warm-up** → leading NaNs, masked out.
- **Causality is non-negotiable**: every feature uses only backward-looking
  windows; the existing truncation test is the enforcement mechanism.

## Testing (TDD)

- **`fracdiff`**: weights satisfy the recurrence; `d=0` ≈ identity; `d=1` ≈ first
  difference; output is causal (truncating the future leaves the past unchanged).
- **Volatility**: all three estimators are ≥ 0 and finite after warm-up; on a
  synthetic series with constant `H/L` ratio, `parkinson` equals the closed form.
- **`roll_spread`**: returns exactly 0 on a series with positive serial covariance
  (e.g. a strong trend); positive on a bid-ask-bounce series.
- **`corwin_schultz`**: never negative.
- **`hurst`**: ≈ 0.5 (±0.15) on a random walk; > 0.6 on a strong trend.
- **`entropy`**: ≈ 0 on constant returns; higher on noisy returns.
- **`amihud`**: monotonically decreasing in volume.
- **Orchestrator**: `len(FEATURE_NAMES(cfg)) == 22`; `compute_features` returns
  those exact columns in order.
- **The existing causality truncation test must still pass** — it iterates every
  output column, so it covers all 12 new features automatically. This is the
  headline guarantee.

## Verification (end-to-end)

1. `pytest` → all existing + new tests green (esp. the causality test).
2. `build-panel --config configs/m0.yaml` → panel shape `(≈4153, 30, 22)`; report
   the new valid-mask fraction (expect a drop from 99.0% due to warm-up).
3. `baseline` → a new RankIC floor with 22 regressors (record it; it replaces
   0.0062 as the number M3 must beat).
4. Retrain the full stack on Colab GPU (TS → CS → fusion → tokenize → predictor)
   and compare against the D=10 results.

## Out of scope

VPIN, Kyle's λ, Higuchi fractal dimension, any CFI/clustering/feature-selection
stage, intraday data ingestion.
