"""Turning the table into the grids the models eat.

Both entries want the same shape of thing — a grid of [companies × days × features]:

    TS   1 company  × 4 days × 26    one sample per company per day
    CS  30 companies × 5 days × 26   one sample per day

This module cuts those grids out of the feature table and hands them to the models
in batches, divided into three periods of history.

⚠️ Three ways a trading model quietly cheats, and what stops each one here:

  1. SHUFFLING TIME. If you split days at random, the model trains on Thursday and
     is tested on Wednesday. We split strictly by DATE — learn on the past, test on
     the future.

  2. SCALING WITH THE FUTURE. Features are re-scaled to a common size. If you
     compute that scale from all of history, the average of days you have not seen
     yet leaks into training. We measure it on the LEARN period only.

  3. A TARGET THAT REACHES OVER THE BORDER. The last day of the learn period has a
     target that is tomorrow's return — and tomorrow is the first day of the tune
     period. We drop that day. It is one row, and it is the difference between an
     honest boundary and a leaky one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, Dataset

PERIODS = ("learn", "tune", "test")


@dataclass
class Arrays:
    """The feature table, as plain arrays: fast to slice, easy to check."""

    dates: pd.DatetimeIndex     # [T]
    tickers: list[str]          # [N]
    x: np.ndarray               # [T, N, F]  the features
    y: np.ndarray               # [T, N]     tomorrow's return (the answer)
    ok: np.ndarray              # [T, N]     is this cell usable at all?
    names: list[str]            # F


def to_arrays(table: pd.DataFrame, settings: dict) -> Arrays:
    """Pivot the (date, ticker) table into dense [days × companies × features] arrays."""
    from bubble_bi.data.features import names as feature_names

    names = feature_names()
    tickers = list(settings["tickers"])

    x = (
        table[names]
        .unstack("ticker")                       # columns become (feature, ticker)
        .reindex(columns=pd.MultiIndex.from_product([names, tickers]))
    )
    dates = pd.DatetimeIndex(x.index)
    grid = x.to_numpy(dtype=np.float32).reshape(len(dates), len(names), len(tickers))
    # ascontiguousarray, not just transpose: a transposed view is read-only and slow
    # to slice, and every window we cut is a slice.
    grid = np.ascontiguousarray(grid.transpose(0, 2, 1))     # [T, N, F]

    y = np.ascontiguousarray(
        table["target"].unstack("ticker").reindex(columns=tickers)
        .to_numpy(dtype=np.float32)              # [T, N]
    )

    # A cell is usable only if every feature is there AND we know the answer.
    ok = np.isfinite(grid).all(axis=2) & np.isfinite(y)
    return Arrays(dates=dates, tickers=tickers, x=grid, y=y, ok=ok, names=names)


def split_days(arrays: Arrays, settings: dict) -> dict[str, np.ndarray]:
    """Cut history into three periods, in order. Never at random.

    The last day of each period is dropped: its target is tomorrow's return, and
    tomorrow belongs to the next period.
    """
    total = len(arrays.dates)
    learn_end = int(total * settings["split"]["learn"])
    tune_end = learn_end + int(total * settings["split"]["tune"])

    edges = {
        "learn": (0, learn_end),
        "tune": (learn_end, tune_end),
        "test": (tune_end, total),
    }
    # `stop - 1` is the embargo: drop the final day so its target cannot reach across.
    return {name: np.arange(start, stop - 1) for name, (start, stop) in edges.items()}


class Scaler:
    """Normalises each company against ITS OWN history — along time, not across companies.

    Companies live on wildly different scales. Measured on the raw features, half of
    `close_frac`'s variation is nothing but *which company it is*; a third of
    `obv_frac`, a third of `amihud`, a fifth of every volatility estimator. Pooling
    them and computing one average per feature would leave that company-identity
    baked into the numbers — and the codebook would spend its precious 512 words
    memorising "this is NVDA" instead of "this is a panic".

    So every company gets its own average and its own spread, computed down the time
    axis. A feature then says the same thing for everyone:

        "how unusual is today — FOR THIS COMPANY?"

    which is exactly what a shared vocabulary needs.

    ⚠️ What this deliberately throws away: that one company is *genuinely* more
    volatile, or less liquid, than another. Afterwards, a sleepy utility sitting at
    its own average volatility and a wild biotech sitting at ITS own average
    volatility look identical. That is the price of a token meaning the same thing
    whichever company it describes.

    Measured on the LEARN period only — using all of history would let days the model
    has not seen yet leak backwards into its training.
    """

    # "Flat" has to be judged RELATIVE to the feature's own size, never against a
    # fixed number. Illiquidity for a mega-cap lives around 1e-6 while varying by 30%
    # of itself — real, useful signal that an absolute threshold would throw away.
    # A genuinely constant feature has a spread made of pure float noise, which is
    # ~1e-7 of its own magnitude in float32; 1e-9 sits comfortably below that.
    FLAT = 1e-9
    # A company with less history than this cannot give a trustworthy average.
    LEAST_DAYS = 60

    def __init__(self, arrays: Arrays, learn_days: np.ndarray, names: list[str] | None = None):
        # float64 throughout: in float32, summing thousands of rows loses enough
        # precision to matter when we are about to divide by the result.
        rows = arrays.x[learn_days].astype(np.float64)   # [t, N, F]
        usable = arrays.ok[learn_days]                   # [t, N]
        days_each = usable.sum(axis=0)                   # [N] -- per company

        thin = [
            t for t, n in zip(arrays.tickers, days_each) if n < self.LEAST_DAYS
        ]
        if thin:
            raise ValueError(
                f"Too little history in the learn period to normalise: {thin}. "
                f"Each company needs at least {self.LEAST_DAYS} usable days — either "
                "start earlier, or drop these companies."
            )

        keep = usable[:, :, None]                        # [t, N, 1]
        counts = days_each[:, None]                      # [N, 1]

        # Each company's own average and spread, down the time axis.
        self.middle = np.where(keep, rows, 0.0).sum(axis=0) / counts          # [N, F]
        gap = np.where(keep, rows - self.middle[None], 0.0)
        self.spread = np.sqrt((gap ** 2).sum(axis=0) / counts)                # [N, F]

        # A feature that never moves for a given company carries no information, and
        # dividing by its (near-zero) spread would amplify rounding error into ±1
        # garbage. Leave it alone, and remember, so the notebook can say so out loud.
        # Judged relative to the feature's own size -- see FLAT.
        size = np.maximum(np.abs(self.middle), np.abs(rows).max(axis=(0, 1))[None, :])
        self.constant = self.spread < self.FLAT * np.maximum(size, 1e-30)     # [N, F]
        self.spread = np.where(self.constant, 1.0, self.spread)

        self.flat_features = sorted({
            names[f]
            for _, f in zip(*np.nonzero(self.constant))
            if names
        }) if names else []

    def apply(self, x: np.ndarray) -> np.ndarray:
        """x: [T, N, F] -> each company normalised against its own history."""
        return ((x - self.middle[None]) / self.spread[None]).astype(np.float32)


def _complete_windows(ok: np.ndarray, days: int) -> np.ndarray:
    """[T, N] -> was every one of the last `days` days usable, for each company?"""
    t, n = ok.shape
    out = np.zeros((t, n), dtype=bool)
    if t >= days:
        # window i covers days i .. i+days-1, so its verdict belongs to the LAST day
        out[days - 1:] = sliding_window_view(ok, days, axis=0).all(axis=2)
    return out


class TSGrids(Dataset):
    """One sample per company per day: that company's last few days."""

    def __init__(self, arrays: Arrays, scaled: np.ndarray, day_pool: np.ndarray, days: int):
        self.x, self.y, self.days = scaled, arrays.y, days
        whole = _complete_windows(arrays.ok, days)
        usable = day_pool[day_pool >= days - 1]
        rows, cols = np.nonzero(whole[usable])           # which (day, company) pairs work
        self.when = usable[rows]
        self.who = cols

    def __len__(self) -> int:
        return len(self.when)

    def __getitem__(self, i: int) -> dict:
        t, j = int(self.when[i]), int(self.who[i])
        window = self.x[t - self.days + 1: t + 1, j]     # [days, F]
        return {
            "grid": torch.from_numpy(window[None]),      # [1, days, F] -- one company
            "present": torch.ones(1, dtype=torch.bool),
            "target": torch.tensor(self.y[t, j]),
            "day": torch.tensor(t),
            "company": torch.tensor(j),
        }


class CSGrids(Dataset):
    """One sample per day: every company's last few days, together."""

    def __init__(self, arrays: Arrays, scaled: np.ndarray, day_pool: np.ndarray, days: int,
                 least: int = 2):
        self.x, self.days = scaled, days
        self.whole = _complete_windows(arrays.ok, days)  # [T, N]
        usable = day_pool[day_pool >= days - 1]
        # A day is worth using only if enough companies actually traded through it.
        self.when = usable[self.whole[usable].sum(axis=1) >= least]

    def __len__(self) -> int:
        return len(self.when)

    def __getitem__(self, i: int) -> dict:
        t = int(self.when[i])
        window = self.x[t - self.days + 1: t + 1]        # [days, N, F]
        present = self.whole[t]                          # [N]
        grid = np.where(present[None, :, None], window, 0.0).transpose(1, 0, 2)
        return {
            "grid": torch.from_numpy(np.ascontiguousarray(grid)),   # [N, days, F]
            "present": torch.from_numpy(present.copy()),            # [N]
            "day": torch.tensor(t),
        }


@dataclass
class Batches:
    """Everything the models need to eat, already split into three periods."""

    arrays: Arrays
    scaler: Scaler
    days: dict[str, np.ndarray]
    ts: dict[str, DataLoader]
    cs: dict[str, DataLoader]

    def sizes(self) -> pd.DataFrame:
        """A small table of what ended up where — for the notebook."""
        return pd.DataFrame(
            {
                "from": [self.arrays.dates[d[0]].date() for d in self.days.values()],
                "to": [self.arrays.dates[d[-1]].date() for d in self.days.values()],
                "days": [len(d) for d in self.days.values()],
                "TS samples": [len(self.ts[p].dataset) for p in PERIODS],
                "CS samples": [len(self.cs[p].dataset) for p in PERIODS],
            },
            index=list(PERIODS),
        )


def make_tensors(table: pd.DataFrame, settings: dict) -> Batches:
    """The feature table in, batches of grids out — split learn / tune / test."""
    arrays = to_arrays(table, settings)
    days = split_days(arrays, settings)

    scaler = Scaler(arrays, days["learn"], arrays.names)   # measured on the past ONLY
    scaled = scaler.apply(arrays.x)

    def loaders(build, window, batch):
        out = {}
        for period in PERIODS:
            dataset = build(arrays, scaled, days[period], window)
            if len(dataset) == 0:
                raise ValueError(
                    f"No usable {window}-day grids in the '{period}' period. The slow "
                    "features need a long run-up (a year, for illiquidity) — either "
                    "start the history earlier, or use fewer companies."
                )
            out[period] = DataLoader(
                dataset,
                batch_size=batch,
                shuffle=(period == "learn"),     # order within a period does not matter;
                # Dropping a ragged final batch is fine -- unless it is the ONLY batch,
                # in which case it would silently hand back an empty loader and the
                # model would appear to train on nothing.
                drop_last=(period == "learn" and len(dataset) > batch),
            )
        return out

    return Batches(
        arrays=arrays,
        scaler=scaler,
        days=days,
        ts=loaders(TSGrids, settings["ts"]["days"], settings["ts"]["batch"]),
        cs=loaders(CSGrids, settings["cs"]["days"], settings["cs"]["batch"]),
    )


def tuning_loaders(batches: Batches, entry: str, days: int,
                   batch: int) -> dict[str, DataLoader]:
    """One entry's grids at a given window length — LEARN AND TUNE ONLY.

    The hyperparameter search varies `days` (the window length), which changes the
    shape of the grids themselves, so they have to be rebuilt from scratch for every
    window the search tries.

    ⚠️ `test` is not in the result, and that is the whole point. A search that could
    reach the test days would quietly tune on the answer, then report a wonderful
    score on it -- because it had already seen it. The defence here is not a comment
    telling the next person to be careful with `test`; it is that the test loader is
    never built in the first place, so there is nothing for the search to reach.
    """
    if entry not in ("ts", "cs"):
        raise ValueError(f"`entry` must be 'ts' or 'cs', got {entry!r}.")

    build = TSGrids if entry == "ts" else CSGrids
    scaled = batches.scaler.apply(batches.arrays.x)

    out = {}
    for period in ("learn", "tune"):
        dataset = build(batches.arrays, scaled, batches.days[period], days)
        if len(dataset) == 0:
            raise ValueError(
                f"No usable {days}-day {entry.upper()} grids in the '{period}' period. "
                f"A {days}-day window needs {days} days of run-up on top of what the slow "
                "features already need -- try a shorter window, or a longer history."
            )
        out[period] = DataLoader(
            dataset,
            batch_size=batch,
            shuffle=(period == "learn"),
            # Same guard as in `make_tensors`: dropping a ragged final batch is fine
            # unless it is the ONLY batch, in which case it would silently hand back
            # an empty loader and the search would appear to train on nothing.
            drop_last=(period == "learn" and len(dataset) > batch),
        )
    return out
