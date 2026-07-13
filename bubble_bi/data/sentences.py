"""The market as a sentence: one time-window, every company, raw grids.

⚠️ Nothing is cached here any more, and that is the point.

The old version ran the FROZEN encoders once over history and stored their output, which is
what freezing buys you. Under joint training nothing is frozen -- the encoders are being
reshaped by the forecast every step -- so a cache would simply be wrong.

But the CS grid is IDENTICAL for every company on a given day. Serving it per company-day
would push the biggest grid in the model through the biggest encoder in the model thirty
times more often than necessary. So a batch is one TIME WINDOW across ALL companies at once:
the market is carried once per day, and encoded once per day.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from bubble_bi.data.tensors import Batches, PERIODS, _complete_windows
from bubble_bi.models.world import CANDLE


class Sentences(Dataset):
    """One item = one WINDOW of `length` consecutive days, for every company at once."""

    def __init__(self, batches: Batches, day_pool: np.ndarray, settings: dict):
        arrays = batches.arrays
        self.x = batches.scaler.apply(arrays.x)              # [T, N, F]  normalised
        self.ok = arrays.ok                                   # [T, N]
        self.length = settings["predictor"]["sentence_length"]
        self.ts_days = settings["ts"]["days"]
        self.cs_days = settings["cs"]["days"]
        self.candle = [arrays.names.index(name) for name in CANDLE]

        # A day is usable as the END of a step only if both grids behind it are whole.
        ts_whole = _complete_windows(self.ok, self.ts_days)   # [T, N]
        cs_whole = _complete_windows(self.ok, self.cs_days)   # [T, N]
        market_ok = cs_whole.sum(axis=1) >= 2                 # enough companies traded

        usable = np.zeros(len(self.x), dtype=bool)
        usable[day_pool] = True
        step_ok = usable & market_ok & ts_whole.all(axis=1)

        # An unbroken run of `length` usable steps is a sentence. A window that jumps over a
        # missing day is not a sentence, it is two sentences stapled together.
        run, self.ends = 0, []
        for day, fine in enumerate(step_ok):
            run = run + 1 if fine else 0
            if run >= self.length:
                self.ends.append(day)

        self.cs_present = cs_whole

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, i: int) -> dict:
        end = self.ends[i]
        days = np.arange(end - self.length + 1, end + 1)      # [T]

        # TS: one grid per company per day -> [T, N, 1, ts_days, F]
        ts = np.stack([self.x[d - self.ts_days + 1: d + 1] for d in days])   # [T, ts_days, N, F]
        ts = ts.transpose(0, 2, 1, 3)[:, :, None, :, :]                       # [T, N, 1, ts_days, F]

        # CS: ONE grid per day -> [T, N, cs_days, F].  No company axis outside the grid:
        # this single copy is what every company on that day will read.
        cs = np.stack([self.x[d - self.cs_days + 1: d + 1] for d in days])   # [T, cs_days, N, F]
        cs = cs.transpose(0, 2, 1, 3)                                         # [T, N, cs_days, F]

        present = self.cs_present[days]                                       # [T, N]
        cs = np.where(present[:, :, None, None], cs, 0.0)

        return {
            "ts_grid": torch.from_numpy(np.ascontiguousarray(ts)).float(),
            "cs_grid": torch.from_numpy(np.ascontiguousarray(cs)).float(),
            "cs_present": torch.from_numpy(present.copy()),
            "candle": torch.from_numpy(
                np.ascontiguousarray(self.x[days][:, :, self.candle])).float(),  # [T, N, 4]
            "days": torch.from_numpy(days.copy()),
        }


def make_sentences(batches: Batches, settings: dict) -> dict[str, DataLoader]:
    """The market as sentences — learn, tune and test.

    `test` is built here (unlike the tuning search, which must not even build it) because
    the world model IS finally scored on it, once, at the very end -- but nothing hands it
    to `train_joint`, which only ever iterates `"learn"` and `"tune"`.
    """
    size = settings["fusion"]["batch"]
    out = {}
    for period in PERIODS:
        data = Sentences(batches, batches.days[period], settings)
        if len(data) == 0:
            raise ValueError(
                f"No usable {settings['predictor']['sentence_length']}-day sentences in the "
                f"'{period}' period. Either shorten `predictor['sentence_length']` or start "
                "the history earlier."
            )
        out[period] = DataLoader(
            data, batch_size=size, shuffle=(period == "learn"),
            drop_last=(period == "learn" and len(data) > size))
    return out
