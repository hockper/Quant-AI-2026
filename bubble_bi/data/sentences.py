"""Turning history into sentences.

The predictor reads a company's life as a sentence — one word per day — and guesses the
next word. So a training example is not a day. It is a **run of consecutive days for one
company**.

    "AAPL, 64 trading days ending 3 March"  →  64 words  →  what is the 65th?

**Everything the frozen encoders say is computed once, here, and cached.**

TS and CS are frozen. Their answers can never change, so recomputing them on every
training step would be pure waste: pushing thirty companies through a transformer, over
and over, to get the same numbers back. Instead we run them across the whole of history
once, keep the result, and training just looks it up. It is a few tens of megabytes, and
it turns each step from "encode the market" into an array index.

⚠️ A sentence must not straddle a gap. If a company stopped trading for a week, the days
either side of the gap are not consecutive, and a model reading them as if they were
would be learning from a lie. So sentences are only ever cut from unbroken runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from bubble_bi.data.tensors import PERIODS


@dataclass
class Memory:
    """What the frozen encoders said about every day of history. Computed once."""

    z_ts: torch.Tensor          # [days, companies, width]  what each company was doing
    market: torch.Tensor        # [days, keys, width]       what the market was offering
    candle: torch.Tensor        # [days, companies, 4]      the day's own candle shape
    ok: np.ndarray              # [days, companies]         was this a usable day?
    dates: object               # [days]
    tickers: list[str]

    def megabytes(self) -> float:
        return (self.z_ts.numel() + self.market.numel()
                + self.candle.numel()) * 4 / 1e6


@torch.no_grad()
def remember(tokenizer, batches, settings: dict) -> Memory:
    """Run the frozen encoders over all of history, once.

    Everything after this is fast, because nothing here will ever change again.
    """
    from bubble_bi.training import pick_device

    where = pick_device(settings)
    tokenizer = tokenizer.to(where).eval()

    arrays = batches.arrays
    days, companies = arrays.ok.shape
    width = tokenizer.width

    z_ts = torch.zeros(days, companies, width)
    market = None
    seen_market = np.zeros(days, dtype=bool)

    # TS: one vector per (company, day).
    for period in PERIODS:
        for batch in batches.ts[period]:
            grid = batch["grid"].to(where)
            said = tokenizer.ts.summarise(grid).cpu()
            z_ts[batch["day"].numpy(), batch["company"].numpy()] = said

    # CS: one set of keys per day. Same for every company on that day, so it is stored
    # once rather than thirty times.
    for period in PERIODS:
        for batch in batches.cs[period]:
            said = tokenizer.cs.context(
                batch["grid"].to(where), batch["present"].to(where), tokenizer.attend_to
            ).cpu()
            when = batch["day"].numpy()
            if market is None:
                market = torch.zeros(days, said.shape[1], width)
            market[when] = said
            seen_market[when] = True

    if market is None:
        raise ValueError("No market days to encode — the CS batches are empty.")

    # The candle each day actually was. This is what the predictor is asked to DRAW for
    # tomorrow -- and, being exactly invertible, what lets us show the drawing.
    from bubble_bi.models.world import CANDLE

    missing = [c for c in CANDLE if c not in arrays.names]
    if missing:
        raise ValueError(f"The candle features are missing: {missing}.")
    where_candle = [arrays.names.index(c) for c in CANDLE]
    scaled = batches.scaler.apply(arrays.x)[:, :, where_candle]      # [days, N, 4]
    candle = torch.from_numpy(np.nan_to_num(scaled)).float()

    # A day is only usable if the market was encoded for it too.
    ok = arrays.ok & seen_market[:, None]
    return Memory(z_ts=z_ts, market=market, candle=candle, ok=ok,
                  dates=arrays.dates, tickers=arrays.tickers)


def _unbroken_runs(usable: np.ndarray, days: np.ndarray, length: int) -> list[int]:
    """The last day of every stretch of `length` consecutive usable days.

    A sentence that straddles a gap in a company's trading is a sentence about days that
    never followed one another. We simply never cut one.
    """
    allowed = np.zeros(len(usable), dtype=bool)
    allowed[days] = True
    good = usable & allowed

    ends, run = [], 0
    for t, alive in enumerate(good):
        run = run + 1 if alive else 0
        if run >= length:
            ends.append(t)
    return ends


class Sentences(Dataset):
    """One company, `length` consecutive days, straight from the cache."""

    def __init__(self, memory: Memory, day_pool: np.ndarray, length: int):
        self.memory, self.length = memory, length
        self.where = [
            (end, company)
            for company in range(memory.ok.shape[1])
            for end in _unbroken_runs(memory.ok[:, company], day_pool, length)
        ]

    def __len__(self) -> int:
        return len(self.where)

    def __getitem__(self, i: int) -> dict:
        end, company = self.where[i]
        start = end - self.length + 1
        return {
            "z_ts": self.memory.z_ts[start:end + 1, company],       # [T, width]
            "market": self.memory.market[start:end + 1],            # [T, keys, width]
            "candle": self.memory.candle[start:end + 1, company],   # [T, 4] -- what to draw
            "company": torch.tensor(company),
            "last_day": torch.tensor(end),
        }


def make_sentences(tokenizer, batches, settings: dict) -> dict:
    """Cache the frozen encoders, then hand back sentence batches per period."""
    memory = remember(tokenizer, batches, settings)
    length = settings["predictor"]["sentence_length"]
    batch = settings["fusion"]["batch"]

    loaders = {}
    for period in PERIODS:
        book = Sentences(memory, batches.days[period], length)
        if len(book) == 0:
            raise ValueError(
                f"No unbroken {length}-day stretches in the '{period}' period. "
                "Shorten `predictor['sentence_length']`, or use a longer history."
            )
        loaders[period] = DataLoader(
            book,
            batch_size=batch,
            shuffle=(period == "learn"),
            drop_last=(period == "learn" and len(book) > batch),
        )
    return {"memory": memory, "loaders": loaders}
