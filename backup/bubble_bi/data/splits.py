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
