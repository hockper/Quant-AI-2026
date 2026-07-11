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
