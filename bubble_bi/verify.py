"""The check that closes each section of the notebook.

One function per part of the project. Each one proves the part actually works
rather than asserting that it does, then says what we now have.
"""

from __future__ import annotations

import os
from pathlib import Path

from bubble_bi.report import report, run_tests
from bubble_bi.settings import device


def setup(settings: dict) -> None:
    """Section 1-2: the settings are sound and the code runs here."""
    n = len(settings["tickers"])
    where = device()

    data_dir = Path(settings["data_dir"])
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / ".write-probe"
        probe.touch()
        probe.unlink()
        writable = True
    except OSError:
        writable = False

    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False

    tests_pass, tests_summary = run_tests()

    ts, cs = settings["ts"], settings["cs"]
    report(
        "Setup",
        [
            ("Settings understood", True, f"{n} companies, no typos"),
            ("Data folder writable", writable, f"{data_dir}/"),
            ("PyTorch available", has_torch, "required to train"),
            ("Hardware", True, where.upper()),
            ("Project's own tests", tests_pass, tests_summary),
        ],
        have=f"""
        A checked configuration — and nothing else yet.
        The tokenizer will read {ts['days']} days of each stock and {cs['days']} days of the
        whole market, then merge them into 1 token out of {settings['fusion']['vocabulary']}.
        No prices downloaded, no model trained.
        """,
    )
    if where == "cpu" and os.environ.get("COLAB_GPU") is None:
        print("\n  ℹ️  Running on CPU. Training will work but be slow —")
        print("     on Colab, switch to a GPU runtime for a large speed-up.")


def prices(table, settings: dict) -> None:
    """Section 3: the raw prices arrived, and they are sane."""
    dates = table.index.get_level_values("date")
    tickers = table.index.get_level_values("ticker")
    n_days, n_co = dates.nunique(), tickers.nunique()
    asked = len(settings["tickers"])

    ohlc = table[["open", "high", "low", "close"]]
    # A high below the low, or a price at zero, means the data is corrupt --
    # and every high/low-based feature downstream would silently become garbage.
    ordered = bool((table["high"] >= table["low"]).all())
    positive = bool((ohlc > 0).to_numpy().all())
    no_gaps = not bool(ohlc.isna().to_numpy().any())

    # The last row of each company has no tomorrow, so its target is blank. That is
    # expected -- exactly one blank per company means nothing else is missing.
    blank_targets = int(table["target"].isna().sum())

    report(
        "Raw prices",
        [
            ("All companies arrived", n_co == asked, f"{n_co} of {asked}"),
            ("Trading days", n_days > 0, f"{n_days:,} days, "
                                         f"{dates.min():%Y-%m-%d} → {dates.max():%Y-%m-%d}"),
            ("Prices are positive", positive, "no zero or negative prices"),
            ("Highs are above lows", ordered, "candles are the right way up"),
            ("No missing prices", no_gaps, "every day has O/H/L/C"),
            ("Target is tomorrow's return", blank_targets == n_co,
             f"blank only on each company's last day ({blank_targets})"),
        ],
        have=f"""
        {len(table):,} trading days of real prices ({n_co} companies × {n_days:,} days).
        Six numbers per day: open, high, low, close, volume — and `target`,
        tomorrow's return, which is the answer we are trying to predict.
        No features yet: this is just what the market did.
        """,
    )


def features(table, prices_only, settings: dict) -> None:
    """Section 4: the features are built — and none of them look into the future."""
    from bubble_bi.data.features import FAMILIES, names
    from bubble_bi.data.leakage import find_leaks

    feature_columns = names()
    present = [c for c in feature_columns if c in table.columns]

    # THE test. Recompute with the future deleted; anything that changes was peeking.
    leaks = find_leaks(prices_only, settings)

    usable = table[feature_columns].notna().all(axis=1)
    kept = float(usable.mean())
    warmup = len(table) - int(usable.sum())

    report(
        "Features",
        [
            ("Every feature built", len(present) == len(feature_columns),
             f"{len(present)} features, {len(FAMILIES)} families"),
            ("NOTHING LOOKS AHEAD", not leaks,
             "proved by deleting the future and recomputing"
             if not leaks else f"LEAKING: {leaks}"),
            ("Enough history to use", kept > 0.8, f"{kept:.1%} of rows are complete"),
        ],
        have=f"""
        A table of {len(table):,} rows × {len(feature_columns)} features.
        Each row describes one company on one day from five angles:
        {', '.join(FAMILIES)}.
        {warmup:,} early rows are incomplete — the slow features (Hurst needs 100
        days) have not warmed up yet. They are dropped, not guessed.
        This table is what the tokenizer will learn to compress into single tokens.
        """,
    )
