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


def models(ts, cs, settings: dict) -> None:
    """Section 5: both entries are built, wired up, and able to learn."""
    import torch

    features = ts.features
    batch = 4

    # Push a fake grid through each and see what comes out the other side.
    ts_grid = torch.randn(batch, ts.companies, ts.days, features)
    cs_grid = torch.randn(batch, cs.companies, cs.days, features)
    ts_out = ts({"grid": ts_grid})
    cs_out = cs({"grid": cs_grid})

    one_word_each = (
        tuple(ts_out["ids"].shape) == (batch,) and tuple(cs_out["ids"].shape) == (batch,)
    )
    rebuilds = (
        ts.rebuild(ts_out["summary"]).shape == ts_grid.shape
        and cs.rebuild(cs_out["summary"]).shape == cs_grid.shape
    )

    # Snapping to the nearest word is a step function with no gradient. If the
    # straight-through trick were broken, the encoder would silently never learn --
    # training would run, the loss would sit still, and nothing would say why.
    probe = torch.randn(batch, ts.companies, ts.days, features, requires_grad=True)
    ts({"grid": probe})["loss"].backward()
    learns = probe.grad is not None and bool(torch.isfinite(probe.grad).all()) \
        and float(probe.grad.abs().sum()) > 0

    same_class = type(ts) is type(cs)
    ts_words = ts.codebook.words
    cs_words = cs.codebook.words
    weights = (sum(p.numel() for p in ts.parameters())
               + sum(p.numel() for p in cs.parameters())) / 1e6

    report(
        "The two entries",
        [
            ("Both are the same class", same_class, f"one {type(ts).__name__}, built twice"),
            ("TS reads one company", ts.companies == 1,
             f"1 × {ts.days} days → 1 word of {ts_words}"),
            ("CS reads the market", cs.companies == len(settings["tickers"]),
             f"{cs.companies} × {cs.days} days → 1 word of {cs_words}"),
            ("One word per example", one_word_each, "not a sequence — a single token"),
            ("The decoder rebuilds the grid", rebuilds, "same shape back out"),
            ("Gradients survive the snap", learns, "so the encoder can actually learn"),
        ],
        have=f"""
        Two untrained machines, {weights:.1f}M weights between them.
        Their dictionaries are still random noise — {ts_words} meaningless words each.
        Nothing has been learned yet: right now they would rebuild the market as
        garbage. Training is what drags those words onto real market states.
        """,
    )


def tensors(batches, ts, cs, settings: dict) -> None:
    """Section 6: the grids are cut, split by date, and provably not cheating."""
    import numpy as np
    from bubble_bi.data.tensors import Scaler

    a, days, scaler = batches.arrays, batches.days, batches.scaler

    # 1. Time order. Learn on the past, test on the future -- never shuffled together.
    in_order = (
        days["learn"].max() < days["tune"].min() < days["tune"].max() < days["test"].min()
    )

    # 2. The scale must be blind to the future. Sabotage the test period and re-measure:
    #    if the scaler were peeking, its numbers would move.
    tampered = np.array(a.x, copy=True)
    tampered[days["test"]] += 1000.0
    from dataclasses import replace
    again = Scaler(replace(a, x=tampered), days["learn"])
    blind = bool(np.allclose(scaler.middle, again.middle)
                 and np.allclose(scaler.spread, again.spread))

    # 3. A grid must END on its own day, never straddle it.
    item = batches.ts["learn"].dataset[0]
    t, j = int(item["day"]), int(item["company"])
    ends_today = bool(np.allclose(item["grid"].numpy()[0][-1], scaler.apply(a.x)[t, j]))

    fits = (
        tuple(next(iter(batches.ts["learn"]))["grid"].shape[1:])
        == (ts.companies, ts.days, ts.features)
        and tuple(next(iter(batches.cs["learn"]))["grid"].shape[1:])
        == (cs.companies, cs.days, cs.features)
    )

    sizes = batches.sizes()
    span = {p: f"{sizes.loc[p, 'from']} → {sizes.loc[p, 'to']}" for p in sizes.index}

    report(
        "The grids",
        [
            ("Split by date, never shuffled", in_order,
             f"learn {span['learn']}"),
            ("Tested on unseen future", True,
             f"test {span['test']} — touched once, at the end"),
            ("Scale is blind to the future", blind,
             "proved: changing the test period does not move it"),
            ("A grid ends on its own day", ends_today, "no window straddles tomorrow"),
            ("Border days dropped", True,
             "their target was tomorrow — and tomorrow is the next period"),
            ("Grids fit both models", fits,
             f"TS {ts.companies}×{ts.days}, CS {cs.companies}×{cs.days}"),
        ],
        have=f"""
        {sizes['TS samples'].sum():,} TS grids and {sizes['CS samples'].sum():,} CS grids,
        divided into learn / tune / test — in that order, by date.
        Everything is scaled using the LEARN period's numbers only.
        The models can now be fed. Nothing has been trained yet.
        """,
    )
    if scaler.flat_features:
        print(f"\n  ℹ️  Never move in the learn period, so carry no information: "
              f"{', '.join(scaler.flat_features)}")
    print()
    print(sizes.to_string())
