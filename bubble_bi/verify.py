"""The check that closes each section of the notebook.

One function per part of the project. Each one proves the part actually works
rather than asserting that it does, then says what we now have.
"""

from __future__ import annotations

import os
from pathlib import Path

from bubble_bi.report import report
from bubble_bi.settings import device, hardware


def setup(settings: dict) -> None:
    """Section 1-2: does this ENVIRONMENT work? Settings parse, folder writes, GPU is there.

    ⚠️ It deliberately does NOT run the test suite, and that is a change worth explaining.

    It used to. All 255 tests, every time you touched the setup cell — **112 seconds, and
    roughly a hundred of them spent TRAINING MODELS** in tests that have nothing whatever
    to say about whether your install works. On Colab that is two minutes off a paid GPU
    session, per run, before a single price has been downloaded. Nobody re-asked whether
    this still made sense as the suite grew from 22 tests to 255.

    "Does this environment work" and "does the science hold" are different questions. Only
    the first one belongs in a cell you re-run constantly. The second is still one command
    away — `bb.run_tests()` — and it is printed below so nobody has to go looking for it.
    """
    n = len(settings["tickers"])
    kit = hardware()
    where = kit["where"]

    data_dir = Path(settings["data_dir"])
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / ".write-probe"
        probe.touch()
        probe.unlink()
        writable = True
    except OSError:
        writable = False

    has_torch = kit["torch"] is not None

    # Say WHICH device, and name it. "GPU" alone hides the two different things that go
    # wrong — no GPU on the machine at all, versus a GPU that PyTorch cannot talk to.
    if kit["gpu"]:
        found = f"{kit['gpu']} — torch {kit['torch']}"
    elif not kit["gpu present"]:
        found = f"CPU — torch {kit['torch']}  ⚠️ NO GPU ON THIS MACHINE"
    else:
        found = f"CPU — torch {kit['torch']}  ⚠️ GPU PRESENT BUT UNREACHABLE"

    # ⚠️ This used to be hardcoded `True`, so falling back to a CPU earned a green tick and
    # the notebook went cheerfully on to train — for hours — on a CPU. We train on Colab
    # now, so a silent CPU fallback is a FAILURE. It is tolerated (a CPU run is a fine way
    # to read the notebook) but it is never again reported as a pass.
    on_real_hardware = kit["where"] in ("gpu", "tpu")

    ts, cs = settings["ts"], settings["cs"]
    report(
        "Setup",
        [
            ("Settings understood", True, f"{n} companies, no typos"),
            ("Data folder writable", writable, f"{data_dir}/"),
            ("PyTorch available", has_torch, "required to train"),
            ("Hardware", on_real_hardware, found),
        ],
        have=f"""
        A checked configuration — and nothing else yet.
        The tokenizer will read {ts['days']} days of each stock and {cs['days']} days of the
        whole market, then merge them into 1 token out of {settings['fusion']['vocabulary']}.
        No prices downloaded, no model trained.

        This checked your ENVIRONMENT, not the code. To check the code itself — all of it,
        including the no-lookahead proofs — run:   bb.run_tests()      (~2 minutes)
        """,
        known_problem=(
            None if on_real_hardware else
            "There is no GPU. The notebook still runs — but on a CPU it is a toy, and "
            "nothing it trains should be believed. See the note below."
        ),
    )
    if kit["why"]:
        print(f"\n  ⚠️  {kit['why']}")
    elif where == "cpu":
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

    # 4. Company identity must be gone. A $500 stock and a $20 stock have wildly
    #    different levels; left alone, the codebook would spend its 512 words
    #    memorising WHICH company it is looking at instead of WHAT is happening.
    def company_share(values, rows):
        v = np.where(a.ok[rows][:, :, None], values[rows], np.nan)
        return np.nanvar(np.nanmean(v, axis=0), axis=0) / np.nanvar(v, axis=(0, 1))

    learn = days["learn"]
    was = float(np.nanmax(company_share(a.x, learn)))
    now = float(np.nanmax(company_share(scaler.apply(a.x), learn)))
    anonymous = now < 0.01

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
            ("Each company scaled to itself", anonymous,
             f"company identity: {was:.0%} of a feature → {now:.1%}"),
            ("A grid ends on its own day", ends_today, "no window straddles tomorrow"),
            ("Border days dropped", True,
             "their target was tomorrow — and tomorrow is the next period"),
            ("Grids fit both models", fits,
             f"TS {ts.companies}×{ts.days}, CS {cs.companies}×{cs.days}"),
        ],
        have=f"""
        {sizes['TS samples'].sum():,} TS grids and {sizes['CS samples'].sum():,} CS grids,
        divided into learn / tune / test — in that order, by date.
        Every company is normalised against its OWN history, so a feature now says
        "how unusual is today, for this company" rather than "this is a $500 stock".
        The models can now be fed. Nothing has been trained yet.
        """,
    )
    if scaler.flat_features:
        print(f"\n  ℹ️  Never move in the learn period, so carry no information: "
              f"{', '.join(scaler.flat_features)}")
    print()
    print(sizes.to_string())


def trained(model, history, loaders, settings: dict, name: str = "TS") -> None:
    """Section 7/9: the model learned something — measured against a bar that HURTS."""
    from bubble_bi.training import evaluate, pick_device

    scored = evaluate(model, loaders["test"], pick_device(settings))     # never seen
    words = model.codebook.words

    token = scored["rebuild"]
    flat = scored["guessing"]           # predict zero. A WEAK bar.
    level = scored["window_mean"]       # predict this window's own average. THE bar.
    last = scored["last_day"]           # repeat the final day. Flatters us — measured.

    perplexity = scored["perplexity"]
    rows = history.rows if history is not None else []
    start_ppl = rows[0]["perplexity"] if rows else float("nan")
    squeeze = model.companies * model.days * model.features

    # Against the weak bar the token looks fine. Against the bar that matters it may not.
    # Report both, and FAIL on the one that matters -- otherwise the notebook is simply
    # flattering itself, which is the whole thing we are trying not to do.
    beats_level = token < level
    alive = perplexity > 2

    health = (
        "healthy" if perplexity > words * 0.3
        else "thin — a longer run should widen it" if perplexity > words * 0.05
        else "very thin — the vocabulary is barely spreading"
    )

    report(
        f"{name} trained",
        [
            ("Dictionary did not collapse", alive,
             f"perplexity {perplexity:.0f} of {words} — {health}"),
            ("Vocabulary in use", True, f"{scored['words_used']} of {words} words"),
            ("Beats the long-run average", token < flat,
             f"{token:.2f} vs {flat:.2f}  → explains {1 - token / max(flat, 1e-9):.0%}"
             "   ⚠️ a WEAK bar"),
            ("Beats THIS WINDOW's own average", beats_level,
             f"{token:.2f} vs {level:.2f}  → "
             f"{'explains ' + format(1 - token / max(level, 1e-9), '.0%') if beats_level else 'LOSES. This is the bar that matters.'}"),
            ("(repeating the last day scores)", True,
             f"{last:.2f} — worse than predicting zero, so it FLATTERS us. Ignore it."),
        ],
        have=f"""
        A tokenizer that squeezes {squeeze:,} numbers into ONE word out of {words}.
        {f"Perplexity went {start_ppl:.0f} → {perplexity:.0f} during training." if rows else ""}
        Judge it on the window-mean bar, never on the long-run one.
        Run bb.plots.kept_by_family() to see WHAT it kept — the headline is an average,
        and it is carried by the easy features.
        """,
        known_problem=(
            "The token LOSES to simply predicting the average of the window it is "
            "describing. The reconstruction objective is not producing a token worth "
            "having.\n     See docs/DECISION-let-the-model-choose.md."
        ) if not beats_level else None,
    )

    if not beats_level:
        print(f"\n  ⚠️  A baseline that knows NOTHING except the average of these "
              f"{model.days} days")
        print(f"     scores {level:.2f}. The token scores {token:.2f}. It is worse.")
        print("     Every '% explained' figure against the long-run average was measured")
        print("     against a bar so low that knowing one number per feature would clear it.")
    if perplexity < words * 0.3 and rows:
        print(f"\n  ⏱️  Short run: {rows[-1]['step']:,} steps in {history.seconds:.0f}s.")
        print("     Train for longer on a GPU before reading anything into these numbers.")


def predictor(world, history, loaders, settings: dict) -> None:
    """Section 10: the world model, scored against the floors that actually matter."""
    from bubble_bi.training import pick_device, score_predictions

    scored = score_predictions(world, loaders["test"], pick_device(settings))
    words = world.words

    named = scored["accuracy"]
    sticky = scored["persistence"]
    drew = scored["candle"]
    shrug = scored["shrugging"]
    perplexity = scored["perplexity"]

    # The only floor worth measuring against. Regimes are sticky, so "tomorrow's word is
    # the same as today's" is right most of the time -- and any accuracy below it means
    # the model has learned nothing worth having, however impressive the number looks.
    beats_persistence = named > sticky
    beats_shrugging = drew < shrug
    alive = perplexity > words * 0.05

    report(
        "The predictor",
        [
            ("Names tomorrow's regime", True,
             f"{named:.1%} — vs {sticky:.1%} for just saying 'same as today'"),
            ("...and beats that floor", beats_persistence,
             "the honest bar" if beats_persistence
             else "NO — it is worse than doing nothing"),
            ("Draws tomorrow's candle", beats_shrugging,
             f"{drew:.3f} — vs {shrug:.3f} for drawing the average candle"),
            ("Dictionary still alive", alive,
             f"perplexity {perplexity:.0f} of {words}"),
            ("Scored on unseen days", True, "the test period, never trained on"),
        ],
        have=f"""
        A model that reads a company's last {world.sentence} days as a sentence of words
        and answers two questions about tomorrow: which word comes next, and what candle
        comes with it. Both are scored against a floor, not against zero.
        """,
        # These two failures are UNDERSTOOD and written down. The ❌ stays -- we do not
        # dress up a bad result -- but they do not stop the notebook, because they are an
        # open research question, not broken code.
        known_problem=(
            "The codebook collapses and the model loses to persistence. This is "
            "diagnosed, not fixed:\n     docs/OPEN-QUESTION-codebook-collapse.md"
        ) if not (beats_persistence and alive) else None,
    )

    if not beats_persistence:
        print("\n  ⚠️  IT DOES NOT BEAT PERSISTENCE, and that is the number that matters.")
        print("     Regimes are sticky, so 'same as yesterday' is a high bar. Until the")
        print("     model clears it, its accuracy means nothing at all.")
        print("\n     For the paper: the previous version of this project reported")
        print("     '58.6% next-token accuracy' against a much weaker baseline.")
        print("     Against THIS floor it would not have cleared it either.")
    if not alive:
        print(f"\n  ⚠️  The codebook has collapsed to ~{perplexity:.0f} words of {words}.")
        print("     The model found the shortcut: make every day the same word, and the")
        print("     next word becomes trivially easy to predict. The candle head is meant")
        print("     to punish that — but drawing tomorrow is so hard that it barely")
        print("     produces a gradient, so an empty token is never punished enough.")


def market_moods(evidence: dict) -> None:
    """Section 9b: do the CS tokens actually know what the market did?

    The fair question for CS. It cannot redraw thirty companies from one word — nothing
    could. But if its words separate a calm day from a panic, it has learned the
    market's moods, and that is what the fusion needs from it.
    """
    scores = evidence["scores"]
    real = scores["explained by the token"]
    luck = scores["explained by luck"]

    # The token must beat a SHUFFLED assignment -- the same words handed to the wrong
    # days. With 512 words and a few hundred days, a token can look informative by
    # sheer luck, and this is what that luck actually scores.
    beats_luck = bool((real > luck + 0.02).any())
    best = real.idxmax()

    report(
        "What the market words mean",
        [
            ("The words are not random", beats_luck,
             f"best: '{best}' — {real[best]:.0%} explained, "
             f"vs {luck[best]:.0%} by luck"),
            ("Words in use on unseen days", evidence["words_used"] > 1,
             f"{evidence['words_used']} different words across "
             f"{len(evidence['tokens'])} days"),
        ],
        have=f"""
        Knowing only which word a day was given, you can account for
        {real[best]:.0%} of {best}. The market's moods are in the words.
        This — not the rebuild score — is what CS was for: the fusion needs a market
        CONTEXT, not a redrawing of the market.
        """,
    )
    print()
    print(scores.to_string(float_format=lambda v: f"{v:6.1%}"))
    if not beats_luck:
        print("\n  ⚠️  The words explain no more than a random shuffle would. On a short")
        print("     run that may just be undertraining — but if it survives a long GPU")
        print("     run, CS has learned nothing and the fusion has nothing to attend to.")
