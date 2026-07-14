"""Looking at what a single word actually remembered.

The tokenizer squeezes several days of a company into ONE word out of 512, then
rebuilds those days from that word alone. This draws both: the real candles, and
the candles the word remembered.

It is the most honest demonstration in the project. Nothing here is a summary
statistic you have to trust — you can simply look at it and see what survived the
squeeze and what did not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from bubble_bi.data.features.candle import rebuild_candles

SHAPE = ["gap", "body", "upper_wick", "lower_wick"]


def _market_grid_for_day(batches, cs, day: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The CS grid + presence mask for one calendar day, built directly from the
    feature arrays and scaler -- the same way `CSGrids.__getitem__` does.

    Why this exists at all: a TS example names a DAY, and drawing what the trained
    model actually said for it requires running that day's market through the SAME
    cross-attention the real model uses (see the module docstring on `remembered`
    below) -- which needs the market's own grid for that day, not just the TS one.
    This builds it on demand, for any day a TS example names, without that day needing
    to already be its own batch inside `batches.cs`.
    """
    from bubble_bi.data.tensors import _complete_windows

    arrays, scaler = batches.arrays, batches.scaler
    scaled = scaler.apply(arrays.x)
    whole = _complete_windows(arrays.ok, cs.days)          # [T, N]
    if day < cs.days - 1 or not whole[day].any():
        raise ValueError(
            f"No usable {cs.days}-day market window ending on day {day} -- either too "
            "early in history, or nobody traded through it."
        )
    window = scaled[day - cs.days + 1: day + 1]             # [days, N, F]
    present = whole[day]                                    # [N]
    grid = np.where(present[None, :, None], window, 0.0).transpose(1, 0, 2)
    return (
        torch.from_numpy(np.ascontiguousarray(grid)).unsqueeze(0),   # [1, N, days, F]
        torch.from_numpy(present.copy()).unsqueeze(0),                 # [1, N]
    )


def _draw_candles(ax, candles: pd.DataFrame, title: str) -> None:
    up, down = "#26a69a", "#ef5350"
    for i, (_, day) in enumerate(candles.iterrows()):
        rising = day["close"] >= day["open"]
        colour = up if rising else down
        # the wick: the whole range of the day
        ax.plot([i, i], [day["low"], day["high"]], color=colour, linewidth=1.4, zorder=1)
        # the body: open to close
        low, high = sorted((day["open"], day["close"]))
        ax.add_patch(
            __import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                (i - 0.28, low), 0.56, max(high - low, (high + low) * 1e-4),
                facecolor=colour, edgecolor=colour, zorder=2,
            )
        )
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xticks(range(len(candles)))
    ax.set_xticklabels([d.strftime("%d %b") for d in candles.index], fontsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def remembered(tokenizer, batches, prices: pd.DataFrame, settings: dict,
               company: str | None = None, when: int | None = None):
    """Draw the real candles beside the ones a single token remembered.

    The model never sees a candle — it sees the candle's SHAPE (gap, body, wicks),
    which is scale-free. So we take the shape it reconstructed, invert it back into
    open/high/low/close, and draw it. Given the previous close, that inversion is
    exact: what you see is precisely what the word remembered, not an impression.

    ⚠️ Takes the TOKENIZER, not a bare TS `VQVAE` -- and routes through `tokenizer(...)`,
    not `VQVAE.forward`. `VQVAE.forward` alone would quantise the TS summary BEFORE the
    cross-attention against the market ever runs (`world.py`'s `fused` is what the real
    model quantises, AFTER fusion) -- a different vector. Measured: the two disagree on
    the chosen word 64% of the time. Drawing from the wrong one would show the reader a
    candle the trained model never actually said for this day.
    """
    import matplotlib.pyplot as plt

    model = tokenizer.ts
    arrays, scaler = batches.arrays, batches.scaler
    grids = batches.ts["test"].dataset          # days it was never trained on
    if len(grids) == 0:
        raise ValueError("No test grids to show.")

    # Pick an example: a named company if asked, otherwise the most eventful day we
    # can find, because a flat week teaches the reader nothing.
    wanted = arrays.tickers.index(company) if company else None
    choices = np.nonzero(grids.who == wanted)[0] if wanted is not None \
        else np.arange(len(grids))
    if when is None:
        moves = [abs(float(grids[int(i)]["target"])) for i in choices[:400]]
        when = int(choices[int(np.argmax(moves))])          # the biggest move
    else:
        when = int(choices[when % len(choices)])

    item = grids[when]
    day, who = int(item["day"]), int(item["company"])
    ticker = arrays.tickers[who]
    window = model.days

    # What the token remembered -- through the REAL chain: encode, cross-attend
    # against THAT day's market, then quantise. See the docstring above.
    tokenizer.eval()
    where = next(tokenizer.parameters()).device
    market_grid, market_present = _market_grid_for_day(batches, tokenizer.cs, day)
    with torch.no_grad():
        out = tokenizer(item["grid"].unsqueeze(0).to(where),
                        market_grid.to(where), market_present.to(where))
        rebuilt = model.rebuild(out["ts_vector"])[0, 0].cpu().numpy()
    word = int(out["ts_token"][0])
    tokenizer.train()

    # Undo the normalisation, so we are back in the units the candle lives in.
    shape_cols = [arrays.names.index(c) for c in SHAPE]
    real_shape = scaler.apply(arrays.x)[day - window + 1: day + 1, who][:, shape_cols]
    real_shape = real_shape * scaler.spread[who, shape_cols] + scaler.middle[who, shape_cols]
    said_shape = rebuilt[:, shape_cols] * scaler.spread[who, shape_cols] \
        + scaler.middle[who, shape_cols]

    dates = arrays.dates[day - window + 1: day + 1]
    before = prices.loc[(arrays.dates[day - window], ticker), "close"]

    real = rebuild_candles(pd.DataFrame(real_shape, columns=SHAPE, index=dates), before)
    said = rebuild_candles(pd.DataFrame(said_shape, columns=SHAPE, index=dates), before)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    _draw_candles(axes[0], real, f"What actually happened — {ticker}")
    _draw_candles(axes[1], said, f"What word #{word} remembered")

    span = pd.concat([real, said])
    pad = (span["high"].max() - span["low"].min()) * 0.12
    axes[0].set_ylim(span["low"].min() - pad, span["high"].max() + pad)
    axes[0].set_ylabel("price ($)")

    fig.suptitle(
        f"{window} days of {ticker}, squeezed into ONE word out of "
        f"{model.codebook.words} — and rebuilt from it",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    return fig


def kept_and_lost(tokenizer, batches, settings: dict, examples: int = 400):
    """Which of the 26 features survived the squeeze, and which were thrown away.

    A single word cannot carry everything. This says, feature by feature, how much
    of it came back — and being clear about what was DISCARDED is as informative as
    what was kept.

    ⚠️ Same routing as `remembered()`: through `tokenizer(...)`, so the word being
    scored here is the one the trained model actually assigns each day (encode ->
    cross-attend against that day's market -> quantise), not the pre-fusion word a
    bare `VQVAE.forward` would have picked.
    """
    import matplotlib.pyplot as plt

    model = tokenizer.ts
    grids = batches.ts["test"].dataset
    names = batches.arrays.names
    where = next(tokenizer.parameters()).device

    real, said = [], []
    tokenizer.eval()
    with torch.no_grad():
        for i in range(0, min(examples, len(grids))):
            item = grids[i]
            day = int(item["day"])
            grid = item["grid"].unsqueeze(0).to(where)
            market_grid, market_present = _market_grid_for_day(batches, tokenizer.cs, day)
            out = tokenizer(grid, market_grid.to(where), market_present.to(where))
            back = model.rebuild(out["ts_vector"])
            real.append(grid[0, 0].cpu().numpy())
            said.append(back[0, 0].cpu().numpy())
    tokenizer.train()

    real = np.stack(real)                       # [n, days, features]
    said = np.stack(said)
    left_over = ((real - said) ** 2).mean(axis=(0, 1))
    there_was = (real ** 2).mean(axis=(0, 1))
    kept = 1 - left_over / np.maximum(there_was, 1e-9)

    order = np.argsort(kept)
    fig, ax = plt.subplots(figsize=(8, 7))
    colours = ["#26a69a" if k > 0.5 else "#ffa726" if k > 0.2 else "#ef5350"
               for k in kept[order]]
    ax.barh([names[i] for i in order], kept[order] * 100, color=colours)
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("how much of this feature the single word kept (%)")
    ax.set_title("What survived the squeeze", loc="left", fontsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    return fig


def predicted_candles(world, book, batches, prices: pd.DataFrame, settings: dict,
                      company: str | None = None, show: int = 8):
    """What the model thinks tomorrow looks like, beside what tomorrow actually was.

    The most directly readable thing the whole model produces. It never sees a candle
    directly -- it reads a sentence of words and, for each day, draws the SHAPE of the
    NEXT one (`gap`, `body`, the two wicks). Those four numbers are exactly invertible,
    so given the previous close we can rebuild the whole candle and put the two side by
    side, in real dollars.

    `book` is one WINDOW of `predictor['sentence_length']` days, every company at once
    (see `bubble_bi.data.sentences.Sentences`) -- so `world(...)` is run ONCE, over
    every company in that window, and this simply picks which company's row to draw.
    `world.forward`'s own `wanted = candle[:, 1:]` is what "one day ahead" means here:
    `out["drawn"][company, k]` is the model's guess at `candle[k + 1]`, so its LAST
    `show` rows line up exactly with the LAST `show` real candles.

    ⚠️ Read this honestly. A model that has learned nothing draws a row of small,
    near-identical candles — the average of everything, which is the safest guess when
    you know nothing. Wide, varied predicted candles mean it is COMMITTING to
    something. Whether it commits *correctly* is a separate question, and only the
    numbers (section 8's own checks, `verify.joint`) answer it -- not this picture.
    """
    import matplotlib.pyplot as plt

    from bubble_bi.models.world import CANDLE
    from bubble_bi.training import _to, pick_device

    where = pick_device(settings)
    world.to(where).eval()

    arrays = batches.arrays
    dataset = book["test"].dataset               # the raw Sentences, not the loader
    if len(dataset) == 0:
        raise ValueError("No test sentences to show.")
    item = dataset[len(dataset) // 2]             # a window from the middle of the test period
    sentence_length = len(item["days"])
    if show >= sentence_length:
        raise ValueError(
            f"`show` ({show}) must be smaller than the sentence length "
            f"({sentence_length}) -- one day of history is needed BEFORE the first "
            "candle shown, to invert its shape back into a price."
        )

    with torch.no_grad():
        batch = _to({k: v.unsqueeze(0) for k, v in item.items()}, where)
        out = world(batch)
    world.train()

    # Which company: the one asked for, or the one with the biggest real move over the
    # shown window -- a flat week teaches the reader nothing.
    if company:
        who = arrays.tickers.index(company)
    else:
        moved = item["candle"][-show:, :, CANDLE.index("body")].abs().sum(dim=0)
        who = int(moved.argmax())
    ticker = arrays.tickers[who]

    columns = [arrays.names.index(c) for c in CANDLE]
    spread = batches.scaler.spread[who, columns]
    middle = batches.scaler.middle[who, columns]

    drawn = out["drawn"][who, -show:].cpu().numpy() * spread + middle
    real = item["candle"][-show:, who, :].numpy() * spread + middle

    days = item["days"].numpy()
    shown_days = arrays.dates[days[-show:]]
    before = prices.loc[(arrays.dates[days[-show - 1]], ticker), "close"]

    truth = rebuild_candles(pd.DataFrame(real, columns=CANDLE, index=shown_days), before)
    guess = rebuild_candles(pd.DataFrame(drawn, columns=CANDLE, index=shown_days), before)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    _draw_candles(axes[0], truth, f"What actually happened — {ticker}")
    _draw_candles(axes[1], guess, "What the model predicted, one day ahead")

    span = pd.concat([truth, guess])
    pad = (span["high"].max() - span["low"].min()) * 0.12
    axes[0].set_ylim(span["low"].min() - pad, span["high"].max() + pad)
    axes[0].set_ylabel("price ($)")
    fig.suptitle("Each predicted candle was drawn knowing only the days BEFORE it",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


def progress(history, title: str = "TS"):
    """The loss curves, and the number that says whether the words mean anything.

    Three things, and you need all three:

      LOSS       what it scores on the days it is learning from, against days it has
                 never seen. If the two part company, it is memorising, not learning.
      PERPLEXITY how many words are actually in use. A loss can fall beautifully while
                 the dictionary collapses to nothing -- this is what exposes that.
      WORDS      how many of the 512 ever get chosen at all.

    Returns None if there is no history — which happens when the model was LOADED from
    disk rather than trained in this session.
    """
    import matplotlib.pyplot as plt

    if history is None or not history.rows:
        print(f"({title} was loaded from disk — no training history to plot.)")
        return None

    frame = history.frame()
    words = frame["words_used"].max()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))
    loss, ppl, used = axes

    if "learning" in frame:
        loss.plot(frame.index, frame["learning"], color="#90a4ae", linewidth=1.4,
                  label="on the days it learns from")
    loss.plot(frame.index, frame["rebuild"], color="#1e88e5", marker="o", markersize=3,
              label="on days it has never seen")
    loss.axhline(frame["guessing"].iloc[-1], color="#ef5350", linestyle="--",
                 linewidth=1, label="just guessing the average")
    loss.set_title(f"{title} — rebuild error", loc="left", fontsize=11)
    loss.set_xlabel("step")
    loss.legend(fontsize=8, frameon=False)

    ppl.plot(frame.index, frame["perplexity"], color="#26a69a", marker="o", markersize=3)
    ppl.axhline(1, color="#ef5350", linestyle="--", linewidth=1,
                label="collapsed to one word")
    ppl.set_title(f"{title} — words in use (perplexity)", loc="left", fontsize=11)
    ppl.set_xlabel("step")
    ppl.legend(fontsize=8, frameon=False)

    used.plot(frame.index, frame["words_used"], color="#7e57c2", marker="o", markersize=3)
    used.set_title(f"{title} — vocabulary reached", loc="left", fontsize=11)
    used.set_xlabel("step")
    used.set_ylabel("words ever chosen")

    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(alpha=0.25, linewidth=0.5)
    del words
    fig.tight_layout()
    return fig


def joint_progress(history, title: str = "Joint"):
    """The joint run's own trajectory -- perplexity, and the forecast against its floor.

    `progress()` above draws a single VQ-VAE's `History` (`rebuild`/`guessing`/
    `words_used` columns, from `train()`). `train_joint`'s `History` rows carry a
    completely different set (`drawing`/`shrugging`/`accuracy`/`persistence`/
    `ts_perplexity`/`cs_perplexity` -- see `training.py`), and handing one to the other
    would simply `KeyError`. This is that chart, for the joint run.

    ⚠️ "Perplexity is the number to watch, from the first line" is this project's own
    headline instruction for section 8 -- and until this function existed, there was no
    chart of it at all: `world_history` was computed, returned, and thrown away. TWO
    dictionaries are watched here, side by side, because either one collapsing (the
    naming loss finding its way back into the vocabulary) is invisible in the loss curve
    but never invisible here.

    Returns None if there is no history — which happens when the model was LOADED from
    disk rather than trained in this session.
    """
    import matplotlib.pyplot as plt

    if history is None or not history.rows:
        print(f"({title} was loaded from disk — no training history to plot.)")
        return None

    frame = history.frame()
    fig, (ppl, draw) = plt.subplots(1, 2, figsize=(11, 3.8))

    ppl.plot(frame.index, frame["ts_perplexity"], color="#1e88e5", marker="o", markersize=3,
             label="TS")
    ppl.plot(frame.index, frame["cs_perplexity"], color="#7e57c2", marker="o", markersize=3,
             label="CS")
    ppl.axhline(1, color="#ef5350", linestyle="--", linewidth=1, label="collapsed to one word")
    ppl.set_title(f"{title} — words in use (perplexity)", loc="left", fontsize=11)
    ppl.set_xlabel("step")
    ppl.legend(fontsize=8, frameon=False)

    draw.plot(frame.index, frame["drawing"], color="#26a69a", marker="o", markersize=3,
              label="draws tomorrow's candle")
    draw.plot(frame.index, frame["shrugging"], color="#ef5350", linestyle="--", linewidth=1,
              label="shrugging (the average candle)")
    draw.set_title(f"{title} — the forecast vs its floor", loc="left", fontsize=11)
    draw.set_xlabel("step")
    draw.legend(fontsize=8, frameon=False)

    for ax in (ppl, draw):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    return fig


def kept_by_family(tokenizer, batches, settings: dict, examples: int = 600):
    """How much of each FAMILY of features survived — the honest headline.

    "The token explains 44% of a held-out day" is an average over 26 features, and it is
    carried entirely by the easy ones. Broken apart, it says something quite different:
    the token keeps the slow, smooth indicators almost perfectly and knows next to
    nothing about the actual candle.

    That is not a bug. One token is 9 bits (log2 of 512). Fifteen days of MACD is a
    SMOOTH CURVE — a few numbers describe it. Fifteen days of candle bodies are fifteen
    INDEPENDENT RANDOM NUMBERS — incompressible. The token spends its 9 bits on what can
    actually be compressed, and there is no way for it not to.

    ⚠️ Same routing as `remembered()`: through `tokenizer(...)`, not `VQVAE.forward` --
    see that function's docstring for why the two disagree on the chosen word 64% of
    the time, and how misleading it would be to draw this chart against the wrong one.
    """
    import matplotlib.pyplot as plt

    model = tokenizer.ts
    grids = batches.ts["test"].dataset
    names = batches.arrays.names
    where = next(tokenizer.parameters()).device

    real, said = [], []
    tokenizer.eval()
    with torch.no_grad():
        for i in range(min(examples, len(grids))):
            item = grids[i]
            day = int(item["day"])
            grid = item["grid"].unsqueeze(0).to(where)
            market_grid, market_present = _market_grid_for_day(batches, tokenizer.cs, day)
            out = tokenizer(grid, market_grid.to(where), market_present.to(where))
            back = model.rebuild(out["ts_vector"])
            real.append(grid[0, 0].cpu().numpy())
            said.append(back[0, 0].cpu().numpy())
    tokenizer.train()

    real, said = np.stack(real), np.stack(said)
    kept = 1 - ((real - said) ** 2).mean(axis=(0, 1)) / np.maximum(
        (real ** 2).mean(axis=(0, 1)), 1e-9)

    # Which feature belongs to which family — ask each family what it produces.
    from bubble_bi.data.features import by_family

    kept_by = {
        family: float(np.mean([kept[names.index(n)] for n in columns]))
        for family, columns in by_family(settings).items()
    }
    frame = pd.Series(kept_by).sort_values()

    fig, ax = plt.subplots(figsize=(8, 3.6))
    colours = ["#26a69a" if v > 0.5 else "#ffa726" if v > 0.2 else "#ef5350"
               for v in frame]
    ax.barh(frame.index, frame.to_numpy() * 100, color=colours)
    ax.set_xlabel("how much of this FAMILY the single word kept (%)")
    ax.set_title("The 44% is an average — here is what it is made of",
                 loc="left", fontsize=12)
    ax.axvline(0, color="#444", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    return fig, frame


def tuning_importance(trials):
    """Which knob actually moved the score? Returns the ranking, and draws it.

    Rank correlation, not Optuna's fANOVA: at a dozen trials fANOVA fits a forest to
    almost no data and reports confident nonsense. A rank correlation over twelve points is
    crude, and it is honest about being crude.

    ⚠️ The knobs are read from `tuning.SPACE` — an ALLOWLIST, and deliberately so. This
    used to be a blocklist ("any column not in this hardcoded set is a knob"), and the
    moment `direction_se`/`volatility_se` were added to the trials table the chart ranked
    them as the second and third most important KNOBS in the whole search. They are not
    knobs. They are the ERROR BARS ON THE SCORE — outputs, perfectly correlated with the
    thing they measure — and the chart was reporting an output as a cause. A blocklist has
    to be updated every time a column is added, and nothing makes anyone do it. An
    allowlist cannot drift: `SPACE` is what the search actually turns.
    """
    import matplotlib.pyplot as plt

    from bubble_bi.tuning import SPACE

    usable = trials[np.isfinite(trials["score"])]
    turned = {name for stage in SPACE.values() for name in stage}
    knobs = [c for c in usable.columns if c in turned]
    strength = {}
    for knob in knobs:
        column = pd.to_numeric(usable[knob], errors="coerce")
        if column.notna().sum() > 2 and column.nunique() > 1:
            strength[knob] = abs(column.corr(usable["score"], method="spearman"))

    ranked = pd.Series(strength).dropna().sort_values(ascending=False)
    if ranked.empty:
        print("Not enough completed trials to say which knob mattered.")
        return ranked

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(ranked) + 1.5))
    ax.barh(ranked.index[::-1], ranked.to_numpy()[::-1], color="#26a69a")
    ax.set_xlabel("how strongly this knob moved the score (rank correlation)")
    ax.set_title("Which knob actually mattered", loc="left", fontsize=12)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    plt.show()
    return ranked
