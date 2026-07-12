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


def remembered(model, batches, prices: pd.DataFrame, settings: dict,
               company: str | None = None, when: int | None = None):
    """Draw the real candles beside the ones a single token remembered.

    The model never sees a candle — it sees the candle's SHAPE (gap, body, wicks),
    which is scale-free. So we take the shape it reconstructed, invert it back into
    open/high/low/close, and draw it. Given the previous close, that inversion is
    exact: what you see is precisely what the word remembered, not an impression.
    """
    import matplotlib.pyplot as plt

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

    # What the token remembered.
    model.eval()
    where = next(model.parameters()).device
    with torch.no_grad():
        batch = {"grid": item["grid"].unsqueeze(0).to(where)}
        out = model(batch)
    word = int(out["ids"][0])
    rebuilt = out["rebuilt"][0, 0].cpu().numpy() if "rebuilt" in out else None
    if rebuilt is None:
        with torch.no_grad():
            rebuilt = model.rebuild(model.codebook(model.summarise(batch["grid"]))
                                    ["snapped"])[0, 0].cpu().numpy()

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
    model.train()
    return fig


def kept_and_lost(model, batches, settings: dict, examples: int = 400):
    """Which of the 26 features survived the squeeze, and which were thrown away.

    A single word cannot carry everything. This says, feature by feature, how much
    of it came back — and being clear about what was DISCARDED is as informative as
    what was kept.
    """
    import matplotlib.pyplot as plt

    grids = batches.ts["test"].dataset
    names = batches.arrays.names
    where = next(model.parameters()).device

    real, said = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, min(examples, len(grids))):
            item = grids[i]
            grid = item["grid"].unsqueeze(0).to(where)
            out = model({"grid": grid})
            back = model.rebuild(model.codebook(model.summarise(grid))["snapped"])
            real.append(grid[0, 0].cpu().numpy())
            said.append(back[0, 0].cpu().numpy())
    model.train()

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

    The model never sees a candle. It reads a sentence of words and, for each day, draws
    the SHAPE of the next one — the gap, the body, the two wicks. Those four numbers are
    exactly invertible, so given the previous close we can rebuild the whole candle and
    put the two side by side.

    ⚠️ Read this honestly. A model that has learned nothing draws a row of small,
    near-identical candles — the average of everything, which is the safest guess when
    you know nothing. Wide, varied predicted candles mean it is COMMITTING to something.
    Whether it commits *correctly* is a separate question, and only the numbers answer it.
    """
    import matplotlib.pyplot as plt
    import torch

    from bubble_bi.data.features import names as feature_names
    from bubble_bi.models.world import CANDLE
    from bubble_bi.training import _to, pick_device

    where = pick_device(settings)
    world.to(where).eval()

    memory = book["memory"]
    sentences = book["loaders"]["test"].dataset
    wanted = memory.tickers.index(company) if company else None
    picks = [i for i, (_, c) in enumerate(sentences.where)
             if wanted is None or c == wanted]
    if not picks:
        raise ValueError(f"No test sentences for {company!r}.")

    item = sentences[picks[len(picks) // 2]]
    with torch.no_grad():
        out = world(_to({k: v.unsqueeze(0) for k, v in item.items()}, where))
    world.train()

    columns = [feature_names().index(c) for c in CANDLE]
    who = int(item["company"])
    end = int(item["last_day"])

    # Undo the normalisation -- back into the units a candle actually lives in.
    spread = batches.scaler.spread[who, columns]
    middle = batches.scaler.middle[who, columns]

    # The model draws day t+1 from day t, so its guesses line up one day LATER.
    drawn = out["drawn"][0, -show:].cpu().numpy() * spread + middle
    real = item["candle"][-show:].numpy() * spread + middle

    days = memory.dates[end - show + 1: end + 1]
    ticker = memory.tickers[who]
    before = prices.loc[(memory.dates[end - show], ticker), "close"]

    truth = rebuild_candles(pd.DataFrame(real, columns=CANDLE, index=days), before)
    guess = rebuild_candles(pd.DataFrame(drawn, columns=CANDLE, index=days), before)

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


def kept_by_family(model, batches, settings: dict, examples: int = 600):
    """How much of each FAMILY of features survived — the honest headline.

    "The token explains 44% of a held-out day" is an average over 26 features, and it is
    carried entirely by the easy ones. Broken apart, it says something quite different:
    the token keeps the slow, smooth indicators almost perfectly and knows next to
    nothing about the actual candle.

    That is not a bug. One token is 9 bits (log2 of 512). Fifteen days of MACD is a
    SMOOTH CURVE — a few numbers describe it. Fifteen days of candle bodies are fifteen
    INDEPENDENT RANDOM NUMBERS — incompressible. The token spends its 9 bits on what can
    actually be compressed, and there is no way for it not to.
    """
    import matplotlib.pyplot as plt

    grids = batches.ts["test"].dataset
    names = batches.arrays.names
    where = next(model.parameters()).device

    real, said = [], []
    model.eval()
    with torch.no_grad():
        for i in range(min(examples, len(grids))):
            grid = grids[i]["grid"].unsqueeze(0).to(where)
            back = model.rebuild(model.codebook(model.summarise(grid))["snapped"])
            real.append(grid[0, 0].cpu().numpy())
            said.append(back[0, 0].cpu().numpy())
    model.train()

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
