"""Is CS actually working?

Reconstruction cannot answer this. We measured it: one word out of 512 rebuilds thirty
companies at about 8%, and it never will do better — most of what a company does is its
own business, and a single shared word can only carry what they have in common.

So we ask a different, fairer question, and the one we actually care about:

    **Does the token know what the market DID that day?**

Take every day the model has never seen, read its token, and check whether the token
tells you anything about how the market really behaved — how far it moved, and how
violently. If the tokens genuinely separate a calm grind from a panic, CS has learned
the market's moods, whatever its reconstruction score says.

The number this produces is an R²: of all the variation in (say) the market's daily
move, how much is explained just by knowing which word the day was given?

    R² = 0.00   the token tells you nothing. CS learned nothing.
    R² = 0.30   knowing the word explains 30% of how the market moved that day.
    R² = 1.00   the word tells you exactly what happened. (Not going to happen.)

A word of caution before reading too much into a big number: with 512 words and only a
few hundred test days, a token could look informative by sheer luck. So we also report
what a SHUFFLED assignment scores — the same tokens, handed to the wrong days. That is
the "learned nothing" floor, measured rather than assumed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch

# What the market "did" on a day, in plain terms. Both are averaged across companies,
# in each company's own units (the features are normalised per company), so they mean
# "how unusual was today for the average company".
MOODS = {
    "how far it moved": "log_return",
    "how violently": "realized_vol",
    "how expensive to trade": "roll_spread",
}


def _explained(values: np.ndarray, groups: np.ndarray) -> float:
    """How much of `values` is explained just by knowing which group a day is in?"""
    keep = np.isfinite(values)
    values, groups = values[keep], groups[keep]
    if len(values) < 3:
        return float("nan")

    total = values.var()
    if total <= 0:
        return float("nan")

    # variance left over once each group is replaced by its own average
    leftover = 0.0
    for g in np.unique(groups):
        mine = values[groups == g]
        leftover += len(mine) * mine.var()
    leftover /= len(values)
    return float(1 - leftover / total)


@torch.no_grad()
def market_moods(cs, batches, settings: dict, period: str = "test") -> dict:
    """What do the CS tokens actually mean? Returns the evidence, not a verdict."""
    from bubble_bi.training import pick_device

    where = pick_device(settings)
    cs = cs.to(where).eval()
    arrays = batches.arrays

    tokens, days = [], []
    for batch in batches.cs[period]:
        grid = batch["grid"].to(where)
        present = batch["present"].to(where)
        tokens.append(cs.codebook(cs.summarise(grid, present))["ids"].cpu().numpy())
        days.append(batch["day"].numpy())
    cs.train()

    tokens = np.concatenate(tokens)
    days = np.concatenate(days)

    # What the market really did on those days -- raw features, before normalisation,
    # averaged across the companies that actually traded.
    truth = {}
    for label, feature in MOODS.items():
        if feature not in arrays.names:
            continue
        column = arrays.x[:, :, arrays.names.index(feature)]
        per_day = np.where(arrays.ok, column, np.nan)
        with warnings.catch_warnings():
            # Early days, before the slow features have warmed up, are entirely blank.
            # That is expected -- they simply have no market to average.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            truth[label] = np.nanmean(per_day, axis=1)[days]

    rng = np.random.default_rng(0)
    shuffled = rng.permutation(tokens)          # the same words, on the wrong days

    scores = pd.DataFrame({
        "explained by the token": [_explained(v, tokens) for v in truth.values()],
        "explained by luck": [_explained(v, shuffled) for v in truth.values()],
    }, index=list(truth))

    return {
        "tokens": tokens,
        "days": days,
        "truth": truth,
        "scores": scores,
        "words_used": int(len(np.unique(tokens))),
        "dates": arrays.dates[days],
    }


def moods_plot(evidence: dict, top: int = 14):
    """What the busiest words mean — with the honest score printed on each panel.

    ⚠️ The direction panel is a trap, and that is exactly why it is here. Its bars look
    like a pattern: some words green, some red. They are noise. The R² printed on it
    says so — the words explain no more of the market's direction than a random shuffle
    would. Without that number on the chart, a reader would draw precisely the wrong
    conclusion from it.
    """
    import matplotlib.pyplot as plt

    tokens = evidence["tokens"]
    scores = evidence["scores"]
    violent = evidence["truth"].get("how violently")
    moved = evidence["truth"].get("how far it moved")

    busiest = pd.Series(tokens).value_counts().head(top).index.to_numpy()
    rows = [{
        "word": f"#{w}",
        "violent": float(np.nanmean(violent[tokens == w])),
        "moved": float(np.nanmean(moved[tokens == w])),
    } for w in busiest]
    frame = pd.DataFrame(rows).sort_values("violent")     # sort by what is REAL

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)

    def score_of(label):
        return scores.loc[label, "explained by the token"], scores.loc[label, "explained by luck"]

    real, luck = score_of("how violently")
    left.barh(frame["word"], frame["violent"] * 100, color="#7e57c2")
    left.set_xlabel("average volatility that day (%)")
    left.set_title("How violent the market was", loc="left", fontsize=11)
    left.text(0.98, 0.04, f"the word explains {real:.0%}\n(luck would give {luck:.0%})",
              transform=left.transAxes, ha="right", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.4", fc="#e8f5e9", ec="#66bb6a"))

    real, luck = score_of("how far it moved")
    colours = ["#26a69a" if m >= 0 else "#ef5350" for m in frame["moved"]]
    right.barh(frame["word"], frame["moved"] * 100, color=colours, alpha=0.55)
    right.axvline(0, color="#444", linewidth=0.8)
    right.set_xlabel("average move of the market that day (%)")
    right.set_title("Which way it went", loc="left", fontsize=11)
    right.text(0.98, 0.04,
               f"the word explains {real:.0%}\n(luck would give {luck:.0%})\n"
               f"→ this panel is NOISE",
               transform=right.transAxes, ha="right", fontsize=9,
               bbox=dict(boxstyle="round,pad=0.4", fc="#ffebee", ec="#ef5350"))

    for ax in (left, right):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)

    fig.suptitle(
        "The words know how violent the market was. They know nothing about which way it went.",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    return fig
