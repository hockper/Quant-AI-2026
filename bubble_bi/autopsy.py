"""Why is the token empty?

The fusion codebook collapses to ~12 words of 512 and the predictor loses to
persistence. We know *that*. This module exists to find out *why*, because the fix
depends entirely on the answer, and there are three quite different culprits:

    1. THE FUSION is producing nothing.      The continuous vector it hands to the
                                             codebook is already uninformative.
    2. THE CODEBOOK is destroying it.        The continuous vector IS informative, but
                                             quantising it throws that away.
    3. THE ANCHOR is too weak.               Both are informative enough, but nothing in
                                             the loss punishes an empty token hard enough
                                             to stop the model collapsing it anyway.

These are distinguishable, and this tells them apart:

    • Fit a probe from the CONTINUOUS fused vector to today's candle.
      Poor  -> culprit 1: the fusion is not producing anything worth quantising.
      Good  -> the fusion is fine, keep looking.

    • Fit the same probe from the QUANTISED token instead.
      Much worse than the continuous one -> culprit 2: quantisation is the bottleneck.
      About the same, and both good        -> culprit 3: it is the loss, not the model.

Run it on Colab after a real training run and bring the numbers back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from bubble_bi.models.world import CANDLE


def _probe(x: np.ndarray, y: np.ndarray, train: float = 0.7) -> float:
    """How much of `y` can a linear probe recover from `x`? (R², on held-out rows.)

    Deliberately LINEAR. If a linear probe can find it, the information is plainly
    there. If it cannot, the information is at best deeply buried — and a codebook of
    512 nearest-neighbour lookups is not going to dig it out either.
    """
    keep = np.isfinite(y).all(axis=1) & np.isfinite(x).all(axis=1)
    x, y = x[keep], y[keep]
    if len(x) < 50:
        return float("nan")

    cut = int(len(x) * train)
    x_fit, y_fit = x[:cut], y[:cut]
    x_test, y_test = x[cut:], y[cut:]

    x_fit = np.c_[x_fit, np.ones(len(x_fit))]
    x_test = np.c_[x_test, np.ones(len(x_test))]

    # ridge, so a wide vector cannot simply memorise the rows
    n = x_fit.shape[1]
    w = np.linalg.solve(x_fit.T @ x_fit + 1e-2 * np.eye(n), x_fit.T @ y_fit)

    left = ((y_test - x_test @ w) ** 2).sum()
    total = ((y_test - y_fit.mean(axis=0)) ** 2).sum()
    return float(1 - left / max(total, 1e-12))


@torch.no_grad()
def gather(world, book, batches, settings: dict, period: str = "test",
           every: int = 4) -> dict:
    """Collect everything needed to tell the three culprits apart.

    ⚠️ Takes every Nth batch across the WHOLE period, never the first N. The sentences
    are ordered by company, so stopping early would sample only the first few companies
    and quietly answer a different question than the one asked.
    """
    from bubble_bi.training import _to, pick_device

    where = pick_device(settings)
    world.to(where).eval()
    tokenizer = world.tokenizer

    fused, tokens, candles, attention, targets = [], [], [], [], []
    arrays = batches.arrays
    y = arrays.y                                        # tomorrow's return

    for i, batch in enumerate(book["loaders"][period]):
        if i % every:                                   # thin it out, but evenly
            continue
        batch = _to(batch, where)
        b, t, width = batch["z_ts"].shape

        vector, weights = tokenizer.fusion(
            batch["z_ts"].reshape(b * t, width),
            batch["market"].reshape(b * t, *batch["market"].shape[2:]),
        )
        chosen = tokenizer.codebook(vector)

        fused.append(vector.cpu().numpy())                          # BEFORE quantising
        tokens.append(chosen["ids"].cpu().numpy())                  # AFTER quantising
        candles.append(batch["candle"].reshape(b * t, -1).cpu().numpy())
        attention.append(weights.cpu().numpy())

        # tomorrow's actual return, for each (company, day) in the sentence
        last = batch["last_day"].cpu().numpy()
        who = batch["company"].cpu().numpy()
        for row in range(b):
            days = np.arange(last[row] - t + 1, last[row] + 1)
            targets.append(y[days, who[row]])

    world.train()

    fused = np.concatenate(fused)
    tokens = np.concatenate(tokens)
    candles = np.concatenate(candles)
    attention = np.concatenate(attention)
    targets = np.concatenate(targets).reshape(-1, 1)

    # The quantised token, as a vector: its one-hot, which is all a codebook lookup is.
    words = tokenizer.codebook.words
    onehot = np.zeros((len(tokens), words), dtype=np.float32)
    onehot[np.arange(len(tokens)), tokens] = 1.0

    return {
        "fused": fused, "tokens": tokens, "candle": candles,
        "attention": attention, "target": targets, "onehot": onehot,
        "words": words,
    }


def report(evidence: dict) -> pd.DataFrame:
    """The table that names the culprit."""
    fused, onehot = evidence["fused"], evidence["onehot"]
    candle, target = evidence["candle"], evidence["target"]

    rows = {
        "TODAY's candle, from the CONTINUOUS vector": _probe(fused, candle),
        "TODAY's candle, from the QUANTISED token": _probe(onehot, candle),
        "TOMORROW's return, from the CONTINUOUS vector": _probe(fused, target),
        "TOMORROW's return, from the QUANTISED token": _probe(onehot, target),
    }
    return pd.DataFrame({"recoverable (R²)": rows.values()}, index=list(rows))


def verdict(evidence: dict) -> str:
    """Read the table, name the culprit, and say what to do about it."""
    fused, onehot = evidence["fused"], evidence["onehot"]
    candle = evidence["candle"]

    before = _probe(fused, candle)
    after = _probe(onehot, candle)
    live = len(np.unique(evidence["tokens"]))

    spread = evidence["attention"].std(axis=1).mean()
    keys = evidence["attention"].shape[1]
    flat = 1.0 / keys

    lines = [
        f"words actually used: {live} of {evidence['words']}",
        f"attention: spread {spread:.3f} across {keys} keys "
        f"({'it is choosing' if spread > flat * 0.2 else 'FLAT — it is not choosing'})",
        "",
    ]

    if not np.isfinite(before) or before < 0.10:
        lines += [
            "CULPRIT: THE FUSION.",
            f"  Even the continuous vector, before any quantising, only carries "
            f"{before:.0%} of today's candle. There is nothing worth quantising. "
            "The codebook is not the problem — it is being handed an empty vector.",
            "  → Look at the fusion and at what the frozen encoders are giving it.",
        ]
    elif after < before * 0.5:
        lines += [
            "CULPRIT: THE CODEBOOK.",
            f"  The continuous vector carries {before:.0%} of today's candle, but the "
            f"quantised token only {after:.0%}. Quantising is throwing the information "
            "away — too few live words to say anything with.",
            "  → Fight the collapse harder: more diversity, more revival, more words.",
        ]
    else:
        lines += [
            "CULPRIT: THE LOSS.",
            f"  Both the vector ({before:.0%}) and the token ({after:.0%}) carry today's "
            "candle. The model CAN say something — it is simply never punished hard "
            "enough for not bothering.",
            "  → Add the anchor: a head straight off the token that reconstructs TODAY's "
            "candle. See docs/OPEN-QUESTION-codebook-collapse.md.",
        ]
    return "\n".join(lines)


def plot(evidence: dict):
    """How the words are spread, and whether the attention is choosing anything."""
    import matplotlib.pyplot as plt

    tokens, attention = evidence["tokens"], evidence["attention"]
    words = evidence["words"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 3.8))

    counts = np.bincount(tokens, minlength=words)
    order = np.argsort(-counts)
    live = int((counts > 0).sum())
    left.bar(range(min(60, words)), counts[order][:60], color="#5c6bc0")
    left.set_xlabel(f"the 60 busiest words (of {words}; only {live} are used at all)")
    left.set_ylabel("days")
    left.set_title("How the vocabulary is spread", loc="left", fontsize=11)

    mean = attention.mean(axis=0)
    right.bar(range(len(mean)), mean, color="#26a69a")
    right.axhline(1 / len(mean), color="#ef5350", linestyle="--", linewidth=1,
                  label="flat = not choosing")
    right.set_xlabel("which part of the market it read")
    right.set_ylabel("average attention")
    right.set_title("Is the cross-attention choosing?", loc="left", fontsize=11)
    right.legend(fontsize=8, frameon=False)

    for ax in (left, right):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    return fig
