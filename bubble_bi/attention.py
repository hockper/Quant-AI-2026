"""What each company chose to read.

The fusion gives every company's token one question to ask the market:

    "of everything the market is showing me, what matters to ME?"

The answer is a set of weights that sum to 1 — and those weights are the most directly
readable thing the whole model produces. Everything else is a number you have to trust.
This you can simply look at.

What you see depends on what CS was told to offer (`fusion.attend_to`):

    "days"       5 keys    -> which of the market's recent days each company read
    "companies"  30 keys   -> WHICH OTHER COMPANIES each company read.
                             This is the interesting one: a bank attending to other
                             banks, a chip-maker to other chip-makers. It is a
                             company-to-company map the model drew by itself.
    "cells"      150 keys  -> both, and we show each marginal separately

⚠️ **The first thing to check is whether it is choosing at all.**

If every weight is about 1/keys, the attention is FLAT: the company is reading the whole
market equally, which is the same as reading an average of it, which is the same as the
cross-attention not being there. A flat map means the fusion is doing none of the work it
was built for — and that will never show up in a loss curve.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def gather(world, book, batches, settings: dict, period: str = "test") -> dict:
    """Average attention, per company, over days the model never trained on.

    ⚠️ Under joint training a batch is one time WINDOW across ALL companies at once
    (see `WorldModel.forward`) -- not one sentence per company. So the bug this
    function used to guard against (sentences ordered BY company, so sampling the
    first few batches only ever reached the first few tickers) cannot happen any
    more: every batch already carries every company as its own axis. What CAN still
    go wrong is `gather` reading that axis incorrectly, which is what the shape
    check below exists to catch loudly instead of silently.

    `batches` is unused -- kept in the signature for interface stability with
    earlier callers.
    """
    del batches
    from bubble_bi.training import _to, pick_device

    where = pick_device(settings)
    world.to(where).eval()

    companies = len(settings["tickers"])
    total = None

    for batch in book["loaders"][period]:
        out = world(_to(batch, where))
        attention = out["attention"]                       # [B, T, N, keys]
        b, t, n, keys = attention.shape
        if n != companies:
            raise ValueError(
                f"a batch carries {n} companies but `settings['tickers']` lists "
                f"{companies} — they must line up one to one, or the attention map "
                "would be silently mislabelled."
            )

        if total is None:
            total = np.zeros((companies, keys))
            seen = 0
        total += attention.sum(dim=(0, 1)).cpu().numpy()    # sum over batch & days
        seen += b * t

    world.train()
    if total is None:
        raise ValueError("No attention to gather — the loaders are empty.")

    attention = total / max(seen, 1)

    keys = attention.shape[1]
    flat = 1.0 / keys
    # How far from "reading everything equally". 0 = flat (not choosing at all).
    sharpness = float(np.nanmean(np.abs(attention - flat)) / flat)

    return {
        "attention": attention,                 # [companies, keys]
        "tickers": list(settings["tickers"]),
        "attend_to": world.tokenizer.attend_to,
        "keys": keys,
        "flat": flat,
        "sharpness": sharpness,
        "choosing": sharpness > 0.15,
        "cs_days": settings["cs"]["days"],
    }


def _key_labels(evidence: dict) -> list[str]:
    how, keys = evidence["attend_to"], evidence["keys"]
    if how == "days":
        n = evidence["cs_days"]
        return ["today"] + [f"-{i}d" for i in range(1, n)][::-1][:keys - 1][::-1] \
            if keys == n else [f"day {i}" for i in range(keys)]
    if how == "companies":
        return evidence["tickers"][:keys]
    return [str(i) for i in range(keys)]


def plot(evidence: dict):
    """The attention map: who read what."""
    import matplotlib.pyplot as plt

    attention = evidence["attention"]
    tickers = evidence["tickers"]
    how = evidence["attend_to"]
    flat = evidence["flat"]

    if how == "cells":
        return _cells_plot(evidence)

    labels = _key_labels(evidence)
    height = max(4.5, 0.28 * len(tickers))
    fig, ax = plt.subplots(figsize=(max(7, 0.42 * len(labels) + 3), height))

    # Centre the colour scale on "flat", so choosing MORE than average is one colour and
    # LESS is the other. A map that is all one flat colour is the model not choosing.
    span = max(np.nanmax(np.abs(attention - flat)), 1e-9)
    picture = ax.imshow(attention, cmap="RdBu_r", aspect="auto",
                        vmin=flat - span, vmax=flat + span)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90 if len(labels) > 8 else 0, fontsize=8)
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers, fontsize=8)
    ax.set_ylabel("this company's token…")
    ax.set_xlabel("…read this much of" + (
        " each recent market day" if how == "days" else " each other company"))

    bar = fig.colorbar(picture, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label(f"attention   (flat = {flat:.2f})", fontsize=9)

    verdict = ("it is choosing" if evidence["choosing"]
               else "FLAT — it is not choosing")
    ax.set_title(
        f"What each company read of the market   —   {verdict}",
        loc="left", fontsize=12,
    )
    if not evidence["choosing"]:
        ax.text(
            0.5, -0.14,
            "Every company is reading the market almost equally, which is the same as "
            "reading an average of it —\nthe cross-attention is doing none of the work it "
            "was built for. This never shows up in a loss curve.",
            transform=ax.transAxes, ha="center", fontsize=9, color="#c62828",
        )
    fig.tight_layout()
    return fig


def _cells_plot(evidence: dict):
    """With 150 keys, show the two marginals instead of an unreadable wall."""
    import matplotlib.pyplot as plt

    attention = evidence["attention"]
    tickers = evidence["tickers"]
    n, days = len(tickers), evidence["cs_days"]
    grid = attention.reshape(len(attention), n, days)

    by_company = np.nansum(grid, axis=2)        # [companies, companies]
    by_day = np.nansum(grid, axis=1)            # [companies, days]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13, max(4.5, 0.28 * n)),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    flat_c = 1.0 / n
    span = max(np.nanmax(np.abs(by_company - flat_c)), 1e-9)
    a = left.imshow(by_company, cmap="RdBu_r", aspect="auto",
                    vmin=flat_c - span, vmax=flat_c + span)
    left.set_xticks(range(n)); left.set_xticklabels(tickers, rotation=90, fontsize=7)
    left.set_yticks(range(n)); left.set_yticklabels(tickers, fontsize=7)
    left.set_title("…read this company", loc="left", fontsize=11)
    left.set_ylabel("this company's token…")
    fig.colorbar(a, ax=left, fraction=0.03, pad=0.02)

    flat_d = 1.0 / days
    span = max(np.nanmax(np.abs(by_day - flat_d)), 1e-9)
    b = right.imshow(by_day, cmap="RdBu_r", aspect="auto",
                     vmin=flat_d - span, vmax=flat_d + span)
    right.set_xticks(range(days))
    right.set_xticklabels([f"-{days - 1 - i}d" for i in range(days)], fontsize=8)
    right.set_yticks([])
    right.set_title("…and this day", loc="left", fontsize=11)
    fig.colorbar(b, ax=right, fraction=0.06, pad=0.02)

    verdict = "it is choosing" if evidence["choosing"] else "FLAT — it is not choosing"
    fig.suptitle(f"What each company read of the market   —   {verdict}",
                 fontsize=12, x=0.02, ha="left")
    fig.tight_layout()
    return fig


def neighbours(evidence: dict, top: int = 3):
    """Who each company reads most — only meaningful with attend_to='companies'."""
    import pandas as pd

    if evidence["attend_to"] not in ("companies", "cells"):
        raise ValueError(
            "This only means something when CS offers COMPANIES as keys. "
            "Set fusion['attend_to'] = 'companies' (or 'cells')."
        )

    attention = evidence["attention"]
    tickers = evidence["tickers"]
    n = len(tickers)
    if evidence["attend_to"] == "cells":
        attention = np.nansum(
            attention.reshape(len(attention), n, evidence["cs_days"]), axis=2)

    rows = []
    for i, who in enumerate(tickers):
        weights = attention[i].copy()
        weights[i] = -np.inf                      # ignore itself
        best = np.argsort(-weights)[:top]
        rows.append({
            "company": who,
            "reads most": ", ".join(f"{tickers[j]} ({attention[i, j]:.1%})" for j in best),
            "reads itself": f"{attention[i, i]:.1%}",
        })
    return pd.DataFrame(rows).set_index("company")
