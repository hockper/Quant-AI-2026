"""A linear-probe culprit test -- and the piece of it that outlived the bug it was built for.

This module used to answer "why is the FUSED token empty?": under the old two-stage
design, TS and CS were frozen and their outputs quantised into a THIRD codebook, trained
only against "can this be predicted?" with no reconstruction anchor of its own. That
codebook collapsed to ~12 words of 512, and `gather()` (removed) plus `plot()` (removed)
existed to tell apart three culprits -- the fusion, the codebook, or the loss -- for
exactly that one collapsing vector.

⚠️ There is no fused codebook any more (see `bubble_bi.models.tokenizer`). TS and CS each
keep their OWN codebook, each anchored by rebuilding its own grid -- the fix the third
branch below (`report`/`verdict`'s "CULPRIT: THE LOSS") would have recommended, now built
into the design instead of diagnosed after the fact. So the question `gather()` was
written to answer no longer arises, and it could not have kept working regardless: it
called `tokenizer.codebook(...)` (deleted) and read cached-latent batch keys (`z_ts`,
`market`, `company`, `last_day`) that the current sentence format
(`bubble_bi.data.sentences.Sentences`) does not produce. `tests/test_autopsy.py` only
ever imported `_probe`, `report` and `verdict` -- never `gather` -- so nothing caught it
being broken. Rather than resurrect a diagnostic for a bug that has been designed out (or
leave a function that would raise the moment anyone called it), `gather()` and `plot()`
are deleted along with the notebook section that called them.

What is left, and why it stays:

    `_probe`    a plain ridge-regression R² helper -- how much of `y` can a LINEAR probe
                recover from `x`, on held-out rows. `bubble_bi.tuning` imports this
                directly for its own "did today survive the bottleneck?" search (see its
                module docstring), so this file is not just diagnostic history.
    `report`    `verdict`   generic culprit-naming logic over an `evidence` dict of
                {fused vector, quantised token, candle, target}. Still exercised by their
                own tests with hand-built evidence. Nothing in the current pipeline builds
                that dict any more -- a future two-token autopsy could, by encoding a real
                sentence batch and pulling `ts_summary`/`ts_token` (or `cs_summary`/
                `cs_token`) out of `Tokenizer.forward` before it is thrown away -- but that
                is new work, not a rewrite of what was here, and nothing in this project
                currently needs it: `verify.joint`'s perplexity check already watches both
                anchored codebooks for the collapse this module used to have to go
                hunting for after the fact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
