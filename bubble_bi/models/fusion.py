"""Where a company's story meets the market's.

TS knows what one company has been doing. CS knows what the market has been doing.
Neither is enough on its own: a stock falling 3% means something very different on a
day the whole market fell 3% than on a day the market was flat.

So each company's token gets to **ask the market a question**:

    query   what THIS company is doing today        1 vector      (from TS)
    keys    what the market has been doing          5..150 vectors (from CS)
        ↓
    the company scores every market vector, softmaxes those scores into weights,
    and blends the market vectors together with them
        ↓
    ONE vector — the company's own state, enriched with the part of the market
    that it decided was relevant

The output has the same length as the QUERY, never the keys. One query in, one vector
out, one token per (company, day). The keys are read, not emitted.

Two details that matter:

**The residual.** The blend on its own is made only of market vectors — the company's
own information chose the weights but does not appear in the result. Two companies
that happened to pick similar weights would then get the SAME token. Adding `z_ts`
back is what keeps each company's own identity in its token.

**The weights are readable.** They say, in numbers, which parts of the market each
company attended to. That is not a diagnostic afterthought — it is the most direct
evidence we have that the fusion is doing anything at all.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _Ask(nn.Module):
    """One round of: ask the market, then think about the answer."""

    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.before_asking = nn.LayerNorm(width)
        self.before_reading = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.think = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )

    def forward(self, question: torch.Tensor, market: torch.Tensor):
        answer, weights = self.attention(
            self.before_asking(question),
            self.before_reading(market),
            self.before_reading(market),
            need_weights=True,
            average_attn_weights=True,
        )
        question = question + answer            # the residual: keep who is asking
        question = question + self.think(question)
        return question, weights


class Fusion(nn.Module):
    """Cross-attention: one company's state, enriched by the market it chose to read.

        fused, weights = fusion(z_ts, market)

        z_ts     [B, width]          the query -- one company, today
        market   [B, keys, width]    what CS is offering to be read
        fused    [B, width]          ONE vector -> ONE token
        weights  [B, keys]           which parts of the market it attended to
    """

    def __init__(self, width: int, depth: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if width % heads:
            raise ValueError(f"width ({width}) must divide evenly by heads ({heads}).")
        if depth < 1:
            raise ValueError(f"fusion depth must be at least 1, got {depth}.")
        self.rounds = nn.ModuleList(_Ask(width, heads, dropout) for _ in range(depth))
        self.settle = nn.LayerNorm(width)

    def forward(self, z_ts: torch.Tensor, market: torch.Tensor):
        if z_ts.shape[0] != market.shape[0]:
            raise ValueError(
                f"Got {z_ts.shape[0]} companies but {market.shape[0]} markets — "
                "they must line up one to one."
            )
        question = z_ts.unsqueeze(1)                     # [B, 1, width]
        weights = None
        for ask in self.rounds:
            question, weights = ask(question, market)
        return self.settle(question.squeeze(1)), weights.squeeze(1)
