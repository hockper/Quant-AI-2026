"""The whole thing, end to end.

    TS encoder (frozen) ──┐
                          ├─► cross-attention ─► codebook ─► ONE token ─► GPT ─► next token
    CS encoder (frozen) ──┘

TS and CS were each trained as a full VQ-VAE: encoder, codebook, decoder. That was
scaffolding. **Here we keep only their encoders** — the codebooks and decoders they
were trained with are thrown away, and the encoders are frozen.

**Why the fusion and the predictor train TOGETHER.**

We could bolt a decoder onto the fusion and train it to rebuild the market, as we did
for TS and CS. It would be a mistake. We already measured what happens when a single
token is asked to redraw thirty companies: it manages about 8%. Optimising a token for
a task it cannot do would produce a token good at nothing.

So the fusion is trained against the job the token actually has: **being predictable**.
The predictor's loss flows back through the codebook into the cross-attention, and the
tokens arrange themselves to be worth predicting rather than worth redrawing.

(The paper does the same thing in its own way — it weights reconstruction at 1e-3 and
return-prediction at 0.1. Its factors are trained to predict, not to redraw.)

**One thing to watch.** The codebook is moving while the predictor learns to predict
its output, so the predictor is chasing a target that is itself still settling. The
codebook is updated by slow moving averages, which is what keeps that stable — but if
training ever looks like it is thrashing, this is the first place to look.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bubble_bi.models.codebook import Codebook
from bubble_bi.models.fusion import Fusion
from bubble_bi.models.llama import RMSNorm, Block


class Tokenizer(nn.Module):
    """Two frozen encoders, a cross-attention, and one codebook. Grids in, tokens out."""

    def __init__(self, ts, cs, vocabulary: int = 512, depth: int = 2, heads: int = 4,
                 dropout: float = 0.1, attend_to: str = "days"):
        super().__init__()
        if ts.features != cs.features:
            raise ValueError("TS and CS must have been trained on the same features.")

        self.ts, self.cs = ts, cs
        self.attend_to = attend_to
        self.width = ts.read.out_features

        # Frozen: they already learned to describe a company and a market. Letting them
        # drift now would mean re-learning what we just spent two sections teaching them.
        for encoder in (self.ts, self.cs):
            encoder.eval()
            for weight in encoder.parameters():
                weight.requires_grad = False

        self.fusion = Fusion(self.width, depth=depth, heads=heads, dropout=dropout)
        self.codebook = Codebook(words=vocabulary, width=self.width)

    def latents(self, stock_grid: torch.Tensor, market_grid: torch.Tensor,
                market_present: torch.Tensor | None = None):
        """The fused vector for each (company, day), plus what it attended to."""
        with torch.no_grad():                      # frozen: no gradient, no drift
            z_ts = self.ts.summarise(stock_grid)                      # [B, width]
            market = self.cs.context(market_grid, market_present, self.attend_to)
        return self.fusion(z_ts, market)                             # [B, width], [B, keys]

    def forward(self, stock_grid: torch.Tensor, market_grid: torch.Tensor,
                market_present: torch.Tensor | None = None) -> dict:
        fused, attention = self.latents(stock_grid, market_grid, market_present)
        chosen = self.codebook(fused)
        return {
            "token": chosen["ids"],                 # [B] -- the word for this company-day
            "vector": chosen["snapped"],            # [B, width] -- and its meaning
            "attention": attention,                 # [B, keys] -- what it read
            "commitment_loss": chosen["commitment_loss"],
            "diversity_loss": chosen["diversity_loss"],
            "perplexity": chosen["perplexity"],
        }


class WorldModel(nn.Module):
    """The tokenizer and the predictor, trained as one.

    A batch is a SENTENCE: one company, over `sentence` consecutive days, each day a
    grid. We turn every day into a token, then ask the GPT to predict the next one.
    """

    def __init__(self, tokenizer: Tokenizer, sentence: int = 64, depth: int = 4,
                 heads: int = 4, kv_heads: int | None = None, theta: float = 10000.0,
                 dropout: float = 0.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.sentence = sentence
        self.words = tokenizer.codebook.words
        width = tokenizer.width

        self.blocks = nn.ModuleList(
            Block(width, heads, kv_heads, theta, dropout) for _ in range(depth)
        )
        self.settle = RMSNorm(width)
        self.guess = nn.Linear(width, self.words, bias=False)

    def understand(self, vectors: torch.Tensor) -> torch.Tensor:
        """Read a sentence of token vectors [B, T, width] -> what it makes of each day.

        This hidden state is the artifact the RL agent will eventually consume: the
        model's understanding of where the company stands, with the prediction head
        taken off.
        """
        for block in self.blocks:
            vectors = block(vectors)
        return self.settle(vectors)

    def forward(self, batch: dict) -> dict:
        """batch:
            stock   [B, T, 1, ts_days, F]        each day of the sentence, that company
            market  [B, T, N, cs_days, F]        each day of the sentence, the market
            present [B, T, N]                    who was trading
        """
        stock, market = batch["stock"], batch["market"]
        present = batch.get("present")
        b, t = stock.shape[:2]

        # Every day of the sentence becomes a token. Flatten the sentence into the batch
        # so the tokenizer sees them all at once.
        flat = self.tokenizer(
            stock.reshape(b * t, *stock.shape[2:]),
            market.reshape(b * t, *market.shape[2:]),
            None if present is None else present.reshape(b * t, -1),
        )
        tokens = flat["token"].view(b, t)                       # [B, T]
        vectors = flat["vector"].view(b, t, -1)                 # [B, T, width]

        # Read days 0..T-2, and try to name day 1..T-1. The attention is causal, so no
        # position can see the answer it is being asked for.
        thought = self.understand(vectors[:, :-1])
        said = self.guess(thought)                             # [B, T-1, words]
        answer = tokens[:, 1:]                                 # [B, T-1]

        naming = F.cross_entropy(said.reshape(-1, self.words), answer.reshape(-1))
        right = (said.argmax(-1) == answer).float().mean()

        return {
            "loss": naming + flat["commitment_loss"] + flat["diversity_loss"],
            "naming_loss": naming,                  # how well it predicts the next word
            "commitment_loss": flat["commitment_loss"],
            "diversity_loss": flat["diversity_loss"],
            "accuracy": right,                      # ...and how often it is exactly right
            "perplexity": flat["perplexity"],       # is the dictionary still alive?
            "tokens": tokens,
            "attention": flat["attention"].view(b, t, -1),
        }

    def describe(self) -> str:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return (
            f"frozen encoders ({frozen/1e6:.2f}M) → cross-attention → "
            f"1 token of {self.words} → GPT ({len(self.blocks)} layers)\n"
            f"   training {trainable/1e6:.2f}M weights, "
            f"reading sentences of {self.sentence} days"
        )
