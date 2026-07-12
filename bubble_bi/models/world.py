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

    def __init__(self, ts, cs, settings: dict, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if ts.features != cs.features:
            raise ValueError("TS and CS must have been trained on the same features.")

        # Unlike VQVAE (`VQVAE(**settings["ts"])`), this one takes the whole `settings`
        # dict rather than a splat of one block: the fusion codebook is the one that
        # COLLAPSES to ~12 words, and it needs both `fusion` (its own knobs) AND
        # `model_size` (which lives outside that block) to be built correctly. Splatting
        # just `**settings["fusion"]` would lose `model_size` entirely.
        fusion = settings["fusion"]

        self.ts, self.cs = ts, cs
        self.attend_to = fusion["attend_to"]
        self.width = ts.read.out_features

        # Frozen: they already learned to describe a company and a market. Letting them
        # drift now would mean re-learning what we just spent two sections teaching them.
        for encoder in (self.ts, self.cs):
            encoder.eval()
            for weight in encoder.parameters():
                weight.requires_grad = False

        self.fusion = Fusion(self.width, depth=fusion["depth"], heads=heads, dropout=dropout)
        self.codebook = Codebook(
            words=fusion["vocabulary"],
            width=settings["model_size"],
            commitment=fusion["commitment"],
            diversity=fusion["diversity"],
            decay=fusion["decay"],
        )

    @torch.no_grad()
    def look(self, stock_grid: torch.Tensor, market_grid: torch.Tensor,
             market_present: torch.Tensor | None = None):
        """The FROZEN half: grids in, latents out.

        Because these encoders never change, their answers never change either — so this
        is run exactly once over the whole history and cached. Training then never has
        to push thirty companies through a transformer again; it just looks the answer
        up. That is what freezing actually buys us.

            z_ts    [B, width]          what this company is doing
            market  [B, keys, width]    what the market is offering to be read
        """
        return (
            self.ts.summarise(stock_grid),
            self.cs.context(market_grid, market_present, self.attend_to),
        )

    def speak(self, z_ts: torch.Tensor, market: torch.Tensor) -> dict:
        """The TRAINABLE half: latents in, one token out."""
        fused, attention = self.fusion(z_ts, market)
        chosen = self.codebook(fused)
        return {
            "token": chosen["ids"],                 # [B] -- the word for this company-day
            "vector": chosen["snapped"],            # [B, width] -- and its meaning
            "attention": attention,                 # [B, keys] -- what it read
            "commitment_loss": chosen["commitment_loss"],
            "diversity_loss": chosen["diversity_loss"],
            "perplexity": chosen["perplexity"],
        }

    def forward(self, stock_grid: torch.Tensor, market_grid: torch.Tensor,
                market_present: torch.Tensor | None = None) -> dict:
        """Grids all the way to a token. Convenient, but slow — prefer look() + speak()."""
        z_ts, market = self.look(stock_grid, market_grid, market_present)
        return self.speak(z_ts, market)


# The four numbers that ARE a candle. Given yesterday's close they rebuild
# open/high/low/close exactly, which is what lets us draw what the model predicted.
CANDLE = ["gap", "body", "upper_wick", "lower_wick"]


class WorldModel(nn.Module):
    """The tokenizer and the predictor, trained as one.

    A batch is a SENTENCE: one company, over `sentence` consecutive days, each day a
    single word. The GPT reads the sentence and answers TWO questions about tomorrow:

        head A   which WORD comes next          -- what kind of day tomorrow will be
        head B   what CANDLE comes next         -- gap, body, and the two wicks

    **Head B is not decoration. It is what stops the model cheating.**

    Train only for the next word, and the model discovers a shortcut: make every day the
    same word. A constant token is perfectly predictable, so the loss collapses to zero
    while the tokens come to mean nothing. We watched exactly this happen — the codebook
    fell to three words and "accuracy" shot to 87%.

    Predicting the next CANDLE cannot be cheated that way. A collapsed token carries no
    information, so it cannot rebuild a candle, and the loss punishes it. And because a
    candle contains the BODY — where it closed against where it opened — the token is
    forced to carry direction, which is the one thing we found both halves of the
    tokenizer throwing away.

    So the two heads pull in exactly the directions we need: one makes the token
    predictable, the other makes it worth predicting.
    """

    def __init__(self, tokenizer: Tokenizer, sentence: int = 64, depth: int = 4,
                 heads: int = 4, kv_heads: int | None = None, theta: float = 10000.0,
                 dropout: float = 0.0, candle: float = 1.0,
                 naming: float = 0.1):
        super().__init__()
        self.tokenizer = tokenizer
        self.sentence = sentence
        self.words = tokenizer.codebook.words
        # Parameters are named `candle`/`naming` (not `candle_weight`/`naming_weight`) so
        # `WorldModel(..., **settings["loss"])` can splat cleanly, the same way
        # `VQVAE(**settings["ts"])` already does. The attributes below keep their old,
        # more descriptive names -- nothing downstream needs to change because of this.
        self.candle_weight = candle
        # Deliberately small by default. See the `loss` block in settings: this is the
        # weight that rewards cheating, and turning it up collapses the codebook.
        self.naming_weight = naming
        width = tokenizer.width

        self.blocks = nn.ModuleList(
            Block(width, heads, kv_heads, theta, dropout) for _ in range(depth)
        )
        self.settle = RMSNorm(width)
        self.guess = nn.Linear(width, self.words, bias=False)     # head A: the next word
        self.draw = nn.Sequential(                                 # head B: the next candle
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, len(CANDLE)),
        )

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
        """A batch is a SENTENCE: one company, over T consecutive days.

        batch:
            z_ts    [B, T, width]           what that company was doing, each day
            market  [B, T, keys, width]     what the market was offering, each day

        Both come straight from the cache — the frozen encoders already did their work.
        """
        z_ts, market = batch["z_ts"], batch["market"]
        b, t, width = z_ts.shape

        # Every day of the sentence becomes a token. Flatten the sentence into the batch
        # so the fusion sees them all at once.
        flat = self.tokenizer.speak(
            z_ts.reshape(b * t, width),
            market.reshape(b * t, *market.shape[2:]),
        )
        tokens = flat["token"].view(b, t)                       # [B, T]
        vectors = flat["vector"].view(b, t, -1)                 # [B, T, width]

        # Read days 0..T-2, and answer about days 1..T-1. The attention is causal, so no
        # position can see the answer it is being asked for.
        thought = self.understand(vectors[:, :-1])             # [B, T-1, width]

        said = self.guess(thought)                             # [B, T-1, words]
        answer = tokens[:, 1:]                                 # [B, T-1]
        naming = F.cross_entropy(said.reshape(-1, self.words), answer.reshape(-1))
        right = (said.argmax(-1) == answer).float().mean()

        out = {
            "naming_loss": naming,                  # how well it names tomorrow's regime
            "commitment_loss": flat["commitment_loss"],
            "diversity_loss": flat["diversity_loss"],
            "accuracy": right,
            "perplexity": flat["perplexity"],       # is the dictionary still alive?
            "tokens": tokens,                       # [B, T]
            "said": said,                           # [B, T-1, words]
            "fused": vectors,                       # [B, T, width] -- for reviving words
            "attention": flat["attention"].view(b, t, -1),
        }

        drawn = self.draw(thought)                             # [B, T-1, 4]
        out["drawn"] = drawn
        loss = (self.naming_weight * naming
                + flat["commitment_loss"] + flat["diversity_loss"])

        if "candle" in batch:
            wanted = batch["candle"][:, 1:]                    # TOMORROW's candle
            drawing = F.mse_loss(drawn, wanted)
            # What you would score by shrugging and drawing the average candle. The
            # features are normalised to spread 1, so this is ~1.0 -- which makes
            # `drawing` mean something on its own.
            shrugging = wanted.pow(2).mean()
            loss = loss + self.candle_weight * drawing
            out["drawing_loss"] = drawing
            out["shrugging"] = shrugging

        out["loss"] = loss
        return out

    def describe(self) -> str:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return (
            f"frozen encoders ({frozen/1e6:.2f}M) → cross-attention → "
            f"1 token of {self.words} → GPT ({len(self.blocks)} layers)\n"
            f"   training {trainable/1e6:.2f}M weights, "
            f"reading sentences of {self.sentence} days"
        )
