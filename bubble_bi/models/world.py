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

from bubble_bi.models.fusion import Fusion
from bubble_bi.models.llama import RMSNorm, Block
from bubble_bi.models.vqvae import VQVAE


class Tokenizer(nn.Module):
    """Two words a day: what THIS stock did, and what the MARKET did.

    ⚠️ There is no fused codebook here any more, and its absence is the design.

    Every codebook in this project that was anchored by a reconstruction loss stayed
    healthy — TS reached perplexity 157. The one that was not (the fused codebook, asked
    only to draw tomorrow's candle) collapsed to 10 words out of 512, on every loss balance
    we tried. Under joint training that would get worse, not better, because the naming loss
    is then free to pull on the encoders too.

    So TS and CS keep their OWN codebooks and their OWN decoders, each anchored by rebuilding
    its own grid, and the predictor reads both words. It is also far cheaper: the CS grid is
    identical for every company on a day, so it is encoded once per DAY rather than once per
    company-day.

    The cross-attention sits INSIDE the TS path, before the TS codebook quantises:

        ts_token  =  "what this stock did, GIVEN what the market was doing"
        cs_token  =  "what the market did"

    It has to be there. After a 9-bit quantisation the fine detail the attention needs is
    already gone. Reconstruction keeps the token honest; PREDICTION is what pays for the
    market context, because rebuilding the TS grid does not need CS at all.
    """

    def __init__(self, ts: VQVAE, cs: VQVAE, *, model_size: int, depth: int = 2,
                 attend_to: str = "companies", heads: int = 4, dropout: float = 0.1,
                 batch: int | None = None):
        # `batch` belongs to the data loader, not the model. Accepted and ignored purely so
        # the notebook can splat `**settings["fusion"]` in one go -- the same convention
        # VQVAE follows for its own `batch`/`steps`.
        del batch
        super().__init__()
        self.ts, self.cs = ts, cs
        self.attend_to = attend_to
        self.width = ts.read.out_features
        if model_size != self.width:
            raise ValueError(
                f"`model_size` is {model_size} but the TS/CS encoders were built "
                f"{self.width} wide. The cross-attention needs both sides the same width, "
                "so these must agree."
            )
        self.fusion = Fusion(self.width, depth=depth, heads=heads, dropout=dropout)

    def forward(self, ts_grid: torch.Tensor, cs_grid: torch.Tensor,
                cs_present: torch.Tensor | None = None) -> dict:
        # --- CS: an ordinary VQ-VAE pass. Its codebook is anchored by rebuilding its grid.
        # read_grid gives BOTH the summary (for the codebook) and the cells (for the
        # fusion), so the biggest encoder in the model runs exactly once.
        cs_summary, cs_cells = self.cs.read_grid(cs_grid, cs_present)
        cs_chosen = self.cs.codebook(cs_summary)
        cs_recon = _rebuild_loss(self.cs, cs_chosen["snapped"], cs_grid, cs_present)
        market = self.cs.keys_from_cells(cs_cells, cs_present, self.attend_to)

        # --- TS: encode, THEN read the market, THEN quantise.
        ts_summary, _ = self.ts.read_grid(ts_grid)
        fused, attention = self.fusion(ts_summary, market)
        ts_chosen = self.ts.codebook(fused)
        ts_recon = _rebuild_loss(self.ts, ts_chosen["snapped"], ts_grid, None)

        return {
            "ts_token": ts_chosen["ids"],
            "cs_token": cs_chosen["ids"],
            "ts_vector": ts_chosen["snapped"],
            "cs_vector": cs_chosen["snapped"],
            "attention": attention,
            "recon_loss": ts_recon + cs_recon,
            "commitment_loss": ts_chosen["commitment_loss"] + cs_chosen["commitment_loss"],
            "diversity_loss": ts_chosen["diversity_loss"] + cs_chosen["diversity_loss"],
            "ts_perplexity": ts_chosen["perplexity"],
            "cs_perplexity": cs_chosen["perplexity"],
            "ts_summary": fused,          # pre-quantisation, for reviving dead words
            "cs_summary": cs_summary,
        }


def _rebuild_loss(model: VQVAE, snapped: torch.Tensor, grid: torch.Tensor,
                  present: torch.Tensor | None) -> torch.Tensor:
    """Rebuild the grid from the snapped word and score it. THE ANCHOR.

    Only companies that actually traded are scored -- rewarding the model for "rebuilding" a
    company that did not trade would be meaningless.
    """
    error = (model.rebuild(snapped) - grid).pow(2)
    if present is None:
        return error.mean()
    weight = present.unsqueeze(-1).unsqueeze(-1).to(error.dtype)
    return (error * weight).sum() / weight.expand_as(error).sum().clamp(min=1)


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
