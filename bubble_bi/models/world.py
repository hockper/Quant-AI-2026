"""The whole thing, end to end -- and nothing is frozen.

    CS grid  ──► CS encoder ──► CS codebook ──► cs_token
       (30 cos, 1 day)               │
                                     ├──► CS decoder ──► rebuild CS grid   [ANCHOR]
                                     │ (keys, one per company)
                                     ▼
    TS grid  ──► TS encoder ──► cross-attention ──► TS codebook ──► ts_token
       (1 co, 2 days)                                    │
                                                          └──► TS decoder ──► rebuild TS grid [ANCHOR]

               GPT reads the sentence:  (ts,cs)₁ (ts,cs)₂ … (ts,cs)_t
                          │
               ┌──────────┴──────────┐
          head A: next WORD          head B: tomorrow's CANDLE
          (reads a DETACHED copy)    (the real objective -- full gradient)

Two words a day, each from its OWN anchored codebook (`Tokenizer`, below), read by a
GPT (`WorldModel`) that trains at the same time as everything upstream of it: both
encoders, both codebooks, the cross-attention, and itself. One loss, one optimiser,
from step zero.

**This replaces a two-stage design** (train TS/CS → freeze the encoders → train a
fused codebook against "can it be predicted?"). That fused codebook was the only one
the predictor ever saw, and it is the one we watched collapse to 10 words out of 512.
Every codebook that had a reconstruction anchor stayed healthy; the one that did not,
did not. So there is no fused codebook any more: TS and CS each keep their own, each
anchored by rebuilding their own grid, and the predictor reads BOTH words.

**⚠️ The landmine joint training steps on, and the one thing in this file that must
never be undone.** This GPT invents its own vocabulary (the two codebooks) and is then
graded on predicting it:

    naming loss = CrossEntropy( GPT(z₁…z_t),  id(z_{t+1}) )
                                └ the model's ┘ └ ALSO the model's ┘

There are two ways to drive that to zero: learn real market dynamics (hard), or make
every day the same word (trivial). Gradient descent takes the second, and it is a
STABLE fixed point -- once the vocabulary is dead, nothing pulls it back out. We
measured exactly this: 92% next-token accuracy at perplexity 2.2, on three live words.
In NLP this cannot happen, because the tokenizer is external and fixed. Here it is not.

The fix lives in `WorldModel.forward`: the naming head reads a **detached** copy of
the token vectors, so it can make the GPT better at reading the language but can never
rewrite the language to be easier. See the comment on that line before you touch it.
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
    """A GPT that reads the market as a sentence of TWO words a day, and everything
    trains at once: both encoders, both codebooks, the fusion, and this.

    ⚠️ THE ONE THING THAT MUST NEVER BE UNDONE -- see `forward`.

    This model invents its own vocabulary and is then graded on predicting it. That
    means "make every day the same word" scores perfectly, and it is a STABLE fixed
    point: once the vocabulary is dead, nothing pulls it back out. We measured exactly
    that -- 92% next-token accuracy at perplexity 2.2, on a codebook of three live
    words.

    So the naming head reads a DETACHED copy of the tokens. It trains the GPT to read
    the language; it cannot rewrite the language to be easier. The forecast (`draw`)
    keeps its full gradient into the tokenizer -- that is the entire point of training
    jointly -- and reconstruction anchors the codebooks so the words stay apart.

    **A batch is one time-window across ALL companies at once, not one company.** The
    CS grid is identical for every company on a day, so encoding it per company-day
    would run the biggest encoder in the model N times more often than necessary --
    that saving is what makes a two-token design affordable at all. See `forward` for
    exactly how the shapes are handled.
    """

    def __init__(self, tokenizer: Tokenizer, sentence: int = 64, depth: int = 4,
                 heads: int = 4, dropout: float = 0.0,
                 predict: float = 1.0, naming: float = 0.1, recon: float = 1.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.sentence = sentence
        width = tokenizer.width

        # Splat-friendly names (`predict`/`naming`/`recon`, not `*_weight`) so
        # `WorldModel(tokenizer, **settings["loss"])` works the same way
        # `VQVAE(**settings["ts"])` already does.
        self.predict_weight = predict
        # Deliberately small by default. This is the weight that rewards cheating --
        # see the module docstring -- and turning it up collapses the codebook.
        self.naming_weight = naming
        self.recon_weight = recon

        self.ts_words = tokenizer.ts.codebook.words
        self.cs_words = tokenizer.cs.codebook.words

        self.blocks = nn.ModuleList(
            Block(width, heads=heads, dropout=dropout) for _ in range(depth))
        self.settle = RMSNorm(width)
        self.guess_ts = nn.Linear(width, self.ts_words, bias=False)   # head A, this stock
        self.guess_cs = nn.Linear(width, self.cs_words, bias=False)   # head A, the market
        self.draw = nn.Sequential(                                     # head B, the candle
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, len(CANDLE)))

    def understand(self, vectors: torch.Tensor) -> torch.Tensor:
        """Read a sentence of token vectors [batch, T, width] -> what it makes of each day.

        This hidden state is the artifact the RL agent will eventually consume: the
        model's understanding of where a company stands, with the prediction heads
        taken off. The attention inside each `Block` is causal, so a position can never
        see a day that comes after it -- see `test_the_predictor_cannot_see_tomorrow`.
        """
        for block in self.blocks:
            vectors = block(vectors)
        return self.settle(vectors)

    def forward(self, batch: dict) -> dict:
        """One time-window, every company at once.

        batch:
            ts_grid      [B, T, N, 1, ts_days, F]   one grid per company per day
            cs_grid      [B, T, C, cs_days, F]      ONE grid per day (N == C == companies)
            cs_present   [B, T, C]                  which companies traded that day
            candle       [B, T, N, 4]                tomorrow's candle, per company

        `cs_grid` has NO company axis of its own outside the grid: the C inside it IS
        the market, all thirty companies bundled into a single reading. That is what
        lets the CS encoder run only `B*T` times below, instead of `B*T*N` -- the
        saving the whole two-token design depends on.
        """
        b, t, n = batch["ts_grid"].shape[:3]
        tokenizer = self.tokenizer

        # ---- CS: encode the market ONCE per (batch, day) -- never once per company.
        cs_grid = batch["cs_grid"].reshape(b * t, *batch["cs_grid"].shape[2:])
        cs_present = (batch["cs_present"].reshape(b * t, n)
                      if batch.get("cs_present") is not None else None)

        cs_summary, cs_cells = tokenizer.cs.read_grid(cs_grid, cs_present)
        cs_chosen = tokenizer.cs.codebook(cs_summary)
        cs_recon = _rebuild_loss(tokenizer.cs, cs_chosen["snapped"], cs_grid, cs_present)
        # The menu the market offers, ONE PER DAY: [B*T, keys, width].
        market = tokenizer.cs.keys_from_cells(cs_cells, cs_present, tokenizer.attend_to)

        # ---- TS: one grid per company per day -- B*T*N of them.
        ts_grid = batch["ts_grid"].reshape(b * t * n, *batch["ts_grid"].shape[3:])
        ts_summary, _ = tokenizer.ts.read_grid(ts_grid)

        # Every company reading a given day's market reads the SAME menu. Repeating an
        # already-computed tensor costs nothing -- no encoder runs a second time -- and
        # this is the ONLY place N re-enters the computation. Losing this line does not
        # break correctness, only the saving: the CS encoder would silently go back to
        # running B*T*N times instead of B*T, and nothing would say so.
        market = market.repeat_interleave(n, dim=0)                    # [B*T*N, keys, W]

        fused, attention = tokenizer.fusion(ts_summary, market)
        ts_chosen = tokenizer.ts.codebook(fused)
        ts_recon = _rebuild_loss(tokenizer.ts, ts_chosen["snapped"], ts_grid, None)

        recon_loss = ts_recon + cs_recon
        commitment_loss = ts_chosen["commitment_loss"] + cs_chosen["commitment_loss"]
        diversity_loss = ts_chosen["diversity_loss"] + cs_chosen["diversity_loss"]

        # ---- Build the sentence. The GPT reads one sequence per (batch, company): T
        # days, two words each. Every company of a given day gets the SAME cs_token --
        # broadcast here, not learned separately, because the market only spoke once.
        ts_ids = ts_chosen["ids"].view(b, t, n)
        cs_ids = cs_chosen["ids"].view(b, t, 1).expand(b, t, n)
        ts_vectors = ts_chosen["snapped"].view(b, t, n, -1)
        cs_vectors = cs_chosen["snapped"].view(b, t, 1, -1).expand(b, t, n, -1)

        # Sum the two words into one vector per day, so the GPT's sequence length stays
        # T rather than 2T. Then move the company axis next to batch, so each company
        # gets its OWN sentence: [B, T, N, W] -> [B, N, T, W] -> [B*N, T, W].
        vectors = (ts_vectors + cs_vectors).permute(0, 2, 1, 3).reshape(b * n, t, -1)
        ts_ids = ts_ids.permute(0, 2, 1).reshape(b * n, t)
        cs_ids = cs_ids.permute(0, 2, 1).reshape(b * n, t)
        candle = batch["candle"].permute(0, 2, 1, 3).reshape(b * n, t, -1)

        # ── head B: tomorrow's CANDLE. Full gradient into the tokenizer. This is the
        # whole reason for training jointly: the FORECAST shapes the token.
        thought = self.understand(vectors[:, :-1])
        drawn = self.draw(thought)

        # ── head A: tomorrow's WORDS. ⚠️ READS A DETACHED COPY.
        #
        # `vectors.detach()` severs the gradient path from this loss back into the
        # encoders, the fusion and the codebooks. The naming loss can make the GPT
        # better at READING the language it is handed. It can NEVER make the language
        # easier -- which is the shortcut it took last time, all the way down to three
        # live words while "accuracy" read 92%.
        #
        # This costs a second pass through the (small) GPT blocks -- the encoders
        # dominate the compute either way. It buys a vocabulary that survives.
        #
        # If you ever find yourself deleting `.detach()` here to "let naming help the
        # tokenizer learn faster": don't. That is exactly the collapse this file
        # exists to prevent, and the loss curve will look FINE while it happens --
        # perplexity is the number that will tell you, and only if you are watching it.
        blind = self.understand(vectors.detach()[:, :-1])
        said_ts = self.guess_ts(blind)
        said_cs = self.guess_cs(blind)

        next_ts, next_cs = ts_ids[:, 1:], cs_ids[:, 1:]
        naming = (F.cross_entropy(said_ts.reshape(-1, self.ts_words), next_ts.reshape(-1))
                  + F.cross_entropy(said_cs.reshape(-1, self.cs_words), next_cs.reshape(-1)))

        accuracy = ((said_ts.argmax(-1) == next_ts).float().mean()
                    + (said_cs.argmax(-1) == next_cs).float().mean()) / 2
        # THE HONEST BAR: "tomorrow's word is today's word". Regimes are sticky, so
        # this is hard to beat -- and it is the bar, never zero.
        persistence = ((ts_ids[:, :-1] == next_ts).float().mean()
                       + (cs_ids[:, :-1] == next_cs).float().mean()) / 2

        wanted = candle[:, 1:]                              # TOMORROW's candle
        drawing = F.mse_loss(drawn, wanted)
        # What you would score by shrugging and drawing the average candle. The
        # features are normalised to spread 1, so this is ~1.0 -- which is what makes
        # `drawing` mean anything on its own.
        shrugging = wanted.pow(2).mean()

        loss = (self.predict_weight * drawing
                + self.naming_weight * naming
                + self.recon_weight * recon_loss
                + commitment_loss
                + diversity_loss)

        return {
            "loss": loss,
            "drawing_loss": drawing,
            "naming_loss": naming,
            "recon_loss": recon_loss,
            "commitment_loss": commitment_loss,
            "diversity_loss": diversity_loss,
            "shrugging": shrugging,
            "accuracy": accuracy,
            "persistence": persistence,
            "ts_perplexity": ts_chosen["perplexity"],
            "cs_perplexity": cs_chosen["perplexity"],
            "ts_summary": fused,             # pre-quantisation, for reviving dead words
            "cs_summary": cs_summary,         # pre-quantisation, for reviving dead words
            "attention": attention.view(b, t, n, -1),
        }

    def describe(self) -> str:
        return (
            f"A GPT reading {self.sentence} days, two words each -- "
            f"{self.ts_words} for the stock, {self.cs_words} for the market. "
            "Everything trains at once."
        )
