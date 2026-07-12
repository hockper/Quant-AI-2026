"""One machine, used twice.

TS and CS look like different problems. They are not — they are the same problem
at two sizes:

    TS   1 company  × 4 days × 22 features  →  one word   "what THIS stock is doing"
    CS  30 companies × 5 days × 22 features →  one word   "what the MARKET is doing"

TS is simply CS watching a single company. So there is one class, configured two
ways, rather than two classes that quietly drift apart.

Each one works the same way:

    encoder   reads the whole grid and boils it down to a single vector
    codebook  snaps that vector to the nearest word in its dictionary
    decoder   tries to rebuild the entire grid from that one word

Training makes the rebuild as accurate as possible. That is the trick: nobody
tells the machine what the words should mean. It is simply forced to describe a
grid of market data using one word out of 512 — and the only way to do that well
is for the words to come to mean something.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bubble_bi.models.codebook import Codebook


def _transformer(width: int, depth: int, heads: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=width,
        nhead=heads,
        dim_feedforward=width * 2,
        dropout=dropout,
        batch_first=True,
        norm_first=True,
    )
    # enable_nested_tensor is incompatible with norm_first and only warns; off.
    return nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)


class VQVAE(nn.Module):
    """Squeeze a grid of market data into one word from a learned dictionary.

    Both entries of the tokenizer are this class:

        ts = VQVAE(companies=1,  days=4, features=22, ...)
        cs = VQVAE(companies=30, days=5, features=22, ...)

    Input is always a grid: [batch, companies, days, features].
    """

    def __init__(
        self,
        companies: int,
        days: int,
        features: int,
        *,
        vocabulary: int = 512,
        width: int = 128,
        encoder_depth: int = 3,
        decoder_depth: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        commitment: float = 0.25,
        diversity: float = 0.1,
        decay: float = 0.99,
        batch: int | None = None,
        steps: int | None = None,
    ):
        # `batch` and `steps` are not model settings -- they belong to the data loader and
        # the trainer. They are accepted and ignored here purely so the notebook can hand
        # over an entry's whole settings block in one go: VQVAE(..., **settings["ts"]).
        del batch, steps
        super().__init__()
        if companies < 1 or days < 1 or features < 1:
            raise ValueError(
                f"A grid needs at least 1 company, 1 day and 1 feature — "
                f"got companies={companies}, days={days}, features={features}."
            )
        if width % heads:
            raise ValueError(f"width ({width}) must divide evenly by heads ({heads}).")

        self.companies, self.days, self.features = companies, days, features

        # --- encoder: grid -> one vector -------------------------------------
        self.read = nn.Linear(features, width)
        # Learned markers so the transformer can tell cells apart: which company a
        # cell belongs to, and how long ago the day was.
        self.which_company = nn.Parameter(torch.zeros(1, companies, 1, width))
        self.how_long_ago = nn.Parameter(torch.zeros(1, 1, days, width))
        # A blank slot that reads the whole grid and ends up holding the summary.
        self.summary_slot = nn.Parameter(torch.zeros(1, 1, width))
        self.encoder = _transformer(width, encoder_depth, heads, dropout)

        # --- the dictionary ---------------------------------------------------
        self.codebook = Codebook(words=vocabulary, width=width, decay=decay,
                                 commitment=commitment, diversity=diversity)

        # --- decoder: one vector -> grid --------------------------------------
        self.decoder = _transformer(width, decoder_depth, heads, dropout)
        self.write = nn.Linear(width, features)

        nn.init.normal_(self.which_company, std=0.02)
        nn.init.normal_(self.how_long_ago, std=0.02)
        nn.init.normal_(self.summary_slot, std=0.02)

    # ------------------------------------------------------------------ encode
    def read_grid(self, grid: torch.Tensor, present: torch.Tensor | None = None):
        """Encode a grid. Returns (summary, cells).

            summary  [B, width]                     the one vector the codebook quantises
            cells    [B, companies, days, width]    every cell, having seen all the others

        `cells` is what the fusion attends over, once this model's codebook and decoder
        have been thrown away. Pooling it all the way down to `summary` would discard
        exactly the detail cross-attention needs.
        """
        b, c, d, f = grid.shape
        if (c, d, f) != (self.companies, self.days, self.features):
            raise ValueError(
                f"This VQ-VAE expects grids of "
                f"[batch, {self.companies}, {self.days}, {self.features}] "
                f"but was given [batch, {c}, {d}, {f}]."
            )

        cells = self.read(grid) + self.which_company + self.how_long_ago
        cells = cells.reshape(b, c * d, -1)
        cells = torch.cat([self.summary_slot.expand(b, 1, -1), cells], dim=1)

        ignore = None
        if present is not None:
            # A missing company's cells are hidden from the transformer entirely,
            # rather than being fed in as zeros and mistaken for real data.
            missing = (~present).unsqueeze(-1).expand(b, c, d).reshape(b, c * d)
            keep_summary = torch.zeros(b, 1, dtype=torch.bool, device=grid.device)
            ignore = torch.cat([keep_summary, missing], dim=1)

        read = self.encoder(cells, src_key_padding_mask=ignore)
        return read[:, 0], read[:, 1:].reshape(b, c, d, -1)

    def summarise(self, grid: torch.Tensor, present: torch.Tensor | None = None) -> torch.Tensor:
        """Boil a grid [B, companies, days, features] down to one vector [B, width]."""
        return self.read_grid(grid, present)[0]

    def context(self, grid: torch.Tensor, present: torch.Tensor | None = None,
                how: str = "days") -> torch.Tensor:
        """What CS hands the fusion to attend over: [B, keys, width].

        The number of keys is a free choice — cross-attention's output length is set by
        the QUERY, not the keys. The keys are simply the menu the company chooses from,
        and this decides how fine-grained that menu is:

            "days"       one vector per market day        (5 keys) -- what the paper does:
                         its CS patch is "all stocks on a single trading day"
            "companies"  one vector per company           (30 keys) -- lets a bank attend
                         to other banks
            "cells"      every (company, day)             (150 keys) -- richest: a specific
                         peer on a specific day

        A single vector would be a NO-OP: softmax over one key is identically 1.0, so
        every company would receive the same market vector and the attention would learn
        nothing. That is why there is no "one summary" option here.

        Cost is linear in the number of keys (our query is a single vector), so even
        "cells" is cheap. Start with "days"; widen it when tuning.

        Companies that did not trade are left out of the averages, not counted as zeros.
        """
        if how not in ("days", "companies", "cells"):
            raise ValueError(
                f"`attend_to` must be 'days', 'companies' or 'cells' — got {how!r}."
            )

        _, cells = self.read_grid(grid, present)                 # [B, C, D, W]
        b, c, d, w = cells.shape

        if how == "cells":
            return cells.reshape(b, c * d, w)
        if how == "companies":                  # average over DAYS -> one per company
            return cells.mean(dim=2)

        if present is None:                     # "days": average over COMPANIES
            return cells.mean(dim=1)
        weight = present.to(cells.dtype).unsqueeze(-1).unsqueeze(-1)     # [B, C, 1, 1]
        return (cells * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)

    def rebuild(self, word: torch.Tensor) -> torch.Tensor:
        """Reconstruct the whole grid from a single vector [B, width]."""
        b = word.shape[0]
        cells = (
            word.view(b, 1, 1, -1).expand(b, self.companies, self.days, -1)
            + self.which_company
            + self.how_long_ago
        )
        cells = self.decoder(cells.reshape(b, self.companies * self.days, -1))
        cells = cells.reshape(b, self.companies, self.days, -1)
        return self.write(cells)

    def tokenize(self, grid: torch.Tensor, present: torch.Tensor | None = None) -> torch.Tensor:
        """The word for each grid: [B]. This is what the whole model exists to produce."""
        with torch.no_grad():
            return self.codebook(self.summarise(grid, present))["ids"]

    # ----------------------------------------------------------------- forward
    def forward(self, batch: dict) -> dict:
        """Encode → snap to a word → rebuild → score how good the rebuild was.

        batch: {"grid": [B, companies, days, features], "present": [B, companies] bool}
        """
        grid = batch["grid"]
        present = batch.get("present")

        summary = self.summarise(grid, present)
        chosen = self.codebook(summary)
        rebuilt = self.rebuild(chosen["snapped"])

        error = (rebuilt - grid).pow(2)
        if present is not None:
            # Only score companies that were actually there. Rewarding the model for
            # "rebuilding" a company that did not trade would be meaningless.
            weight = present.unsqueeze(-1).unsqueeze(-1).to(error.dtype)
            rebuild_loss = (error * weight).sum() / weight.expand_as(error).sum().clamp(min=1)
        else:
            rebuild_loss = error.mean()

        return {
            # Three pressures, pulling in different directions:
            #   rebuild      "say enough about the grid that I can redraw it"
            #   commitment   "and say it in words that are already in the dictionary"
            #   diversity    "and stop crowding onto the same few words"
            "loss": rebuild_loss + chosen["commitment_loss"] + chosen["diversity_loss"],
            "rebuild_loss": rebuild_loss,
            "commitment_loss": chosen["commitment_loss"],
            "diversity_loss": chosen["diversity_loss"],
            "perplexity": chosen["perplexity"],
            "ids": chosen["ids"],
            "summary": summary,
        }

    def describe(self) -> str:
        """What this VQ-VAE is, in one line — for the notebook."""
        watching = (
            "1 company" if self.companies == 1 else f"all {self.companies} companies"
        )
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"{watching} × {self.days} days × {self.features} features "
            f"→ 1 word out of {self.codebook.words}   ({params/1e6:.2f}M weights)"
        )
