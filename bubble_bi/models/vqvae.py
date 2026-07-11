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
    ):
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
        self.codebook = Codebook(words=vocabulary, width=width)

        # --- decoder: one vector -> grid --------------------------------------
        self.decoder = _transformer(width, decoder_depth, heads, dropout)
        self.write = nn.Linear(width, features)

        nn.init.normal_(self.which_company, std=0.02)
        nn.init.normal_(self.how_long_ago, std=0.02)
        nn.init.normal_(self.summary_slot, std=0.02)

    # ------------------------------------------------------------------ encode
    def summarise(self, grid: torch.Tensor, present: torch.Tensor | None = None) -> torch.Tensor:
        """Boil a grid [B, companies, days, features] down to one vector [B, width]."""
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
        return read[:, 0]                                   # the summary slot

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
            "loss": rebuild_loss + chosen["commitment_loss"],
            "rebuild_loss": rebuild_loss,
            "commitment_loss": chosen["commitment_loss"],
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
