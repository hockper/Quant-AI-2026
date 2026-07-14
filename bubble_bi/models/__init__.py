"""The models.

    Codebook    the dictionary of market words
    VQVAE       squeezes a grid of market data into ONE of those words
    Fusion      where a company's story meets the market's (cross-attention)
    Tokenizer   TS + CS + fusion -> TWO tokens per company-day, each anchored by its
                own reconstruction -- there is no third, fused codebook
    WorldModel  the tokenizer and a Llama-style GPT, trained TOGETHER

The tokenizer's two entries are the same VQVAE class, configured differently:

    ts = VQVAE(companies=1,  days=4, features=26, ...)   what THIS stock is doing
    cs = VQVAE(companies=30, days=5, features=26, ...)   what the MARKET is doing

Nothing here is frozen. Every codebook that was anchored by rebuilding its own grid
stayed healthy; the one codebook that was not -- a single fused token, asked only to
be predictable -- collapsed to 10 words out of 512. So TS and CS keep their own
codebooks and decoders, each anchored, and the GPT reads both words at once. See
`bubble_bi/models/world.py` for the severed-gradient guard that makes joint training
of all of this safe.
"""

from bubble_bi.models.codebook import Codebook
from bubble_bi.models.fusion import Fusion
from bubble_bi.models.llama import Attention, Block, RMSNorm, Rotary, SwiGLU
from bubble_bi.models.vqvae import VQVAE
from bubble_bi.models.world import Tokenizer, WorldModel

__all__ = [
    "Codebook", "VQVAE",                       # the two entries
    "Fusion",                                  # where they meet
    "Tokenizer", "WorldModel",                 # the whole thing
    "RMSNorm", "Rotary", "SwiGLU", "Attention", "Block",   # Llama-3 parts
]
