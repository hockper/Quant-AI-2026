"""The models.

    Codebook    the dictionary of market words
    VQVAE       squeezes a grid of market data into ONE of those words
    Fusion      where a company's story meets the market's (cross-attention)
    Tokenizer   frozen TS + frozen CS + fusion + ONE codebook -> one token per company-day
    WorldModel  the tokenizer and a Llama-style GPT, trained together

The tokenizer's two entries are the same VQVAE class, configured differently:

    ts = VQVAE(companies=1,  days=4, features=26, ...)   what THIS stock is doing
    cs = VQVAE(companies=30, days=5, features=26, ...)   what the MARKET is doing

They are trained separately, then FROZEN. Their codebooks and decoders were only ever
scaffolding: we keep the encoders and throw the rest away.
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
