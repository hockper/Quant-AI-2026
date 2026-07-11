"""The models.

    Codebook   the dictionary of market words
    VQVAE      squeezes a grid of market data into ONE of those words

The tokenizer's two entries are the same VQVAE class, configured differently:

    ts = VQVAE(companies=1,  days=4, features=22, ...)   what THIS stock is doing
    cs = VQVAE(companies=30, days=5, features=22, ...)   what the MARKET is doing
"""

from bubble_bi.models.codebook import Codebook
from bubble_bi.models.vqvae import VQVAE

__all__ = ["Codebook", "VQVAE"]
