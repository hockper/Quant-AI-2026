"""The predictor: a GPT that reads market sentences.

Once every (company, day) is a single word, a company's history is literally a
sentence — and predicting the next word is exactly what a language model does. So we
use one.

The parts here are the ones Llama 3 uses, and each earns its place:

  RMSNorm    a cheaper LayerNorm that only rescales, never re-centres
  RoPE       position by ROTATING the query and key, so the model sees how far apart
             two days are rather than where they sit in the window
  SwiGLU     a gated feed-forward layer that consistently beats a plain one
  GQA        several query heads share one key/value head — much less memory
  causal     day t may look at days <= t and NEVER at t+1. In a language model this
             is a nicety. Here it is the whole no-lookahead rule again, in the model
             instead of the data: without it the predictor would simply read tomorrow.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Normalise by size alone. No mean subtraction, no bias — cheaper, and as good."""

    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.scale * (x * size)


class Rotary(nn.Module):
    """Rotary position embedding: position as a ROTATION of the query and key.

    A dot product between two rotated vectors depends only on the ANGLE between them —
    that is, on how many days apart they are — not on where either sits in the window.
    So the model learns "three days ago" rather than "day 47".
    """

    def __init__(self, head_width: int, theta: float = 10000.0):
        super().__init__()
        if head_width % 2:
            raise ValueError(f"head width must be even for rotation, got {head_width}.")
        angles = 1.0 / (theta ** (torch.arange(0, head_width, 2).float() / head_width))
        self.register_buffer("angles", angles, persistent=False)

    def forward(self, length: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        steps = torch.arange(length, device=device).float()
        turn = torch.outer(steps, self.angles.to(device))     # [length, head_width/2]
        turn = torch.cat([turn, turn], dim=-1)                # [length, head_width]
        return turn.cos()[None, None], turn.sin()[None, None]


def _half_turn(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat([-right, left], dim=-1)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _half_turn(x) * sin


class SwiGLU(nn.Module):
    """A gated feed-forward layer: one branch decides how much of the other gets through."""

    def __init__(self, width: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or int(width * 8 / 3 / 64 + 1) * 64      # Llama's sizing rule
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    """Causal self-attention with rotary positions and grouped key/value heads."""

    def __init__(self, width: int, heads: int, kv_heads: int | None = None,
                 theta: float = 10000.0, dropout: float = 0.0):
        super().__init__()
        if width % heads:
            raise ValueError(f"width ({width}) must divide evenly by heads ({heads}).")
        kv_heads = kv_heads or heads
        if heads % kv_heads:
            raise ValueError(
                f"heads ({heads}) must divide evenly by kv_heads ({kv_heads}) — "
                "each key/value head is shared by a whole group of query heads."
            )

        self.heads, self.kv_heads = heads, kv_heads
        self.head_width = width // heads
        self.share = heads // kv_heads
        self.dropout = dropout

        self.to_q = nn.Linear(width, heads * self.head_width, bias=False)
        self.to_k = nn.Linear(width, kv_heads * self.head_width, bias=False)
        self.to_v = nn.Linear(width, kv_heads * self.head_width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.rotary = Rotary(self.head_width, theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.to_q(x).view(b, t, self.heads, self.head_width).transpose(1, 2)
        k = self.to_k(x).view(b, t, self.kv_heads, self.head_width).transpose(1, 2)
        v = self.to_v(x).view(b, t, self.kv_heads, self.head_width).transpose(1, 2)

        cos, sin = self.rotary(t, x.device)
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)

        if self.share > 1:                          # one k/v head serves several q heads
            k = k.repeat_interleave(self.share, dim=1)
            v = v.repeat_interleave(self.share, dim=1)

        # is_causal: day t sees days <= t and NEVER t+1. This is the no-lookahead rule,
        # enforced inside the model.
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(out.transpose(1, 2).reshape(b, t, -1))


class Block(nn.Module):
    """One Llama block: attend, then think. Normalise before each, add back after."""

    def __init__(self, width: int, heads: int, kv_heads: int | None = None,
                 theta: float = 10000.0, dropout: float = 0.0):
        super().__init__()
        self.before_attending = RMSNorm(width)
        self.attend = Attention(width, heads, kv_heads, theta, dropout)
        self.before_thinking = RMSNorm(width)
        self.think = SwiGLU(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attend(self.before_attending(x))
        return x + self.think(self.before_thinking(x))
