from __future__ import annotations

import torch
import torch.nn as nn

from bubble_bi.config import ModelConfig


class MarketToStockFusion(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        H, heads, L = cfg.d_model, cfg.heads, cfg.fusion_layers

        def mk():
            return nn.MultiheadAttention(H, heads, dropout=cfg.dropout, batch_first=True)

        self.attn = nn.ModuleList([mk() for _ in range(L)])
        self.norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(L)])

    def forward(self, z_ts: torch.Tensor, z_cs: torch.Tensor) -> torch.Tensor:
        kv = z_cs.unsqueeze(1)                        # [B, 1, H]  single market key
        h = z_ts
        for attn, norm in zip(self.attn, self.norm):
            a, _ = attn(h, kv, kv)                    # [B, N, H]  each stock reads the market
            h = norm(h + a)                           # residual keeps stocks distinct
        return h
