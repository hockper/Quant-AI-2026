from __future__ import annotations

import torch
import torch.nn as nn

from bubble_bi.config import ModelConfig


class CrossAttentionFusion(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        H, heads, L = cfg.d_model, cfg.heads, cfg.fusion_layers

        def mk():
            return nn.MultiheadAttention(H, heads, dropout=cfg.dropout, batch_first=True)

        self.ts_attn = nn.ModuleList([mk() for _ in range(L)])
        self.cs_attn = nn.ModuleList([mk() for _ in range(L)])
        self.ts_norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(L)])
        self.cs_norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(L)])

    def forward(self, z_ts: torch.Tensor, z_cs: torch.Tensor, valid: torch.Tensor):
        cs = z_cs.unsqueeze(1)                        # [B, 1, H]
        pad = ~valid                                  # [B, N] True = ignore
        for ts_attn, cs_attn, tn, cn in zip(self.ts_attn, self.cs_attn,
                                            self.ts_norm, self.cs_norm):
            a, _ = ts_attn(z_ts, cs, cs)              # stock <- market
            z_ts = tn(z_ts + a)
            b, _ = cs_attn(cs, z_ts, z_ts, key_padding_mask=pad)   # market <- stocks
            cs = cn(cs + b)
        return z_ts, cs.squeeze(1)
