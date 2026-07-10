from __future__ import annotations

import torch
import torch.nn as nn

from bubble_bi.config import ModelConfig
from bubble_bi.models.ts_vqvae import _encoder_stack


class CSEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, d_in: int, n_stocks: int):
        super().__init__()
        self.embed = nn.Linear(d_in, cfg.d_model)
        self.stock_id = nn.Parameter(torch.zeros(1, n_stocks, cfg.d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.enc = _encoder_stack(cfg, cfg.enc_layers)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        h = self.embed(x) + self.stock_id             # [B, N, H]
        B = h.shape[0]
        cls = self.cls.expand(B, 1, -1)
        h = torch.cat([cls, h], dim=1)                # [B, N+1, H]
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        pad = torch.cat([cls_pad, ~valid], dim=1)     # True = ignore
        h = self.enc(h, src_key_padding_mask=pad)
        return h[:, 0]                                 # [B, H]


class CSDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig, d_out: int, n_stocks: int):
        super().__init__()
        self.stock_query = nn.Parameter(torch.zeros(1, n_stocks, cfg.d_model))
        self.dec = _encoder_stack(cfg, cfg.dec_layers)
        self.out = nn.Linear(cfg.d_model, d_out)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        h = z_q.unsqueeze(1) + self.stock_query       # [B, N, H]
        return self.out(self.dec(h))                  # [B, N, D]
