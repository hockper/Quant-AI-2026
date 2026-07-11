from __future__ import annotations

import torch
import torch.nn as nn

from bubble_bi.config import ModelConfig
from bubble_bi.models.ts_vqvae import _encoder_stack


class JointDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig, d_out: int, n_stocks: int):
        super().__init__()
        self.p = cfg.p
        self.stock_id = nn.Parameter(torch.zeros(1, n_stocks, cfg.d_model))
        self.cross = _encoder_stack(cfg, cfg.dec_layers)          # across stocks
        self.day_pos = nn.Parameter(torch.zeros(1, cfg.p, cfg.d_model))
        self.temporal = _encoder_stack(cfg, cfg.dec_layers)       # over days, per stock
        self.out = nn.Linear(cfg.d_model, d_out)

    def forward(self, z_q: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        B, N, H = z_q.shape
        h = z_q + self.stock_id
        h = self.cross(h, src_key_padding_mask=~valid)            # [B, N, H]
        h = h.reshape(B * N, 1, H) + self.day_pos                 # [B*N, p, H]
        h = self.temporal(h)                                      # [B*N, p, H]
        return self.out(h).reshape(B, N, self.p, -1)              # [B, N, p, D]
