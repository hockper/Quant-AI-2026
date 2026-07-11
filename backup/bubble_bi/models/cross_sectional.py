from __future__ import annotations

import torch
import torch.nn as nn

from bubble_bi.config import ModelConfig
from bubble_bi.models.ts_vqvae import _encoder_stack


class CSFieldEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, d_in: int, n_stocks: int):
        super().__init__()
        self.embed = nn.Linear(d_in, cfg.d_model)
        self.stock_id = nn.Parameter(torch.zeros(1, n_stocks, 1, cfg.d_model))
        self.day_pos = nn.Parameter(torch.zeros(1, 1, cfg.cs_p, cfg.d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.enc = _encoder_stack(cfg, cfg.cs_enc_layers)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        B, N, P, D = x.shape
        h = self.embed(x) + self.stock_id + self.day_pos     # [B, N, P, H]
        h = h.reshape(B, N * P, -1)                          # [B, N*P, H]
        cls = self.cls.expand(B, 1, -1)
        h = torch.cat([cls, h], dim=1)                       # [B, 1+N*P, H]
        pad_stock = (~valid).unsqueeze(-1).expand(B, N, P).reshape(B, N * P)
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        pad = torch.cat([cls_pad, pad_stock], dim=1)
        h = self.enc(h, src_key_padding_mask=pad)
        return h[:, 0]                                        # [B, H]


class CSFieldDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig, d_out: int, n_stocks: int):
        super().__init__()
        self.n_stocks = n_stocks
        self.cs_p = cfg.cs_p
        self.stock_id = nn.Parameter(torch.zeros(1, n_stocks, 1, cfg.d_model))
        self.day_pos = nn.Parameter(torch.zeros(1, 1, cfg.cs_p, cfg.d_model))
        self.dec = _encoder_stack(cfg, cfg.cs_dec_layers)
        self.out = nn.Linear(cfg.d_model, d_out)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        B = z_q.shape[0]
        N, P = self.n_stocks, self.cs_p
        q = z_q.view(B, 1, 1, -1) + self.stock_id + self.day_pos   # [B, N, P, H]
        h = self.dec(q.reshape(B, N * P, -1)).reshape(B, N, P, -1)
        return self.out(h)                                          # [B, N, P, D]
