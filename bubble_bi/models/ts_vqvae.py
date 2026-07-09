from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bubble_bi.config import ModelConfig
from bubble_bi.models.vq import VectorQuantizerEMA


def _encoder_layer(cfg: ModelConfig) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=cfg.d_model, nhead=cfg.heads, dim_feedforward=cfg.ff,
        dropout=cfg.dropout, batch_first=True, norm_first=True,
    )


def _encoder_stack(cfg: ModelConfig, layers: int) -> nn.TransformerEncoder:
    # enable_nested_tensor is incompatible with norm_first; disable to avoid a warning.
    return nn.TransformerEncoder(_encoder_layer(cfg), layers, enable_nested_tensor=False)


class TSEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, d_in: int):
        super().__init__()
        self.embed = nn.Linear(d_in, cfg.d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos = nn.Parameter(torch.zeros(1, cfg.p + 1, cfg.d_model))
        self.enc = _encoder_stack(cfg, cfg.enc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)                          # [B, p, H]
        cls = self.cls.expand(h.shape[0], 1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos  # [B, p+1, H]
        return self.enc(h)[:, 0]                    # CLS -> [B, H]


class TSDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig, d_out: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, cfg.p, cfg.d_model))
        self.dec = _encoder_stack(cfg, cfg.dec_layers)
        self.out = nn.Linear(cfg.d_model, d_out)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        h = z_q.unsqueeze(1) + self.pos            # [B, p, H]
        return self.out(self.dec(h))               # [B, p, D]


class TSVQVAE(nn.Module):
    def __init__(self, cfg: ModelConfig, d_in: int):
        super().__init__()
        self.enc = TSEncoder(cfg, d_in)
        self.vq = VectorQuantizerEMA(cfg.codebook_size, cfg.d_model,
                                     cfg.ema_decay, commitment_beta=cfg.beta_commit)
        self.dec = TSDecoder(cfg, d_in)
        self.lambda_div = cfg.lambda_div
        self.lambda_ortho = cfg.lambda_ortho
        self.dead_code_reinit_every = cfg.dead_code_reinit_every

    def forward(self, x: torch.Tensor) -> dict:
        z_e = self.enc(x)
        q = self.vq(z_e)
        recon = self.dec(q["z_q"])
        recon_loss = F.mse_loss(recon, x)
        # Orthogonality is a diagnostic only: the codebook is an EMA buffer with no
        # gradient, so an orthogonality *loss* on it cannot train anything.
        ortho = self.vq.orthogonality_loss()
        loss = (recon_loss + q["commit"]
                + self.lambda_div * q["diversity"])
        return {"recon": recon, "loss": loss, "recon_loss": recon_loss,
                "commit": q["commit"], "diversity": q["diversity"],
                "ortho": ortho, "perplexity": q["perplexity"],
                "ids": q["ids"], "z_e": z_e}

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vq(self.enc(x))["ids"].long()
