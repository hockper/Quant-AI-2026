from __future__ import annotations

import torch
import torch.nn as nn

from bubble_bi.config import ModelConfig
from bubble_bi.models.cross_sectional import CSFieldDecoder, CSFieldEncoder
from bubble_bi.models.vq import VectorQuantizerEMA


def _masked_field_mse(recon: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # recon/target [B, N, P, D] ; valid [B, N]
    err = ((recon - target) ** 2).mean(dim=(2, 3))     # [B, N]
    v = valid.float()
    return (err * v).sum() / v.sum().clamp(min=1.0)


class CSVQVAE(nn.Module):
    def __init__(self, cfg: ModelConfig, d_in: int, n_stocks: int):
        super().__init__()
        self.cs_p = cfg.cs_p
        self.enc = CSFieldEncoder(cfg, d_in, n_stocks)
        self.vq = VectorQuantizerEMA(cfg.cs_codebook_size, cfg.d_model,
                                     cfg.ema_decay, commitment_beta=cfg.beta_commit)
        self.dec = CSFieldDecoder(cfg, d_in, n_stocks)
        self.lambda_div = cfg.lambda_div
        self.dead_code_reinit_every = cfg.dead_code_reinit_every

    def forward(self, batch: dict) -> dict:
        x = batch["block"][:, :, -self.cs_p:, :]       # [B, N, cs_p, D]
        valid = batch["valid"]
        z_e = self.enc(x, valid)                        # [B, H]
        q = self.vq(z_e)
        recon = self.dec(q["z_q"])                      # [B, N, cs_p, D]
        recon_loss = _masked_field_mse(recon, x, valid)
        loss = recon_loss + q["commit"] + self.lambda_div * q["diversity"]
        return {"recon": recon, "loss": loss, "recon_loss": recon_loss,
                "commit": q["commit"], "diversity": q["diversity"],
                "perplexity": q["perplexity"], "ids": q["ids"], "z_e": z_e}

    def reinit_dead_codes(self, out: dict) -> None:
        self.vq.reset_dead_codes(out["z_e"])
