from __future__ import annotations

import torch
import torch.nn as nn

from bubble_bi.config import ModelConfig
from bubble_bi.models.cross_sectional import CSDecoder, CSEncoder
from bubble_bi.models.fusion import CrossAttentionFusion
from bubble_bi.models.ts_vqvae import TSDecoder, TSEncoder
from bubble_bi.models.vq import VectorQuantizerEMA


def _masked_mse(recon: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    err = (recon - target) ** 2
    err = err.reshape(err.shape[0], err.shape[1], -1).mean(-1)   # [B, N]
    v = valid.float()
    return (err * v).sum() / v.sum().clamp(min=1.0)


class DualVQVAE(nn.Module):
    def __init__(self, cfg: ModelConfig, d_in: int, n_stocks: int):
        super().__init__()
        self.active = list(cfg.active_modules)
        self.lambda_div = cfg.lambda_div
        self.dead_code_reinit_every = cfg.dead_code_reinit_every
        self.use_fusion = bool(cfg.use_fusion) and ("ts" in self.active and "cs" in self.active)
        if "ts" in self.active:
            self.ts_enc = TSEncoder(cfg, d_in)
            self.ts_vq = VectorQuantizerEMA(cfg.codebook_size, cfg.d_model,
                                            cfg.ema_decay, commitment_beta=cfg.beta_commit)
            self.ts_dec = TSDecoder(cfg, d_in)
        if "cs" in self.active:
            self.cs_enc = CSEncoder(cfg, d_in, n_stocks)
            self.cs_vq = VectorQuantizerEMA(cfg.cs_codebook_size, cfg.d_model,
                                            cfg.ema_decay, commitment_beta=cfg.beta_commit)
            self.cs_dec = CSDecoder(cfg, d_in, n_stocks)
        if self.use_fusion:
            self.fusion = CrossAttentionFusion(cfg)

    def _encode_latents(self, windows, valid):
        B, N, p, D = windows.shape
        z_ts = z_cs = None
        if "ts" in self.active:
            z_ts = self.ts_enc(windows.reshape(B * N, p, D)).reshape(B, N, -1)
        if "cs" in self.active:
            z_cs = self.cs_enc(windows[:, :, -1, :], valid)
        if self.use_fusion:
            z_ts, z_cs = self.fusion(z_ts, z_cs, valid)
        return z_ts, z_cs

    def forward(self, batch: dict) -> dict:
        windows, valid = batch["windows"], batch["valid"]
        B, N, p, D = windows.shape
        z_ts, z_cs = self._encode_latents(windows, valid)
        loss = windows.new_zeros(())
        out: dict = {}
        if "ts" in self.active:
            q = self.ts_vq(z_ts.reshape(B * N, -1))
            recon = self.ts_dec(q["z_q"]).reshape(B, N, p, D)
            ts_recon = _masked_mse(recon, windows, valid)
            loss = loss + ts_recon + q["commit"] + self.lambda_div * q["diversity"]
            out.update(ts_recon=ts_recon, ts_perplexity=q["perplexity"],
                       ts_commit=q["commit"], ts_diversity=q["diversity"],
                       ts_z_e=z_ts.reshape(B * N, -1).detach())
        if "cs" in self.active:
            cs_target = windows[:, :, -1, :]
            q = self.cs_vq(z_cs)
            recon = self.cs_dec(q["z_q"])
            cs_recon = _masked_mse(recon, cs_target, valid)
            loss = loss + cs_recon + q["commit"] + self.lambda_div * q["diversity"]
            out.update(cs_recon=cs_recon, cs_perplexity=q["perplexity"],
                       cs_commit=q["commit"], cs_diversity=q["diversity"],
                       cs_z_e=z_cs.detach())
        out["loss"] = loss
        out["recon_loss"] = out.get("ts_recon", out.get("cs_recon"))
        out["perplexity"] = out.get("ts_perplexity", out.get("cs_perplexity"))
        return out

    @torch.no_grad()
    def encode(self, batch: dict):
        windows, valid = batch["windows"], batch["valid"]
        B, N, p, D = windows.shape
        z_ts, z_cs = self._encode_latents(windows, valid)
        ts_tok = cs_tok = None
        if "ts" in self.active:
            ts_tok = self.ts_vq(z_ts.reshape(B * N, -1))["ids"].reshape(B, N).long()
        if "cs" in self.active:
            cs_tok = self.cs_vq(z_cs)["ids"].long()
        return ts_tok, cs_tok

    def reinit_dead_codes(self, out: dict) -> None:
        if "ts" in self.active and "ts_z_e" in out:
            self.ts_vq.reset_dead_codes(out["ts_z_e"])
        if "cs" in self.active and "cs_z_e" in out:
            self.cs_vq.reset_dead_codes(out["cs_z_e"])
