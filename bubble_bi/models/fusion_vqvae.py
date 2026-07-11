from __future__ import annotations

import torch
import torch.nn as nn

from bubble_bi.config import ModelConfig
from bubble_bi.models.cross_sectional import CSFieldEncoder
from bubble_bi.models.fusion import MarketToStockFusion
from bubble_bi.models.joint_decoder import JointDecoder
from bubble_bi.models.ts_vqvae import TSEncoder
from bubble_bi.models.vq import VectorQuantizerEMA


def _masked_mse(recon: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    err = ((recon - target) ** 2).reshape(recon.shape[0], recon.shape[1], -1).mean(-1)  # [B,N]
    v = valid.float()
    return (err * v).sum() / v.sum().clamp(min=1.0)


class FusionVQVAE(nn.Module):
    def __init__(self, cfg: ModelConfig, d_in: int, n_stocks: int):
        super().__init__()
        self.p = cfg.p
        self.cs_p = cfg.cs_p
        self.use_fusion = bool(cfg.use_fusion)
        self.lambda_div = cfg.lambda_div
        self.dead_code_reinit_every = cfg.dead_code_reinit_every
        self.ts_enc = TSEncoder(cfg, d_in)
        if self.use_fusion:
            self.cs_enc = CSFieldEncoder(cfg, d_in, n_stocks)
            self.fusion = MarketToStockFusion(cfg)
        self.vq = VectorQuantizerEMA(cfg.fusion_codebook_size, cfg.d_model,
                                     cfg.ema_decay, commitment_beta=cfg.beta_commit)
        self.dec = JointDecoder(cfg, d_in, n_stocks)

    @staticmethod
    def _load_prefixed(module: nn.Module, state: dict, prefix: str) -> None:
        sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        module.load_state_dict(sub)

    def load_frozen(self, ts_ckpt: str, cs_ckpt: str) -> None:
        ts = torch.load(ts_ckpt, map_location="cpu", weights_only=False)["model"]
        self._load_prefixed(self.ts_enc, ts, "enc.")
        self.ts_enc.eval()
        for p in self.ts_enc.parameters():
            p.requires_grad = False
        if self.use_fusion:
            cs = torch.load(cs_ckpt, map_location="cpu", weights_only=False)["model"]
            self._load_prefixed(self.cs_enc, cs, "enc.")
            self.cs_enc.eval()
            for p in self.cs_enc.parameters():
                p.requires_grad = False

    def _fused(self, block: torch.Tensor, valid: torch.Tensor):
        B, N, L, D = block.shape
        ts_in = block[:, :, -self.p:, :]                              # [B,N,p,D]
        z_ts = self.ts_enc(ts_in.reshape(B * N, self.p, D)).reshape(B, N, -1)
        if self.use_fusion:
            z_cs = self.cs_enc(block[:, :, -self.cs_p:, :], valid)    # [B,H]
            fused = self.fusion(z_ts, z_cs)
        else:
            fused = z_ts
        return ts_in, fused

    def forward(self, batch: dict) -> dict:
        block, valid = batch["block"], batch["valid"]
        B, N = valid.shape
        ts_in, fused = self._fused(block, valid)
        q = self.vq(fused.reshape(B * N, -1))
        recon = self.dec(q["z_q"].reshape(B, N, -1), valid)          # [B,N,p,D]
        recon_loss = _masked_mse(recon, ts_in, valid)
        loss = recon_loss + q["commit"] + self.lambda_div * q["diversity"]
        return {"recon": recon, "loss": loss, "recon_loss": recon_loss,
                "commit": q["commit"], "diversity": q["diversity"],
                "perplexity": q["perplexity"], "ids": q["ids"].reshape(B, N),
                "z_e": fused.reshape(B * N, -1).detach()}

    @torch.no_grad()
    def encode(self, batch: dict) -> torch.Tensor:
        block, valid = batch["block"], batch["valid"]
        B, N = valid.shape
        _, fused = self._fused(block, valid)
        return self.vq(fused.reshape(B * N, -1))["ids"].reshape(B, N).long()

    def reinit_dead_codes(self, out: dict) -> None:
        self.vq.reset_dead_codes(out["z_e"])
