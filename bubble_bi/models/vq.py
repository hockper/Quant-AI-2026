from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_codes: int, dim: int, decay: float = 0.99,
                 eps: float = 1e-5, commitment_beta: float = 0.25):
        super().__init__()
        self.K = num_codes
        self.H = dim
        self.decay = decay
        self.eps = eps
        self.beta = commitment_beta
        embed = torch.randn(num_codes, dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z_e: torch.Tensor) -> dict:
        d = (z_e.pow(2).sum(1, keepdim=True)
             - 2 * z_e @ self.embed.t()
             + self.embed.pow(2).sum(1))          # [B, K]
        ids = d.argmin(1)                          # [B]
        z_q = self.embed[ids]                      # [B, H]

        probs = F.softmax(-d, dim=1)
        mean_probs = probs.mean(0)
        diversity = (mean_probs * (mean_probs + 1e-10).log()).sum()  # -entropy

        onehot = F.one_hot(ids, self.K).type_as(z_e)   # [B, K]
        avg = onehot.mean(0)
        perplexity = torch.exp(-(avg * (avg + 1e-10).log()).sum())

        if self.training:
            self._ema_update(z_e.detach(), onehot.detach())

        commit = self.beta * (z_e - z_q.detach()).pow(2).mean()
        z_q_st = z_e + (z_q - z_e).detach()        # straight-through
        return {"z_q": z_q_st, "ids": ids, "commit": commit,
                "diversity": diversity, "perplexity": perplexity}

    @torch.no_grad()
    def _ema_update(self, z_e: torch.Tensor, onehot: torch.Tensor) -> None:
        n = onehot.sum(0)                          # [K]
        self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
        dw = onehot.t() @ z_e                      # [K, H]
        self.embed_avg.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        total = self.cluster_size.sum()
        cluster = ((self.cluster_size + self.eps)
                   / (total + self.K * self.eps) * total)
        self.embed.copy_(self.embed_avg / cluster.unsqueeze(1))

    def orthogonality_loss(self) -> torch.Tensor:
        e = F.normalize(self.embed, dim=1)
        g = e @ e.t()
        eye = torch.eye(self.K, device=e.device)
        return (g - eye).pow(2).mean()

    @torch.no_grad()
    def reset_dead_codes(self, z_e: torch.Tensor) -> int:
        used = self.cluster_size > 1e-2
        dead = ~used
        n_dead = int(dead.sum())
        if n_dead == 0 or z_e.shape[0] == 0:
            return 0
        pick = torch.randint(0, z_e.shape[0], (n_dead,), device=z_e.device)
        self.embed[dead] = z_e[pick]
        self.embed_avg[dead] = z_e[pick]
        self.cluster_size[dead] = 1.0
        return n_dead
