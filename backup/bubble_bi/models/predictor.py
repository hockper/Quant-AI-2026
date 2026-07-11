from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bubble_bi.config import ModelConfig
from bubble_bi.models.llama import LlamaBlock, RMSNorm


class NextTokenPredictor(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab: int):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, cfg.d_model)
        self.blocks = nn.ModuleList([LlamaBlock(cfg) for _ in range(cfg.pred_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab, bias=False)
        self.dead_code_reinit_every = 10 ** 9

    def hidden_state(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)

    def forward(self, batch: dict) -> dict:
        tokens, targets = batch["tokens"], batch["targets"]
        logits = self.head(self.hidden_state(tokens))          # [B, W, vocab]
        loss = F.cross_entropy(logits.reshape(-1, self.vocab), targets.reshape(-1))
        with torch.no_grad():
            acc = (logits.argmax(-1) == targets).float().mean()
            ppl = loss.detach().exp()
        return {"loss": loss, "recon_loss": loss.detach(), "perplexity": ppl,
                "accuracy": acc, "logits": logits}

    def reinit_dead_codes(self, out: dict) -> None:
        pass
