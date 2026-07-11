from __future__ import annotations

import math

import torch


def train_marginal_token(loader) -> int:
    counts: dict[int, int] = {}
    for batch in loader:
        for tok in batch["targets"].reshape(-1).tolist():
            counts[tok] = counts.get(tok, 0) + 1
    return max(counts, key=counts.get) if counts else 0


@torch.no_grad()
def evaluate_predictor(model, loader, device, marginal: int) -> dict:
    model.eval()
    correct, total, ce_sum, batches, base_correct = 0, 0, 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        preds = out["logits"].argmax(-1)
        targets = batch["targets"]
        correct += int((preds == targets).sum())
        base_correct += int((targets == marginal).sum())
        total += targets.numel()
        ce_sum += float(out["loss"]); batches += 1
    return {"accuracy": correct / max(total, 1),
            "baseline_accuracy": base_correct / max(total, 1),
            "perplexity": math.exp(ce_sum / max(batches, 1))}


@torch.no_grad()
def rollout_accuracy(model, loader, device, horizon: int = 5) -> float:
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        targets = batch["targets"].to(device)
        W = tokens.shape[1]
        h = min(horizon, W - 1)
        seq = tokens[:, :W - h].clone()                      # context prefix
        for step in range(h):
            nxt = model({"tokens": seq, "targets": seq})["logits"][:, -1].argmax(-1)
            seq = torch.cat([seq, nxt[:, None]], dim=1)
            correct += int((nxt == targets[:, W - h - 1 + step]).sum())
            total += nxt.numel()
        break                                                # one batch is enough for a sanity check
    return correct / max(total, 1)
