from __future__ import annotations

import torch


@torch.no_grad()
def evaluate_tokenizer(model, loader, device) -> dict:
    model.eval()
    se, n, ppl_sum, batches = 0.0, 0, 0.0, 0
    used: set[int] = set()
    baseline_se = 0.0
    for xb in loader:
        xb = xb.to(device)
        out = model(xb)
        se += float(((out["recon"] - xb) ** 2).sum())
        baseline_se += float((xb ** 2).sum())      # mean of standardized feats ≈ 0
        n += xb.numel()
        ppl_sum += float(out["perplexity"])
        batches += 1
        used.update(out["ids"].tolist())
    K = model.vq.K
    return {
        "recon_mse": se / max(n, 1),
        "mean_baseline_mse": baseline_se / max(n, 1),
        "perplexity": ppl_sum / max(batches, 1),
        "codes_used_frac": len(used) / K,
    }
