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


@torch.no_grad()
def evaluate_cs(model, loader, device) -> dict:
    model.eval()
    se, base, n, ppl, batches = 0.0, 0.0, 0, 0.0, 0
    used: set[int] = set()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        x = batch["block"][:, :, -model.cs_p:, :]
        valid = batch["valid"].float()
        denom = float(valid.sum().clamp(min=1.0))
        se += float(out["recon_loss"]) * denom
        base += float((x.pow(2).mean(dim=(2, 3)) * valid).sum())
        n += denom
        ppl += float(out["perplexity"]); batches += 1
        used.update(out["ids"].tolist())
    return {"recon_mse": se / max(n, 1), "mean_baseline_mse": base / max(n, 1),
            "perplexity": ppl / max(batches, 1), "codes_used_frac": len(used) / model.vq.K}


@torch.no_grad()
def evaluate_fusion(model, loader, device) -> dict:
    model.eval()
    se, base, n, ppl, batches = 0.0, 0.0, 0, 0.0, 0
    used: set[int] = set()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        ts_in = batch["block"][:, :, -model.p:, :]
        valid = batch["valid"].float()
        denom = float(valid.sum().clamp(min=1.0))
        se += float(out["recon_loss"]) * denom
        base += float((ts_in.pow(2).mean(dim=(2, 3)) * valid).sum())
        n += denom
        ppl += float(out["perplexity"]); batches += 1
        used.update(out["ids"][batch["valid"]].tolist())
    return {"recon_mse": se / max(n, 1), "mean_baseline_mse": base / max(n, 1),
            "perplexity": ppl / max(batches, 1), "codes_used_frac": len(used) / model.vq.K}

