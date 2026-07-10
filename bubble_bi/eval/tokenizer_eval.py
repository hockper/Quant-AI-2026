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
def evaluate_dual(model, loader, device) -> dict:
    model.eval()
    agg = {mod: {"se": 0.0, "base": 0.0, "count": 0.0, "ppl": 0.0, "batches": 0, "used": set()}
           for mod in model.active}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        windows, valid = batch["windows"], batch["valid"]
        vf = valid.float()
        denom = float(vf.sum().clamp(min=1.0))
        ts_tok, cs_tok = model.encode(batch)
        if "ts" in model.active:
            g = agg["ts"]
            g["se"] += float(out["ts_recon"]) * denom
            g["base"] += float(((windows ** 2).mean(dim=(2, 3)) * vf).sum())
            g["count"] += denom
            g["ppl"] += float(out["ts_perplexity"]); g["batches"] += 1
            g["used"].update(ts_tok[valid].tolist())
        if "cs" in model.active:
            g = agg["cs"]
            cs_t = windows[:, :, -1, :]
            g["se"] += float(out["cs_recon"]) * denom
            g["base"] += float(((cs_t ** 2).mean(dim=2) * vf).sum())
            g["count"] += denom
            g["ppl"] += float(out["cs_perplexity"]); g["batches"] += 1
            g["used"].update(cs_tok.tolist())
    result = {}
    if "ts" in model.active:
        g = agg["ts"]
        result.update(ts_recon_mse=g["se"] / max(g["count"], 1),
                      ts_baseline_mse=g["base"] / max(g["count"], 1),
                      ts_perplexity=g["ppl"] / max(g["batches"], 1),
                      ts_codes_used=len(g["used"]) / model.ts_vq.K)
    if "cs" in model.active:
        g = agg["cs"]
        result.update(cs_recon_mse=g["se"] / max(g["count"], 1),
                      cs_baseline_mse=g["base"] / max(g["count"], 1),
                      cs_perplexity=g["ppl"] / max(g["batches"], 1),
                      cs_codes_used=len(g["used"]) / model.cs_vq.K)
    return result
