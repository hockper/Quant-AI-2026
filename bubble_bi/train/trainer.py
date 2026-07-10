from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from bubble_bi.train.metrics_logger import MetricsLogger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _dead_every(model) -> int:
    return getattr(model, "dead_code_reinit_every", 250)


def _to_device(batch, device):
    if isinstance(batch, dict):
        return {k: v.to(device) for k, v in batch.items()}
    return batch.to(device)


def _batch_size(batch) -> int:
    if isinstance(batch, dict):
        return next(iter(batch.values())).shape[0]
    return batch.shape[0]


def _scalars(out: dict) -> dict:
    result = {}
    for k, v in out.items():
        if isinstance(v, (int, float)):
            result[k] = float(v)
        elif torch.is_tensor(v) and v.ndim == 0:
            result[k] = float(v.detach())
    return result


class Trainer:
    def __init__(self, model, loaders, cfg, ckpt_dir, standardizer=None, device=None, run_dir=None):
        self.cfg = cfg
        self.device = torch.device(device) if device else resolve_device(cfg.device)
        self.model = model.to(self.device)
        self.loaders = loaders
        self.standardizer = standardizer
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        self.use_amp = bool(cfg.amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
        self.global_step = 0
        self.best_val = float("inf")
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.logger = MetricsLogger(run_dir if run_dir is not None else ckpt_dir)

    def train(self) -> dict:
        cfg = self.cfg
        model = self.model
        summary: dict = {}
        model.train()
        while self.global_step < cfg.max_steps:
            for xb in self.loaders["train"]:
                if self.global_step >= cfg.max_steps:
                    break
                xb = _to_device(xb, self.device)
                self.opt.zero_grad()
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    out = model(xb)
                    loss = out["loss"]
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.global_step += 1

                if self.global_step % _dead_every(model) == 0:
                    if hasattr(model, "reinit_dead_codes"):
                        model.reinit_dead_codes(out)
                    else:
                        model.vq.reset_dead_codes(out["z_e"].detach())

                scal = _scalars(out)
                summary = {"step": self.global_step,
                           "recon": scal.get("recon_loss", scal.get("loss", float("nan"))),
                           "perplexity": scal.get("perplexity", float("nan"))}
                if self.global_step % cfg.log_every == 0:
                    self.logger.log({"phase": "train", "step": self.global_step, **scal})
                if self.global_step % cfg.val_every == 0:
                    val = self.evaluate("val")
                    self.best_val = min(self.best_val, val)
                    summary["val_mse"] = val
                    self.logger.log({"phase": "val", "step": self.global_step, "val_mse": val})
                    model.train()
                if self.global_step % cfg.ckpt_every == 0:
                    self.save_checkpoint(str(self.ckpt_dir / "last.pt"))
        if "val_mse" not in summary:
            summary["val_mse"] = self.evaluate("val")
        self.save_checkpoint(str(self.ckpt_dir / "last.pt"))
        self.logger.to_csv()
        return summary

    @torch.no_grad()
    def evaluate(self, split: str = "val") -> float:
        self.model.eval()
        total, n = 0.0, 0
        for xb in self.loaders[split]:
            xb = _to_device(xb, self.device)
            out = self.model(xb)
            bs = _batch_size(xb)
            total += float(out["recon_loss"]) * bs
            n += bs
        return total / max(n, 1)

    def save_checkpoint(self, path: str) -> None:
        state = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.global_step,
            "best_val": self.best_val,
            "rng_torch": torch.get_rng_state(),
            "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
        }
        if self.standardizer is not None:
            state["standardizer"] = self.standardizer.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["opt"])
        self.scaler.load_state_dict(state["scaler"])
        self.global_step = state["step"]
        self.best_val = state["best_val"]
        torch.set_rng_state(state["rng_torch"])
        np.random.set_state(state["rng_numpy"])
        random.setstate(state["rng_python"])
        if self.standardizer is not None and "standardizer" in state:
            self.standardizer.load_state_dict(state["standardizer"])
