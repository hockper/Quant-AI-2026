from __future__ import annotations

import torch


def _xla():
    """Return torch_xla's xla_model module, or None when unavailable."""
    try:
        import torch_xla.core.xla_model as xm

        return xm
    except Exception:
        return None


def detect_runtime() -> str:
    """Report the active accelerator: "tpu" | "cuda" | "cpu"."""
    xm = _xla()
    if xm is not None:
        try:
            xm.xla_device()
            return "tpu"
        except Exception:
            pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = detect_runtime()
    if name == "tpu":
        xm = _xla()
        if xm is None:
            raise RuntimeError("device 'tpu' requested but torch_xla is not installed")
        return xm.xla_device()
    return torch.device(name)


def is_xla(device) -> bool:
    return getattr(device, "type", None) == "xla"


def optimizer_step(opt, scaler, device) -> None:
    """Step the optimizer the way the active runtime requires."""
    if is_xla(device):
        _xla().optimizer_step(opt)
    elif getattr(device, "type", None) == "cuda" and scaler is not None and scaler.is_enabled():
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()


def mark_step(device) -> None:
    """XLA needs an explicit graph boundary each iteration; a no-op elsewhere."""
    if is_xla(device):
        _xla().mark_step()


def save_state(state: dict, path: str, device) -> None:
    if is_xla(device):
        _xla().save(state, path)
    else:
        torch.save(state, path)
