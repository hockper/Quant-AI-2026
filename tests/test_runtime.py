import types

import torch

from bubble_bi import runtime


def test_detect_runtime_is_cpu_here():
    # this dev machine has no CUDA and no torch_xla
    assert runtime.detect_runtime() == "cpu"


def test_resolve_device_cpu_and_auto():
    assert runtime.resolve_device("cpu").type == "cpu"
    assert runtime.resolve_device("auto").type in {"cpu", "cuda"}
    assert runtime.is_xla(torch.device("cpu")) is False


def test_optimizer_step_cpu_calls_opt_step():
    stepped = []
    opt = types.SimpleNamespace(step=lambda: stepped.append(1))
    runtime.optimizer_step(opt, None, torch.device("cpu"))
    assert stepped == [1]


def test_optimizer_step_cuda_uses_scaler():
    calls = []
    scaler = types.SimpleNamespace(is_enabled=lambda: True,
                                   step=lambda o: calls.append("step"),
                                   update=lambda: calls.append("update"))
    runtime.optimizer_step(object(), scaler, torch.device("cuda"))
    assert calls == ["step", "update"]


def test_xla_dispatch_with_stubbed_xm(monkeypatch, tmp_path):
    """Proves the TPU code path without owning a TPU."""
    calls = []
    fake_xm = types.SimpleNamespace(
        optimizer_step=lambda opt: calls.append("optimizer_step"),
        mark_step=lambda: calls.append("mark_step"),
        save=lambda state, path: calls.append("save"),
    )
    monkeypatch.setattr(runtime, "_xla", lambda: fake_xm)
    dev = types.SimpleNamespace(type="xla")          # stand-in XLA device
    assert runtime.is_xla(dev) is True
    runtime.optimizer_step(object(), None, dev)
    runtime.mark_step(dev)
    runtime.save_state({"a": 1}, str(tmp_path / "x.pt"), dev)
    assert calls == ["optimizer_step", "mark_step", "save"]


def test_save_state_cpu_writes_file(tmp_path):
    p = tmp_path / "s.pt"
    runtime.save_state({"a": 1}, str(p), torch.device("cpu"))
    assert p.exists()
    assert torch.load(str(p), weights_only=False)["a"] == 1
