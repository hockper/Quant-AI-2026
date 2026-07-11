import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bubble_bi.config import TrainConfig
from bubble_bi.train.trainer import Trainer, set_seed, resolve_device


class _M(nn.Module):
    def __init__(self, d=6):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.dead_code_reinit_every = 10 ** 9

    def forward(self, batch):
        x = batch["block"]
        mse = ((self.lin(x) - x) ** 2).mean()
        return {"loss": mse, "recon_loss": mse, "perplexity": torch.tensor(1.0)}

    def reinit_dead_codes(self, out):
        pass


class _DS(Dataset):
    def __init__(self, n=32, N=5, L=4, D=6):
        self.b = torch.randn(n, N, L, D)
        self.v = torch.ones(n, N, dtype=torch.bool)

    def __len__(self):
        return self.b.shape[0]

    def __getitem__(self, i):
        return {"block": self.b[i], "valid": self.v[i]}


def _loaders():
    ld = DataLoader(_DS(), batch_size=8, shuffle=True, drop_last=True)
    return {"train": ld, "val": ld, "test": ld}


def test_resolve_device_reexported_from_trainer():
    assert resolve_device("cpu").type == "cpu"


def test_amp_disabled_off_cuda(tmp_path):
    set_seed(0)
    cfg = TrainConfig(max_steps=2, batch_size=8, val_every=2, ckpt_every=2,
                      device="cpu", amp=True)          # amp requested but device is CPU
    tr = Trainer(_M(), _loaders(), cfg, str(tmp_path), device="cpu")
    assert tr.use_amp is False                          # must not enable AMP off CUDA
    assert tr.scaler.is_enabled() is False
    tr.train()                                          # still trains fine
    assert tr.global_step == 2
