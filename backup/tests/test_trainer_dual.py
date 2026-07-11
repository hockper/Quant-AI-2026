import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bubble_bi.config import TrainConfig
from bubble_bi.train.trainer import Trainer, set_seed


class _DictModel(nn.Module):
    """Minimal dict-batch model to exercise the Trainer independent of any tokenizer."""

    def __init__(self, d=6):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.dead_code_reinit_every = 10 ** 9

    def forward(self, batch):
        x = batch["windows"]
        mse = ((self.lin(x) - x) ** 2).mean()
        return {"loss": mse, "recon_loss": mse, "perplexity": torch.tensor(1.0),
                "ts_recon": mse, "cs_recon": mse}

    def reinit_dead_codes(self, out):
        pass


class _DayDS(Dataset):
    def __init__(self, n=64, N=5, p=4, D=6):
        self.w = torch.randn(n, N, p, D)
        self.v = torch.ones(n, N, dtype=torch.bool)

    def __len__(self):
        return self.w.shape[0]

    def __getitem__(self, i):
        return {"windows": self.w[i], "valid": self.v[i]}


def _loaders():
    ld = DataLoader(_DayDS(), batch_size=8, shuffle=True, drop_last=True)
    return {"train": ld, "val": ld, "test": ld}


def test_trainer_handles_dict_batches(tmp_path):
    set_seed(0)
    cfg = TrainConfig(max_steps=4, batch_size=8, val_every=4, ckpt_every=4,
                      device="cpu", amp=False)
    tr = Trainer(_DictModel(), _loaders(), cfg, str(tmp_path), device="cpu")
    metrics = tr.train()
    assert tr.global_step == 4
    assert "val_mse" in metrics


def test_trainer_dual_resume(tmp_path):
    set_seed(0)
    cfg = TrainConfig(max_steps=3, batch_size=8, val_every=3, ckpt_every=3,
                      device="cpu", amp=False)
    tr = Trainer(_DictModel(), _loaders(), cfg, str(tmp_path), device="cpu")
    tr.train()
    fresh = Trainer(_DictModel(), _loaders(), cfg, str(tmp_path), device="cpu")
    fresh.load_checkpoint(str(tmp_path / "last.pt"))
    assert fresh.global_step == 3
