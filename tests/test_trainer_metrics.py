import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bubble_bi.config import TrainConfig
from bubble_bi.train.trainer import Trainer, set_seed, _scalars


class _DictModel(nn.Module):
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


def test_scalars_skips_non_scalar_tensors():
    out = {"loss": torch.tensor(1.5), "ts_recon": torch.tensor(0.3),
           "ts_z_e": torch.randn(4, 16), "ids": torch.zeros(4, dtype=torch.long)}
    s = _scalars(out)
    assert s["loss"] == pytest.approx(1.5) and s["ts_recon"] == pytest.approx(0.3)
    assert "ts_z_e" not in s              # multi-element tensor skipped
    assert "ids" not in s                 # 1-D vector skipped


def test_trainer_writes_dense_history(tmp_path):
    set_seed(0)
    cfg = TrainConfig(max_steps=25, batch_size=8, val_every=10, ckpt_every=25,
                      log_every=10, device="cpu", amp=False)
    tr = Trainer(_DictModel(), _loaders(), cfg, str(tmp_path / "ck"),
                 run_dir=str(tmp_path / "run"), device="cpu")
    tr.train()
    lines = [json.loads(x) for x in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    train_recs = [r for r in lines if r["phase"] == "train"]
    val_recs = [r for r in lines if r["phase"] == "val"]
    assert len(train_recs) >= 2
    assert "ts_recon" in train_recs[0] and "cs_recon" in train_recs[0]
    assert len(val_recs) >= 1
    assert (tmp_path / "run" / "metrics.csv").exists()
