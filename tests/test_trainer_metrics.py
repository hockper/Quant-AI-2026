import json

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from bubble_bi.config import ModelConfig, TrainConfig
from bubble_bi.models.dual_vqvae import DualVQVAE
from bubble_bi.train.trainer import Trainer, set_seed, _scalars


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


def _model():
    return DualVQVAE(ModelConfig(p=4, d_model=16, codebook_size=16, cs_codebook_size=16,
                                 enc_layers=1, dec_layers=1, fusion_layers=1, heads=2,
                                 ff=32, dropout=0.0), d_in=6, n_stocks=5)


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
    tr = Trainer(_model(), _loaders(), cfg, str(tmp_path / "ck"),
                 run_dir=str(tmp_path / "run"), device="cpu")
    tr.train()
    lines = [json.loads(x) for x in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    train_recs = [r for r in lines if r["phase"] == "train"]
    val_recs = [r for r in lines if r["phase"] == "val"]
    assert len(train_recs) >= 2                    # steps 10, 20
    assert "ts_recon" in train_recs[0] and "cs_recon" in train_recs[0]
    assert len(val_recs) >= 1
    assert (tmp_path / "run" / "metrics.csv").exists()
