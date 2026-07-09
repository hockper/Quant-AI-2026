import numpy as np
import torch
from torch.utils.data import DataLoader

from bubble_bi.config import ModelConfig, TrainConfig
from bubble_bi.models.ts_vqvae import TSVQVAE
from bubble_bi.train.trainer import Trainer, set_seed


def _loaders(p=4, d=5, n=64, bs=16):
    x = torch.randn(n, p, d)

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            return x[i]

    loader = DataLoader(_DS(), batch_size=bs, shuffle=True, drop_last=True)
    return {"train": loader, "val": loader, "test": loader}


def _model():
    return TSVQVAE(ModelConfig(p=4, d_model=16, codebook_size=16, enc_layers=1,
                               dec_layers=1, heads=2, ff=32, dropout=0.0), d_in=5)


def test_trainer_runs_and_reports_metrics(tmp_path):
    set_seed(0)
    cfg = TrainConfig(max_steps=5, batch_size=16, val_every=5, ckpt_every=5,
                      device="cpu", amp=False)
    tr = Trainer(_model(), _loaders(), cfg, str(tmp_path), device="cpu")
    metrics = tr.train()
    assert tr.global_step == 5
    assert np.isfinite(metrics["val_mse"])


def test_checkpoint_resume_restores_state(tmp_path):
    set_seed(0)
    cfg = TrainConfig(max_steps=3, batch_size=16, val_every=3, ckpt_every=3,
                      device="cpu", amp=False)
    tr = Trainer(_model(), _loaders(), cfg, str(tmp_path), device="cpu")
    tr.train()
    ckpt = tmp_path / "last.pt"
    assert ckpt.exists()

    fresh = Trainer(_model(), _loaders(), cfg, str(tmp_path), device="cpu")
    fresh.load_checkpoint(str(ckpt))
    assert fresh.global_step == 3
    a = dict(tr.model.named_parameters())
    for name, param in fresh.model.named_parameters():
        assert torch.allclose(param, a[name])
