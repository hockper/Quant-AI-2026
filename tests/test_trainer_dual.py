import torch
from torch.utils.data import DataLoader, Dataset

from bubble_bi.config import ModelConfig, TrainConfig
from bubble_bi.models.dual_vqvae import DualVQVAE
from bubble_bi.train.trainer import Trainer, set_seed


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


def test_trainer_handles_dict_batches(tmp_path):
    set_seed(0)
    cfg = TrainConfig(max_steps=4, batch_size=8, val_every=4, ckpt_every=4,
                      device="cpu", amp=False)
    tr = Trainer(_model(), _loaders(), cfg, str(tmp_path), device="cpu")
    metrics = tr.train()
    assert tr.global_step == 4
    assert "val_mse" in metrics


def test_trainer_dual_resume(tmp_path):
    set_seed(0)
    cfg = TrainConfig(max_steps=3, batch_size=8, val_every=3, ckpt_every=3,
                      device="cpu", amp=False)
    tr = Trainer(_model(), _loaders(), cfg, str(tmp_path), device="cpu")
    tr.train()
    fresh = Trainer(_model(), _loaders(), cfg, str(tmp_path), device="cpu")
    fresh.load_checkpoint(str(tmp_path / "last.pt"))
    assert fresh.global_step == 3
