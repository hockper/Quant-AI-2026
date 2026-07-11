import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bubble_bi.config import TrainConfig
from bubble_bi.train.trainer import Trainer, set_seed


class _FrozenHalf(nn.Module):
    def __init__(self, d=6):
        super().__init__()
        self.frozen = nn.Linear(d, d)
        self.trainable = nn.Linear(d, d)
        self.dead_code_reinit_every = 10 ** 9
        for p in self.frozen.parameters():
            p.requires_grad = False

    def forward(self, batch):
        x = batch["block"]
        mse = ((self.trainable(self.frozen(x)) - x) ** 2).mean()
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


def test_trainer_excludes_frozen_params(tmp_path):
    set_seed(0)
    model = _FrozenHalf()
    ld = DataLoader(_DS(), batch_size=8, shuffle=True, drop_last=True)
    cfg = TrainConfig(max_steps=2, batch_size=8, val_every=2, ckpt_every=2,
                      device="cpu", amp=False)
    tr = Trainer(model, {"train": ld, "val": ld, "test": ld}, cfg, str(tmp_path), device="cpu")
    opt_ids = {id(p) for g in tr.opt.param_groups for p in g["params"]}
    assert id(model.frozen.weight) not in opt_ids       # frozen excluded from optimizer
    assert id(model.trainable.weight) in opt_ids
    before = model.frozen.weight.clone()
    tr.train()
    assert torch.allclose(before, model.frozen.weight)   # and stays unchanged
