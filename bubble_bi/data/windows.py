from __future__ import annotations

import numpy as np


def chronological_split(T: int, train_frac: float, val_frac: float):
    a = int(T * train_frac)
    b = int(T * (train_frac + val_frac))
    return (0, a), (a, b), (b, T)


class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, features: np.ndarray, mask: np.ndarray, day_range) -> "Standardizer":
        lo, hi = day_range
        X = features[lo:hi]              # [t, N, D]
        m = mask[lo:hi]                  # [t, N]
        flat = X[m]                      # [valid, D]
        self.mean = flat.mean(axis=0).astype(np.float32)
        self.std = (flat.std(axis=0) + 1e-8).astype(np.float32)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.std).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"]
        self.std = d["std"]


import torch
from torch.utils.data import DataLoader, Dataset


class WindowDataset(Dataset):
    def __init__(self, std_features: np.ndarray, mask: np.ndarray, p: int, day_range):
        self.X = std_features
        self.p = p
        lo, hi = day_range
        index = []
        for t in range(max(lo, p - 1), hi):
            for j in range(mask.shape[1]):
                if mask[t - p + 1:t + 1, j].all():
                    index.append((t, j))
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> torch.Tensor:
        t, j = self.index[i]
        w = self.X[t - self.p + 1:t + 1, j, :]
        return torch.from_numpy(np.ascontiguousarray(w)).float()


def build_loaders(panel, cfg):
    T = len(panel.dates)
    tr, va, te = chronological_split(T, cfg.data.train_frac, cfg.data.val_frac)
    std = Standardizer().fit(panel.features, panel.mask, tr)
    Xs = std.transform(panel.features)
    p = cfg.model.p
    bs = cfg.train.batch_size
    nw = cfg.train.num_workers
    loaders = {
        "train": DataLoader(WindowDataset(Xs, panel.mask, p, tr), batch_size=bs,
                            shuffle=True, num_workers=nw, drop_last=True),
        "val": DataLoader(WindowDataset(Xs, panel.mask, p, va), batch_size=bs,
                          shuffle=False, num_workers=nw),
        "test": DataLoader(WindowDataset(Xs, panel.mask, p, te), batch_size=bs,
                           shuffle=False, num_workers=nw),
    }
    return loaders, std
