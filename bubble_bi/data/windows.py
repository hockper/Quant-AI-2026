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
