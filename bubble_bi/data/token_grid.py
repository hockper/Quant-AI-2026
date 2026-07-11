from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from bubble_bi.data.windows import chronological_split


@torch.no_grad()
def build_token_grid(model, std_features: np.ndarray, mask: np.ndarray,
                     window_len: int, device, batch_days: int = 64) -> np.ndarray:
    model.eval().to(device)
    T, N, D = std_features.shape
    grid = np.full((T, N), -1, dtype=np.int64)
    days = list(range(window_len - 1, T))
    for i in range(0, len(days), batch_days):
        chunk = days[i:i + batch_days]
        blocks = np.zeros((len(chunk), N, window_len, D), dtype=np.float32)
        valids = np.zeros((len(chunk), N), dtype=bool)
        for bi, t in enumerate(chunk):
            for j in range(N):
                if mask[t - window_len + 1:t + 1, j].all():
                    blocks[bi, j] = std_features[t - window_len + 1:t + 1, j, :]
                    valids[bi, j] = True
        batch = {"block": torch.from_numpy(blocks).to(device),
                 "valid": torch.from_numpy(valids).to(device)}
        ids = model.encode(batch).cpu().numpy()        # [len(chunk), N]
        for bi, t in enumerate(chunk):
            grid[t, valids[bi]] = ids[bi, valids[bi]]
    return grid


class TokenSeqDataset(Dataset):
    def __init__(self, grid: np.ndarray, W: int, day_range):
        self.grid = grid
        self.W = W
        lo, hi = day_range
        T, N = grid.shape
        self.samples: list[tuple[int, int]] = []
        for j in range(N):
            col = grid[:, j]
            t = 0
            while t < T:
                if col[t] == -1:
                    t += 1
                    continue
                start = t
                while t < T and col[t] != -1:
                    t += 1
                # windows within [start, t): need W+1 tokens; assign by last target day
                for s in range(start, t - W):
                    if lo <= s + W < hi:
                        self.samples.append((j, s))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        j, s = self.samples[i]
        toks = self.grid[s:s + self.W, j]
        tgts = self.grid[s + 1:s + self.W + 1, j]
        return {"tokens": torch.from_numpy(np.ascontiguousarray(toks)).long(),
                "targets": torch.from_numpy(np.ascontiguousarray(tgts)).long()}


def build_token_loaders(grid: np.ndarray, cfg) -> dict:
    T = grid.shape[0]
    tr, va, te = chronological_split(T, cfg.data.train_frac, cfg.data.val_frac)
    W = cfg.model.pred_window
    bs, nw = cfg.train.batch_size, cfg.train.num_workers

    def mk(rng, shuffle):
        return DataLoader(TokenSeqDataset(grid, W, rng), batch_size=bs,
                          shuffle=shuffle, num_workers=nw, drop_last=shuffle)

    return {"train": mk(tr, True), "val": mk(va, False), "test": mk(te, False)}
