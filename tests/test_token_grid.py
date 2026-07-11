import numpy as np
import torch

from bubble_bi.config import ModelConfig
from bubble_bi.models.fusion_vqvae import FusionVQVAE
from bubble_bi.data.token_grid import build_token_grid, TokenSeqDataset, build_token_loaders


def _cfg():
    return ModelConfig(p=4, cs_p=3, d_model=16, fusion_codebook_size=16, enc_layers=1,
                       dec_layers=1, fusion_layers=1, heads=2, ff=32, dropout=0.0,
                       pred_window=5)


def test_build_token_grid_shapes_and_invalid():
    torch.manual_seed(0)
    cfg = _cfg()
    model = FusionVQVAE(cfg, d_in=6, n_stocks=4).eval()
    T, N, D, L = 20, 4, 6, 4
    feats = np.random.default_rng(0).normal(size=(T, N, D)).astype(np.float32)
    mask = np.ones((T, N), dtype=bool)
    mask[:3, 0] = False                                # stock 0 invalid on first days
    grid = build_token_grid(model, feats, mask, window_len=L, device="cpu")
    assert grid.shape == (T, N)
    assert grid.dtype == np.int64
    assert (grid[:L - 1] == -1).all()
    assert grid[3, 0] == -1                            # window [0..3] includes invalid day 0
    valid_tokens = grid[grid != -1]
    assert (valid_tokens >= 0).all() and (valid_tokens < cfg.fusion_codebook_size).all()


def test_token_seq_dataset_windows_and_shift():
    grid = np.arange(1, 41).reshape(20, 2).astype(np.int64) % 16   # [T=20, N=2], all valid
    ds = TokenSeqDataset(grid, W=5, day_range=(0, 20))
    item = ds[0]
    assert item["tokens"].shape == (5,) and item["targets"].shape == (5,)
    assert torch.equal(item["targets"][:-1], item["tokens"][1:])


def test_token_seq_dataset_excludes_invalid_windows():
    grid = np.ones((12, 1), dtype=np.int64)
    grid[6, 0] = -1                                    # a gap
    ds = TokenSeqDataset(grid, W=4, day_range=(0, 12))
    for i in range(len(ds)):
        assert (ds[i]["tokens"] != -1).all() and (ds[i]["targets"] != -1).all()


def test_build_token_loaders_splits():
    from bubble_bi.config import Config, DataConfig
    grid = np.tile(np.arange(300, dtype=np.int64).reshape(300, 1) % 16, (1, 3))
    cfg = Config(data=DataConfig(tickers=["A", "B", "C"]), model=_cfg())
    cfg.train.batch_size = 8
    loaders = build_token_loaders(grid, cfg)
    assert set(loaders) == {"train", "val", "test"}
    batch = next(iter(loaders["train"]))
    assert batch["tokens"].shape[1] == cfg.model.pred_window
