import numpy as np

from bubble_bi.data.windows import chronological_split, Standardizer


def test_chronological_split_ranges():
    tr, va, te = chronological_split(100, 0.7, 0.15)
    assert tr == (0, 70) and va == (70, 85) and te == (85, 100)


def test_standardizer_fits_on_train_only_and_is_leakfree():
    T, N, D = 100, 3, 2
    feats = np.ones((T, N, D), dtype=np.float32)
    feats[50:] = 100.0                       # future dates are huge
    mask = np.ones((T, N), dtype=bool)
    std = Standardizer().fit(feats, mask, (0, 50))
    # mean must reflect only the train block (all ones), not the future
    assert np.allclose(std.mean, 1.0)
    z = std.transform(feats)
    assert np.allclose(z[:50], 0.0)          # train block standardizes to 0
    assert (z[50:] > 1.0).all()              # future is far from train mean


def test_standardizer_ignores_masked_rows():
    feats = np.zeros((10, 2, 1), dtype=np.float32)
    feats[:, 1, 0] = 999.0                   # invalid stock
    mask = np.ones((10, 2), dtype=bool)
    mask[:, 1] = False
    std = Standardizer().fit(feats, mask, (0, 10))
    assert np.allclose(std.mean, 0.0)
