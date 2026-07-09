from bubble_bi.config import SplitConfig
from bubble_bi.data.splits import walk_forward_splits


def test_splits_are_ordered_and_non_overlapping():
    cfg = SplitConfig(train_days=100, val_days=20, test_days=20, step_days=20)
    splits = walk_forward_splits(200, cfg)
    assert len(splits) >= 1
    for s in splits:
        assert s.train[0] < s.train[1] == s.val[0] < s.val[1] == s.test[0] < s.test[1]
        assert s.test[1] <= 200


def test_windows_advance_by_step():
    cfg = SplitConfig(train_days=100, val_days=20, test_days=20, step_days=20)
    splits = walk_forward_splits(260, cfg)
    assert splits[1].train[0] - splits[0].train[0] == 20


def test_no_split_when_insufficient_history():
    cfg = SplitConfig(train_days=100, val_days=20, test_days=20, step_days=20)
    assert walk_forward_splits(50, cfg) == []
