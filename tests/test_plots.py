import json
from pathlib import Path

import pytest

from bubble_bi.viz.plots import plot_run, plot_compare


def _write_history(run_dir):
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    recs = []
    for s in (10, 20, 30):
        recs.append({"phase": "train", "step": s, "loss": 1.0 / s,
                     "ts_recon": 0.5 / s, "cs_recon": 0.9 / s,
                     "ts_perplexity": s, "cs_perplexity": s / 2, "recon_loss": 0.5 / s})
    recs.append({"phase": "val", "step": 20, "val_mse": 0.4})
    with open(Path(run_dir) / "metrics.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


def test_plot_run_writes_pngs(tmp_path):
    run = tmp_path / "r1"
    _write_history(run)
    paths = plot_run(str(run))
    assert len(paths) == 3
    for p in paths:
        assert Path(p).exists() and Path(p).stat().st_size > 0


def test_plot_compare_writes_pngs(tmp_path):
    _write_history(tmp_path / "r1")
    _write_history(tmp_path / "r2")
    paths = plot_compare([str(tmp_path / "r1"), str(tmp_path / "r2")], str(tmp_path / "cmp"))
    assert len(paths) == 2
    for p in paths:
        assert Path(p).exists() and Path(p).stat().st_size > 0


def test_plot_run_missing_history_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        plot_run(str(tmp_path / "nope"))
