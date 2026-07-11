from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _read(run_dir: str):
    p = Path(run_dir) / "metrics.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"no metrics history at {p}")
    recs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    train = [r for r in recs if r.get("phase") == "train"]
    val = [r for r in recs if r.get("phase") == "val"]
    return train, val


def _series(recs, key):
    xs = [r["step"] for r in recs if key in r]
    ys = [r[key] for r in recs if key in r]
    return xs, ys


def _line_plot(series, title, xlabel, ylabel, path, markers=False):
    fig, ax = plt.subplots()
    drew = False
    for label, (xs, ys) in series.items():
        if xs:
            ax.plot(xs, ys, marker="o" if markers else None, label=label)
            drew = True
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if drew:
        ax.legend()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_run(run_dir: str) -> list[str]:
    train, val = _read(run_dir)
    out_dir = Path(run_dir) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_keys = ["loss", "ts_recon", "cs_recon", "recon_loss",
                 "ts_commit", "cs_commit", "ts_diversity", "cs_diversity"]
    paths = [
        _line_plot({k: _series(train, k) for k in loss_keys},
                   "training losses", "step", "loss", out_dir / "losses.png"),
        _line_plot({k: _series(train, k) for k in ["ts_perplexity", "cs_perplexity", "perplexity"]},
                   "codebook perplexity", "step", "perplexity", out_dir / "perplexity.png"),
        _line_plot({"val_mse": _series(val, "val_mse"), "train_recon": _series(train, "recon_loss")},
                   "validation", "step", "mse", out_dir / "val.png", markers=True),
    ]
    return paths


def plot_compare(run_dirs: list[str], out_dir: str) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    loss_series, val_series = {}, {}
    for rd in run_dirs:
        train, val = _read(rd)
        name = Path(rd).name
        loss_series[name] = _series(train, "loss")
        val_series[name] = _series(val, "val_mse")
    return [
        _line_plot(loss_series, "total loss (compare)", "step", "loss", out / "compare_loss.png"),
        _line_plot(val_series, "val_mse (compare)", "step", "val_mse", out / "compare_val.png", markers=True),
    ]
