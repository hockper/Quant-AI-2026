from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bubble_bi.config import Config, load_config
from bubble_bi.data.ingest import YFinanceSource, ingest
from bubble_bi.data.panel import Panel, build_panel, load_panel, save_panel
from bubble_bi.data.splits import walk_forward_splits
from bubble_bi.data.universe import load_universe
from bubble_bi.baselines.ridge import evaluate_baseline
from bubble_bi.data.windows import build_loaders, build_day_loaders
from bubble_bi.eval.tokenizer_eval import evaluate_tokenizer
from bubble_bi.models.ts_vqvae import TSVQVAE
from bubble_bi.train.trainer import Trainer, resolve_device, set_seed
from bubble_bi.viz.plots import plot_run, plot_compare


def run_ingest(cfg: Config) -> dict[str, str]:
    tickers = load_universe(cfg.data)
    return ingest(tickers, YFinanceSource(), cfg.data.raw_dir, cfg.data.start, cfg.data.end)


def build_panel_from_raw(cfg: Config) -> Panel:
    tickers = load_universe(cfg.data)
    per: dict[str, pd.DataFrame] = {}
    for t in tickers:
        path = Path(cfg.data.raw_dir) / f"{t}.parquet"
        if path.exists():
            per[t] = pd.read_parquet(path)
    if not per:
        raise FileNotFoundError(f"no parquet files in {cfg.data.raw_dir}; run ingest first")
    panel = build_panel(per, cfg.data, cfg.features)
    Path(cfg.data.cache_dir).mkdir(parents=True, exist_ok=True)
    save_panel(panel, str(Path(cfg.data.cache_dir) / "panel.npz"))
    return panel


def run_baseline(cfg: Config) -> dict:
    cache = Path(cfg.data.cache_dir) / "panel.npz"
    panel = load_panel(str(cache)) if cache.exists() else build_panel_from_raw(cfg)
    splits = walk_forward_splits(len(panel.dates), cfg.splits)
    result = evaluate_baseline(panel, splits)
    print(f"walk-forward splits: {result['n_splits']}")
    print(f"RankIC:    {result['rank_ic']:.4f}")
    print(f"RankICIR:  {result['rank_icir']:.4f}")
    return result


def _load_or_build_panel(cfg: Config) -> Panel:
    cache = Path(cfg.data.cache_dir) / "panel.npz"
    return load_panel(str(cache)) if cache.exists() else build_panel_from_raw(cfg)


def _run_dir(cfg: Config, run_name: str) -> Path:
    return Path(cfg.data.cache_dir) / "runs" / run_name


def _write_eval_json(cfg: Config, run_name: str, result: dict) -> None:
    import json

    rd = _run_dir(cfg, run_name)
    rd.mkdir(parents=True, exist_ok=True)
    with open(rd / "eval.json", "w") as fh:
        json.dump(result, fh, indent=2)


def train_tokenizer(cfg: Config, run_name: str | None = None) -> dict:
    set_seed(cfg.seed)
    panel = _load_or_build_panel(cfg)
    loaders, std = build_loaders(panel, cfg)
    model = TSVQVAE(cfg.model, d_in=panel.features.shape[2])
    run_name = run_name or f"tsvqvae_{cfg.train.max_steps}"
    ckpt_dir = Path(cfg.data.cache_dir) / "checkpoints"
    trainer = Trainer(model, loaders, cfg.train, str(ckpt_dir), standardizer=std,
                      run_dir=str(_run_dir(cfg, run_name)))
    metrics = trainer.train()
    trainer.logger.write_meta({"model": "tsvqvae", "d_model": cfg.model.d_model,
                               "codebook_size": cfg.model.codebook_size,
                               "n_features": int(panel.features.shape[2]),
                               "max_steps": cfg.train.max_steps,
                               "log_every": cfg.train.log_every, "final": metrics})
    print(f"trained {metrics['step']} steps | recon {metrics['recon']:.4f} "
          f"| val_mse {metrics['val_mse']:.4f} | ppl {metrics['perplexity']:.1f}")
    return metrics


def eval_tokenizer(cfg: Config, run_name: str | None = None) -> dict:
    set_seed(cfg.seed)
    panel = _load_or_build_panel(cfg)
    loaders, std = build_loaders(panel, cfg)
    device = resolve_device(cfg.train.device)
    model = TSVQVAE(cfg.model, d_in=panel.features.shape[2]).to(device)
    ckpt = Path(cfg.data.cache_dir) / "checkpoints" / "last.pt"
    if ckpt.exists():
        import torch

        state = torch.load(str(ckpt), map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    result = evaluate_tokenizer(model, loaders["test"], device)
    print(f"test recon_mse {result['recon_mse']:.4f} "
          f"(baseline {result['mean_baseline_mse']:.4f}) | "
          f"ppl {result['perplexity']:.1f} | codes {result['codes_used_frac']:.2%}")
    if run_name:
        _write_eval_json(cfg, run_name, result)
    return result


def plot_metrics(cfg: Config, run_names: list[str]) -> list[str]:
    dirs = [str(_run_dir(cfg, n)) for n in run_names]
    if len(dirs) == 1:
        paths = plot_run(dirs[0])
    else:
        paths = plot_compare(dirs, str(Path(dirs[0]) / "plots"))
    print("wrote plots:")
    for p in paths:
        print(f"  {p}")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bubble_bi")
    parser.add_argument("command", choices=["ingest", "build-panel", "baseline",
                                            "train-tokenizer", "eval-tokenizer",
                                            "plot-metrics"])
    parser.add_argument("--config", default="configs/m0.yaml")
    parser.add_argument("--run-name", nargs="+", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_name = args.run_name[0] if args.run_name else None
    if args.command == "ingest":
        paths = run_ingest(cfg)
        print(f"ingested {len(paths)} tickers into {cfg.data.raw_dir}")
    elif args.command == "build-panel":
        panel = build_panel_from_raw(cfg)
        print(f"panel: {panel.features.shape} dates={len(panel.dates)}")
    elif args.command == "baseline":
        run_baseline(cfg)
    elif args.command == "train-tokenizer":
        train_tokenizer(cfg, run_name=run_name)
    elif args.command == "eval-tokenizer":
        eval_tokenizer(cfg, run_name=run_name)
    elif args.command == "plot-metrics":
        plot_metrics(cfg, args.run_name or [_default_dual_run(cfg)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
