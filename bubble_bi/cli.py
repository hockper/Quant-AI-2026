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
import numpy as np

from bubble_bi.eval.tokenizer_eval import evaluate_tokenizer, evaluate_cs, evaluate_fusion
from bubble_bi.eval.predictor_eval import evaluate_predictor, rollout_accuracy, train_marginal_token
from bubble_bi.data.token_grid import build_token_grid, build_token_loaders
from bubble_bi.data.windows import Standardizer
from bubble_bi.models.ts_vqvae import TSVQVAE
from bubble_bi.models.cs_vqvae import CSVQVAE
from bubble_bi.models.fusion_vqvae import FusionVQVAE
from bubble_bi.models.predictor import NextTokenPredictor
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


def train_cs(cfg: Config, run_name: str | None = None) -> dict:
    set_seed(cfg.seed)
    panel = _load_or_build_panel(cfg)
    loaders, std = build_day_loaders(panel, cfg, window_len=cfg.model.cs_p)
    model = CSVQVAE(cfg.model, d_in=panel.features.shape[2], n_stocks=len(panel.tickers))
    run_name = run_name or f"cs_{cfg.train.max_steps}"
    ckpt_dir = Path(cfg.data.cache_dir) / "checkpoints_cs"
    trainer = Trainer(model, loaders, cfg.train, str(ckpt_dir), standardizer=std,
                      run_dir=str(_run_dir(cfg, run_name)))
    metrics = trainer.train()
    trainer.logger.write_meta({"model": "cs", "cs_p": cfg.model.cs_p,
                               "cs_codebook_size": cfg.model.cs_codebook_size,
                               "n_features": int(panel.features.shape[2]),
                               "n_stocks": len(panel.tickers),
                               "max_steps": cfg.train.max_steps, "final": metrics})
    print(f"[cs] trained {metrics['step']} steps | recon {metrics['recon']:.4f} "
          f"| val_mse {metrics['val_mse']:.4f} | ppl {metrics['perplexity']:.1f}")
    return metrics


def eval_cs(cfg: Config, run_name: str | None = None) -> dict:
    set_seed(cfg.seed)
    panel = _load_or_build_panel(cfg)
    loaders, std = build_day_loaders(panel, cfg, window_len=cfg.model.cs_p)
    device = resolve_device(cfg.train.device)
    model = CSVQVAE(cfg.model, d_in=panel.features.shape[2],
                    n_stocks=len(panel.tickers)).to(device)
    ckpt = Path(cfg.data.cache_dir) / "checkpoints_cs" / "last.pt"
    if ckpt.exists():
        import torch

        state = torch.load(str(ckpt), map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    result = evaluate_cs(model, loaders["test"], device)
    print(f"[cs] recon {result['recon_mse']:.4f} (baseline {result['mean_baseline_mse']:.4f}) "
          f"| ppl {result['perplexity']:.1f} | codes {result['codes_used_frac']:.2%}")
    if run_name:
        _write_eval_json(cfg, run_name, result)
    return result


def train_fusion(cfg: Config, run_name: str | None = None) -> dict:
    set_seed(cfg.seed)
    panel = _load_or_build_panel(cfg)
    window_len = max(cfg.model.p, cfg.model.cs_p)
    loaders, std = build_day_loaders(panel, cfg, window_len=window_len)
    model = FusionVQVAE(cfg.model, d_in=panel.features.shape[2], n_stocks=len(panel.tickers))
    ts_ckpt = Path(cfg.data.cache_dir) / "checkpoints" / "last.pt"
    cs_ckpt = Path(cfg.data.cache_dir) / "checkpoints_cs" / "last.pt"
    if not ts_ckpt.exists():
        raise FileNotFoundError(f"missing TS checkpoint {ts_ckpt}; run train-tokenizer first")
    if cfg.model.use_fusion and not cs_ckpt.exists():
        raise FileNotFoundError(f"missing CS checkpoint {cs_ckpt}; run train-cs first")
    model.load_frozen(str(ts_ckpt), str(cs_ckpt))
    run_name = run_name or f"fusion_{cfg.train.max_steps}"
    ckpt_dir = Path(cfg.data.cache_dir) / "checkpoints_fusion"
    trainer = Trainer(model, loaders, cfg.train, str(ckpt_dir), standardizer=std,
                      run_dir=str(_run_dir(cfg, run_name)))
    metrics = trainer.train()
    trainer.logger.write_meta({"model": "fusion", "p": cfg.model.p, "cs_p": cfg.model.cs_p,
                               "fusion_codebook_size": cfg.model.fusion_codebook_size,
                               "use_fusion": cfg.model.use_fusion,
                               "n_features": int(panel.features.shape[2]),
                               "n_stocks": len(panel.tickers),
                               "max_steps": cfg.train.max_steps, "final": metrics})
    print(f"[fusion] trained {metrics['step']} steps | recon {metrics['recon']:.4f} "
          f"| val_mse {metrics['val_mse']:.4f} | ppl {metrics['perplexity']:.1f}")
    return metrics


def eval_fusion(cfg: Config, run_name: str | None = None) -> dict:
    set_seed(cfg.seed)
    panel = _load_or_build_panel(cfg)
    window_len = max(cfg.model.p, cfg.model.cs_p)
    loaders, std = build_day_loaders(panel, cfg, window_len=window_len)
    device = resolve_device(cfg.train.device)
    model = FusionVQVAE(cfg.model, d_in=panel.features.shape[2],
                        n_stocks=len(panel.tickers)).to(device)
    ckpt = Path(cfg.data.cache_dir) / "checkpoints_fusion" / "last.pt"
    if ckpt.exists():
        import torch

        state = torch.load(str(ckpt), map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    result = evaluate_fusion(model, loaders["test"], device)
    print(f"[fusion] recon {result['recon_mse']:.4f} (baseline {result['mean_baseline_mse']:.4f}) "
          f"| ppl {result['perplexity']:.1f} | codes {result['codes_used_frac']:.2%}")
    if run_name:
        _write_eval_json(cfg, run_name, result)
    return result


def _load_frozen_fusion(cfg: Config, device):
    import torch

    panel = _load_or_build_panel(cfg)
    ckpt = Path(cfg.data.cache_dir) / "checkpoints_fusion" / "last.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing fusion tokenizer {ckpt}; run train-fusion first")
    state = torch.load(str(ckpt), map_location=device, weights_only=False)
    model = FusionVQVAE(cfg.model, d_in=panel.features.shape[2], n_stocks=len(panel.tickers))
    model.load_state_dict(state["model"])
    std = Standardizer()
    std.load_state_dict(state["standardizer"])
    return panel, model, std


def tokenize_panel(cfg: Config) -> np.ndarray:
    device = resolve_device(cfg.train.device)
    panel, model, std = _load_frozen_fusion(cfg, device)
    window_len = max(cfg.model.p, cfg.model.cs_p)
    grid = build_token_grid(model, std.transform(panel.features), panel.mask, window_len, device)
    out = Path(cfg.data.cache_dir) / "tokens.npz"
    np.savez_compressed(out, grid=grid)
    print(f"token grid {grid.shape} -> {out}  (valid {(grid != -1).mean():.1%})")
    return grid


def _load_or_build_token_grid(cfg: Config) -> np.ndarray:
    p = Path(cfg.data.cache_dir) / "tokens.npz"
    if p.exists():
        return np.load(str(p))["grid"]
    return tokenize_panel(cfg)


def train_predictor(cfg: Config, run_name: str | None = None) -> dict:
    set_seed(cfg.seed)
    grid = _load_or_build_token_grid(cfg)
    loaders = build_token_loaders(grid, cfg)
    model = NextTokenPredictor(cfg.model, vocab=cfg.model.fusion_codebook_size)
    run_name = run_name or f"pred_{cfg.train.max_steps}"
    ckpt_dir = Path(cfg.data.cache_dir) / "checkpoints_predictor"
    trainer = Trainer(model, loaders, cfg.train, str(ckpt_dir),
                      run_dir=str(_run_dir(cfg, run_name)))
    metrics = trainer.train()
    trainer.logger.write_meta({"model": "predictor", "pred_window": cfg.model.pred_window,
                               "pred_layers": cfg.model.pred_layers,
                               "vocab": cfg.model.fusion_codebook_size,
                               "max_steps": cfg.train.max_steps, "final": metrics})
    print(f"[pred] trained {metrics['step']} steps | ce {metrics['recon']:.4f} "
          f"| val {metrics['val_mse']:.4f} | ppl {metrics['perplexity']:.1f}")
    return metrics


def eval_predictor(cfg: Config, run_name: str | None = None) -> dict:
    set_seed(cfg.seed)
    grid = _load_or_build_token_grid(cfg)
    loaders = build_token_loaders(grid, cfg)
    device = resolve_device(cfg.train.device)
    model = NextTokenPredictor(cfg.model, vocab=cfg.model.fusion_codebook_size).to(device)
    ckpt = Path(cfg.data.cache_dir) / "checkpoints_predictor" / "last.pt"
    if ckpt.exists():
        import torch

        state = torch.load(str(ckpt), map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    marginal = train_marginal_token(loaders["train"])
    result = evaluate_predictor(model, loaders["test"], device, marginal)
    result["rollout_accuracy"] = rollout_accuracy(model, loaders["test"], device)
    print(f"[pred] acc {result['accuracy']:.2%} (baseline {result['baseline_accuracy']:.2%}) "
          f"| ppl {result['perplexity']:.1f} | rollout {result['rollout_accuracy']:.2%}")
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
                                            "train-cs", "eval-cs",
                                            "train-fusion", "eval-fusion",
                                            "tokenize", "train-predictor", "eval-predictor",
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
    elif args.command == "train-cs":
        train_cs(cfg, run_name=run_name)
    elif args.command == "eval-cs":
        eval_cs(cfg, run_name=run_name)
    elif args.command == "train-fusion":
        train_fusion(cfg, run_name=run_name)
    elif args.command == "eval-fusion":
        eval_fusion(cfg, run_name=run_name)
    elif args.command == "tokenize":
        tokenize_panel(cfg)
    elif args.command == "train-predictor":
        train_predictor(cfg, run_name=run_name)
    elif args.command == "eval-predictor":
        eval_predictor(cfg, run_name=run_name)
    elif args.command == "plot-metrics":
        plot_metrics(cfg, args.run_name or [_default_dual_run(cfg)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
