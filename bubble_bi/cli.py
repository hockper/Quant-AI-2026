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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bubble_bi")
    parser.add_argument("command", choices=["ingest", "build-panel", "baseline"])
    parser.add_argument("--config", default="configs/m0.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "ingest":
        paths = run_ingest(cfg)
        print(f"ingested {len(paths)} tickers into {cfg.data.raw_dir}")
    elif args.command == "build-panel":
        panel = build_panel_from_raw(cfg)
        print(f"panel: {panel.features.shape} dates={len(panel.dates)}")
    elif args.command == "baseline":
        run_baseline(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
