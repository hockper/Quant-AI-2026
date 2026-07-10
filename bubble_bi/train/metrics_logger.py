from __future__ import annotations

import csv
import json
from pathlib import Path


class MetricsLogger:
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.records: list[dict] = []

    def log(self, record: dict) -> None:
        self.records.append(record)
        with open(self.jsonl_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    def to_csv(self) -> None:
        if not self.records:
            return
        cols = sorted({k for r in self.records for k in r})
        with open(self.run_dir / "metrics.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for r in self.records:
                writer.writerow(r)

    def write_meta(self, meta: dict) -> None:
        with open(self.run_dir / "meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)
