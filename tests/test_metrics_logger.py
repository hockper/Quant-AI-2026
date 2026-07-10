import json

from bubble_bi.train.metrics_logger import MetricsLogger


def test_log_appends_jsonl_and_records(tmp_path):
    ml = MetricsLogger(str(tmp_path))
    ml.log({"phase": "train", "step": 1, "loss": 0.5})
    ml.log({"phase": "val", "step": 1, "val_mse": 0.7})
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["loss"] == 0.5
    assert len(ml.records) == 2


def test_to_csv_uses_union_of_keys(tmp_path):
    ml = MetricsLogger(str(tmp_path))
    ml.log({"step": 1, "loss": 0.5})
    ml.log({"step": 2, "val_mse": 0.7})
    ml.to_csv()
    header = (tmp_path / "metrics.csv").read_text().splitlines()[0]
    assert set(header.split(",")) == {"loss", "step", "val_mse"}


def test_write_meta(tmp_path):
    ml = MetricsLogger(str(tmp_path))
    ml.write_meta({"active_modules": ["ts", "cs"], "max_steps": 10})
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["max_steps"] == 10
