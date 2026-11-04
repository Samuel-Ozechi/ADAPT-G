"""Integration test: driver writes the expected artifacts on a synthetic run."""
import json
import subprocess
import sys
from pathlib import Path

from scripts.train_tabular import _load_cli_config


def test_repeated_config_arguments_are_deep_merged(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("project: {name: adapt-g, seed: 42}\npaths: {runs: runs}\n")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("project: {seed: 0}\n")

    cfg = _load_cli_config([
        "--config", str(base), "--config", str(overlay),
    ])

    assert cfg == {
        "project": {"name": "adapt-g", "seed": 0},
        "paths": {"runs": "runs"},
    }


def test_driver_produces_frozen_reference(tmp_path):
    # Point runs dir to tmp; raw data absent -> synthetic fixture used.
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""
project: {{name: adapt-g, seed: 0}}
paths: {{data_raw: {tmp_path}/raw, runs: {tmp_path}/runs}}
logging: {{level: INFO, json: true}}
data:
  transaction_file: nope.csv
  identity_file: nope_id.csv
  id_column: TransactionID
  target_column: isFraud
  time_column: TransactionDT
split: {{train_end_q: 0.8, val_end_q: 0.9}}
""")
    out = subprocess.run(
        [sys.executable, "scripts/train_tabular.py", "--config", str(cfg)],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1
    run_dir = runs[0]
    assert (run_dir / "baseline_table.json").exists()
    assert (run_dir / "frozen_reference.json").exists()
    table = json.loads((run_dir / "baseline_table.json").read_text())
    names = {r["model"] for r in table}
    assert names == {"majority", "rule", "logreg", "lightgbm"}
    for r in table:
        assert "pr_auc" in r and "precision" in r and "n_parameters" in r