"""Config argument tests for the temporal training driver."""
from scripts.train_temporal import _load_cli_config


def test_repeated_config_arguments_are_deep_merged(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("project: {name: adapt-g, seed: 42}\npaths: {runs: runs}\n")
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("project: {seed: 0}\n")

    cfg = _load_cli_config([
        "--config", str(base), "--config", str(overlay),
        "--epochs", "5", "--batch-size", "512", "--lr", "1e-3",
    ])

    assert cfg == {
        "project": {"name": "adapt-g", "seed": 0},
        "paths": {"runs": "runs"},
    }