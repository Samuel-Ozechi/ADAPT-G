import json

from adapt_g.manifest import write_manifest


def test_manifest_fields(tmp_path):
    cfg = {"data": {"transaction_file": "x.csv"}, "project": {"seed": 42}}
    path = write_manifest(
        run_dir=tmp_path / "runs" / "test_run",
        cfg=cfg,
        dataset_version="ieee_cis_v1",
        split_boundaries={"train_max_dt": 100.0, "val_max_dt": 200.0},
        extra={"ticket": "AG-003"},
    )
    data = json.loads(path.read_text())
    for key in ("run_id", "created_utc", "git_commit", "config_hash",
                "config", "dataset_version", "split_boundaries", "extra"):
        assert key in data
    assert data["extra"]["ticket"] == "AG-003"