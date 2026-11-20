import json

from adapt_g.manifest import new_run_id, write_manifest


def test_new_run_id_accepts_evaluation_tag():
    assert new_run_id("evaluation").startswith("evaluation-")


def test_new_run_id_accepts_descriptive_custom_tag():
    assert new_run_id("eval-lightgbm-top5pct").startswith(
        "eval-lightgbm-top5pct-"
    )


def test_new_run_id_rejects_invalid_custom_tag():
    import pytest

    with pytest.raises(ValueError, match="invalid run tag"):
        new_run_id("Eval lightgbm")


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