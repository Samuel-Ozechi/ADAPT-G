"""Run manifest creation. Every run writes runs/<run_id>/run_manifest.json."""
from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adapt_g.config import config_hash


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def write_manifest(
    run_dir: Path,
    cfg: dict[str, Any],
    dataset_version: str | None,
    split_boundaries: dict[str, Any] | None,
    extra: dict[str, Any] | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config_hash": config_hash(cfg),
        "config": cfg,
        "dataset_version": dataset_version,
        "split_boundaries": split_boundaries,
        "extra": extra or {},
    }
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path