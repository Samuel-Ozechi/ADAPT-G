"""Run manifest creation. Every run writes runs/<run_id>/run_manifest.json."""
from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adapt_g.config import config_hash


VALID_TAGS = {
    "baseline", "transformer", "gnn", "unified",
    "ablation", "graph", "smoke",
}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def new_run_id(tag: str) -> str:
    """Return '<tag>-<YYYY-MM-DD>-<HHMMSS>-<short_hash>'.

    Tag must be one of VALID_TAGS; raises ValueError otherwise.
    This keeps run IDs self-describing and consistent.
    """
    if tag not in VALID_TAGS:
        raise ValueError(
            f"unknown tag {tag!r}; expected one of {sorted(VALID_TAGS)}"
        )
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{tag}-{stamp}-{short}"


def write_manifest(
    run_dir: Path,
    cfg: dict[str, Any],
    dataset_version: str | None,
    split_boundaries: dict[str, Any] | None,
    extra: dict[str, Any] | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    extra = extra or {}
    manifest = {
        "run_id": run_dir.name,
        "tag": extra.get("tag", "unknown"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config_hash": config_hash(cfg),
        "config": cfg,
        "dataset_version": dataset_version,
        "split_boundaries": split_boundaries,
        "synthetic_fixture": False,
        "extra": extra,
    }
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path