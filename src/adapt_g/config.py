"""Configuration loading with deep merge and hashing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(*paths: str | Path) -> dict[str, Any]:
    """Load and deep-merge YAML configs in order."""
    merged: dict[str, Any] = {}
    for p in paths:
        with open(p, "r") as f:
            merged = _deep_merge(merged, yaml.safe_load(f) or {})
    return merged


def config_hash(cfg: dict[str, Any]) -> str:
    """Deterministic hash of a config dict (for run manifests)."""
    payload = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]