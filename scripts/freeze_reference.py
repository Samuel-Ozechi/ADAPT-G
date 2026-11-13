"""Freeze a component run as a Tier-2 reference for AG-010.

Usage:
  python scripts/freeze_reference.py \
    --component baseline \
    --run-id baseline-2026-11-13-142530-a3f9c1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


COMPONENT_TO_TAG = {
    "baseline": "baseline",
    "transformer": "transformer",
    "gnn": "gnn",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", required=True, choices=list(COMPONENT_TO_TAG))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", default="configs/frozen_references.json")
    args = ap.parse_args()

    run_dir = Path("runs") / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    if manifest.get("synthetic_fixture", False):
        raise SystemExit(
            f"refusing to freeze synthetic run {args.run_id}; "
            "rerun on real data"
        )
    if manifest.get("tag") != COMPONENT_TO_TAG[args.component]:
        raise SystemExit(
            f"tag mismatch: run has tag {manifest.get('tag')!r}, "
            f"expected {COMPONENT_TO_TAG[args.component]!r}"
        )

    out = Path(args.out)
    data = json.loads(out.read_text()) if out.exists() else {}
    data.setdefault(args.component, {})["run_id"] = args.run_id
    data[args.component]["tag"] = manifest["tag"]
    data[args.component]["ticket"] = manifest.get("extra", {}).get("ticket")
    data[args.component]["preds_dir"] = str(run_dir / "preds")
    data[args.component]["evaluation_dir"] = str(run_dir / "evaluation")

    if args.component == "baseline":
        data[args.component]["config"] = str(run_dir / "frozen_reference.json")
    elif args.component == "transformer":
        data[args.component]["config"] = str(run_dir / "frozen_temporal_reference.json")
    else:
        data[args.component]["config"] = str(run_dir / "frozen_gnn_reference.json")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    print(f"froze {args.component} reference from run {args.run_id} -> {out}")


if __name__ == "__main__":
    main()