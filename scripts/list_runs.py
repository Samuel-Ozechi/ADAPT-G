"""List runs grouped by tag, newest first."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--tag", default=None, help="Filter by tag, e.g. baseline")
    ap.add_argument("--ticket", default=None, help="Filter by ticket, e.g. AG-006")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = []
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        mf = d / "run_manifest.json"
        if not mf.exists():
            continue
        m = json.loads(mf.read_text())
        extra = m.get("extra", {}) or {}
        tag = m.get("tag") or extra.get("tag", "?")
        ticket = extra.get("ticket", "?")
        synthetic = m.get("synthetic_fixture", False)
        if args.tag and tag != args.tag:
            continue
        if args.ticket and ticket != args.ticket:
            continue
        rows.append((d.name, tag, ticket, synthetic))

    if not rows:
        print("no matching runs")
        return
    print(f"{'run_id':<44} {'tag':<12} {'ticket':<10} {'synthetic':<10}")
    print("-" * 78)
    for run_id, tag, ticket, synthetic in rows:
        print(f"{run_id:<44} {tag:<12} {ticket:<10} {str(synthetic):<10}")


if __name__ == "__main__":
    main()