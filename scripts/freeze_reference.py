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
COMPONENT_TO_PREDS_MODEL = {
    "baseline": "lightgbm",
    "transformer": "temporal",
    "gnn": "gnn",
}
COMPONENT_TO_EVAL_MODELS = {
    "baseline": {"lightgbm", "lightgbm-top5pct"},
    "transformer": {"temporal", "transformer", "transformer-top5pct"},
    "gnn": {"gnn", "gnn-top5pct"},
}


def find_top5pct_evaluation(component: str, runs_dir: Path = Path("runs")) -> str | None:
    """Return the newest matching 5% top-k evaluation report directory, if any."""
    expected_models = COMPONENT_TO_EVAL_MODELS[component]
    candidates: list[tuple[str, Path]] = []
    for candidate_dir in runs_dir.iterdir():
        manifest_path = candidate_dir / "run_manifest.json"
        report_path = candidate_dir / "evaluation_report.json"
        if not candidate_dir.is_dir() or not manifest_path.is_file() or not report_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        extra = manifest.get("extra", {}) or {}
        selection = report.get("threshold_selection", {}) or {}
        model_name = str(extra.get("model", "")).lower().replace("_", "-")
        alert_rate = selection.get("alert_rate")
        if (
            model_name in expected_models
            and extra.get("threshold_rule") == "top_k_alert_rate"
            and selection.get("rule") == "top_k_alert_rate"
            and isinstance(alert_rate, (int, float))
            and abs(alert_rate - 0.05) < 1e-9
        ):
            candidates.append((str(manifest.get("created_utc", "")), candidate_dir))
    if not candidates:
        return None
    _, latest_dir = max(candidates, key=lambda item: item[0])
    return str(latest_dir)


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
    expected_tag = COMPONENT_TO_TAG[args.component]
    run_tag = manifest.get("tag")
    if not isinstance(run_tag, str) or run_tag.casefold() != expected_tag.casefold():
        raise SystemExit(
            f"tag mismatch: run has tag {run_tag!r}, "
            f"expected {expected_tag!r}"
        )

    out = Path(args.out)
    data = json.loads(out.read_text()) if out.exists() else {}
    data.setdefault(args.component, {})["run_id"] = args.run_id
    data[args.component]["tag"] = expected_tag
    data[args.component]["ticket"] = manifest.get("extra", {}).get("ticket")
    data[args.component]["preds_dir"] = str(
        run_dir / "preds" / COMPONENT_TO_PREDS_MODEL[args.component]
    )
    data[args.component]["evaluation_dir"] = str(run_dir / "evaluation")
    top5pct_evaluation = find_top5pct_evaluation(args.component)
    if top5pct_evaluation is not None:
        data[args.component]["evaluation_top5pct"] = top5pct_evaluation
    else:
        data[args.component].pop("evaluation_top5pct", None)

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