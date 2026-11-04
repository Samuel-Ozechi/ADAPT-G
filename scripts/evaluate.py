"""Standard evaluation command (AG-005).

Inputs: predictions CSVs with columns TransactionID, y_true, y_score,
optionally y_uncertainty. One file per fold: train/val/test.

Usage:
  python scripts/evaluate.py \
    --config configs/base.yaml \
    --train-preds runs/<id>/train_preds.csv \
    --val-preds   runs/<id>/val_preds.csv \
    --test-preds  runs/<id>/test_preds.csv \
    --model-name  tabular_baseline
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from adapt_g.config import load_config
from adapt_g.evaluation.report import build_report, write_report
from adapt_g.logging_utils import get_logger
from adapt_g.manifest import new_run_id, write_manifest
from adapt_g.seeding import set_seed


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    df = pd.read_csv(path)
    y = df["y_true"].to_numpy()
    s = df["y_score"].to_numpy()
    u = df["y_uncertainty"].to_numpy() if "y_uncertainty" in df.columns else None
    return y, s, u


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", action="append", default=None,
        help="YAML config to load; repeat to deep-merge configs in order",
    )
    ap.add_argument("--extra-config", action="append", default=[])
    ap.add_argument("--train-preds", required=True)
    ap.add_argument("--val-preds", required=True)
    ap.add_argument("--test-preds", required=True)
    ap.add_argument("--model-name", default="model")
    ap.add_argument("--threshold-rule", default="max_f1")
    args = ap.parse_args()

    config_paths = args.config or ["configs/base.yaml"]
    config_paths.extend(args.extra_config)
    cfg = load_config(*config_paths)
    set_seed(int(cfg["project"]["seed"]))
    run_id = new_run_id()
    run_dir = Path(cfg["paths"]["runs"]) / run_id
    logger = get_logger("adapt_g.evaluate", run_dir / "run.log",
                        cfg["logging"]["level"], cfg["logging"]["json"])

    y_tr, s_tr, _ = _load(Path(args.train_preds))
    y_va, s_va, _ = _load(Path(args.val_preds))
    y_te, s_te, u_te = _load(Path(args.test_preds))

    report = build_report(
        y_train=y_tr, s_train=s_tr,
        y_val=y_va, s_val=s_va,
        y_test=y_te, s_test=s_te,
        threshold_rule=args.threshold_rule,
        uncertainty_test=u_te,
        model_name=args.model_name,
    )
    json_path, md_path = write_report(report, run_dir)
    write_manifest(
        run_dir=run_dir, cfg=cfg,
        dataset_version=None,
        split_boundaries=None,
        extra={"ticket": "AG-005", "model": args.model_name,
               "threshold_rule": args.threshold_rule},
    )
    logger.info(f"evaluation complete: {json_path}", extra={"run_id": run_id})
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()