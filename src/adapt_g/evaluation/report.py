"""Standard evaluation report: JSON + Markdown summary."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from adapt_g.evaluation.metrics import full_metrics
from adapt_g.evaluation.thresholds import select_threshold
from adapt_g.evaluation.variance import bootstrap_ci


def build_report(
    y_train: np.ndarray, s_train: np.ndarray,
    y_val: np.ndarray, s_val: np.ndarray,
    y_test: np.ndarray, s_test: np.ndarray,
    threshold_rule: str = "max_f1",
    uncertainty_test: np.ndarray | None = None,
    model_name: str = "model",
) -> dict[str, Any]:
    """Select threshold on validation; score validation and test; compute CIs."""
    sel = select_threshold(y_val, s_val, rule=threshold_rule)
    t = sel["threshold"]

    val_metrics = full_metrics(y_val, s_val, t)
    test_metrics = full_metrics(y_test, s_test, t, uncertainty=uncertainty_test)

    # Bootstrap CIs on test for headline metrics.
    cis = {m: bootstrap_ci(y_test, s_test, metric=m)
           for m in ("pr_auc", "roc_auc", "brier")}

    return {
        "model": model_name,
        "threshold_selection": sel,
        "validation": val_metrics,
        "test": test_metrics,
        "test_bootstrap_ci": cis,
        "notes": {
            "threshold_source": "validation_only",
            "test_scored_once": True,
        },
    }


def write_report(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evaluation_report.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))

    md = [f"# Evaluation Report: {report['model']}", ""]
    md.append(f"Threshold rule: `{report['threshold_selection']['rule']}` "
              f"→ t={report['threshold_selection']['threshold']:.4f}")
    md.append("")
    md.append("## Test metrics")
    tm = report["test"]
    md.append(f"- PR-AUC: {tm['ranking']['pr_auc']:.4f} "
              f"(95% CI {report['test_bootstrap_ci']['pr_auc']['ci_low']:.4f}–"
              f"{report['test_bootstrap_ci']['pr_auc']['ci_high']:.4f})")
    md.append(f"- ROC-AUC: {tm['ranking']['roc_auc']:.4f}")
    md.append(f"- Precision: {tm['operational']['precision']:.4f}")
    md.append(f"- Recall: {tm['operational']['recall']:.4f}")
    md.append(f"- F1: {tm['operational']['f1']:.4f}")
    md.append(f"- FPR: {tm['operational']['fpr']:.4f}")
    md.append(f"- Brier: {tm['calibration']['brier']:.4f}")
    md.append(f"- Log loss: {tm['calibration']['log_loss']:.4f}")
    md.append(f"- ECE (equal-frequency, 10 bins): {tm['calibration']['ece']:.4f}")
    if tm.get("uncertainty"):
        md.append(f"- Error-detection AUROC: {tm['uncertainty']['error_auroc']:.4f}")
    md_path = out_dir / "evaluation_report.md"
    md_path.write_text("\n".join(md))
    return json_path, md_path