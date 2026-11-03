"""Metric calculation. All functions accept 1D arrays of labels and scores.

Definitions (recorded in docs/evaluation_schema.md):
- PR-AUC: sklearn.metrics.average_precision_score (area under precision-recall).
- ROC-AUC: sklearn.metrics.roc_auc_score.
- Precision/Recall/F1/FPR/FN at a given threshold.
- Brier: mean((p - y)^2).
- Log loss: sklearn.metrics.log_loss.
- ECE: expected calibration error with equal-frequency bins.
- Error-detection AUROC: uncertainty column scored as a detector of |p - y|.
- Risk-coverage: selective accuracy/error at coverage levels.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss, log_loss,
    precision_score, recall_score, f1_score, confusion_matrix,
)


def _safe(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


def ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    if len(np.unique(y_true)) < 2:
        return {"pr_auc": float("nan"), "roc_auc": float("nan"),
                "flag": "single_class_in_fold"}
    if np.allclose(y_score, y_score[0]):
        return {"pr_auc": float("nan"), "roc_auc": float("nan"),
                "flag": "constant_score"}
    return {
        "pr_auc": _safe(average_precision_score, y_true, y_score),
        "roc_auc": _safe(roc_auc_score, y_true, y_score),
        "flag": None,
    }


def operational_metrics(y_true: np.ndarray, y_score: np.ndarray,
                        threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    return {
        "threshold": float(threshold),
        "precision": _safe(precision_score, y_true, y_pred, zero_division=0),
        "recall": _safe(recall_score, y_true, y_pred, zero_division=0),
        "f1": _safe(f1_score, y_true, y_pred, zero_division=0),
        "fpr": float(fpr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def calibration_metrics(y_true: np.ndarray, y_score: np.ndarray,
                        n_bins: int = 10) -> dict:
    """Brier, log loss, ECE, and reliability-diagram data.

    ECE uses equal-frequency bins (quantile-based). The binning method is
    recorded in the output so downstream plots and text use the same
    definition.
    """
    brier = _safe(brier_score_loss, y_true, y_score)
    # Keep original probabilities for Brier/ECE; clipping them changes exact
    # perfect predictions. log_loss handles endpoint probabilities safely.
    ll = _safe(log_loss, y_true, np.clip(y_score, 1e-9, 1 - 1e-9), labels=[0, 1])

    # Equal-frequency binning.
    order = np.argsort(y_score)
    y_sorted = y_score[order]
    y_true_sorted = y_true[order]
    n = len(y_score)
    bins = np.array_split(np.arange(n), n_bins)
    bin_data = []
    ece = 0.0
    for idx in bins:
        if len(idx) == 0:
            continue
        conf = float(y_sorted[idx].mean())
        acc = float(y_true_sorted[idx].mean())
        weight = len(idx) / n
        ece += weight * abs(conf - acc)
        bin_data.append({"n": int(len(idx)), "mean_confidence": conf,
                         "accuracy": acc, "weight": weight})
    return {
        "brier": brier,
        "log_loss": ll,
        "ece": float(ece),
        "n_bins": n_bins,
        "binning_method": "equal_frequency",
        "reliability_diagram": bin_data,
    }


def uncertainty_metrics(y_true: np.ndarray, y_score: np.ndarray,
                        uncertainty: np.ndarray) -> dict:
    """Error-detection AUROC/AUPRC and risk-coverage.

    Error label: |p - y| > 0.5 treated as a "hard" example for detection.
    (Documented heuristic; AG-011 may refine.)
    """
    hard = (np.abs(y_score - y_true) > 0.5).astype(int)
    out = {}
    if len(np.unique(hard)) < 2:
        out["error_auroc"] = float("nan")
        out["error_auprc"] = float("nan")
        out["risk_coverage"] = []
        return out
    from sklearn.metrics import roc_auc_score, average_precision_score
    out["error_auroc"] = _safe(roc_auc_score, hard, uncertainty)
    out["error_auprc"] = _safe(average_precision_score, hard, uncertainty)

    # Risk-coverage: sort by uncertainty ascending, retain top-k%.
    order = np.argsort(uncertainty)
    y_s = y_true[order]
    p_s = y_score[order]
    coverages = [0.5, 0.7, 0.8, 0.9, 0.95]
    rc = []
    for c in coverages:
        k = int(len(y_s) * c)
        if k == 0:
            continue
        preds = (p_s[:k] >= 0.5).astype(int)
        acc = float((preds == y_s[:k]).mean())
        rc.append({"coverage": c, "retained": k, "accuracy": acc})
    out["risk_coverage"] = rc
    return out


def full_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float,
                 uncertainty: np.ndarray | None = None,
                 n_bins: int = 10) -> dict:
    """Aggregate all metric groups into one dict."""
    return {
        "ranking": ranking_metrics(y_true, y_score),
        "operational": operational_metrics(y_true, y_score, threshold),
        "calibration": calibration_metrics(y_true, y_score, n_bins=n_bins),
        "uncertainty": (uncertainty_metrics(y_true, y_score, uncertainty)
                        if uncertainty is not None else None),
    }