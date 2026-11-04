"""Validation-only threshold selection. Deterministic tie-breaking."""
from __future__ import annotations

import numpy as np


def select_threshold(
    y_val: np.ndarray,
    s_val: np.ndarray,
    rule: str = "max_f1",
    fixed_recall: float = 0.8,
    fixed_fpr: float = 0.05,
) -> dict:
    """Return a dict with chosen threshold and the rule used.

    Rules:
    - max_f1 (default): threshold maximising validation F1. Ties -> smallest threshold.
    - precision_at_recall: among thresholds meeting the recall target, maximize
      precision; ties use the highest threshold.
    - recall_at_fpr: among thresholds with FPR <= fixed_fpr, maximize precision;
      ties use the highest threshold.
    """
    y_val = np.asarray(y_val)
    s_val = np.asarray(s_val, dtype=float)
    if y_val.ndim != 1 or s_val.ndim != 1 or len(y_val) != len(s_val):
        raise ValueError("Validation labels and scores must be 1D arrays of equal length")
    if len(y_val) == 0:
        raise ValueError("Cannot select a threshold from an empty validation fold")
    if not np.isfinite(s_val).all():
        raise ValueError("Validation scores must all be finite")
    if ((s_val < 0.0) | (s_val > 1.0)).any():
        raise ValueError("Validation scores must be probabilities in [0, 1]")
    if not np.isin(y_val, [0, 1]).all():
        raise ValueError("Validation labels must be binary 0/1")

    # Predictions only change at a score value; include a threshold above the
    # maximum score to represent predicting no positives.
    thresholds = np.unique(np.concatenate((
        np.array([0.0]), s_val, np.array([np.nextafter(s_val.max(), np.inf)]),
    )))
    sorted_scores = np.sort(s_val)[::-1]
    sorted_labels = y_val[np.argsort(-s_val)]
    cumulative_tp = np.cumsum(sorted_labels == 1)
    predicted_count = np.searchsorted(-sorted_scores, -thresholds, side="right")
    tp = np.zeros(len(thresholds), dtype=float)
    has_predictions = predicted_count > 0
    tp[has_predictions] = cumulative_tp[predicted_count[has_predictions] - 1]
    fp = predicted_count - tp
    positives = int(np.sum(y_val == 1))
    negatives = len(y_val) - positives
    precision = np.divide(tp, predicted_count, out=np.zeros_like(tp), where=predicted_count > 0)
    recall = tp / positives if positives else np.zeros_like(tp)
    f1 = np.divide(2 * tp, 2 * tp + fp + (positives - tp),
                   out=np.zeros_like(tp), where=(2 * tp + fp + positives - tp) > 0)
    if rule == "max_f1":
        best_idx = int(np.argmax(f1))  # thresholds are ascending; ties -> smallest
        return {"rule": rule, "threshold": float(thresholds[best_idx]),
                "val_f1": float(f1[best_idx])}

    if rule == "precision_at_recall":
        if not 0.0 <= fixed_recall <= 1.0:
            raise ValueError("fixed_recall must be in [0, 1]")
        eligible = np.flatnonzero(recall >= fixed_recall)
        if not len(eligible):
            raise ValueError(
                f"No validation threshold achieves recall >= {fixed_recall}"
            )
        best_prec = np.max(precision[eligible])
        best_idx = eligible[np.flatnonzero(precision[eligible] == best_prec)[-1]]
        return {"rule": rule, "threshold": float(thresholds[best_idx]),
                "val_precision_at_recall": float(best_prec),
                "target_recall": fixed_recall}

    if rule == "recall_at_fpr":
        if not 0.0 <= fixed_fpr <= 1.0:
            raise ValueError("fixed_fpr must be in [0, 1]")
        if negatives == 0:
            raise ValueError("recall_at_fpr requires at least one negative validation row")
        fpr = fp / negatives
        eligible = np.flatnonzero(fpr <= fixed_fpr)
        best_prec = np.max(precision[eligible])
        best_idx = eligible[np.flatnonzero(precision[eligible] == best_prec)[-1]]
        return {"rule": rule, "threshold": float(thresholds[best_idx]),
                "val_precision_at_fpr": float(best_prec),
                "target_fpr": fixed_fpr}

    raise ValueError(f"Unknown threshold rule: {rule}")