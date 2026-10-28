"""Validation-only threshold selection. Deterministic tie-breaking."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def select_threshold(
    y_val: np.ndarray,
    s_val: np.ndarray,
    rule: str = "max_f1",
    fixed_recall: float = 0.8,
    fixed_fpr: float = 0.05,
) -> dict:
    """Return a dict with chosen threshold and the rule used.

    Rules:
    - max_f1 (default): threshold maximising F1 on validation. Ties -> smallest threshold.
    - recall_at_fpr: smallest threshold achieving recall with FPR <= fixed_fpr,
      then among eligible thresholds pick the one with highest precision.
    - precision_at_recall: highest threshold achieving recall >= fixed_recall,
      then among those pick highest precision.
    """
    thresholds = np.unique(np.round(np.linspace(0, 1, 1001), 6))
    if rule == "max_f1":
        best_t, best_f1 = None, -1.0
        for t in thresholds:
            f1 = f1_score(y_val, (s_val >= t).astype(int), zero_division=0)
            if f1 > best_f1 + 1e-12:
                best_f1, best_t = f1, float(t)
        return {"rule": rule, "threshold": best_t, "val_f1": float(best_f1)}

    if rule == "precision_at_recall":
        best_t, best_prec = None, -1.0
        for t in thresholds:
            preds = (s_val >= t).astype(int)
            rec = recall_score(y_val, preds, zero_division=0)
            if rec >= fixed_recall:
                prec = precision_score(y_val, preds, zero_division=0)
                if prec > best_prec + 1e-12:
                    best_prec, best_t = prec, float(t)
        return {"rule": rule, "threshold": best_t,
                "val_precision_at_recall": float(best_prec),
                "target_recall": fixed_recall}

    if rule == "recall_at_fpr":
        best_t, best_prec = None, -1.0
        for t in thresholds:
            preds = (s_val >= t).astype(int)
            tn = int(((preds == 0) & (y_val == 0)).sum())
            fp = int(((preds == 1) & (y_val == 0)).sum())
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 1.0
            if fpr <= fixed_fpr:
                prec = precision_score(y_val, preds, zero_division=0)
                if prec > best_prec + 1e-12:
                    best_prec, best_t = prec, float(t)
        return {"rule": rule, "threshold": best_t,
                "val_precision_at_fpr": float(best_prec),
                "target_fpr": fixed_fpr}

    raise ValueError(f"Unknown threshold rule: {rule}")