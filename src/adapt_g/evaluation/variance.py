"""Bootstrap confidence intervals for headline metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


def _metric_fn(name: str, y: np.ndarray, s: np.ndarray) -> float:
    if name == "pr_auc":
        return float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    if name == "roc_auc":
        return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    if name == "brier":
        return float(brier_score_loss(y, s))
    raise ValueError(name)


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray,
                 metric: str = "pr_auc", n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> dict:
    if len(y_true) < 30:
        return {"metric": metric, "ci_low": None, "ci_high": None,
                "point": _metric_fn(metric, y_true, y_score),
                "flag": "insufficient_n"}
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(_metric_fn(metric, y_true[idx], y_score[idx]))
    if not vals:
        return {"metric": metric, "ci_low": None, "ci_high": None,
                "point": _metric_fn(metric, y_true, y_score),
                "flag": "no_valid_bootstraps"}
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"metric": metric, "point": _metric_fn(metric, y_true, y_score),
            "ci_low": float(lo), "ci_high": float(hi),
            "n_boot": len(vals), "alpha": alpha, "flag": None}