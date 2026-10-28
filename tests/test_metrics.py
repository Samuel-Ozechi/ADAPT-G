import numpy as np
import pytest

from adapt_g.evaluation.metrics import (
    ranking_metrics, operational_metrics, calibration_metrics,
    uncertainty_metrics, full_metrics,
)
from adapt_g.evaluation.thresholds import select_threshold
from adapt_g.evaluation.variance import bootstrap_ci
from adapt_g.evaluation.report import build_report


def _toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = np.clip(0.5 * y + rng.normal(0.5, 0.2, n), 0, 1)
    return y, s


def test_ranking_metrics_shapes():
    y, s = _toy()
    out = ranking_metrics(y, s)
    assert 0 <= out["pr_auc"] <= 1
    assert 0 <= out["roc_auc"] <= 1
    assert out["flag"] is None


def test_ranking_metrics_single_class():
    y = np.zeros(10, dtype=int); s = np.linspace(0, 1, 10)
    out = ranking_metrics(y, s)
    assert np.isnan(out["pr_auc"]) and out["flag"] == "single_class_in_fold"


def test_operational_metrics_known_values():
    y = np.array([0, 0, 1, 1]); s = np.array([0.1, 0.4, 0.35, 0.8])
    out = operational_metrics(y, s, threshold=0.5)
    assert out["tp"] == 1 and out["fp"] == 0 and out["fn"] == 1 and out["tn"] == 2
    assert out["precision"] == 1.0
    assert out["recall"] == 0.5


def test_calibration_metrics_range():
    y, s = _toy()
    out = calibration_metrics(y, s)
    assert 0 <= out["brier"] <= 1
    assert 0 <= out["ece"] <= 1
    assert len(out["reliability_diagram"]) == 10
    assert out["binning_method"] == "equal_frequency"


def test_calibration_perfect_predictions():
    y = np.array([0, 0, 1, 1]); s = np.array([0, 0, 1, 1])
    out = calibration_metrics(y, s)
    assert out["brier"] == 0.0
    assert out["ece"] == 0.0


def test_threshold_max_f1_deterministic():
    y, s = _toy()
    t1 = select_threshold(y, s, rule="max_f1")
    t2 = select_threshold(y, s, rule="max_f1")
    assert t1["threshold"] == t2["threshold"]
    assert t1["rule"] == "max_f1"


def test_threshold_precision_at_recall():
    y = np.array([0, 0, 1, 1, 1, 0]); s = np.array([0.1, 0.3, 0.6, 0.7, 0.8, 0.9])
    out = select_threshold(y, s, rule="precision_at_recall", fixed_recall=0.66)
    assert out["threshold"] is not None
    assert out["target_recall"] == 0.66


def test_threshold_rejects_unknown_rule():
    with pytest.raises(ValueError):
        select_threshold(np.array([0, 1]), np.array([0.2, 0.8]), rule="nope")


def test_bootstrap_ci_insufficient_n():
    y = np.array([0, 1, 0]); s = np.array([0.1, 0.9, 0.4])
    out = bootstrap_ci(y, s, metric="pr_auc")
    assert out["flag"] == "insufficient_n"


def test_bootstrap_ci_bounds_contain_point():
    y, s = _toy()
    out = bootstrap_ci(y, s, metric="pr_auc", n_boot=200)
    assert out["ci_low"] <= out["point"] <= out["ci_high"]


def test_uncertainty_metrics_are_valid():
    y, s = _toy()
    rng = np.random.default_rng(1)
    u = rng.uniform(0, 1, size=len(y))
    out = uncertainty_metrics(y, s, u)
    assert 0 <= out["error_auroc"] <= 1
    assert len(out["risk_coverage"]) > 0


def test_full_metrics_aggregates():
    y, s = _toy()
    out = full_metrics(y, s, threshold=0.5)
    assert set(out.keys()) == {"ranking", "operational", "calibration", "uncertainty"}
    assert out["uncertainty"] is None


def test_build_report_uses_validation_threshold():
    y, s = _toy(n=300)
    report = build_report(y, s, y, s, y, s, model_name="toy")
    assert report["threshold_selection"]["rule"] == "max_f1"
    assert report["notes"]["threshold_source"] == "validation_only"
    assert report["notes"]["test_scored_once"] is True
    assert "pr_auc" in report["test_bootstrap_ci"]