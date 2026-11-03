def test_majority_baseline_predicts_majority_class():
    # Imbalanced: majority is 0.
    X = pd.DataFrame({"a": [0.0] * 80 + [1.0] * 20})
    y = pd.Series([0] * 80 + [1] * 20)
    m = MajorityBaseline().fit(X, y)
    s = m.predict_score(X)
    assert m.majority_class == 0
    assert np.allclose(s, 0.0)

    # Balanced-toward-positive: majority is 1.
    y2 = pd.Series([0] * 30 + [1] * 70)
    m2 = MajorityBaseline().fit(X, y2)
    s2 = m2.predict_score(X)
    assert m2.majority_class == 1
    assert np.allclose(s2, 1.0)


def test_majority_baseline_flagged_by_harness():
    from adapt_g.evaluation.metrics import ranking_metrics
    y = np.array([0] * 90 + [1] * 10)
    s = np.zeros_like(y, dtype=float)  # constant score
    out = ranking_metrics(y, s)
    assert out["flag"] == "constant_score"
    assert np.isnan(out["pr_auc"])