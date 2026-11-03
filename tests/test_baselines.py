import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adapt_g.models.baselines import (
    MajorityBaseline, RuleBaseline, LogRegBaseline, LightGBMBaseline,
)


def _xy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({
        "TransactionAmt": rng.uniform(1, 500, n) + y * 100,
        "ProductCD": rng.choice(["W", "C", "H", "S", "R"], n),
        "card4": rng.choice(["visa", "mastercard"], n),
    })
    return X, pd.Series(y)


def test_majority_baseline():
    X, y = _xy()
    m = MajorityBaseline().fit(X, y)
    s = m.predict_score(X)
    assert len(s) == len(X)
    assert np.allclose(s, y.mean())


def test_rule_baseline_produces_binary_scores():
    X, y = _xy()
    m = RuleBaseline().fit(X, y)
    s = m.predict_score(X)
    assert set(np.unique(s)).issubset({0.0, 1.0})


def test_logreg_predicts_probabilities():
    X, y = _xy()
    m = LogRegBaseline(seed=0).fit(X, y)
    s = m.predict_score(X)
    assert (s >= 0).all() and (s <= 1).all()
    assert m.n_parameters() > 0


def test_lightgbm_tunes_on_validation():
    X, y = _xy(n=400)
    X_tr, y_tr = X.iloc[:300], y.iloc[:300]
    X_va, y_va = X.iloc[300:], y.iloc[300:]
    m = LightGBMBaseline(seed=0, grid_size=3).fit(X_tr, y_tr, X_va, y_va)
    assert m.best_cfg is not None
    assert len(m.tuning_log) == 3
    s = m.predict_score(X_va)
    assert (s >= 0).all() and (s <= 1).all()
    assert m.n_parameters() > 0


def test_lightgbm_falls_back_without_validation():
    X, y = _xy()
    m = LightGBMBaseline(seed=0).fit(X, y)
    assert m.best_cfg is not None
    s = m.predict_score(X)
    assert (s >= 0).all() and (s <= 1).all()