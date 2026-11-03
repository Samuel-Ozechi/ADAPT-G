"""Majority-class and hand-crafted rule baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd


class MajorityBaseline:
    """Majority-class predictor expressed as a calibrated score.

    Semantics:
    - Determine the majority class c* from the training fold.
    - Predict P(y = 1 | majority-class strategy) as:
        - p_pos if c* == 1  (always predict positive)
        - 1 - p_pos if c* == 0  (always predict negative -> score for positive = 0)
      where p_pos = y.mean() on train.

    This produces a *constant* score, which is the honest expression of
    "no information beyond the class prior." Ranking metrics are
    degenerate by construction; the harness flags this.
    """
    name = "majority"

    def __init__(self):
        self.p_pos = 0.0
        self.majority_class = 0
        self.constant_score = 0.0

    def fit(self, X, y):
        self.p_pos = float(y.mean())
        self.majority_class = 1 if self.p_pos >= 0.5 else 0
        # Score for the "positive" class under always-predict-majority.
        self.constant_score = 1.0 if self.majority_class == 1 else 0.0
        return self

    def predict_score(self, X):
        return np.full(len(X), self.constant_score, dtype=float)

    def n_parameters(self):
        return 1


class RuleBaseline:
    """Decline if TransactionAmt > train 99th percentile AND ProductCD == 'C'.

    Deliberately simple. Documented as an arbitrary hand rule; exists to
    anchor the floor of detection performance and to demonstrate that naive
    rules fail under real class imbalance.
    """

    name = "rule_amt_p99_productc"

    def __init__(self, amount_col: str = "TransactionAmt",
                 product_col: str = "ProductCD", product_value: str = "C") -> None:
        self.amount_col = amount_col
        self.product_col = product_col
        self.product_value = product_value
        self.amount_threshold: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RuleBaseline":
        # Preprocessor has already scaled TransactionAmt. Use the raw column if
        # available; otherwise use the 99th percentile of the scaled column.
        if self.amount_col in X.columns:
            self.amount_threshold = float(X[self.amount_col].quantile(0.99))
        else:
            self.amount_threshold = 1.0
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        cond_amt = X[self.amount_col] > self.amount_threshold
        cond_prod = X[self.product_col] == self.product_value
        return (cond_amt & cond_prod).astype(float).to_numpy()

    def n_parameters(self) -> int:
        return 1