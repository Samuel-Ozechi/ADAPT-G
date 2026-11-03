"""Logistic regression baseline with class weighting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class LogRegBaseline:
    name = "logreg"

    def __init__(self, max_iter: int = 200, seed: int = 0) -> None:
        self.max_iter = max_iter
        self.seed = seed
        self.pipe: Pipeline | None = None
        self._cat_cols: list[str] = []
        self._num_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogRegBaseline":
        self._cat_cols = [c for c in X.columns if not is_numeric_dtype(X[c].dtype)]
        self._num_cols = [c for c in X.columns if is_numeric_dtype(X[c].dtype)]
        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), self._num_cols),
                ("cat", Pipeline([
                    ("imp", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
                ]), self._cat_cols),
            ],
            remainder="drop",
        )
        self.pipe = Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(
                max_iter=self.max_iter, class_weight="balanced",
                random_state=self.seed, solver="lbfgs",
            )),
        ])
        self.pipe.fit(X, y)
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        assert self.pipe is not None
        return self.pipe.predict_proba(X)[:, 1]

    def n_parameters(self) -> int:
        assert self.pipe is not None
        clf = self.pipe.named_steps["clf"]
        return int(clf.coef_.size + clf.intercept_.size)
