"""LightGBM baseline with a small validation-only hyperparameter grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from sklearn.metrics import average_precision_score

try:
    import lightgbm as lgb
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "LightGBM not installed. Run: pip install -e '.[baselines]'"
    ) from e


@dataclass
class LGBMConfig:
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0


class LightGBMBaseline:
    """LightGBM with a documented small grid, selected on validation PR-AUC."""

    name = "lightgbm"

    def __init__(self, seed: int = 0, grid_size: int = 6) -> None:
        self.seed = seed
        self.grid_size = grid_size
        self.best_cfg: LGBMConfig | None = None
        self.model: lgb.LGBMClassifier | None = None
        self.tuning_log: list[dict] = []
        self._cat_cols: list[str] = []
        self._cat_categories: dict[str, pd.Index] = {}

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        prepared = X.copy()
        for column in self._cat_cols:
            if column not in prepared.columns:
                continue
            categories = self._cat_categories.get(column)
            if categories is None:
                prepared[column] = prepared[column].astype("category")
                self._cat_categories[column] = prepared[column].cat.categories
            else:
                prepared[column] = pd.Categorical(prepared[column], categories=categories)
        return prepared

    def _candidate_grid(self) -> list[LGBMConfig]:
        # Small, bounded grid; <= grid_size fits total.
        candidates = [
            LGBMConfig(),
            LGBMConfig(n_estimators=600, learning_rate=0.03, num_leaves=63),
            LGBMConfig(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=40),
            LGBMConfig(n_estimators=500, learning_rate=0.03, num_leaves=31,
                       subsample=0.8, colsample_bytree=0.8),
            LGBMConfig(n_estimators=400, learning_rate=0.07, num_leaves=63, reg_lambda=5.0),
            LGBMConfig(n_estimators=800, learning_rate=0.02, num_leaves=127, min_child_samples=10),
        ]
        return candidates[: self.grid_size]

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None) -> "LightGBMBaseline":
        """If X_val/y_val provided, tune on validation PR-AUC; else use default cfg."""
        self._cat_cols = [
            column for column in X.columns
            if is_object_dtype(X[column].dtype)
            or is_string_dtype(X[column].dtype)
            or isinstance(X[column].dtype, pd.CategoricalDtype)
        ]
        self._cat_categories = {}
        X = self._prepare_features(X)
        if X_val is not None:
            X_val = self._prepare_features(X_val)

        pos = int(y.sum())
        neg = int(len(y) - pos)
        spw = neg / max(pos, 1)

        if X_val is None or y_val is None:
            self.best_cfg = LGBMConfig()
            self.model = self._fit_one(X, y, self.best_cfg, spw)
            return self

        best = None
        for i, cfg in enumerate(self._candidate_grid()):
            model = self._fit_one(X, y, cfg, spw, seed_offset=i)
            s_val = model.predict_proba(X_val)[:, 1]
            pr = float(average_precision_score(y_val, s_val))
            self.tuning_log.append({"trial": i, "cfg": cfg.__dict__, "val_pr_auc": pr})
            if best is None or pr > best[1]:
                best = (cfg, pr, model)
        self.best_cfg, _, self.model = best
        return self

    def _fit_one(self, X: pd.DataFrame, y: pd.Series, cfg: LGBMConfig,
                 spw: float, seed_offset: int = 0) -> lgb.LGBMClassifier:
        model = lgb.LGBMClassifier(
            n_estimators=cfg.n_estimators, learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves, min_child_samples=cfg.min_child_samples,
            subsample=cfg.subsample, colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda, scale_pos_weight=spw,
            random_state=self.seed + seed_offset, n_jobs=-1, verbose=-1,
        )
        model.fit(X, y)
        return model

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        X = self._prepare_features(X)
        return self.model.predict_proba(X)[:, 1]

    def n_parameters(self) -> int:
        assert self.model is not None
        booster = self.model.booster_

        def count_leaves(node: dict) -> int:
            if "leaf_value" in node:
                return 1
            return count_leaves(node["left_child"]) + count_leaves(node["right_child"])

        return sum(
            count_leaves(tree["tree_structure"])
            for tree in booster.dump_model()["tree_info"]
        )