"""LightGBM baseline with a small validation-only hyperparameter grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
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
        self._cat_cols = [c for c in X.columns if X[c].dtype == object]
        for c in self._cat_cols:
            X = X.copy()
            X_val = X_val.copy() if X_val is not None else None
        # LightGBM accepts category dtype directly.
        for d in (X, X_val):
            if d is None:
                continue
            for c in self._cat_cols:
                d[c] = d[c].astype("category")

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
        X = X.copy()
        for c in self._cat_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")
        return self.model.predict_proba(X)[:, 1]

    def n_parameters(self) -> int:
        assert self.model is not None
        booster = self.model.booster_
        return int(sum(len(t.split()) for t in booster.dump_model()["tree_info"]))