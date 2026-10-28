"""Training-only-fitted preprocessing for IEEE-CIS.

Design:
- Preprocessor.fit(train_df) learns all statistics from the training fold.
- Preprocessor.transform(df) applies those statistics to any fold.
- No method inspects val/test during fit. This is enforced by construction:
  fit only accepts a single DataFrame and never receives the other folds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

UNSEEN_TOKEN = "__UNSEEN__"
MISSING_CAT_TOKEN = "__MISSING__"


@dataclass
class PreprocessorConfig:
    id_col: str = "TransactionID"
    time_col: str = "TransactionDT"
    target_col: str = "isFraud"
    # Columns never used as model features.
    passthrough_drop: tuple[str, ...] = ()
    # Categorical columns to frequency/one-hot encode; inferred if empty.
    categorical_cols: tuple[str, ...] = ()
    # Numeric columns to impute + scale; inferred if empty.
    numeric_cols: tuple[str, ...] = ()
    # Max categories kept per column; rarer categories -> UNSEEN.
    max_categories: int = 64
    # Columns to drop outright (logged with reason).
    drop_columns: tuple[str, ...] = ()
    drop_reasons: dict[str, str] = field(default_factory=dict)


class Preprocessor:
    """Fit on train; transform any fold. Statistics stored on self."""

    def __init__(self, cfg: PreprocessorConfig):
        self.cfg = cfg
        self._numeric_medians: dict[str, float] = {}
        self._numeric_means: dict[str, float] = {}
        self._numeric_stds: dict[str, float] = {}
        self._category_lists: dict[str, list[str]] = {}
        self._dropped_columns: list[tuple[str, str]] = []
        self._fitted = False

    # ---------- inference of column roles ----------
    def _infer_columns(self, df: pd.DataFrame) -> None:
        if self.cfg.categorical_cols and self.cfg.numeric_cols:
            return
        exclude = {self.cfg.id_col, self.cfg.time_col, self.cfg.target_col,
                   "has_identity", *self.cfg.passthrough_drop}
        cat, num = [], []
        for c in df.columns:
            if c in exclude:
                continue
            if df[c].dtype == object:
                cat.append(c)
            elif np.issubdtype(df[c].dtype, np.number):
                num.append(c)
        if not self.cfg.categorical_cols:
            self.cfg.categorical_cols = tuple(cat)
        if not self.cfg.numeric_cols:
            self.cfg.numeric_cols = tuple(num)

    # ---------- fit ----------
    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        self._infer_columns(train_df)

        dropped = set(self.cfg.drop_columns)
        # Constant-on-train columns carry no signal; drop and log.
        for c in list(train_df.columns):
            if c in dropped:
                continue
            if c in (self.cfg.id_col, self.cfg.time_col, self.cfg.target_col):
                continue
            if train_df[c].nunique(dropna=False) <= 1:
                self._dropped_columns.append((c, "constant_on_train"))
                dropped.add(c)
        for c in self.cfg.drop_columns:
            self._dropped_columns.append(
                (c, self.cfg.drop_reasons.get(c, "explicit_drop"))
            )

        for c in self.cfg.numeric_cols:
            if c in dropped:
                continue
            s = pd.to_numeric(train_df[c], errors="coerce")
            self._numeric_medians[c] = float(s.median()) if not s.dropna().empty else 0.0
            filled = s.fillna(self._numeric_medians[c])
            self._numeric_means[c] = float(filled.mean())
            std = float(filled.std(ddof=0))
            self._numeric_stds[c] = std if std > 1e-12 else 1.0

        for c in self.cfg.categorical_cols:
            if c in dropped:
                continue
            vc = train_df[c].astype("string").fillna(MISSING_CAT_TOKEN).value_counts()
            kept = vc.head(self.cfg.max_categories).index.tolist()
            self._category_lists[c] = kept

        self._dropped_columns_unique = sorted(set(self._dropped_columns))
        self._fitted = True
        return self

    # ---------- transform ----------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit on training fold first.")
        out = df.copy()

        for c in self.cfg.numeric_cols:
            if c not in out.columns:
                continue
            s = pd.to_numeric(out[c], errors="coerce")
            out[f"{c}__is_missing"] = s.isna().astype("int8")
            filled = s.fillna(self._numeric_medians[c])
            out[c] = (filled - self._numeric_means[c]) / self._numeric_stds[c]

        for c in self.cfg.categorical_cols:
            if c not in out.columns:
                continue
            kept = set(self._category_lists.get(c, []))
            s = out[c].astype("string").fillna(MISSING_CAT_TOKEN)
            out[c] = s.where(s.isin(kept), UNSEEN_TOKEN)

        drop_cols = [c for c, _ in self._dropped_columns_unique if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)
        return out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)

    # ---------- reporting ----------
    def report(self) -> dict[str, Any]:
        return {
            "n_numeric": len(self._numeric_medians),
            "n_categorical": len(self._category_lists),
            "dropped_columns": self._dropped_columns_unique,
            "numeric_medians": self._numeric_medians,
            "category_sizes": {k: len(v) for k, v in self._category_lists.items()},
        }