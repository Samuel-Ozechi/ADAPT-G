"""Tabular feature assembly for baselines.

Pipeline:
1. Raw transactions + identity joined.
2. Chronological split.
3. Preprocessor.fit(train) -> transform each fold.
4. RollingAggregator on each fold independently.
5. Concatenate features; drop identifier/time/target columns from X.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from adapt_g.data.loader import join_identity
from adapt_g.data.splits import SplitBoundaries, bounds_as_dict, chronological_split
from adapt_g.features.preprocessing import Preprocessor, PreprocessorConfig
from adapt_g.features.rolling import RollingAggregator, RollingConfig


@dataclass
class TabularFolds:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_ids: pd.Series
    val_ids: pd.Series
    test_ids: pd.Series
    bounds: SplitBoundaries
    feature_names: list[str]
    preprocessor_report: dict


def build_tabular_folds(
    txn: pd.DataFrame,
    ident: pd.DataFrame,
    id_col: str = "TransactionID",
    time_col: str = "TransactionDT",
    target_col: str = "isFraud",
    train_end_q: float = 0.8,
    val_end_q: float = 0.9,
    rolling_windows: tuple[int, ...] = (3600, 86400, 7 * 86400),
) -> TabularFolds:
    df = join_identity(txn, ident, id_col)

    train, val, test, bounds = chronological_split(
        df, time_col=time_col, id_col=id_col, target_col=target_col,
        train_end_q=train_end_q, val_end_q=val_end_q,
    )

    pre_cfg = PreprocessorConfig(id_col=id_col, time_col=time_col, target_col=target_col)
    pre = Preprocessor(pre_cfg).fit(train)
    train_p = pre.transform(train)
    val_p = pre.transform(val)
    test_p = pre.transform(test)

    agg = RollingAggregator(RollingConfig(
        entity_col="card1", time_col=time_col, amount_col="TransactionAmt",
        windows_seconds=rolling_windows,
    ))
    train_r = agg.fit_transform(train_p)
    val_r = agg.fit_transform(val_p)
    test_r = agg.fit_transform(test_p)

    drop = {id_col, time_col, target_col}
    feats = [c for c in train_r.columns if c not in drop]

    def split_xy(d: pd.DataFrame):
        return d[feats].copy(), d[target_col].astype(int).copy(), d[id_col].copy()

    X_tr, y_tr, ids_tr = split_xy(train_r)
    X_va, y_va, ids_va = split_xy(val_r)
    X_te, y_te, ids_te = split_xy(test_r)

    return TabularFolds(
        X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va,
        X_test=X_te, y_test=y_te,
        train_ids=ids_tr, val_ids=ids_va, test_ids=ids_te,
        bounds=bounds, feature_names=feats,
        preprocessor_report=pre.report(),
    )