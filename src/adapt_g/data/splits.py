"""Chronological splitting on TransactionDT with entity-overlap reporting."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SplitBoundaries:
    train_max_dt: float
    val_max_dt: float
    train_q: float
    val_q: float
    n_train: int
    n_val: int
    n_test: int
    train_fraud_rate: float
    val_fraud_rate: float
    test_fraud_rate: float


@dataclass(frozen=True)
class EntityOverlap:
    entity: str
    train_entities: int
    val_entities: int
    test_entities: int
    val_seen_in_train: int
    test_seen_in_train: int
    test_seen_in_val: int
    val_seen_frac: float
    test_seen_frac: float


def chronological_split(
    df: pd.DataFrame,
    time_col: str = "TransactionDT",
    id_col: str = "TransactionID",
    target_col: str = "isFraud",
    train_end_q: float = 0.8,
    val_end_q: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitBoundaries]:
    """Split by quantiles of time_col. Deterministic tie-break on id_col."""
    if not (0.0 < train_end_q < val_end_q < 1.0):
        raise ValueError("Require 0 < train_end_q < val_end_q < 1")

    df = df.sort_values([time_col, id_col], kind="mergesort").reset_index(drop=True)
    train_max_dt = float(df[time_col].quantile(train_end_q))
    val_max_dt = float(df[time_col].quantile(val_end_q))

    train = df[df[time_col] <= train_max_dt]
    val = df[(df[time_col] > train_max_dt) & (df[time_col] <= val_max_dt)]
    test = df[df[time_col] > val_max_dt]

    bounds = SplitBoundaries(
        train_max_dt=train_max_dt,
        val_max_dt=val_max_dt,
        train_q=train_end_q,
        val_q=val_end_q,
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
        train_fraud_rate=float(train[target_col].mean()) if len(train) else float("nan"),
        val_fraud_rate=float(val[target_col].mean()) if len(val) else float("nan"),
        test_fraud_rate=float(test[target_col].mean()) if len(test) else float("nan"),
    )
    return train, val, test, bounds


def entity_overlap_report(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    entity_cols: Iterable[str],
) -> list[EntityOverlap]:
    """Compute how many entities in val/test were seen in earlier folds.

    Documented for AG-004; used by the split report. High overlap means
    entity-level leakage is a real risk; low overlap means cold-start stress.
    """
    rows: list[EntityOverlap] = []
    for col in entity_cols:
        if col not in train.columns:
            continue
        train_ids = set(train[col].dropna().unique())
        val_ids = set(val[col].dropna().unique())
        test_ids = set(test[col].dropna().unique())
        val_seen = len(val_ids & train_ids)
        test_seen_tr = len(test_ids & train_ids)
        test_seen_val = len(test_ids & val_ids)
        rows.append(EntityOverlap(
            entity=col,
            train_entities=len(train_ids),
            val_entities=len(val_ids),
            test_entities=len(test_ids),
            val_seen_in_train=val_seen,
            test_seen_in_train=test_seen_tr,
            test_seen_in_val=test_seen_val,
            val_seen_frac=val_seen / len(val_ids) if val_ids else 0.0,
            test_seen_frac=test_seen_tr / len(test_ids) if test_ids else 0.0,
        ))
    return rows


def bounds_as_dict(bounds: SplitBoundaries) -> dict:
    return asdict(bounds)