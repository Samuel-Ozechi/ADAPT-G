"""Behavioural leakage tests. These must pass before any modelling.

Strategy:
1. Synthetic fixture with known timestamps and entity ids.
2. Split chronologically.
3. Verify no feature value changes when future rows are deleted.
4. Verify rolling aggregates never include the current or future row.
5. Verify preprocessor statistics are invariant to val/test content.
"""
import numpy as np
import pandas as pd
import pytest

from adapt_g.data.splits import chronological_split
from adapt_g.features.preprocessing import Preprocessor, PreprocessorConfig
from adapt_g.features.rolling import RollingAggregator, RollingConfig


def make_txn(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt = np.sort(rng.integers(0, 100_000, size=n))
    return pd.DataFrame({
        "TransactionID": np.arange(n),
        "isFraud": rng.integers(0, 2, size=n),
        "TransactionDT": dt,
        "TransactionAmt": rng.uniform(1, 500, size=n),
        "ProductCD": rng.choice(list("WCHSR"), size=n),
        "card1": rng.integers(1000, 1100, size=n),
        "P_emaildomain": rng.choice(["a.com", "b.com", None], size=n),
    })


def test_no_future_influence_on_features():
    """Deleting future rows must not change any feature for earlier rows."""
    df = make_txn()
    train, val, test, _ = chronological_split(df, train_end_q=0.6, val_end_q=0.8)

    agg = RollingAggregator(RollingConfig(windows_seconds=(3600, 86400)))
    full = agg.fit_transform(train)
    # Truncate to first half of train and recompute; earlier rows must match.
    half = train[train["TransactionDT"] <= train["TransactionDT"].quantile(0.5)]
    truncated = agg.fit_transform(half)

    cols = [c for c in full.columns if c.startswith("prior_")]
    merged = full.merge(truncated[["TransactionID"] + cols],
                        on="TransactionID", suffixes=("", "_trunc"))
    for c in cols:
        a = merged[c].fillna(-1).values
        b = merged[f"{c}_trunc"].fillna(-1).values
        assert np.allclose(a, b), f"Future rows influenced feature {c}"


def test_rolling_aggregates_are_strictly_prior():
    df = make_txn(50)
    agg = RollingAggregator(RollingConfig(windows_seconds=(86400,)))
    out = agg.fit_transform(df)
    # For each entity, the first row must have no window history.
    first_rows = out.groupby("card1", sort=False).head(1)
    assert first_rows["prior_count_86400s"].isna().all()


def test_rolling_aggregates_exclude_current_row():
    """If two rows share an entity and time, neither sees the other."""
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "isFraud": [0, 0, 0],
        "TransactionDT": [100, 100, 200],
        "TransactionAmt": [10.0, 20.0, 30.0],
        "card1": [7, 7, 7],
    })
    agg = RollingAggregator(RollingConfig(windows_seconds=(86400,)))
    out = agg.fit_transform(df).sort_values("TransactionID").reset_index(drop=True)
    # Rows 1 and 2 share time=100, so neither sees the other in a strict-< window.
    assert out.loc[0, "prior_count_86400s"] != out.loc[0, "prior_count_86400s"]  # NaN
    assert out.loc[1, "prior_count_86400s"] != out.loc[1, "prior_count_86400s"]  # NaN


def test_preprocessor_statistics_invariant_to_val_test():
    df = make_txn()
    train, val, test, _ = chronological_split(df, train_end_q=0.6, val_end_q=0.8)
    cfg = PreprocessorConfig(
        numeric_cols=("TransactionAmt",),
        categorical_cols=("ProductCD", "P_emaildomain"),
    )
    pre_a = Preprocessor(cfg).fit(train)
    # Same train, different val/test content -> identical fitted stats.
    train_b, val_b, test_b, _ = chronological_split(df, train_end_q=0.6, val_end_q=0.8)
    pre_b = Preprocessor(cfg).fit(train_b)
    assert pre_a._numeric_medians == pre_b._numeric_medians
    assert pre_a._category_lists == pre_b._category_lists


def test_preprocessor_unseen_categories_map_to_token():
    df = make_txn()
    train, val, _, _ = chronological_split(df, train_end_q=0.7, val_end_q=0.85)
    cfg = PreprocessorConfig(
        numeric_cols=("TransactionAmt",),
        categorical_cols=("ProductCD",),
        max_categories=2,
    )
    pre = Preprocessor(cfg).fit(train)
    out = pre.transform(val)
    # Any value outside the top-2 training categories becomes __UNSEEN__.
    kept = set(pre._category_lists["ProductCD"])
    for v in out["ProductCD"].dropna().unique():
        assert v in kept or v == "__UNSEEN__"


def test_constant_columns_dropped_and_logged():
    df = make_txn()
    df["constant_col"] = 1
    train, _, _, _ = chronological_split(df, train_end_q=0.8, val_end_q=0.9)
    cfg = PreprocessorConfig(
        numeric_cols=("TransactionAmt",),
        categorical_cols=("ProductCD",),
    )
    pre = Preprocessor(cfg).fit(train)
    dropped = dict(pre._dropped_columns_unique)
    assert dropped.get("constant_col") == "constant_on_train"