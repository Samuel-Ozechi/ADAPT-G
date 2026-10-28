"""AG-003 smoke test: fresh env can load, validate, join, and split."""
import pandas as pd
import pytest

from adapt_g.data.loader import join_identity
from adapt_g.data.splits import chronological_split
from adapt_g.data.validation import SchemaError, validate_schema


def test_load_and_validate(sample_transactions):
    validate_schema(sample_transactions)


def test_validation_rejects_bad_label(sample_transactions):
    bad = sample_transactions.copy()
    bad.loc[0, "isFraud"] = 2
    with pytest.raises(SchemaError):
        validate_schema(bad)


def test_identity_join_marks_missing(sample_transactions, sample_identity):
    joined = join_identity(sample_transactions, sample_identity)
    assert "has_identity" in joined.columns
    # Neither sample transaction ID (2987000/2987001) appears in identity fixture.
    assert (joined["has_identity"] == 0).all()


def test_identity_join_marks_present(sample_transactions, sample_identity):
    # Introduce a matching identity row to exercise the positive branch.
    identity_ext = pd.concat(
        [sample_identity,
         pd.DataFrame([{"TransactionID": 2987000, "DeviceType": "mobile"}])],
        ignore_index=True,
    )
    joined = join_identity(sample_transactions, identity_ext)
    assert joined.loc[joined["TransactionID"] == 2987000, "has_identity"].iloc[0] == 1


def test_chronological_split_orders_and_partitions():
    df = pd.DataFrame({
        "TransactionID": list(range(10)),
        "TransactionDT": [100, 100, 200, 300, 400, 500, 600, 700, 800, 900],
        "isFraud": [0] * 10,
    })
    train, val, test, bounds = chronological_split(df, train_end_q=0.6, val_end_q=0.8)
    assert train["TransactionDT"].max() <= bounds.train_max_dt
    assert val["TransactionDT"].min() > bounds.train_max_dt
    assert val["TransactionDT"].max() <= bounds.val_max_dt
    assert test["TransactionDT"].min() > bounds.val_max_dt
    assert len(train) + len(val) + len(test) == len(df)