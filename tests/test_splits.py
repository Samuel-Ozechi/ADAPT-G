import pandas as pd
import pytest

from adapt_g.data.splits import chronological_split


def test_split_boundary_quantiles():
    df = pd.DataFrame({
        "TransactionID": list(range(100)),
        "TransactionDT": list(range(100)),
        "isFraud": [0] * 100,
    })
    train, val, test, bounds = chronological_split(df, train_end_q=0.8, val_end_q=0.9)
    assert bounds.train_max_dt == 79.0
    assert bounds.val_max_dt == 89.0
    assert len(train) == 80 and len(val) == 10 and len(test) == 10


def test_split_rejects_invalid_quantiles():
    df = pd.DataFrame({"TransactionID": [1], "TransactionDT": [10], "isFraud": [0]})
    with pytest.raises(ValueError):
        chronological_split(df, train_end_q=0.9, val_end_q=0.8)