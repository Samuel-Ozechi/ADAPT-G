import pandas as pd

from adapt_g.data.loader import load_identity, load_transactions


def test_csv_loaders_support_column_pruning(tmp_path):
    transactions = pd.DataFrame({
        "TransactionID": [1],
        "isFraud": [0],
        "unused_feature": ["not loaded"],
    })
    identity = pd.DataFrame({
        "TransactionID": [1],
        "DeviceInfo": ["device"],
        "unused_identity_feature": ["not loaded"],
    })
    transactions.to_csv(tmp_path / "transactions.csv", index=False)
    identity.to_csv(tmp_path / "identity.csv", index=False)

    loaded_transactions = load_transactions(
        tmp_path, "transactions.csv", usecols=["TransactionID", "isFraud"]
    )
    loaded_identity = load_identity(
        tmp_path, "identity.csv", usecols=["TransactionID", "DeviceInfo"]
    )

    assert loaded_transactions.columns.tolist() == ["TransactionID", "isFraud"]
    assert loaded_identity.columns.tolist() == ["TransactionID", "DeviceInfo"]
