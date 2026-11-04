"""Schema validation for IEEE-CIS transaction and identity tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_TRANSACTION_COLS = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
]
REQUIRED_IDENTITY_COLS = ["TransactionID"]


class SchemaError(ValueError):
    pass


def validate_schema(txn: pd.DataFrame, identity: pd.DataFrame | None = None) -> None:
    missing = [c for c in REQUIRED_TRANSACTION_COLS if c not in txn.columns]
    if missing:
        raise SchemaError(f"Transaction table missing columns: {missing}")
    if not txn["TransactionID"].is_unique:
        raise SchemaError("TransactionID not unique in transaction table")
    if not txn["isFraud"].isin([0, 1]).all():
        raise SchemaError("isFraud must be binary 0/1")
    if (txn["TransactionDT"] < 0).any():
        raise SchemaError("TransactionDT contains negative values")
    if txn["TransactionID"].isna().any():
        raise SchemaError("TransactionID contains nulls")
    for col in ("TransactionDT", "TransactionAmt"):
        values = pd.to_numeric(txn[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise SchemaError(f"{col} must contain only finite numeric values")

    if identity is not None:
        missing_id = [c for c in REQUIRED_IDENTITY_COLS if c not in identity.columns]
        if missing_id:
            raise SchemaError(f"Identity table missing columns: {missing_id}")
        if not identity["TransactionID"].is_unique:
            raise SchemaError("TransactionID not unique in identity table")
        if identity["TransactionID"].isna().any():
            raise SchemaError("Identity TransactionID contains nulls")


def detect_forbidden_columns(
    txn: pd.DataFrame, forbidden_substrings: tuple[str, ...] = (
        "future", "post_", "chargeback", "label_", "target_",
    )
) -> list[str]:
    """Flag column names that suggest post-outcome or target-derived content.

    This is a name-based heuristic; the leakage test suite provides the
    behavioural check. Column list is logged for human review.
    """
    hits = []
    for c in txn.columns:
        low = c.lower()
        if any(s in low for s in forbidden_substrings) and c != "isFraud":
            hits.append(c)
    return hits