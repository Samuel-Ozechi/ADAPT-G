"""Load IEEE-CIS CSVs from the raw data directory."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _path(data_dir: str | Path, filename: str) -> Path:
    p = Path(data_dir) / filename
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download IEEE-CIS data per scripts/download_data.sh."
        )
    return p


def load_transactions(data_dir: str | Path, filename: str) -> pd.DataFrame:
    return pd.read_csv(_path(data_dir, filename))


def load_identity(data_dir: str | Path, filename: str) -> pd.DataFrame:
    return pd.read_csv(_path(data_dir, filename))


def join_identity(
    txn: pd.DataFrame, identity: pd.DataFrame, id_col: str = "TransactionID"
) -> pd.DataFrame:
    """Left-join identity. Adds has_identity indicator (no imputation here)."""
    out = txn.merge(identity, on=id_col, how="left", suffixes=("", "_id"))
    out["has_identity"] = out[id_col].isin(identity[id_col]).astype("int8")
    return out