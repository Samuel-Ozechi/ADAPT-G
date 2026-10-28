from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "sample_transaction.csv")


@pytest.fixture
def sample_identity() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "sample_identity.csv")