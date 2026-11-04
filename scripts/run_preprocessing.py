"""AG-004 driver: load data, validate, split, fit preprocessor, write report.

Usage:
  python scripts/run_preprocessing.py \
    --config configs/base.yaml \
    --config configs/data/ieee_cis.yaml

If the raw data directory is absent, runs on a synthetic fixture and
writes a report marked `synthetic_fixture: true`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adapt_g.config import load_config
from adapt_g.data.loader import join_identity, load_identity, load_transactions
from adapt_g.data.splits import (
    bounds_as_dict, chronological_split, entity_overlap_report,
)
from adapt_g.data.validation import detect_forbidden_columns, validate_schema
from adapt_g.features.preprocessing import Preprocessor, PreprocessorConfig
from adapt_g.features.rolling import RollingAggregator, RollingConfig
from adapt_g.logging_utils import get_logger
from adapt_g.manifest import new_run_id, write_manifest
from adapt_g.seeding import set_seed


def _synthetic_fixture(n: int = 500, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dt = np.sort(rng.integers(0, 1_000_000, size=n))
    txn = pd.DataFrame({
        "TransactionID": np.arange(n),
        "isFraud": rng.integers(0, 2, size=n),
        "TransactionDT": dt,
        "TransactionAmt": rng.uniform(1, 1000, size=n),
        "ProductCD": rng.choice(list("WCHSR"), size=n),
        "card1": rng.integers(1000, 1500, size=n),
        "card4": rng.choice(["visa", "mastercard"], size=n),
        "P_emaildomain": rng.choice(["a.com", "b.com", None], size=n),
    })
    ident = pd.DataFrame({"TransactionID": txn["TransactionID"].iloc[::7]})
    return txn, ident


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", action="append", default=None,
        help="YAML config to load; repeat to deep-merge configs in order",
    )
    ap.add_argument("--extra-config", action="append", default=[])
    args = ap.parse_args()

    config_paths = args.config or ["configs/base.yaml"]
    config_paths.extend(args.extra_config)
    cfg = load_config(*config_paths)
    set_seed(int(cfg["project"]["seed"]))

    run_id = new_run_id()
    run_dir = Path(cfg["paths"]["runs"]) / run_id
    logger = get_logger("adapt_g.ag004", run_dir / "run.log",
                        cfg["logging"]["level"], cfg["logging"]["json"])

    dcfg = cfg["data"]
    data_dir = Path(cfg["paths"]["data_raw"])
    synthetic = False
    try:
        txn = load_transactions(data_dir, dcfg["transaction_file"])
        ident = load_identity(data_dir, dcfg["identity_file"])
    except FileNotFoundError:
        if data_dir.exists():
            raise
        logger.info("raw data absent; using synthetic fixture",
                    extra={"run_id": run_id, "ticket": "AG-004"})
        txn, ident = _synthetic_fixture()
        synthetic = True

    validate_schema(txn, ident)
    df = join_identity(txn, ident, dcfg["id_column"])

    forbidden = detect_forbidden_columns(df)
    if forbidden:
        logger.info(f"forbidden-suspect columns flagged: {forbidden}",
                    extra={"run_id": run_id, "event": "forbidden_columns"})

    train, val, test, bounds = chronological_split(
        df,
        time_col=dcfg["time_column"],
        id_col=dcfg["id_column"],
        target_col=dcfg["target_column"],
        train_end_q=cfg["split"]["train_end_q"],
        val_end_q=cfg["split"]["val_end_q"],
    )

    # Preprocessor fitted on train only.
    pre_cfg = PreprocessorConfig(
        id_col=dcfg["id_column"],
        time_col=dcfg["time_column"],
        target_col=dcfg["target_column"],
    )
    pre = Preprocessor(pre_cfg).fit(train)
    train_p = pre.transform(train)
    val_p = pre.transform(val)
    test_p = pre.transform(test)

    # Rolling features use only past transactions, carrying history across folds.
    agg = RollingAggregator(RollingConfig(entity_col="card1",
                                          time_col=dcfg["time_column"],
                                          amount_col="TransactionAmt"))
    n_train, n_val = len(train_p), len(val_p)
    all_p = pd.concat([train_p, val_p, test_p], ignore_index=True)
    all_raw = pd.concat([train, val, test], ignore_index=True)
    all_r = agg.fit_transform(all_raw)
    rolling_cols = [c for c in all_r.columns if c.startswith("prior_")]
    for column in rolling_cols:
        all_p[column] = all_r[column].to_numpy()
    val_r = all_p.iloc[n_train:n_train + n_val].copy()
    test_r = all_p.iloc[n_train + n_val:].copy()
    train_r = all_p.iloc[:n_train].copy()

    # Entity overlap report.
    overlap = entity_overlap_report(
        train, val, test,
        entity_cols=("card1", "card4", "P_emaildomain", "has_identity"),
    )
    overlap_payload = [o.__dict__ for o in overlap]

    # Write reports.
    split_report = {
        "synthetic_fixture": synthetic,
        "bounds": bounds_as_dict(bounds),
        "entity_overlap": overlap_payload,
        "forbidden_columns_flagged": forbidden,
        "preprocessor": pre.report(),
    }
    (run_dir / "split_report.json").write_text(json.dumps(split_report, indent=2))
    for name, d in (("train", train_r), ("val", val_r), ("test", test_r)):
        d.to_parquet(run_dir / f"{name}_processed.parquet", index=False) if hasattr(d, "to_parquet") else None

    write_manifest(
        run_dir=run_dir,
        cfg=cfg,
        dataset_version="synthetic_fixture" if synthetic else f"ieee_cis@{dcfg['transaction_file']}",
        split_boundaries=bounds_as_dict(bounds),
        extra={"ticket": "AG-004", "synthetic_fixture": synthetic},
    )
    logger.info("AG-004 preprocessing complete",
                extra={"run_id": run_id, "ticket": "AG-004"})


if __name__ == "__main__":
    main()