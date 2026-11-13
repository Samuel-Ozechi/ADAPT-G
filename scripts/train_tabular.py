"""Train the tabular baselines (AG-006) and freeze the tabular reference.

Usage:
  python scripts/train_tabular.py \
    --config configs/base.yaml \
    --config configs/data/ieee_cis.yaml

Writes to runs/<run_id>/:
  - baseline_table.json          (all baselines, one row each)
  - frozen_reference.json        (paths + configs + parameter counts)
  - preds/<model>/train.csv, val.csv, test.csv
  - evaluation/<model>/evaluation_report.{json,md}
  - run.log, run_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adapt_g.config import load_config
from adapt_g.data.loader import load_identity, load_transactions
from adapt_g.evaluation.report import build_report, write_report
from adapt_g.features.tabular import build_tabular_folds
from adapt_g.logging_utils import get_logger
from adapt_g.manifest import new_run_id, write_manifest
from adapt_g.models.baselines import (
    LightGBMBaseline, LogRegBaseline, MajorityBaseline, RuleBaseline,
)
from adapt_g.seeding import set_seed


def _synthetic_fixture(n: int = 800, seed: int = 0):
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
    ident = pd.DataFrame({"TransactionID": txn["TransactionID"].iloc[::5]})
    return txn, ident


def _save_preds(path: Path, ids: pd.Series, y: pd.Series, s: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"TransactionID": ids.values, "y_true": y.values, "y_score": s}).to_csv(
        path, index=False
    )


def _evaluate_and_save(model_name: str, preds_dir: Path, runs_dir: Path,
                       cfg: dict) -> dict:
    def load(kind: str):
        d = pd.read_csv(preds_dir / f"{kind}.csv")
        return d["y_true"].to_numpy(), d["y_score"].to_numpy()

    y_tr, s_tr = load("train")
    y_va, s_va = load("val")
    y_te, s_te = load("test")
    report = build_report(y_tr, s_tr, y_va, s_va, y_te, s_te, model_name=model_name)
    out = runs_dir / "evaluation" / model_name
    write_report(report, out)
    return report


def _load_cli_config(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", action="append", default=None,
        help="YAML config to load; repeat to deep-merge configs in order",
    )
    ap.add_argument("--extra-config", action="append", default=[])
    args = ap.parse_args(argv)

    config_paths = args.config or ["configs/base.yaml"]
    config_paths.extend(args.extra_config)
    return load_config(*config_paths)


def main() -> None:
    cfg = _load_cli_config()
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    run_id = new_run_id("baseline")
    run_dir = Path(cfg["paths"]["runs"]) / run_id
    logger = get_logger("adapt_g", run_dir / "run.log",
                        cfg["logging"]["level"], cfg["logging"]["json"])
    logger.info("AG-006 start", extra={"run_id": run_id, "run": "classical baselines", "seed": seed})

    dcfg = cfg["data"]
    data_dir = Path(cfg["paths"]["data_raw"])
    synthetic = False
    
    txn = load_transactions(data_dir, dcfg["transaction_file"])
    ident = load_identity(data_dir, dcfg["identity_file"])
    
    folds = build_tabular_folds(
        txn, ident,
        id_col=dcfg["id_column"], time_col=dcfg["time_column"],
        target_col=dcfg["target_column"],
        train_end_q=cfg["split"]["train_end_q"], val_end_q=cfg["split"]["val_end_q"],
    )
    logger.info(f"folds: train={len(folds.X_train)} val={len(folds.X_val)} "
                f"test={len(folds.X_test)} feats={len(folds.feature_names)}",
                extra={"run_id": run_id, "ticket": "AG-006"})

    models = {
        "majority": MajorityBaseline(),
        "rule": RuleBaseline(),
        "logreg": LogRegBaseline(seed=seed),
        "lightgbm": LightGBMBaseline(seed=seed),
    }

    results = {}
    frozen = {"synthetic_fixture": synthetic, "folds": {}, "models": {}}

    for name, model in models.items():
        logger.info(f"training {name}", extra={"run_id": run_id, "event": "train_start", "extra": {"model": name}})
        if name == "lightgbm":
            model.fit(folds.X_train, folds.y_train, folds.X_val, folds.y_val)
            tuning_log = getattr(model, "tuning_log", [])
            frozen["models"][name] = {"tuning_log": tuning_log,
                                      "best_cfg": model.best_cfg.__dict__ if model.best_cfg else None,
                                      "n_parameters": model.n_parameters()}
        else:
            model.fit(folds.X_train, folds.y_train)
            frozen["models"][name] = {"n_parameters": model.n_parameters()}

        preds_dir = run_dir / "preds" / name
        _save_preds(preds_dir / "train.csv", folds.train_ids, folds.y_train, model.predict_score(folds.X_train))
        _save_preds(preds_dir / "val.csv", folds.val_ids, folds.y_val, model.predict_score(folds.X_val))
        _save_preds(preds_dir / "test.csv", folds.test_ids, folds.y_test, model.predict_score(folds.X_test))

        report = _evaluate_and_save(name, preds_dir, run_dir, cfg)
        results[name] = report

    # Baseline table: one row per model, headline metrics.
    table = []
    for name, rep in results.items():
        t = rep["test"]
        table.append({
            "model": name,
            "pr_auc": t["ranking"]["pr_auc"],
            "roc_auc": t["ranking"]["roc_auc"],
            "precision": t["operational"]["precision"],
            "recall": t["operational"]["recall"],
            "f1": t["operational"]["f1"],
            "fpr": t["operational"]["fpr"],
            "brier": t["calibration"]["brier"],
            "ece": t["calibration"]["ece"],
            "threshold": rep["threshold_selection"]["threshold"],
            "threshold_rule": rep["threshold_selection"]["rule"],
            "n_parameters": frozen["models"][name]["n_parameters"],
            "pr_auc_ci_low": rep["test_bootstrap_ci"]["pr_auc"]["ci_low"],
            "pr_auc_ci_high": rep["test_bootstrap_ci"]["pr_auc"]["ci_high"],
        })
    (run_dir / "baseline_table.json").write_text(json.dumps(table, indent=2, default=str))

    frozen["folds"] = {
        "bounds": folds.bounds.__dict__,
        "n_features": len(folds.feature_names),
        "feature_names": folds.feature_names,
        "preprocessor_report": folds.preprocessor_report,
    }
    (run_dir / "frozen_reference.json").write_text(json.dumps(frozen, indent=2, default=str))

    write_manifest(
        run_dir=run_dir, cfg=cfg,
        dataset_version=f"ieee_cis@{dcfg['transaction_file']}",
        split_boundaries=folds.bounds.__dict__,
        extra={"tag": "baseline", "models": list(models.keys()), "ticket": "AG-006"},
    )
    logger.info("Baseline run complete", extra={"run_id": run_id, "run": "classical baselines", "seed": seed})
    print(f"Wrote baseline table and frozen reference to {run_dir}")


if __name__ == "__main__":
    main()