"""Train the temporal Transformer baseline.

Writes:
  runs/<id>/frozen_temporal_reference.json
  runs/<id>/preds/temporal/{train,val,test}.csv
  runs/<id>/evaluation/temporal/evaluation_report.{json,md}
  runs/<id>/temporal_profile.json    (latency, memory, params)
  runs/<id>/run_manifest.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from adapt_g.config import load_config
from adapt_g.data.loader import load_identity, load_transactions
from adapt_g.evaluation.report import build_report, write_report
from adapt_g.features.sequences import SequenceConfig, build_sequence_folds
from adapt_g.logging_utils import get_logger
from adapt_g.manifest import new_run_id, write_manifest
from adapt_g.models.temporal import TemporalConfig, TemporalTransformer
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
        "card1": rng.integers(1000, 1100, size=n),
    })
    ident = pd.DataFrame({"TransactionID": txn["TransactionID"].iloc[::5]})
    return txn, ident


def _to_tensors(seq, mask, cur, y=None, device="cpu"):
    out = {
        "seq": torch.from_numpy(seq).float().to(device),
        "mask": torch.from_numpy(mask).bool().to(device),
        "cur": torch.from_numpy(cur).float().to(device),
    }
    if y is not None:
        out["y"] = torch.from_numpy(y).float().to(device)
    return out


def _predict(model, seq, mask, cur, batch_size=512, device="cpu"):
    model.eval()
    n = len(seq)
    scores = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            out = model(
                torch.from_numpy(seq[i:j]).float().to(device),
                torch.from_numpy(mask[i:j]).bool().to(device),
                torch.from_numpy(cur[i:j]).float().to(device),
            )
            scores[i:j] = torch.sigmoid(out).cpu().numpy()
    return scores


def _train(model, tensors, y_train, device, epochs=5, batch_size=512,
           lr=1e-3, pos_weight=None):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    n = len(y_train)
    idx = np.arange(n)
    model.train()
    for ep in range(epochs):
        np.random.shuffle(idx)
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            b = idx[i:j]
            logits = model(
                tensors["seq"][b], tensors["mask"][b], tensors["cur"][b],
            )
            loss = loss_fn(logits, torch.from_numpy(y_train[b]).float().to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def _measure_profile(model, seq, mask, cur, device, batch_size=512):
    # 3 warmup + 3 timed runs; report mean latency.
    model.eval()
    times = []
    n = len(seq)
    with torch.no_grad():
        for _ in range(3):
            _predict(model, seq[:1024], mask[:1024], cur[:1024], batch_size, device)
        for _ in range(3):
            t0 = time.perf_counter()
            _predict(model, seq, mask, cur, batch_size, device)
            times.append(time.perf_counter() - t0)
    return {
        "latency_full_test_mean_s": float(np.mean(times)),
        "latency_full_test_std_s": float(np.std(times)),
        "n_test": int(n),
        "batch_size": batch_size,
        "device": device,
        "n_parameters": model.n_parameters(),
    }


def _load_cli_config(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", action="append", default=None,
        help="YAML config to load; repeat to deep-merge configs in order",
    )
    ap.add_argument("--extra-config", action="append", default=[])
    args, _ = ap.parse_known_args(argv)

    config_paths = args.config or ["configs/base.yaml"]
    config_paths.extend(args.extra_config)
    return load_config(*config_paths)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", action="append", default=None,
        help="YAML config to load; repeat to deep-merge configs in order",
    )
    ap.add_argument("--extra-config", action="append", default=[])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    cfg = _load_cli_config()
    seed = int(cfg["project"]["seed"])
    set_seed(seed)
    torch.manual_seed(seed)

    run_id = new_run_id()
    run_dir = Path(cfg["paths"]["runs"]) / run_id
    logger = get_logger("adapt_g.ag007", run_dir / "run.log",
                        cfg["logging"]["level"], cfg["logging"]["json"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}", extra={"run_id": run_id, "run": "transformer baseline", "seed": seed})

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
                    extra={"run_id": run_id, "run": "transformer baseline", "seed": seed})
        txn, ident = _synthetic_fixture()
        synthetic = True

    sf = build_sequence_folds(
        txn, ident, id_col=dcfg["id_column"], time_col=dcfg["time_column"],
        target_col=dcfg["target_column"],
        train_end_q=cfg["split"]["train_end_q"], val_end_q=cfg["split"]["val_end_q"],
        cfg=SequenceConfig(
            time_col=dcfg["time_column"], id_col=dcfg["id_column"],
            amount_col="TransactionAmt", product_col="ProductCD", max_len=32,
        ),
    )
    logger.info(f"sequences: train={len(sf.y_train)} val={len(sf.y_val)} "
                f"test={len(sf.y_test)} max_len={sf.seq_train.shape[1]}",
                extra={"run_id": run_id, "run": "transformer baseline", "seed": seed})

    tensors_tr = _to_tensors(sf.seq_train, sf.mask_train, sf.cur_train, sf.y_train, device)

    pos = int(sf.y_train.sum()); neg = len(sf.y_train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float, device=device)

    model = TemporalTransformer(TemporalConfig(
        n_step_features=sf.seq_train.shape[2],
        n_current_features=sf.cur_train.shape[1],
        max_len=sf.seq_train.shape[1],
    )).to(device)
    logger.info(f"params={model.n_parameters()}",
                extra={"run_id": run_id, "run": "transformer baseline", "seed": seed})

    _train(model, tensors_tr, sf.y_train, device, epochs=args.epochs,
           batch_size=args.batch_size, lr=args.lr, pos_weight=pos_weight)

    # Predict all folds.
    s_tr = _predict(model, sf.seq_train, sf.mask_train, sf.cur_train, args.batch_size, device)
    s_va = _predict(model, sf.seq_val, sf.mask_val, sf.cur_val, args.batch_size, device)
    s_te = _predict(model, sf.seq_test, sf.mask_test, sf.cur_test, args.batch_size, device)

    # Save predictions.
    preds_dir = run_dir / "preds" / "temporal"
    preds_dir.mkdir(parents=True, exist_ok=True)
    for name, ids, y, s in (
        ("train", sf.ids_train, sf.y_train, s_tr),
        ("val", sf.ids_val, sf.y_val, s_va),
        ("test", sf.ids_test, sf.y_test, s_te),
    ):
        pd.DataFrame({"TransactionID": ids, "y_true": y, "y_score": s}).to_csv(
            preds_dir / f"{name}.csv", index=False,
        )

    # Evaluate via AG-005 harness.
    report = build_report(sf.y_train, s_tr, sf.y_val, s_va, sf.y_test, s_te,
                         model_name="temporal")
    write_report(report, run_dir / "evaluation" / "temporal")

    # Profile.
    profile = _measure_profile(model, sf.seq_test, sf.mask_test, sf.cur_test,
                               device, args.batch_size)
    (run_dir / "temporal_profile.json").write_text(json.dumps(profile, indent=2))

    # Freeze temporal reference.
    frozen = {
        "synthetic_fixture": synthetic,
        "bounds": sf.bounds,
        "product_vocab_size": len(sf.product_vocab),
        "step_feature_names": sf.step_feature_names,
        "current_feature_names": sf.current_feature_names,
        "max_len": int(sf.seq_train.shape[1]),
        "n_parameters": model.n_parameters(),
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "pos_weight": float(pos_weight.item()),
        "device": device,
    }
    (run_dir / "frozen_temporal_reference.json").write_text(json.dumps(frozen, indent=2))

    torch.save(model.state_dict(), run_dir / "temporal_model.pt")

    write_manifest(
        run_dir=run_dir, cfg=cfg,
        dataset_version="synthetic_fixture" if synthetic else f"ieee_cis@{dcfg['transaction_file']}",
        split_boundaries=sf.bounds,
        extra={"run": "transformer baseline", "synthetic_fixture": synthetic,
               "n_parameters": model.n_parameters()},
    )
    logger.info("Transformer baseline complete", extra={"run_id": run_id, "run": "transformer baseline", "seed": seed})
    print(f"Wrote temporal reference and profile to {run_dir}")


if __name__ == "__main__":
    main()