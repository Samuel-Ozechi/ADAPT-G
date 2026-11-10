"""Train the GNN-only baseline (AG-009).

Usage:
  python scripts/train_gnn.py \
    --config configs/base.yaml \
    --config configs/data/ieee_cis.yaml
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

from adapt_g.config import load_config
from adapt_g.data.loader import join_identity, load_identity, load_transactions
from adapt_g.data.splits import chronological_split
from adapt_g.evaluation.report import build_report, write_report
from adapt_g.graph.builder import GraphConfig, build_temporal_graph
from adapt_g.graph.sampler import SamplerConfig, sample_neighbourhood
from adapt_g.logging_utils import get_logger
from adapt_g.manifest import new_run_id, write_manifest
from adapt_g.models.gnn import GNNConfig, GraphSAGE, collate_snapshots
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
        "addr1": rng.integers(100, 200, size=n).astype(float),
        "P_emaildomain": rng.choice(["a.com", "b.com", None], size=n),
    })
    ident = pd.DataFrame({
        "TransactionID": txn["TransactionID"].iloc[::5],
        "DeviceInfo": rng.choice(["devA", "devB", None],
                                 size=len(txn["TransactionID"].iloc[::5])),
    })
    return txn, ident


def _make_snapshots(df: pd.DataFrame, build, cfg, sampler_cfg: SamplerConfig,
                    time_col: str, label_col: str | None) -> list[tuple]:
    """Return list of (sg, dt_t, label_or_None) in df row order."""
    out = []
    for _, row in df.iterrows():
        card = row["card1"] if pd.notna(row["card1"]) else None
        sg = sample_neighbourhood(build, card_id=card,
                                  dt_t=float(row[time_col]), cfg=sampler_cfg)
        label = int(row[label_col]) if label_col and label_col in row else None
        out.append((sg, float(row[time_col]), label))
    return out


def _predict(model, snapshots, k_hops, batch_size, device):
    model.eval()
    scores = np.zeros(len(snapshots), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(snapshots), batch_size):
            chunk = snapshots[i:i + batch_size]
            batch = collate_snapshots(chunk, k_hops=k_hops)
            logits = model(batch.X.to(device), batch.adj_index.to(device),
                           batch.root_index.to(device))
            scores[i:i + len(chunk)] = torch.sigmoid(logits).cpu().numpy()
    return scores


def _train(model, snapshots, k_hops, device, epochs, batch_size, lr, pos_weight):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    n = len(snapshots)
    idx = np.arange(n)
    model.train()
    for ep in range(epochs):
        np.random.shuffle(idx)
        for i in range(0, n, batch_size):
            chunk = [snapshots[j] for j in idx[i:i + batch_size]]
            batch = collate_snapshots(chunk, k_hops=k_hops)
            logits = model(batch.X.to(device), batch.adj_index.to(device),
                           batch.root_index.to(device))
            loss = loss_fn(logits, batch.y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()


def _connectivity_analysis(snapshots, scores, labels, threshold, sampler_cfg):
    """Bucket by snapshot size and report precision/recall at threshold."""
    buckets = {"isolated": [], "sparse": [], "dense": []}
    for (sg, _, _), s, y in zip(snapshots, scores, labels):
        if sg.is_isolated:
            buckets["isolated"].append((s, y))
        elif len(sg.nodes) < 20:
            buckets["sparse"].append((s, y))
        else:
            buckets["dense"].append((s, y))
    out = {"threshold": threshold, "k_hops": sampler_cfg.k_hops,
           "fan_out": sampler_cfg.fan_out, "buckets": {}}
    for name, pairs in buckets.items():
        if not pairs:
            out["buckets"][name] = {"n": 0}
            continue
        s = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
        pred = (s >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        out["buckets"][name] = {"n": len(pairs), "precision": precision,
                                "recall": recall, "tp": tp, "fp": fp, "fn": fn}
    return out


def _cost_report(snapshots, batch_size, device, model, sampler_cfg, k_hops):
    model.eval()
    t0 = time.perf_counter()
    _predict(model, snapshots[:2048], k_hops, batch_size, device)
    sampling_time = time.perf_counter() - t0

    times = []
    with torch.no_grad():
        for _ in range(3):
            t0 = time.perf_counter()
            _predict(model, snapshots, k_hops, batch_size, device)
            times.append(time.perf_counter() - t0)
    mean_lat = float(np.mean(times))
    return {
        "n_snapshots": len(snapshots),
        "sampling_plus_forward_2048_s": float(sampling_time),
        "full_forward_mean_s": mean_lat,
        "full_forward_std_s": float(np.std(times)),
        "batch_size": batch_size,
        "device": device,
        "n_parameters": model.n_parameters(),
        "k_hops": sampler_cfg.k_hops, "fan_out": sampler_cfg.fan_out,
    }


def _load_cli_config(argv: list[str] | None = None) -> tuple[dict, argparse.Namespace]:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", action="append", default=None,
        help="YAML config to load; repeat to deep-merge configs in order",
    )
    ap.add_argument("--extra-config", action="append", default=[])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args(argv)

    config_paths = args.config or ["configs/base.yaml"]
    config_paths.extend(args.extra_config)
    return load_config(*config_paths), args


def main() -> None:
    cfg, args = _load_cli_config()
    set_seed(int(cfg["project"]["seed"]))
    torch.manual_seed(int(cfg["project"]["seed"]))
    run_id = new_run_id()
    run_dir = Path(cfg["paths"]["runs"]) / run_id
    logger = get_logger("adapt_g.ag009", run_dir / "run.log",
                        cfg["logging"]["level"], cfg["logging"]["json"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dcfg = cfg["data"]
    data_dir = Path(cfg["paths"]["data_raw"])
    synthetic = False
    try:
        txn = load_transactions(data_dir, dcfg["transaction_file"])
        ident = load_identity(data_dir, dcfg["identity_file"])
    except FileNotFoundError:
        logger.info("raw data absent; using synthetic fixture",
                    extra={"run_id": run_id, "ticket": "AG-009"})
        txn, ident = _synthetic_fixture()
        synthetic = True

    df = join_identity(txn, ident, dcfg["id_column"])
    train, val, test, bounds = chronological_split(
        df, time_col=dcfg["time_column"], id_col=dcfg["id_column"],
        target_col=dcfg["target_column"],
        train_end_q=cfg["split"]["train_end_q"], val_end_q=cfg["split"]["val_end_q"],
    )

    # Build graph from train ∪ val (visible to test snapshots with strict-<).
    build = build_temporal_graph(
        pd.concat([train, val], ignore_index=True),
        GraphConfig(id_col=dcfg["id_column"], time_col=dcfg["time_column"]),
    )

    sampler_cfg = SamplerConfig(k_hops=2, fan_out=10)
    logger.info("building snapshots...", extra={"run_id": run_id, "ticket": "AG-009"})
    train_snaps = _make_snapshots(train, build, cfg, sampler_cfg,
                                  dcfg["time_column"], dcfg["target_column"])
    val_snaps = _make_snapshots(val, build, cfg, sampler_cfg,
                                dcfg["time_column"], dcfg["target_column"])
    test_snaps = _make_snapshots(test, build, cfg, sampler_cfg,
                                 dcfg["time_column"], dcfg["target_column"])
    logger.info(f"snapshots: train={len(train_snaps)} val={len(val_snaps)} test={len(test_snaps)}",
                extra={"run_id": run_id, "ticket": "AG-009"})

    pos = int(sum(1 for _, _, l in train_snaps if l == 1))
    neg = len(train_snaps) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float, device=device)

    model = GraphSAGE(GNNConfig()).to(device)
    _train(model, train_snaps, sampler_cfg.k_hops, device,
           args.epochs, args.batch_size, args.lr, pos_weight)

    s_tr = _predict(model, train_snaps, sampler_cfg.k_hops, args.batch_size, device)
    s_va = _predict(model, val_snaps, sampler_cfg.k_hops, args.batch_size, device)
    s_te = _predict(model, test_snaps, sampler_cfg.k_hops, args.batch_size, device)

    preds_dir = run_dir / "preds" / "gnn"
    preds_dir.mkdir(parents=True, exist_ok=True)
    for name, df_fold, scores in (("train", train, s_tr), ("val", val, s_va), ("test", test, s_te)):
        pd.DataFrame({
            "TransactionID": df_fold[dcfg["id_column"]].values,
            "y_true": df_fold[dcfg["target_column"]].values,
            "y_score": scores,
        }).to_csv(preds_dir / f"{name}.csv", index=False)

    report = build_report(
        train[dcfg["target_column"]].values, s_tr,
        val[dcfg["target_column"]].values, s_va,
        test[dcfg["target_column"]].values, s_te,
        model_name="gnn",
    )
    write_report(report, run_dir / "evaluation" / "gnn")

    # Connectivity analysis on test.
    threshold = report["threshold_selection"]["threshold"]
    connectivity = _connectivity_analysis(
        test_snaps, s_te, test[dcfg["target_column"]].values, threshold, sampler_cfg
    )
    (run_dir / "gnn_connectivity_analysis.json").write_text(json.dumps(connectivity, indent=2))

    # Cost report.
    cost = _cost_report(test_snaps, args.batch_size, device, model, sampler_cfg, sampler_cfg.k_hops)
    cost["sampling_is_inference_path"] = True
    (run_dir / "gnn_cost_report.json").write_text(json.dumps(cost, indent=2))

    # Freeze reference.
    frozen = {
        "synthetic_fixture": synthetic,
        "bounds": bounds.__dict__,
        "n_parameters": model.n_parameters(),
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "pos_weight": float(pos_weight.item()),
        "device": device,
        "sampler": {"k_hops": sampler_cfg.k_hops, "fan_out": sampler_cfg.fan_out},
    }
    (run_dir / "frozen_gnn_reference.json").write_text(json.dumps(frozen, indent=2))
    torch.save(model.state_dict(), run_dir / "gnn_model.pt")

    write_manifest(
        run_dir=run_dir, cfg=cfg,
        dataset_version="synthetic_fixture" if synthetic else f"ieee_cis@{dcfg['transaction_file']}",
        split_boundaries=bounds.__dict__,
        extra={"ticket": "AG-009", "synthetic_fixture": synthetic,
               "n_parameters": model.n_parameters()},
    )
    logger.info("AG-009 complete", extra={"run_id": run_id, "ticket": "AG-009"})
    print(f"Wrote GNN reference and reports to {run_dir}")


if __name__ == "__main__":
    main()