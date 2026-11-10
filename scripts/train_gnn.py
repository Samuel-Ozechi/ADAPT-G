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
from adapt_g.graph.sampler import (
    SamplerConfig,
    build_adjacency,
    sample_neighbourhood,
)
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


def _make_snapshots(
    df: pd.DataFrame,
    build,
    sampler_cfg: SamplerConfig,
    time_col: str,
    label_col: str | None,
    adjacency,
) -> list[tuple]:
    """Return list of (sg, dt_t, label_or_None) in df row order."""
    out = []
    for _, row in df.iterrows():
        card = row["card1"] if pd.notna(row["card1"]) else None
        sg = sample_neighbourhood(build, card_id=card,
                                  dt_t=float(row[time_col]), cfg=sampler_cfg,
                                  adjacency=adjacency)
        label = int(row[label_col]) if label_col and label_col in row else None
        out.append((sg, float(row[time_col]), label))
    return out


def _predict(
    model,
    df: pd.DataFrame,
    build,
    sampler_cfg: SamplerConfig,
    adjacency,
    time_col: str,
    label_col: str,
    batch_size: int,
    device: str,
):
    """Predict in bounded-memory chunks and return snapshot connectivity flags."""
    model.eval()
    scores = np.zeros(len(df), dtype=np.float32)
    isolated = np.zeros(len(df), dtype=bool)
    node_counts = np.zeros(len(df), dtype=np.int16)
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            end = min(i + batch_size, len(df))
            snapshots = _make_snapshots(
                df.iloc[i:end], build, sampler_cfg, time_col, label_col, adjacency
            )
            batch = collate_snapshots(snapshots, k_hops=sampler_cfg.k_hops)
            logits = model(batch.X.to(device), batch.adj_index.to(device),
                           batch.root_index.to(device))
            scores[i:end] = torch.sigmoid(logits).cpu().numpy()
            isolated[i:end] = [sg.is_isolated for sg, _, _ in snapshots]
            node_counts[i:end] = [len(sg.nodes) for sg, _, _ in snapshots]
    return scores, isolated, node_counts


def _train(
    model,
    df: pd.DataFrame,
    build,
    sampler_cfg: SamplerConfig,
    adjacency,
    time_col: str,
    label_col: str,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    pos_weight: torch.Tensor,
):
    """Sample and train one batch at a time instead of retaining all snapshots."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    n = len(df)
    idx = np.arange(n)
    for ep in range(epochs):
        model.train()
        np.random.shuffle(idx)
        for i in range(0, n, batch_size):
            rows = df.iloc[idx[i:i + batch_size]]
            snapshots = _make_snapshots(
                rows, build, sampler_cfg, time_col, label_col, adjacency
            )
            batch = collate_snapshots(snapshots, k_hops=sampler_cfg.k_hops)
            logits = model(batch.X.to(device), batch.adj_index.to(device),
                           batch.root_index.to(device))
            loss = loss_fn(logits, batch.y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Completed GNN epoch {ep + 1}/{epochs}")


def _connectivity_analysis(isolated, node_counts, scores, labels, threshold,
                           sampler_cfg):
    """Bucket by snapshot size and report precision/recall at threshold."""
    bucket_masks = {
        "isolated": isolated,
        "sparse": (~isolated) & (node_counts < 20),
        "dense": (~isolated) & (node_counts >= 20),
    }
    out = {"threshold": threshold, "k_hops": sampler_cfg.k_hops,
           "fan_out": sampler_cfg.fan_out, "buckets": {}}
    for name, mask in bucket_masks.items():
        if not mask.any():
            out["buckets"][name] = {"n": 0}
            continue
        s = scores[mask]
        y = labels[mask]
        pred = (s >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        out["buckets"][name] = {"n": int(mask.sum()), "precision": precision,
                                "recall": recall, "tp": tp, "fp": fp, "fn": fn}
    return out


def _cost_report(df, build, batch_size, device, model, sampler_cfg,
                 adjacency, time_col, label_col):
    model.eval()
    t0 = time.perf_counter()
    _predict(model, df.iloc[:2048], build, sampler_cfg, adjacency,
             time_col, label_col, batch_size, device)
    sampling_plus_forward_time = time.perf_counter() - t0

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _predict(model, df, build, sampler_cfg, adjacency,
                 time_col, label_col, batch_size, device)
        times.append(time.perf_counter() - t0)
    mean_lat = float(np.mean(times))
    return {
        "n_snapshots": len(df),
        "sampling_plus_forward_2048_s": float(sampling_plus_forward_time),
        "full_inference_mean_s": mean_lat,
        "full_inference_std_s": float(np.std(times)),
        "timing_includes_sampling": True,
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
    transaction_cols = list(dict.fromkeys([
        dcfg["id_column"], dcfg["target_column"], dcfg["time_column"],
        "TransactionAmt", "ProductCD", "card1", "addr1", "P_emaildomain",
    ]))
    identity_cols = list(dict.fromkeys([dcfg["id_column"], "DeviceInfo"]))
    try:
        txn = load_transactions(
            data_dir, dcfg["transaction_file"], usecols=transaction_cols
        )
        ident = load_identity(
            data_dir, dcfg["identity_file"], usecols=identity_cols
        )
    except FileNotFoundError:
        logger.info("raw data absent; using synthetic fixture",
                    extra={"run_id": run_id, "ticket": "AG-009"})
        txn, ident = _synthetic_fixture()
        synthetic = True

    df = join_identity(txn, ident, dcfg["id_column"])
    del txn, ident
    keep_cols = [
        dcfg["id_column"], dcfg["target_column"], dcfg["time_column"],
        "card1", "addr1", "P_emaildomain", "DeviceInfo",
    ]
    df = df[[col for col in keep_cols if col in df.columns]]
    train, val, test, bounds = chronological_split(
        df, time_col=dcfg["time_column"], id_col=dcfg["id_column"],
        target_col=dcfg["target_column"],
        train_end_q=cfg["split"]["train_end_q"], val_end_q=cfg["split"]["val_end_q"],
    )

    # Build graph from train ∪ val (visible to test snapshots with strict-<).
    graph_df = pd.concat([train, val], ignore_index=True)
    build = build_temporal_graph(
        graph_df,
        GraphConfig(id_col=dcfg["id_column"], time_col=dcfg["time_column"]),
    )
    del graph_df
    adjacency = build_adjacency(build)

    sampler_cfg = SamplerConfig(k_hops=2, fan_out=10)
    logger.info("starting chunked snapshot training",
                extra={"run_id": run_id, "ticket": "AG-009"})

    pos = int(train[dcfg["target_column"]].sum())
    neg = len(train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float, device=device)

    model = GraphSAGE(GNNConfig()).to(device)
    _train(model, train, build, sampler_cfg, adjacency,
           dcfg["time_column"], dcfg["target_column"], device,
           args.epochs, args.batch_size, args.lr, pos_weight)

    s_tr, _, _ = _predict(
        model, train, build, sampler_cfg, adjacency, dcfg["time_column"],
        dcfg["target_column"], args.batch_size, device,
    )
    s_va, _, _ = _predict(
        model, val, build, sampler_cfg, adjacency, dcfg["time_column"],
        dcfg["target_column"], args.batch_size, device,
    )
    s_te, test_isolated, test_node_counts = _predict(
        model, test, build, sampler_cfg, adjacency, dcfg["time_column"],
        dcfg["target_column"], args.batch_size, device,
    )

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
        test_isolated, test_node_counts, s_te,
        test[dcfg["target_column"]].values, threshold, sampler_cfg,
    )
    (run_dir / "gnn_connectivity_analysis.json").write_text(json.dumps(connectivity, indent=2))

    # Cost report.
    cost = _cost_report(
        test, build, args.batch_size, device, model, sampler_cfg, adjacency,
        dcfg["time_column"], dcfg["target_column"],
    )
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