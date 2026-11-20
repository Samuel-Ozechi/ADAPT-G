"""Recreate validation/test predictions from a saved GNN run.

Usage:
  python scripts/export_gnn_predictions.py --run-id gnn-<date>-<time>-<hash>

Writes runs/<run-id>/preds/gnn/{val,test}.csv using the saved checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adapt_g.data.loader import join_identity, load_identity, load_transactions
from adapt_g.data.splits import chronological_split
from adapt_g.graph.builder import GraphConfig, build_temporal_graph
from adapt_g.graph.sampler import SamplerConfig, build_adjacency
from adapt_g.models.gnn import GNNConfig, GraphSAGE, collate_snapshots
from adapt_g.seeding import set_seed
from train_gnn import _make_snapshots


def _predict_fold(model, fold, build, sampler_cfg, adjacency, time_col,
                  label_col, batch_size, device):
    model.eval()
    scores = np.empty(len(fold), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(fold), batch_size):
            end = min(start + batch_size, len(fold))
            snapshots = _make_snapshots(
                fold.iloc[start:end], build, sampler_cfg, time_col, label_col,
                adjacency,
            )
            batch = collate_snapshots(snapshots, k_hops=sampler_cfg.k_hops)
            logits = model(
                batch.X.to(device), batch.adj_index.to(device),
                batch.root_index.to(device),
            )
            scores[start:end] = torch.sigmoid(logits).cpu().numpy()
    return scores


def _get_device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is unavailable")
        mps = getattr(torch.backends, "mps", None)
        if requested == "mps" and (mps is None or not mps.is_available()):
            raise SystemExit("MPS was requested but is unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.runs_dir) / args.run_id
    manifest_path = run_dir / "run_manifest.json"
    checkpoint_path = run_dir / "gnn_model.pt"
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise SystemExit(f"run manifest or GNN checkpoint missing under {run_dir}")

    manifest = json.loads(manifest_path.read_text())
    cfg = manifest["config"]
    dcfg = cfg["data"]
    set_seed(int(cfg["project"]["seed"]))
    torch.manual_seed(int(cfg["project"]["seed"]))
    device = _get_device(args.device)

    data_dir = Path(cfg["paths"]["data_raw"])
    transaction_cols = list(dict.fromkeys([
        dcfg["id_column"], dcfg["target_column"], dcfg["time_column"],
        "TransactionAmt", "ProductCD", "card1", "addr1", "P_emaildomain",
    ]))
    identity_cols = list(dict.fromkeys([dcfg["id_column"], "DeviceInfo"]))
    txn = load_transactions(data_dir, dcfg["transaction_file"], usecols=transaction_cols)
    ident = load_identity(data_dir, dcfg["identity_file"], usecols=identity_cols)
    df = join_identity(txn, ident, dcfg["id_column"])
    keep_cols = [
        dcfg["id_column"], dcfg["target_column"], dcfg["time_column"],
        "card1", "addr1", "P_emaildomain", "DeviceInfo",
    ]
    df = df[[col for col in keep_cols if col in df.columns]]
    train, val, test, bounds = chronological_split(
        df,
        time_col=dcfg["time_column"],
        id_col=dcfg["id_column"],
        target_col=dcfg["target_column"],
        train_end_q=cfg["split"]["train_end_q"],
        val_end_q=cfg["split"]["val_end_q"],
    )

    expected_bounds = manifest.get("split_boundaries") or {}
    for key in ("n_train", "n_val", "n_test"):
        if key in expected_bounds and getattr(bounds, key) != expected_bounds[key]:
            raise SystemExit(f"dataset split mismatch at {key}; refusing to score checkpoint")

    graph_df = pd.concat([train, val], ignore_index=True)
    build = build_temporal_graph(
        graph_df,
        GraphConfig(id_col=dcfg["id_column"], time_col=dcfg["time_column"]),
    )
    adjacency = build_adjacency(build)
    sampler_info = json.loads((run_dir / "frozen_gnn_reference.json").read_text()).get("sampler", {})
    sampler_cfg = SamplerConfig(
        k_hops=int(sampler_info.get("k_hops", 2)),
        fan_out=int(sampler_info.get("fan_out", 10)),
    )
    batch_size = args.batch_size or int(
        json.loads((run_dir / "frozen_gnn_reference.json").read_text()).get("batch_size", 256)
    )

    model = GraphSAGE(GNNConfig()).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    preds_dir = run_dir / "preds" / "gnn"
    preds_dir.mkdir(parents=True, exist_ok=True)
    for name, fold in (("val", val), ("test", test)):
        print(f"Scoring {name} fold ({len(fold):,} rows) on {device}...")
        scores = _predict_fold(
            model, fold, build, sampler_cfg, adjacency,
            dcfg["time_column"], dcfg["target_column"], batch_size, device,
        )
        pd.DataFrame({
            "TransactionID": fold[dcfg["id_column"]].values,
            "y_true": fold[dcfg["target_column"]].values,
            "y_score": scores,
        }).to_csv(preds_dir / f"{name}.csv", index=False)
        print(f"Wrote {preds_dir / f'{name}.csv'}")


if __name__ == "__main__":
    main()
