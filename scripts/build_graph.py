"""AG-008: build the temporal graph from the training fold and report stats.

Usage:
  python scripts/build_graph.py --config configs/base.yaml --config configs/data/ieee_cis.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adapt_g.config import load_config
from adapt_g.data.loader import join_identity, load_identity, load_transactions
from adapt_g.data.splits import chronological_split
from adapt_g.graph.builder import GraphConfig, build_temporal_graph
from adapt_g.graph.sampler import SamplerConfig, sample_neighbourhood
from adapt_g.logging_utils import get_logger
from adapt_g.manifest import new_run_id, write_manifest
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


def _load_cli_config(argv: list[str] | None = None) -> tuple[dict, argparse.Namespace]:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", action="append", default=None,
        help="YAML config to load; repeat to deep-merge configs in order",
    )
    ap.add_argument("--extra-config", action="append", default=[])
    ap.add_argument("--sample-queries", type=int, default=200)
    args = ap.parse_args(argv)

    config_paths = args.config or ["configs/base.yaml"]
    config_paths.extend(args.extra_config)
    return load_config(*config_paths), args


def main() -> None:
    cfg, args = _load_cli_config()
    set_seed(int(cfg["project"]["seed"]))
    run_id = new_run_id("graph")
    run_dir = Path(cfg["paths"]["runs"]) / run_id
    logger = get_logger("adapt_g.ag008", run_dir / "run.log",
                        cfg["logging"]["level"], cfg["logging"]["json"])

    dcfg = cfg["data"]
    data_dir = Path(cfg["paths"]["data_raw"])
    synthetic = False
    txn = load_transactions(data_dir, dcfg["transaction_file"])
    ident = load_identity(data_dir, dcfg["identity_file"])
    

    df = join_identity(txn, ident, dcfg["id_column"])
    train, val, test, bounds = chronological_split(
        df, time_col=dcfg["time_column"], id_col=dcfg["id_column"],
        target_col=dcfg["target_column"],
        train_end_q=cfg["split"]["train_end_q"], val_end_q=cfg["split"]["val_end_q"],
    )

    # Build edges from train ∪ val; used at test inference with strict-< cutoff.
    # (Train-only edges would starve test snapshots of recency; val edges
    # are still strictly before test dt values because the split is
    # chronological.)
    build = build_temporal_graph(
        pd.concat([train, val], ignore_index=True),
        GraphConfig(id_col=dcfg["id_column"], time_col=dcfg["time_column"]),
        out_dir=run_dir,
    )
    logger.info(f"edges={build.stats['n_edges']} nodes={build.stats['n_nodes']}",
                extra={"run_id": run_id, "ticket": "AG-008"})

    # Sample neighbourhoods for a random subset of test rows to report
    # empirical neighbourhood sizes and isolated fraction.
    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    sample_n = min(args.sample_queries, len(test))
    sample_idx = rng.choice(len(test), size=sample_n, replace=False)
    sampler_cfg = SamplerConfig(k_hops=2, fan_out=10)

    n_isolated = 0
    n_nodes_list: list[int] = []
    n_edges_list: list[int] = []
    for i in sample_idx:
        row = test.iloc[i]
        card = row["card1"] if pd.notna(row["card1"]) else None
        sg = sample_neighbourhood(build, card_id=card, dt_t=float(row[dcfg["time_column"]]),
                                  cfg=sampler_cfg)
        if sg.is_isolated:
            n_isolated += 1
        n_nodes_list.append(len(sg.nodes))
        n_edges_list.append(len(sg.edges))

    stats_extra = {
        "sample_n": int(sample_n),
        "isolated_fraction": float(n_isolated / sample_n) if sample_n else 0.0,
        "nodes_per_snapshot_mean": float(np.mean(n_nodes_list)) if n_nodes_list else 0.0,
        "nodes_per_snapshot_std": float(np.std(n_nodes_list)) if n_nodes_list else 0.0,
        "edges_per_snapshot_mean": float(np.mean(n_edges_list)) if n_edges_list else 0.0,
        "edges_per_snapshot_std": float(np.std(n_edges_list)) if n_edges_list else 0.0,
        "k_hops": sampler_cfg.k_hops,
        "fan_out": sampler_cfg.fan_out,
        "synthetic_fixture": synthetic,
    }
    (run_dir / "graph_sampling_statistics.json").write_text(json.dumps(stats_extra, indent=2))

    write_manifest(
        run_dir=run_dir, cfg=cfg,
        dataset_version=f"ieee_cis@{dcfg['transaction_file']}",
        split_boundaries=bounds.__dict__,
        extra={"tag": "graph", "ticket": "AG-008"},
    )
    logger.info("Graph build and sampling complete", extra={"run_id": run_id, "run": "graph build"})
    print(f"Wrote graph statistics to {run_dir}")


if __name__ == "__main__":
    main()