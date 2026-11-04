"""Build temporal entity edges from IEEE-CIS transactions.

Produces, per fold, an edge list of entity co-occurrences with
(first_seen_dt, last_seen_dt) so the sampler can apply a strict-< cutoff
at query time.

Node namespaces:
  card:<card1>
  addr:<addr1>
  email:<P_emaildomain>
  device:<DeviceInfo>
  txn:<TransactionID>   (target transactions; no outgoing edges used at
                         inference, only for identification)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class GraphConfig:
    id_col: str = "TransactionID"
    time_col: str = "TransactionDT"
    target_col: str = "isFraud"
    card_col: str = "card1"
    addr_col: str = "addr1"
    email_col: str = "P_emaildomain"
    device_col: str = "DeviceInfo"
    # Whether to build device edges (identity table is sparse).
    use_device: bool = True


@dataclass
class EntityEdge:
    src: str          # node id, e.g. "card:1234"
    dst: str
    edge_type: str    # "card_card", "card_addr", "addr_email", "card_device"
    first_seen_dt: float
    last_seen_dt: float
    n_observations: int = 1


@dataclass
class GraphBuild:
    edges: list[EntityEdge]
    node_last_active: dict[str, float]  # node id -> max dt observed
    node_degree_train: dict[str, int] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


def _node(kind: str, value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (int, np.integer)) and kind != "card":
        return f"{kind}:{float(value)}"
    if isinstance(value, (float, np.floating)):
        return f"{kind}:{float(value)}"
    return f"{kind}:{value}"


def _aggregate_edges(df: pd.DataFrame, cfg: GraphConfig) -> list[EntityEdge]:
    """For each pair (kind_a, kind_b), build co-occurrence edges with
    temporal extents. Uses rowwise and timewise grouping to keep the graph
    consistent with the temporal snapshot tests.
    """
    edges: list[EntityEdge] = []

    def _pair_edges(kind_a: str, col_a: str, kind_b: str, col_b: str,
                    edge_type: str) -> None:
        sub = df[[col_a, col_b, cfg.time_col]].dropna()
        if sub.empty:
            return
        a = sub[col_a].map(lambda v: _node(kind_a, v))
        b = sub[col_b].map(lambda v: _node(kind_b, v))
        valid = a.notna() & b.notna()
        valid_idx = valid.to_numpy()
        if not valid_idx.any():
            return
        sub = sub.iloc[valid_idx].copy()
        sub["__src"] = a.to_numpy()[valid_idx]
        sub["__dst"] = b.to_numpy()[valid_idx]
        g = sub.groupby(["__src", "__dst"], sort=False)[cfg.time_col]
        agg = g.agg(["min", "max", "count"]).reset_index()
        for _, row in agg.iterrows():
            src, dst = row["__src"], row["__dst"]
            if kind_a > kind_b or (kind_a == kind_b and src > dst):
                src, dst = dst, src
            edges.append(EntityEdge(
                src=src, dst=dst, edge_type=edge_type,
                first_seen_dt=float(row["min"]),
                last_seen_dt=float(row["max"]),
                n_observations=int(row["count"]),
            ))

    def _same_kind_edges(kind: str, col: str, edge_type: str) -> None:
        if col not in df.columns:
            return
        sub = df[[col, cfg.time_col]].dropna().copy()
        if sub.empty:
            return
        time_groups = sub.groupby(cfg.time_col, sort=False)
        pair_seen: dict[tuple[str, str], dict[str, float | int]] = {}
        for dt, group in time_groups:
            vals = [
                _node(kind, v) for v in group[col].dropna().tolist() if _node(kind, v) is not None
            ]
            uniq_vals = sorted(set(vals), key=lambda x: str(x))
            for left, right in zip(uniq_vals, uniq_vals[1:]):
                pass
            for i, left in enumerate(uniq_vals):
                for right in uniq_vals[i + 1:]:
                    key = tuple(sorted((left, right)))
                    bucket = pair_seen.setdefault(key, {"first": float("inf"), "last": float("-inf"), "count": 0})
                    bucket["first"] = min(bucket["first"], float(dt))
                    bucket["last"] = max(bucket["last"], float(dt))
                    bucket["count"] = int(bucket["count"]) + 1
        for (src, dst), info in pair_seen.items():
            edges.append(EntityEdge(
                src=src, dst=dst, edge_type=edge_type,
                first_seen_dt=float(info["first"]),
                last_seen_dt=float(info["last"]),
                n_observations=int(info["count"]),
            ))

    _same_kind_edges("card", cfg.card_col, "card_card")
    _pair_edges("card", cfg.card_col, "addr", cfg.addr_col, "card_addr")
    _pair_edges("addr", cfg.addr_col, "email", cfg.email_col, "addr_email")
    if cfg.use_device and cfg.device_col in df.columns:
        _pair_edges("card", cfg.card_col, "device", cfg.device_col, "card_device")
    return edges


def _node_last_active(edges: list[EntityEdge]) -> dict[str, float]:
    out: dict[str, float] = {}
    for e in edges:
        out[e.src] = max(out.get(e.src, -1.0), e.last_seen_dt)
        out[e.dst] = max(out.get(e.dst, -1.0), e.last_seen_dt)
    return out


def build_temporal_graph(
    train_df: pd.DataFrame,
    cfg: GraphConfig | None = None,
    out_dir: Path | None = None,
) -> GraphBuild:
    """Build entity edges from the training fold only.

    The graph is built from training transactions because inference snapshots
    must be assembled from edges observable *before* each test transaction —
    and the training fold is the only fold whose edges can precede test rows.
    (This is intentional: the graph itself is a "training artifact" that
    encodes history; using it at inference is not leakage because each
    inference-time snapshot filters by strict-< dt.)

    A separate `all_fold_df` argument is provided in the driver so the caller
    can decide exactly which rows are eligible as edge sources. This function
    takes whichever fold(s) the caller designates — by contract, the caller
    must only pass rows that satisfy the strict-< rule for the target test
    transaction. The driver enforces this by passing train ∪ val for test
    inference and train for val inference.
    """
    cfg = cfg or GraphConfig()
    edges = _aggregate_edges(train_df, cfg)
    node_last_active = _node_last_active(edges)

    stats = {
        "n_edges": len(edges),
        "n_nodes": len(node_last_active),
        "edge_type_counts": {},
        "node_kind_counts": {},
    }
    for e in edges:
        stats["edge_type_counts"][e.edge_type] = stats["edge_type_counts"].get(e.edge_type, 0) + 1
    for nid in node_last_active:
        kind = nid.split(":", 1)[0]
        stats["node_kind_counts"][kind] = stats["node_kind_counts"].get(kind, 0) + 1

    build = GraphBuild(edges=edges, node_last_active=node_last_active, stats=stats)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "graph_statistics.json").write_text(
            __import__("json").dumps(stats, indent=2)
        )
    return build