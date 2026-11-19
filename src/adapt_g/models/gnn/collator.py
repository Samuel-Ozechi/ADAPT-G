"""Batch snapshots into fixed-shape tensors for GraphSAGE forward."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from adapt_g.graph.sampler import SampledSubgraph
from adapt_g.models.gnn.encoder import NODE_FEATURE_DIM, encode_snapshot_nodes


@dataclass
class SnapshotBatch:
    X: torch.Tensor           # (B, N_max, F_node)
    adj_index: torch.Tensor   # (B, N_max, max_deg) with -1 pad
    root_index: torch.Tensor  # (B,)
    y: torch.Tensor | None    # (B,) or None
    meta: list[dict]          # per-row: {"n_nodes": ..., "n_edges": ..., "is_isolated": ...}


def collate_snapshots(
    snapshots: list[tuple[SampledSubgraph, float, int | None]],  # (sg, dt_t, label)
    k_hops: int = 2,
    pad_to: int | None = None,
) -> SnapshotBatch:
    B = len(snapshots)
    n_max = pad_to or max(len(sg.nodes) for sg, _, _ in snapshots)
    max_deg = 0
    for sg, _, _ in snapshots:
        deg = np.zeros(len(sg.nodes), dtype=int)
        idx = {nid: i for i, nid in enumerate(sg.nodes)}
        for e in sg.edges:
            if e.src in idx and e.dst in idx:
                deg[idx[e.src]] += 1
                deg[idx[e.dst]] += 1
        if len(deg):
            max_deg = max(max_deg, int(deg.max()))
    max_deg = max(max_deg, 1)

    X = np.zeros((B, n_max, NODE_FEATURE_DIM), dtype=np.float32)
    adj = np.full((B, n_max, max_deg), -1, dtype=np.int64)
    root_idx = np.zeros(B, dtype=np.int64)
    y = np.zeros(B, dtype=np.float32)
    meta: list[dict] = []

    # Per-snapshot degree cursor for O(1) slot insertion.
    deg_cursor = np.zeros((B, n_max), dtype=np.int32)

    for b, (sg, dt_t, label) in enumerate(snapshots):
        feats = encode_snapshot_nodes(sg, dt_t, k_hops)
        n = len(sg.nodes)
        X[b, :n] = feats
        node_pos = {nid: i for i, nid in enumerate(sg.nodes)}
        for e in sg.edges:
            if e.src not in node_pos or e.dst not in node_pos:
                continue
            si = node_pos[e.src]
            di = node_pos[e.dst]
            for src_i, dst_i in ((si, di), (di, si)):
                slot = deg_cursor[b, src_i]
                if slot < max_deg:
                    adj[b, src_i, slot] = dst_i
                    deg_cursor[b, src_i] = slot + 1
        root_idx[b] = node_pos.get(sg.root, 0)
        if label is not None:
            y[b] = float(label)
        meta.append({"n_nodes": n, "n_edges": len(sg.edges),
                     "is_isolated": sg.is_isolated})

    batch = SnapshotBatch(
        X=torch.from_numpy(X),
        adj_index=torch.from_numpy(adj),
        root_index=torch.from_numpy(root_idx),
        y=torch.from_numpy(y) if any(l is not None for _, _, l in snapshots) else None,
        meta=meta,
    )
    return batch