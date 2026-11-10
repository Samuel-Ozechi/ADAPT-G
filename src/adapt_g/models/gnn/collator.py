"""Batch snapshots into fixed-shape tensors for GraphSAGE forward."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from adapt_g.graph.sampler import SampledSubgraph
from adapt_g.models.gnn.encoder import NODE_FEATURE_DIM, encode_snapshot_nodes


EDGE_KIND_TO_IDX = {
    "card_card": 0, "card_addr": 1, "addr_email": 2, "card_device": 3,
}
EDGE_FEATURE_DIM = 6  # 4 one-hot + log n_obs + log recency


@dataclass
class SnapshotBatch:
    X: torch.Tensor           # (B, N_max, F_node)
    adj_index: torch.Tensor   # (B, N_max, max_deg) with -1 pad
    root_index: torch.Tensor  # (B,)
    node_mask: torch.Tensor   # (B, N_max) bool: True at real nodes
    y: torch.Tensor | None    # (B,) or None
    meta: list[dict]          # per-row: {"ids": ..., "n_nodes": ..., "n_edges": ...}


def collate_snapshots(
    snapshots: list[tuple[SampledSubgraph, float, int | None]],  # (sg, dt_t, label)
    k_hops: int = 2,
    pad_to: int | None = None,
) -> SnapshotBatch:
    """snapshots: list of (sg, dt_t, label_or_None). Pads to max size in batch."""
    B = len(snapshots)
    n_max = pad_to or max(len(sg.nodes) for sg, _, _ in snapshots)
    # Cap degree per node; F + 1 is enough for k=2, F=10; use a safe cap.
    max_deg = 0
    for sg, _, _ in snapshots:
        deg = np.zeros(len(sg.nodes), dtype=int)
        idx = {nid: i for i, nid in enumerate(sg.nodes)}
        for e in sg.edges:
            if e.src in idx and e.dst in idx:
                deg[idx[e.src]] += 1
                deg[idx[e.dst]] += 1
        max_deg = max(max_deg, int(deg.max()) if len(deg) else 0)
    max_deg = max(max_deg, 1)

    X = np.zeros((B, n_max, NODE_FEATURE_DIM), dtype=np.float32)
    adj = np.full((B, n_max, max_deg), -1, dtype=np.int64)
    root_idx = np.zeros(B, dtype=np.int64)
    node_mask = np.zeros((B, n_max), dtype=bool)
    y = np.zeros(B, dtype=np.float32)
    meta: list[dict] = []

    for b, (sg, dt_t, label) in enumerate(snapshots):
        feats = encode_snapshot_nodes(sg, dt_t, k_hops)
        n = len(sg.nodes)
        X[b, :n] = feats
        node_mask[b, :n] = True
        node_pos = {nid: i for i, nid in enumerate(sg.nodes)}
        for e in sg.edges:
            if e.src not in node_pos or e.dst not in node_pos:
                continue
            si, di = node_pos[e.src], node_pos[e.dst]
            for endpoint, other in ((si, di), (di, si)):
                slot = 0
                while slot < max_deg and adj[b, endpoint, slot] != -1:
                    if adj[b, endpoint, slot] == other:
                        break
                    slot += 1
                if slot < max_deg and adj[b, endpoint, slot] == -1:
                    adj[b, endpoint, slot] = other
        root_idx[b] = node_pos.get(sg.root, 0)
        if label is not None:
            y[b] = float(label)
        meta.append({"n_nodes": n, "n_edges": len(sg.edges),
                     "is_isolated": sg.is_isolated})

    batch = SnapshotBatch(
        X=torch.from_numpy(X),
        adj_index=torch.from_numpy(adj),
        root_index=torch.from_numpy(root_idx),
        node_mask=torch.from_numpy(node_mask),
        y=torch.from_numpy(y) if any(l is not None for _, _, l in snapshots) else None,
        meta=meta,
    )
    return batch