"""Node feature encoding for graph snapshots.

Intentionally excludes raw tabular features of the target transaction —
the GNN-only baseline must derive signal solely from graph structure.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np

from adapt_g.graph.sampler import SampledSubgraph


class NodeKind(IntEnum):
    CARD = 0
    ADDR = 1
    EMAIL = 2
    DEVICE = 3
    ISOLATED = 4
    UNKNOWN = 5


KIND_TO_ONEHOT: dict[str, int] = {
    "card": NodeKind.CARD,
    "addr": NodeKind.ADDR,
    "email": NodeKind.EMAIL,
    "device": NodeKind.DEVICE,
    "__ISOLATED__": NodeKind.ISOLATED,
}

NODE_FEATURE_DIM = 7  # 6 one-hot + recency


def encode_snapshot_nodes(
    sg: SampledSubgraph,
    dt_t: float,
    k_hops: int,
    node_feature_dim: int = NODE_FEATURE_DIM,
) -> np.ndarray:
    """Return (n_nodes, node_feature_dim) float32 array.

    Features per node:
      - one-hot of kind (6 dims: card, addr, email, device, isolated, unknown)
      - log1p(dt_t - last_active)   (recency; 1 dim)
    Depth is not used (would inflate dense-node advantage).
    """
    n = len(sg.nodes)
    X = np.zeros((n, node_feature_dim), dtype=np.float32)
    if sg.node_kinds:
        kinds = np.array(sg.node_kinds, dtype=np.int32)
        X[np.arange(n), kinds] = 1.0
    else:
        # Fallback: compute kinds from node ids.
        for i, nid in enumerate(sg.nodes):
            if nid == "__ISOLATED__":
                X[i, NodeKind.ISOLATED] = 1.0
            else:
                prefix = nid.split(":", 1)[0]
                kind = KIND_TO_ONEHOT.get(prefix, NodeKind.UNKNOWN)
                X[i, int(kind)] = 1.0

    last_active = sg.node_last_active
    for i, nid in enumerate(sg.nodes):
        last = last_active.get(nid, -1.0)
        if last >= 0:
            X[i, 6] = float(np.log1p(max(0.0, dt_t - last)))
    return X