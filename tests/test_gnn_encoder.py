import numpy as np
import pytest

from adapt_g.graph.builder import EntityEdge
from adapt_g.graph.sampler import SampledSubgraph
from adapt_g.models.gnn.encoder import NODE_FEATURE_DIM, NodeKind, encode_snapshot_nodes


def _sg(kinds: list[str], last_active: list[float]) -> SampledSubgraph:
    nodes = []
    for k in kinds:
        if k == "isolated":
            nodes.append("__ISOLATED__")
        else:
            nodes.append(f"{k}:1")
    la = {n: v for n, v in zip(nodes, last_active)}
    return SampledSubgraph(root=nodes[0], nodes=nodes, edges=[], node_last_active=la)


def test_shape():
    sg = _sg(["card", "addr", "email"], [100.0, 200.0, 50.0])
    X = encode_snapshot_nodes(sg, dt_t=300.0, k_hops=2)
    assert X.shape == (3, NODE_FEATURE_DIM)
    assert X.dtype == np.float32


def test_onehot_kind():
    sg = _sg(["card", "addr", "email", "isolated"], [0.0] * 4)
    X = encode_snapshot_nodes(sg, dt_t=100.0, k_hops=2)
    assert X[0, int(NodeKind.CARD)] == 1.0
    assert X[1, int(NodeKind.ADDR)] == 1.0
    assert X[2, int(NodeKind.EMAIL)] == 1.0
    assert X[3, int(NodeKind.ISOLATED)] == 1.0
    # Each row sums one-hot to 1.0 (excluding recency col).
    assert np.allclose(X[:, :6].sum(axis=1), 1.0)


def test_recency_feature():
    sg = _sg(["card"], [50.0])
    X = encode_snapshot_nodes(sg, dt_t=150.0, k_hops=2)
    assert np.isclose(X[0, 6], np.log1p(100.0), rtol=1e-6, atol=1e-8)


def test_isolated_recency_is_zero():
    sg = _sg(["isolated"], [-1.0])
    X = encode_snapshot_nodes(sg, dt_t=150.0, k_hops=2)
    assert X[0, 6] == 0.0