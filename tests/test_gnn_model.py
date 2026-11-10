import torch

from adapt_g.models.gnn import GNNConfig, GraphSAGE


def _batch(B=2, N=5, D=6):
    torch.manual_seed(0)
    X = torch.randn(B, N, 7)
    adj = torch.full((B, N, D), -1, dtype=torch.long)
    adj[:, 0, 0] = 1; adj[:, 1, 0] = 0  # bidirectional edge 0 <-> 1
    root = torch.zeros(B, dtype=torch.long)
    return X, adj, root


def test_forward_shape():
    m = GraphSAGE(GNNConfig(hidden_dim=16, n_layers=2))
    X, adj, root = _batch()
    out = m(X, adj, root)
    assert out.shape == (2,)


def test_isolated_node_has_no_neighbours():
    """If root has no neighbours, model still produces finite output."""
    m = GraphSAGE(GNNConfig(hidden_dim=16, n_layers=2)).eval()
    X = torch.randn(1, 3, 7)
    adj = torch.full((1, 3, 4), -1, dtype=torch.long)
    root = torch.tensor([0])
    with torch.no_grad():
        out = m(X, adj, root)
    assert torch.isfinite(out).all()


def test_gradients_reach_all_layers():
    m = GraphSAGE(GNNConfig(hidden_dim=16, n_layers=2))
    X, adj, root = _batch()
    loss = m(X, adj, root).sum()
    loss.backward()
    for prefix in ("input_proj", "layers.0", "layers.1", "head"):
        names = [n for n, p in m.named_parameters()
                 if n.startswith(prefix) and p.grad is not None and p.grad.abs().sum() > 0]
        assert names, prefix