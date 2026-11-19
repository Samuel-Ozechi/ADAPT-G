"""GraphSAGE-style GNN with mean aggregation. No PyG dependency.

Forward:
  X (N, F_node) -> linear -> L x [mean-aggregate from neighbours; combine]
                -> root embedding -> MLP head -> scalar logit.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GNNConfig:
    node_feature_dim: int = 7
    edge_feature_dim: int = 6
    hidden_dim: int = 64
    n_layers: int = 2
    dropout: float = 0.1


class _SageLayer(nn.Module):
    """One GraphSAGE layer: h_i' = ReLU(W_self h_i + W_nbr mean_{j in N(i)} h_j)."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.w_self = nn.Linear(hidden_dim, hidden_dim)
        self.w_nbr = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, adj_mean: torch.Tensor) -> torch.Tensor:
        out = self.w_self(h) + self.w_nbr(adj_mean)
        out = self.norm(torch.relu(self.drop(out)))
        return out


def _mean_aggregate(h: torch.Tensor, adj_index: torch.Tensor) -> torch.Tensor:
    """Compute mean neighbour embedding for each node in each graph.

    h: (B, N, H) node embeddings.
    adj_index: (B, N, max_deg) with -1 for padding.

    Returns (B, N, H) mean neighbour embeddings.
    """
    B, N, H = h.shape
    max_deg = adj_index.size(-1)

    # Flatten batch+node dims so we can use a single index_select.
    h_flat = h.reshape(B * N, H)  # (B*N, H)
    valid = (adj_index >= 0)  # (B, N, max_deg)
    safe_idx = adj_index.clamp(min=0)  # (B, N, max_deg)

    # Add batch offset so each graph's node indices map to disjoint rows.
    offsets = torch.arange(B, device=h.device).unsqueeze(1).unsqueeze(2) * N
    global_idx = safe_idx + offsets  # (B, N, max_deg)

    neigh_h = h_flat[global_idx]  # (B, N, max_deg, H)
    neigh_h = neigh_h * valid.unsqueeze(-1).float()
    denom = valid.sum(dim=2).clamp(min=1.0).unsqueeze(-1)  # (B, N, 1)
    return neigh_h.sum(dim=2) / denom


class GraphSAGE(nn.Module):
    def __init__(self, cfg: GNNConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.node_feature_dim, cfg.hidden_dim)
        self.layers = nn.ModuleList([
            _SageLayer(cfg.hidden_dim, cfg.dropout) for _ in range(cfg.n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, X: torch.Tensor, adj_index: torch.Tensor,
                root_index: torch.Tensor) -> torch.Tensor:
        """X: (B, N, F_node); adj_index: (B, N, max_deg) with -1 pad;
        root_index: (B,) index of root node.

        Returns (B,) logits.
        """
        B, N, _ = X.shape
        h = self.input_proj(X)  # (B, N, H)
        for layer in self.layers:
            adj_mean = _mean_aggregate(h, adj_index)
            h = layer(h, adj_mean)
        # Root embedding via batched index_select.
        h_flat = h.reshape(B * N, -1)
        offsets = torch.arange(B, device=h.device) * N
        root_h = h_flat[root_index + offsets]  # (B, H)
        return self.head(root_h).squeeze(-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)