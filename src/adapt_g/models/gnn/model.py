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
        # h: (N, H); adj_mean: (N, H) = mean of neighbours (or h_i for isolated).
        out = self.w_self(h) + self.w_nbr(adj_mean)
        out = self.norm(torch.relu(self.drop(out)))
        return out


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
            # Gather neighbour embeddings: adj_index (B, N, max_deg).
            # Mask -1 -> 0 by adding a zero row per batch.
            safe = adj_index.clamp(min=0)
            neigh = torch.gather(
                h.unsqueeze(2).expand(-1, -1, safe.size(-1), -1),
                1,
                safe.unsqueeze(-1).expand(-1, -1, -1, h.size(-1)),
            )  # (B, N, max_deg, H)
            valid = (adj_index >= 0).unsqueeze(-1).float()  # (B, N, max_deg, 1)
            denom = valid.sum(dim=2).clamp(min=1.0)
            adj_mean = (neigh * valid).sum(dim=2) / denom  # (B, N, H)
            # For nodes with no neighbours, adj_mean is zero; combine with self.
            h = layer(h, adj_mean)
        # Root embedding.
        root_h = torch.gather(h, 1, root_index.unsqueeze(1).unsqueeze(2).expand(-1, 1, h.size(-1))).squeeze(1)
        return self.head(root_h).squeeze(-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)