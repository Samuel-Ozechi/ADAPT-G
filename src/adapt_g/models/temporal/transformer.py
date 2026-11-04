"""Transformer over prior-transaction sequences + small current-feature MLP."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TemporalConfig:
    n_step_features: int = 4
    n_current_features: int = 4
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    max_len: int = 32

    def __post_init__(self) -> None:
        if self.n_heads < 1 or self.d_model < 1 or self.d_model % self.n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.n_layers < 1 or self.max_len < 1:
            raise ValueError("n_layers and max_len must be positive")
        if self.d_ff < 1 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("d_ff must be positive and dropout must be in [0, 1)")
        if self.n_step_features < 1 or self.n_current_features < 1:
            raise ValueError("feature dimensions must be positive")


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TemporalTransformer(nn.Module):
    """One module. One forward pass. Single checkpoint.

    Architecture:
    - Linear projection of step features -> d_model
    - Positional encoding + TransformerEncoder (masked)
    - Masked mean pooling over real steps
    - Concat with current-feature MLP embedding
    - Linear classifier -> scalar logit
    """

    def __init__(self, cfg: TemporalConfig):
        super().__init__()
        self.cfg = cfg
        self.step_proj = nn.Linear(cfg.n_step_features, cfg.d_model)
        self.pos = _PositionalEncoding(cfg.d_model, cfg.max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_layers, enable_nested_tensor=False,
        )
        self.cur_mlp = nn.Sequential(
            nn.Linear(cfg.n_current_features, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        # seq: (B, L, F_step); mask: (B, L) bool, True at real positions
        x = self.step_proj(seq)                      # (B, L, d)
        x = self.pos(x)
        # TransformerEncoder's src_key_padding_mask expects True at pad positions.
        pad_mask = ~mask
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        # Avoid all-masked attention rows; the original mask still excludes
        # these placeholder outputs from pooling.
        safe_pad_mask = pad_mask.clone()
        empty_rows = ~mask.any(dim=1)
        safe_pad_mask[empty_rows, 0] = False
        enc = self.encoder(x, src_key_padding_mask=safe_pad_mask)  # (B, L, d)
        # Masked mean pooling.
        m = mask.unsqueeze(-1).float()                # (B, L, 1)
        denom = m.sum(dim=1).clamp(min=1.0)
        pooled = (enc * m).sum(dim=1) / denom         # (B, d)
        cur_emb = self.cur_mlp(cur)                   # (B, d)
        z = torch.cat([pooled, cur_emb], dim=-1)      # (B, 2d)
        return self.head(z).squeeze(-1)               # (B,)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)