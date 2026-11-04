"""Leakage-safe temporal sequence builder.

For each row, build the last L prior transactions of the same entity
(strictly < current TransactionDT, ties excluded). Pads at the front.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SequenceConfig:
    entity_col: str = "card1"
    time_col: str = "TransactionDT"
    id_col: str = "TransactionID"
    amount_col: str = "TransactionAmt"
    product_col: str = "ProductCD"
    max_len: int = 32
    # Reserved code for pad positions in product encoding.
    pad_product_code: int = 0

    def __post_init__(self) -> None:
        if self.max_len < 1:
            raise ValueError("max_len must be positive")
        if self.pad_product_code < 0:
            raise ValueError("pad_product_code must be non-negative")


@dataclass
class SequenceFolds:
    # arrays shape (n, L, n_step_features)
    seq_train: np.ndarray
    seq_val: np.ndarray
    seq_test: np.ndarray
    # masks shape (n, L); True at real (non-pad) positions
    mask_train: np.ndarray
    mask_val: np.ndarray
    mask_test: np.ndarray
    # per-row "current transaction" features shape (n, n_current_features)
    cur_train: np.ndarray
    cur_val: np.ndarray
    cur_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    ids_train: np.ndarray
    ids_val: np.ndarray
    ids_test: np.ndarray
    product_vocab: dict[str, int] = field(default_factory=dict)
    step_feature_names: list[str] = field(default_factory=list)
    current_feature_names: list[str] = field(default_factory=list)
    bounds: dict = field(default_factory=dict)


def _build_product_vocab(
    train_df: pd.DataFrame, col: str, pad_code: int = 0,
) -> dict[str, int]:
    vals = train_df[col].astype("string").fillna("__MISSING__").unique().tolist()
    # Keep padding and unseen categories distinct from real categories.
    vocab = {"__PAD__": pad_code}
    next_code = 0
    for token in ("__UNSEEN__", *sorted(vals)):
        if token in vocab:
            continue
        while next_code == pad_code or next_code in vocab.values():
            next_code += 1
        vocab[token] = next_code
        next_code += 1
    return vocab


def _encode_products(s: pd.Series, vocab: dict[str, int]) -> np.ndarray:
    return (
        s.astype("string").fillna("__MISSING__").map(vocab)
        .fillna(vocab["__UNSEEN__"]).astype("int64").to_numpy()
    )


def _build_sequences_for_fold(
    df: pd.DataFrame,
    cfg: SequenceConfig,
    product_vocab: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (sequences, masks, current_features) arrays in df row order."""
    n = len(df)
    L = cfg.max_len
    # Keep local row positions for restoring output order; the incoming index
    # may contain fold positions rather than 0..n-1.
    df = df.reset_index(drop=True).copy()
    df["__orig_idx"] = np.arange(n)
    df = df.sort_values([cfg.entity_col, cfg.time_col, cfg.id_col],
                        kind="mergesort").reset_index(drop=True)

    amt = df[cfg.amount_col].astype(float).to_numpy()
    # Fixed transform keeps a transaction's feature values invariant when
    # later rows are added to the dataset.
    amt_scaled = np.log1p(np.maximum(amt, 0.0))
    dt = df[cfg.time_col].astype(float).to_numpy()
    prod = _encode_products(df[cfg.product_col], product_vocab)

    entities = df[cfg.entity_col].to_numpy()
    # For each entity, iterate its rows in time order.
    seq_out = np.zeros((n, L, 4), dtype=np.float32)
    seq_out[:, :, 3] = cfg.pad_product_code
    mask_out = np.zeros((n, L), dtype=bool)
    cur_out = np.zeros((n, 4), dtype=np.float32)

    order = np.arange(len(df))
    entity_to_rows: dict = {}
    for pos in order:
        entity = entities[pos]
        key = ("__missing_entity__", pos) if pd.isna(entity) else entity
        entity_to_rows.setdefault(key, []).append(pos)

    for ent_rows in entity_to_rows.values():
        # Rows are sorted by time then ID. Search the timestamp boundary once
        # per row to exclude all ties and avoid scanning arbitrarily long tie
        # groups (IEEE-CIS contains many repeated timestamps).
        entity_dt = dt[ent_rows]
        for pos in ent_rows:
            t_now = dt[pos]
            end = int(np.searchsorted(entity_dt, t_now, side="left"))
            start = max(0, end - L)
            prior_positions = ent_rows[start:end]
            m = len(prior_positions)
            if m > 0:
                gaps = (t_now - dt[prior_positions]).astype(np.float32)
                for ii, p in enumerate(prior_positions):
                    seq_out[pos, L - m + ii, 0] = amt_scaled[p]
                    seq_out[pos, L - m + ii, 1] = gaps[ii]
                    seq_out[pos, L - m + ii, 2] = np.log1p(max(gaps[ii], 0.0))
                    seq_out[pos, L - m + ii, 3] = float(prod[p])
                    mask_out[pos, L - m + ii] = True
            cur_out[pos, 0] = amt_scaled[pos]
            cur_out[pos, 1] = float(amt[pos] == 0.0)
            cur_out[pos, 2] = float(prod[pos])
            cur_out[pos, 3] = float(df.iloc[pos]["has_identity"]) if "has_identity" in df.columns else 0.0

    # Restore original row order via __orig_idx.
    orig_idx = df["__orig_idx"].to_numpy()
    out_seq = np.empty_like(seq_out)
    out_mask = np.empty_like(mask_out)
    out_cur = np.empty_like(cur_out)
    out_seq[orig_idx] = seq_out
    out_mask[orig_idx] = mask_out
    out_cur[orig_idx] = cur_out
    return out_seq, out_mask, out_cur


def build_sequence_folds(
    txn: pd.DataFrame,
    ident: pd.DataFrame,
    id_col: str = "TransactionID",
    time_col: str = "TransactionDT",
    target_col: str = "isFraud",
    train_end_q: float = 0.8,
    val_end_q: float = 0.9,
    cfg: SequenceConfig | None = None,
) -> SequenceFolds:
    from adapt_g.data.loader import join_identity
    from adapt_g.data.splits import chronological_split

    cfg = cfg or SequenceConfig(time_col=time_col, id_col=id_col)
    df = join_identity(txn, ident, id_col)
    train, val, test, bounds = chronological_split(
        df, time_col=time_col, id_col=id_col, target_col=target_col,
        train_end_q=train_end_q, val_end_q=val_end_q,
    )

    product_vocab = _build_product_vocab(train, cfg.product_col, cfg.pad_product_code)
    def build(d: pd.DataFrame):
        return _build_sequences_for_fold(d, cfg, product_vocab)

    # Build on the combined chronological rows so validation/test examples can
    # use earlier-fold transactions as history, never later ones. Statistics
    # and vocab remain training-only.
    ordered = pd.concat([train, val, test], axis=0).reset_index(drop=True)
    seq_all, mask_all, cur_all = build(ordered)
    n_train, n_val = len(train), len(val)
    seq_tr, seq_va, seq_te = np.split(seq_all, [n_train, n_train + n_val])
    mask_tr, mask_va, mask_te = np.split(mask_all, [n_train, n_train + n_val])
    cur_tr, cur_va, cur_te = np.split(cur_all, [n_train, n_train + n_val])

    return SequenceFolds(
        seq_train=seq_tr, seq_val=seq_va, seq_test=seq_te,
        mask_train=mask_tr, mask_val=mask_va, mask_test=mask_te,
        cur_train=cur_tr, cur_val=cur_va, cur_test=cur_te,
        y_train=train[target_col].astype(int).to_numpy(),
        y_val=val[target_col].astype(int).to_numpy(),
        y_test=test[target_col].astype(int).to_numpy(),
        ids_train=train[id_col].to_numpy(),
        ids_val=val[id_col].to_numpy(),
        ids_test=test[id_col].to_numpy(),
        product_vocab=product_vocab,
        step_feature_names=["log1p_amount", "gap_seconds", "log1p_gap", "product_code"],
        current_feature_names=["log1p_amount", "amount_zero", "product_code", "has_identity"],
        bounds=bounds.__dict__,
    )