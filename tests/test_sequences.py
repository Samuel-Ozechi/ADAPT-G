"""Leakage and correctness tests for the sequence builder."""
import numpy as np
import pandas as pd
import pytest

from adapt_g.features.sequences import SequenceConfig, build_sequence_folds


def _toy(n=20):
    return pd.DataFrame({
        "TransactionID": np.arange(n),
        "isFraud": np.zeros(n, dtype=int),
        "TransactionDT": np.arange(n) * 10,
        "TransactionAmt": np.arange(n, dtype=float),
        "ProductCD": ["W"] * n,
        "card1": [7] * n,  # single entity
    })


def _ident(n=20):
    return pd.DataFrame({"TransactionID": np.arange(n)})


def test_sequence_uses_only_prior_rows():
    txn = _toy(10)
    sf = build_sequence_folds(txn, _ident(10), train_end_q=0.6, val_end_q=0.8,
                              cfg=SequenceConfig(max_len=5))
    # For a row with k prior same-entity rows, mask.sum() must be min(k, L).
    # Train fold is first 6 rows (quantile 0.6 -> dt <= 50).
    full_mask = np.concatenate([sf.mask_train, sf.mask_val, sf.mask_test])
    full_ids = np.concatenate([sf.ids_train, sf.ids_val, sf.ids_test])
    order = np.argsort(full_ids)
    counts = full_mask.sum(axis=1)[order]
    expected = np.minimum(np.arange(10), 5)
    assert np.array_equal(counts, expected)


def test_no_future_influence_on_sequences():
    txn = _toy(10)
    cfg = SequenceConfig(max_len=5)
    sf_full = build_sequence_folds(txn, _ident(10), train_end_q=0.6, val_end_q=0.8, cfg=cfg)
    # Truncate to first 5 rows (all before dt=50); rebuild as a single fold.
    txn_short = txn.iloc[:5].copy()
    sf_short = build_sequence_folds(txn_short, _ident(5), train_end_q=0.6,
                                    val_end_q=0.8, cfg=cfg)
    # First 5 rows' sequences must match between full and short.
    full_seq = np.concatenate([sf_full.seq_train, sf_full.seq_val, sf_full.seq_test])
    full_ids = np.concatenate([sf_full.ids_train, sf_full.ids_val, sf_full.ids_test])
    short_seq = np.concatenate([sf_short.seq_train, sf_short.seq_val, sf_short.seq_test])
    short_ids = np.concatenate([sf_short.ids_train, sf_short.ids_val, sf_short.ids_test])
    full_map = {tid: i for i, tid in enumerate(full_ids)}
    short_map = {tid: i for i, tid in enumerate(short_ids)}
    for tid in short_ids:
        assert np.allclose(full_seq[full_map[tid]], short_seq[short_map[tid]])


def test_sequences_respect_max_len():
    txn = _toy(50)
    sf = build_sequence_folds(txn, _ident(50), train_end_q=0.6, val_end_q=0.8,
                              cfg=SequenceConfig(max_len=7))
    assert sf.seq_train.shape[1] == 7


def test_no_tie_leakage_at_equal_timestamps():
    txn = pd.DataFrame({
        "TransactionID": [0, 1, 2, 3],
        "isFraud": [0, 0, 0, 0],
        "TransactionDT": [100, 100, 200, 300],
        "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
        "ProductCD": ["W"] * 4,
        "card1": [7] * 4,
    })
    sf = build_sequence_folds(txn, _ident(4), train_end_q=0.6, val_end_q=0.8,
                              cfg=SequenceConfig(max_len=5))
    ids = np.concatenate([sf.ids_train, sf.ids_val, sf.ids_test])
    masks = np.concatenate([sf.mask_train, sf.mask_val, sf.mask_test])
    order = np.argsort(ids)
    counts = masks.sum(axis=1)[order]
    # ID 0: 0 prior. ID 1: tie with ID 0 -> 0 prior. ID 2: 2 prior (0,1, both < 200).
    # ID 3: 3 prior.
    assert counts.tolist() == [0, 0, 2, 3]


def test_unseen_product_code_does_not_use_padding_code():
    txn = _toy(10)
    txn.loc[6:, "ProductCD"] = "Z"
    sf = build_sequence_folds(
        txn, _ident(10), train_end_q=0.6, val_end_q=0.8,
        cfg=SequenceConfig(max_len=5),
    )
    unseen_code = sf.product_vocab["__UNSEEN__"]
    assert unseen_code != sf.product_vocab["__PAD__"]
    assert sf.cur_val[0, 2] == unseen_code