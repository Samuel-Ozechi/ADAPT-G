import numpy as np
import pandas as pd

from adapt_g.features.cache import cache_fold, load_cached_fold
from adapt_g.features.sequences import SequenceConfig, _build_product_vocab
from adapt_g.graph.builder import GraphBuild
from adapt_g.graph.sampler import SamplerConfig


def test_cache_fold_uses_current_sequence_builder_signature(tmp_path):
    fold = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "isFraud": [0, 1, 0],
        "TransactionDT": [10, 20, 30],
        "TransactionAmt": [10.0, 20.0, 30.0],
        "ProductCD": ["W", "W", "C"],
        "card1": [7, 7, 7],
    })
    seq_cfg = SequenceConfig(max_len=2)
    vocab = _build_product_vocab(fold, seq_cfg.product_col)

    cached = cache_fold(
        df=fold,
        build=GraphBuild(edges=[], node_last_active={}),
        out_dir=tmp_path / "train",
        product_vocab=vocab,
        seq_cfg=seq_cfg,
        sampler_cfg=SamplerConfig(k_hops=2, fan_out=2),
        id_col="TransactionID",
        time_col="TransactionDT",
        target_col="isFraud",
        fold_name="train",
    )

    assert cached.seq.shape == (3, 2, 4)
    assert cached.mask.shape == (3, 2)
    assert cached.cur.shape == (3, 4)
    assert np.array_equal(cached.mask.sum(axis=1), np.array([0, 1, 2]))
    reloaded = load_cached_fold(tmp_path, "train")
    assert reloaded.seq.shape == cached.seq.shape
    assert reloaded.meta["sequence_config"]["max_len"] == 2
