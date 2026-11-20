"""AG-010: cache (sequence, snapshot) inputs for train/val/test.

Run once. Subsequent unified training loads from cache.

Usage:
  python scripts/cache_inputs.py \
    --config configs/base.yaml \
    --config configs/data/ieee_cis.yaml \
    --out cache/ieee_cis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adapt_g.config import load_config
from adapt_g.data.loader import join_identity, load_identity, load_transactions
from adapt_g.data.splits import chronological_split
from adapt_g.features.cache import cache_fold
from adapt_g.features.sequences import SequenceConfig, _build_product_vocab
from adapt_g.graph.builder import GraphConfig, build_temporal_graph
from adapt_g.graph.sampler import SamplerConfig
from adapt_g.logging_utils import get_logger
from adapt_g.seeding import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--extra-config", action="append", default=[])
    ap.add_argument("--out", default="cache/ieee_cis")
    args = ap.parse_args()

    config_paths = args.config or ["configs/base.yaml"]
    config_paths.extend(args.extra_config)
    cfg = load_config(*config_paths)
    set_seed(int(cfg["project"]["seed"]))
    logger = get_logger("adapt_g.cache", Path(args.out) / "cache.log",
                        cfg["logging"]["level"], cfg["logging"]["json"])

    dcfg = cfg["data"]
    txn = load_transactions(Path(cfg["paths"]["data_raw"]), dcfg["transaction_file"])
    ident = load_identity(Path(cfg["paths"]["data_raw"]), dcfg["identity_file"])
    df = join_identity(txn, ident, dcfg["id_column"])
    train, val, test, bounds = chronological_split(
        df, time_col=dcfg["time_column"], id_col=dcfg["id_column"],
        target_col=dcfg["target_column"],
        train_end_q=cfg["split"]["train_end_q"], val_end_q=cfg["split"]["val_end_q"],
    )

    build = build_temporal_graph(
        pd.concat([train, val], ignore_index=True),
        GraphConfig(id_col=dcfg["id_column"], time_col=dcfg["time_column"]),
    )

    seq_cfg = SequenceConfig(
        time_col=dcfg["time_column"], id_col=dcfg["id_column"],
        amount_col="TransactionAmt", product_col="ProductCD", max_len=32,
    )
    sampler_cfg = SamplerConfig(k_hops=2, fan_out=10)

    product_vocab = _build_product_vocab(train, seq_cfg.product_col)
    out = Path(args.out)
    for name, fold in (("train", train), ("val", val), ("test", test)):
        logger.info(f"caching fold {name} ({len(fold)} rows)",
                    extra={"event": "cache_start", "extra": {"fold": name}})
        c = cache_fold(
            fold, build, out / name, product_vocab,
            seq_cfg, sampler_cfg,
            dcfg["id_column"], dcfg["time_column"], dcfg["target_column"],
            name,
        )
        logger.info(f"cached fold {name}: {len(c.snapshots)} snapshots, "
                    f"{c.meta['n_isolated']} isolated",
                    extra={"event": "cache_done", "extra": {"fold": name}})

    (out / "cache_meta.json").write_text(json.dumps({
        "bounds": bounds.__dict__,
        "product_vocab_size": len(product_vocab),
        "sampler": {"k_hops": sampler_cfg.k_hops, "fan_out": sampler_cfg.fan_out},
        "sequence": {"max_len": seq_cfg.max_len},
    }, indent=2))
    logger.info("cache complete", extra={"event": "cache_complete"})


if __name__ == "__main__":
    main()