"""Offline caching of (sequence, graph snapshot) inputs for AG-010.

The sampler is deterministic (AG-008 tested), so caching snapshots once
per fold is safe: the same transaction will produce the same snapshot
regardless of when it is sampled, given the same graph build.

Cache layout per fold:
  <cache_dir>/<fold>/
      snapshots.pkl        list[SampledSubgraph]
      sequences.npz        seq, mask, cur arrays
      meta.json            ids, y, n_rows, timestamps, config hashes
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from adapt_g.features.sequences import SequenceConfig, _build_product_vocab, _build_sequences_for_fold
from adapt_g.graph.builder import GraphBuild
from adapt_g.graph.sampler import SamplerConfig, SampledSubgraph, sample_neighbourhood


@dataclass
class CachedFold:
    snapshots: list[SampledSubgraph]
    seq: np.ndarray          # (n, L, 4)
    mask: np.ndarray         # (n, L)
    cur: np.ndarray          # (n, 4)
    ids: np.ndarray          # (n,)
    y: np.ndarray            # (n,)
    meta: dict


def _build_snapshots_for_fold(
    df: pd.DataFrame,
    build: GraphBuild,
    sampler_cfg: SamplerConfig,
    id_col: str,
    time_col: str,
) -> list[SampledSubgraph]:
    out: list[SampledSubgraph] = []
    for _, row in df.iterrows():
        card = row["card1"] if pd.notna(row.get("card1")) else None
        dt_t = float(row[time_col])
        out.append(sample_neighbourhood(build, card_id=card, dt_t=dt_t, cfg=sampler_cfg))
    return out


def cache_fold(
    df: pd.DataFrame,
    build: GraphBuild,
    out_dir: Path,
    product_vocab: dict[str, int],
    seq_cfg: SequenceConfig,
    sampler_cfg: SamplerConfig,
    id_col: str,
    time_col: str,
    target_col: str,
    fold_name: str,
) -> CachedFold:
    out_dir.mkdir(parents=True, exist_ok=True)

    seq, mask, cur = _build_sequences_for_fold(df, seq_cfg, product_vocab)
    snapshots = _build_snapshots_for_fold(df, build, sampler_cfg, id_col, time_col)

    ids = df[id_col].to_numpy()
    y = df[target_col].astype(int).to_numpy()

    np.savez_compressed(out_dir / "sequences.npz", seq=seq, mask=mask, cur=cur)
    with open(out_dir / "snapshots.pkl", "wb") as f:
        pickle.dump(snapshots, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "fold": fold_name,
        "n_rows": int(len(df)),
        "n_fraud": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "sampler_config": {"k_hops": sampler_cfg.k_hops, "fan_out": sampler_cfg.fan_out},
        "sequence_config": {
            "max_len": seq_cfg.max_len,
            "entity_col": seq_cfg.entity_col,
        },
        "n_isolated": int(sum(1 for s in snapshots if s.is_isolated)),
        "n_snapshots_with_nodes": int(sum(1 for s in snapshots if not s.is_isolated)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return CachedFold(
        snapshots=snapshots, seq=seq, mask=mask, cur=cur, ids=ids, y=y, meta=meta,
    )


def load_cached_fold(cache_dir: Path, fold_name: str) -> CachedFold:
    fold_dir = cache_dir / fold_name
    if not fold_dir.exists():
        raise FileNotFoundError(
            f"cached fold not found: {fold_dir}. "
            f"Run scripts/cache_inputs.py first."
        )
    with open(fold_dir / "snapshots.pkl", "rb") as f:
        snapshots = pickle.load(f)
    npz = np.load(fold_dir / "sequences.npz")
    meta = json.loads((fold_dir / "meta.json").read_text())
    return CachedFold(
        snapshots=snapshots,
        seq=npz["seq"], mask=npz["mask"], cur=npz["cur"],
        ids=np.array([]),   # ids stored separately in predictions
        y=np.array([]),
        meta=meta,
    )