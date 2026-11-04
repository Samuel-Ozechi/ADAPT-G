import numpy as np
import pandas as pd

from adapt_g.graph.builder import GraphConfig, build_temporal_graph


def _toy():
    return pd.DataFrame({
        "TransactionID": [0, 1, 2, 3, 4],
        "TransactionDT": [100, 200, 200, 300, 400],
        "card1": [1, 1, 2, 1, 2],
        "addr1": [10, 10, 20, 10, 20],
        "P_emaildomain": ["a.com", "a.com", "b.com", "a.com", "b.com"],
    })


def test_card_card_edges_collapse_direction():
    build = build_temporal_graph(_toy(), GraphConfig(use_device=False))
    cc = [e for e in build.edges if e.edge_type == "card_card"]
    # Cards 1 and 2 co-occur at dt=200 (only once).
    assert len(cc) == 1
    e = cc[0]
    assert {e.src, e.dst} == {"card:1", "card:2"}
    assert e.first_seen_dt == 200 and e.last_seen_dt == 200


def test_temporal_extents_are_min_max():
    build = build_temporal_graph(_toy(), GraphConfig(use_device=False))
    # card 1 -> addr 10 appears at dt 100,200,300
    ca = [e for e in build.edges if e.edge_type == "card_addr"
          and e.src == "addr:10.0" and e.dst == "card:1"]
    # Direction normalisation may flip (addr < card lexicographically? "addr" < "card" yes)
    ca = [e for e in build.edges if e.edge_type == "card_addr"
          and {e.src, e.dst} == {"addr:10.0", "card:1"}]
    assert len(ca) == 1
    e = ca[0]
    assert e.first_seen_dt == 100 and e.last_seen_dt == 300
    assert e.n_observations == 3


def test_node_last_active_tracks_maximum():
    build = build_temporal_graph(_toy(), GraphConfig(use_device=False))
    assert build.node_last_active["card:1"] == 300
    assert build.node_last_active["card:2"] == 400
    assert build.node_last_active["addr:10.0"] == 300


def test_statistics_shape():
    build = build_temporal_graph(_toy(), GraphConfig(use_device=False))
    assert build.stats["n_edges"] > 0
    assert "card_card" in build.stats["edge_type_counts"]
    assert build.stats["node_kind_counts"]["card"] == 2