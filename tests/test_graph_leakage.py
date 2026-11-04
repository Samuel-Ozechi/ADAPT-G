"""Behavioural leakage tests for graph construction + sampling."""
import numpy as np
import pandas as pd

from adapt_g.graph.builder import GraphConfig, build_temporal_graph
from adapt_g.graph.sampler import sample_neighbourhood


def _toy_with_future():
    """Deliberate future-leak fixture.

    card 1 appears at dt=100. card 2 appears at dt=200 sharing addr with 1.
    At dt=150 (query on card 1), card 1 must have no neighbour 2 yet.
    """
    return pd.DataFrame({
        "TransactionID": [0, 1],
        "TransactionDT": [100, 200],
        "card1": [1, 2],
        "addr1": [10, 10],
        "P_emaildomain": ["a", "a"],
    })


def test_future_edge_not_visible():
    df = _toy_with_future()
    build = build_temporal_graph(df, GraphConfig(use_device=False))
    # Query card 1 at dt=150; the card1-card2 edge has last_seen_dt=200.
    sg = sample_neighbourhood(build, card_id=1, dt_t=150)
    for e in sg.edges:
        assert e.last_seen_dt < 150
    assert "card:2" not in sg.nodes


def test_deleting_future_rows_does_not_change_earlier_snapshots():
    df = _toy_with_future()
    build_full = build_temporal_graph(df, GraphConfig(use_device=False))
    df_trunc = df[df["TransactionDT"] < 150].copy()
    build_trunc = build_temporal_graph(df_trunc, GraphConfig(use_device=False))
    sg_full = sample_neighbourhood(build_full, card_id=1, dt_t=150)
    sg_trunc = sample_neighbourhood(build_trunc, card_id=1, dt_t=150)
    assert sg_full.nodes == sg_trunc.nodes
    assert sg_full.is_isolated == sg_trunc.is_isolated


def test_snapshot_bounded_by_dt_t():
    """For every node reached, its contributing edges satisfy last_seen_dt < dt_t."""
    df = pd.DataFrame({
        "TransactionID": np.arange(10),
        "TransactionDT": np.arange(10) * 100,
        "card1": np.arange(10) % 3,
        "addr1": np.arange(10) % 2 + 10,
        "P_emaildomain": ["a", "b"] * 5,
    })
    build = build_temporal_graph(df, GraphConfig(use_device=False))
    for dt_t in (250, 450, 850):
        sg = sample_neighbourhood(build, card_id=0, dt_t=float(dt_t))
        for e in sg.edges:
            assert e.last_seen_dt < dt_t, (dt_t, e)