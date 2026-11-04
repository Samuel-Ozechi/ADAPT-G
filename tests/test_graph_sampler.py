import pandas as pd

from adapt_g.graph.builder import GraphConfig, build_temporal_graph
from adapt_g.graph.sampler import SamplerConfig, sample_neighbourhood


def _build():
    df = pd.DataFrame({
        "TransactionID": [0, 1, 2, 3, 4],
        "TransactionDT": [100, 200, 300, 400, 500],
        "card1": [1, 2, 1, 2, 1],
        "addr1": [10, 20, 10, 20, 10],
        "P_emaildomain": ["a", "b", "a", "b", "a"],
    })
    return build_temporal_graph(df, GraphConfig(use_device=False))


def test_strict_cutoff_excludes_ties():
    build = _build()
    # card 1 has appeared at dt=100,300,500; a query at dt=300 must not see 300.
    sg = sample_neighbourhood(build, card_id=1, dt_t=300, cfg=SamplerConfig(k_hops=2, fan_out=10))
    # No edge with last_seen_dt >= 300 may appear.
    for e in sg.edges:
        assert e.last_seen_dt < 300


def test_isolated_card_returns_token():
    build = _build()
    sg = sample_neighbourhood(build, card_id=999, dt_t=300)
    assert sg.is_isolated is True
    assert sg.nodes == ["__ISOLATED__"]


def test_null_card_returns_isolated():
    build = _build()
    sg = sample_neighbourhood(build, card_id=None, dt_t=300)
    assert sg.is_isolated is True


def test_fan_out_respected():
    build = _build()
    sg = sample_neighbourhood(build, card_id=1, dt_t=500,
                              cfg=SamplerConfig(k_hops=2, fan_out=1))
    # Root always present; per visited node at most 1 neighbour.
    root_degree = sum(1 for e in sg.edges if sg.root in (e.src, e.dst))
    assert root_degree <= 1


def test_deterministic_across_calls():
    build = _build()
    a = sample_neighbourhood(build, card_id=1, dt_t=500)
    b = sample_neighbourhood(build, card_id=1, dt_t=500)
    assert a.nodes == b.nodes
    assert [(e.src, e.dst, e.edge_type) for e in a.edges] == \
           [(e.src, e.dst, e.edge_type) for e in b.edges]