from adapt_g.graph.builder import EntityEdge
from adapt_g.graph.sampler import SampledSubgraph
from adapt_g.models.gnn import collate_snapshots


def _sg(root, nodes, edges=()):
    la = {n: 100.0 for n in nodes}
    return SampledSubgraph(root=root, nodes=nodes, edges=list(edges),
                           node_last_active=la)


def test_collate_shapes():
    sgs = [(
        _sg("card:1", ["card:1", "addr:1"], edges=[
            EntityEdge("card:1", "addr:1", "card_addr", 100.0, 100.0)
        ]),
        150.0,
        1,
    ), (
        _sg("__ISOLATED__", ["__ISOLATED__"]),
        150.0,
        0,
    )]
    batch = collate_snapshots(sgs, k_hops=2)
    assert batch.X.shape[0] == 2
    assert batch.X.shape[1] == 2
    assert batch.X.shape[2] == 7
    assert batch.adj_index.shape[0] == 2
    assert batch.root_index.shape == (2,)
    assert batch.y.shape == (2,)
    assert batch.meta[0]["n_nodes"] == 2
    assert batch.meta[1]["n_nodes"] == 1
