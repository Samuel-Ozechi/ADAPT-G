# Graph Statistics (AG-008)

The driver writes two files per run:

## `graph_statistics.json`
- `n_edges`: total entity-co-occurrence edges
- `n_nodes`: total distinct entity nodes
- `edge_type_counts`: per-type counts
- `node_kind_counts`: per-kind node counts

## `graph_sampling_statistics.json`
- `sample_n`: number of test rows sampled for snapshot statistics
- `isolated_fraction`: fraction of sampled snapshots that were isolated
  (null or unseen card1)
- `nodes_per_snapshot_{mean,std}`: snapshot size in nodes
- `edges_per_snapshot_{mean,std}`: snapshot size in edges
- `k_hops`, `fan_out`: sampler config
- `synthetic_fixture`: whether this run used synthetic data

## Interpretation guide
- `isolated_fraction` on real IEEE-CIS should be < 0.02 given card1's
  coverage. If much higher, investigate the join or card1 null rate.
- `nodes_per_snapshot_mean` > `(1 + fan_out + fan_out^2)` indicates a bug;
  the sampler caps neighbours per visited node, so the mean should stay
  near `1 + F + F^2` (≈ 111) for `k=2, F=10`.
- `edges_per_snapshot_mean` should grow more slowly than node count for
  sparse IEEE-CIS graphs.