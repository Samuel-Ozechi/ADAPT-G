# GNN-Only Baseline Report (AG-009)

## Purpose
Test whether relational context alone (no tabular, no temporal sequence)
adds signal. This is the "graph reference" the unified model must beat
on relational structure, independent of tabular or temporal gains.

## Model
- Input: per-snapshot graph (SampledSubgraph from AG-008).
- Node features (7 dims): one-hot of kind (6) + recency log1p(dt_t - last_active) (1).
  **No raw tabular features.** This is deliberate; it keeps graph-only honest.
- Edges: used for mean aggregation only (no edge features in messages; edge
  features retained in collator for potential use in AG-010, unused here).
- Architecture: input proj -> 2x GraphSAGE layer (mean aggregation,
  LayerNorm, residual W_self/W_nbr) -> root-node embedding -> MLP head -> logit.
- Single forward pass per snapshot.
- BCEWithLogits with pos_weight = neg/pos on train fold.

## Protocol
- Splits: AG-004 boundaries (identical to AG-006/AG-007).
- Graph: built from train ∪ val (strict-< cutoff ensures test cannot
  see future edges); sampler `k_hops=2`, `fan_out=10`.
- Threshold: validation only, via AG-005 harness.
- Test scored once.

## Artifacts per run
- `preds/gnn/{train,val,test}.csv`
- `evaluation/gnn/evaluation_report.{json,md}`
- `gnn_connectivity_analysis.json`: precision/recall by bucket
  (isolated / sparse <20 / dense ≥20 nodes)
- `gnn_cost_report.json`: latency including sampler; parameters
- `frozen_gnn_reference.json`
- `gnn_model.pt`

## Interpretation guide
- If PR-AUC is near the base rate, graph-only adds nothing; the fusion
  hypothesis is not supported by graph-only evidence, and AG-010 must
  justify fusion empirically.
- `connectivity_analysis` shows where the GNN fails: if isolated
  rows dominate and precision is low there, graph sparsity is the
  bottleneck. `dense` bucket precision/recall is the graph's real signal.
- `cost_report.full_forward_mean_s` includes sampler overhead; in AG-010
  this is a component of the unified latency — not additive to a separate
  temporal latency unless the sampler is reused.