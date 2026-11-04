# Temporal Baseline Report (AG-007)

## Purpose
Establish the temporal reference: a Transformer over each entity's prior
transaction history. Any claim that temporal context improves on tabular
detection must survive comparison against this reference on the same
protocol.

## Model
- Input per row:
  - Sequence of last L=32 prior transactions (same `card1`, strictly
    earlier `TransactionDT`, ties excluded).
  - Each step: [amount_scaled, gap_seconds, log1p_gap, product_code].
  - Current-transaction features: [amount_scaled, amount_zero,
    product_code, has_identity].
- Architecture: linear projection -> positional encoding ->
  2-layer Transformer encoder (masked) -> masked mean pooling ->
  concat with current-feature MLP -> linear head -> scalar logit.
- Loss: BCEWithLogits with pos_weight = neg/pos on train fold.
- One forward pass per row; single checkpoint.

## Leakage controls
- Sequence uses only rows with `TransactionDT_j < TransactionDT_i`
  for the same entity. Ties are excluded, matching AG-004's rolling
  aggregator.
- Tests: `tests/test_sequences.py` covers (a) prior-only usage,
  (b) future-row deletion invariance, (c) max_len truncation,
  (d) tie exclusion.
- `tests/test_temporal_model.py` covers padding invariance and
  gradient reach.

## Protocol
- Splits: identical chronological quantiles to AG-004/AG-006.
- Threshold selection: validation only, via AG-005 harness.
- Test fold scored once.
- Seeds: fixed project seed for torch and numpy.

## Artifacts per run
- `frozen_temporal_reference.json`: bounds, feature names, max_len,
  parameter count, training hyperparameters, device.
- `preds/temporal/{train,val,test}.csv`: frozen predictions.
- `evaluation/temporal/evaluation_report.{json,md}`: standard report.
- `temporal_profile.json`: latency (mean, std), n_test, batch_size,
  device, parameter count.
- `temporal_model.pt`: state_dict for reproducibility and warm-start
  variant reporting (brief Section 5).
- `run_manifest.json`: links everything to config, seed, ticket.

## Interpretation guide
- Compare PR-AUC/ROC-AUC against the LightGBM tabular reference from
  AG-006. If temporal-only does not beat tabular, that is informative:
  the temporal signal alone is not enough; the fusion hypothesis (RQ1)
  must be justified by the fused model, not by the temporal-only one.
- Latency and parameter count are recorded so the unified model's cost
  can be attributed fairly.
- Do not compare temporal-only against tabular on a different seed or
  split; both are frozen to the same AG-004 boundaries.