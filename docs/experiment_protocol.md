## Evaluation harness
- Thresholds selected on validation only; rule recorded.
- Test scored once, after design decisions frozen.
- Every result includes PR-AUC and threshold-dependent error measures.
- Calibration reported separately from discrimination.
- Bootstrap 95% CIs for PR-AUC, ROC-AUC, Brier when n_test >= 30.
- Metric definitions frozen in docs/evaluation_schema.md and versioned in the manifest.

## Standard evaluation command
python scripts/evaluate.py \
  --train-preds runs/<id>/train_preds.csv \
  --val-preds   runs/<id>/val_preds.csv \
  --test-preds  runs/<id>/test_preds.csv \
  --model-name  <name> \
  --threshold-rule max_f1

## Temporal reference
- Entity grouping: `card1`.
- Sequence window: last 32 prior transactions; ties excluded.
- Model: 2-layer Transformer, d_model=64, 4 heads, d_ff=128, dropout=0.1.
- Pos-weight BCE; 5 epochs; batch 512; lr 1e-3.
- Frozen predictions written to `preds/temporal/`.
- Latency measured on declared device, batch size 512, 3 warmup + 3 timed runs.