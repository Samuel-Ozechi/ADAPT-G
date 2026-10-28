## Evaluation harness (AG-005)
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