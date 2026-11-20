# Evaluation Schema (AG-005)

## Prediction input files
CSV per fold with columns:
- `TransactionID` (int)
- `y_true` (0/1)
- `y_score` (float in [0,1])
- `y_uncertainty` (float, optional; required only for uncertainty metrics)

## Threshold selection
Rule set on validation predictions:
- `max_f1` (default): threshold maximising F1; ties -> smallest threshold.
- `precision_at_recall`: highest threshold achieving recall >= fixed_recall.
- `recall_at_fpr`: smallest threshold achieving FPR <= fixed_fpr, then highest precision.
- `top_k_alert_rate` (secondary): validation-score quantile for the requested alert
  fraction; provide `--alert-rate` (for example, `0.05` for a 5% validation alert rate).
  Tied scores at the threshold can result in a larger alert fraction.

The selected threshold and rule are recorded in `evaluation_report.json`
under `threshold_selection`. The test fold is scored **once** with that
threshold.

## Metric groups
| Group | Metrics |
|-------|---------|
| ranking | PR-AUC, ROC-AUC |
| operational | precision, recall, F1, FPR, TN/FP/FN/TP at selected threshold |
| calibration | Brier, log loss, ECE (equal-frequency, 10 bins), reliability diagram |
| uncertainty (optional) | error-detection AUROC/AUPRC, risk-coverage curve |

## Variance
Bootstrap (1000 resamples, percentile, 95% CI) for PR-AUC, ROC-AUC, Brier
on the test fold when n >= 30. Otherwise `flag: "insufficient_n"`.

## Report artifacts
- `evaluation_report.json` — full machine-readable report
- `evaluation_report.md` — human-readable summary
- `run_manifest.json` — links report to config, seed, ticket