# Experiment Protocol (living document)

## Prediction unit
One transaction at authorisation time.

## Label
`isFraud` (binary).

## Temporal split rule
Chronological quantiles on `TransactionDT`:
- train: first 80%
- val: next 10%
- test: final 10%
Boundaries recorded in every run manifest.

## Threshold selection
Validation only. Test set is untouched until final evaluation.

## Metrics (to be finalized in AG-005)
- Ranking: PR-AUC, ROC-AUC
- Operational: precision, recall, F1, FPR, FN at chosen thresholds
- Calibration: Brier, log loss, ECE (10 bins, methodology documented)

## Seeds
Single seed for scaffolding; repeated seeds introduced in AG-017.

## Parameter reporting
All model tiers report parameter counts and tuning budget.