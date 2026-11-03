# Tabular Baseline Report (AG-006)

## Purpose
Establish the strong tabular reference for all later comparisons. Any model
that adds temporal, graph, diffusion, or uncertainty machinery must beat
this reference on an identical protocol to justify its complexity.

## Baselines
| Model | Role |
|-------|------|
| majority | Floor; predicts training positive rate for all rows |
| rule | Hand-crafted rule (amt > train p99 & ProductCD == C); floor for detection |
| logreg | Linear strong baseline with balanced class weights |
| lightgbm | Strong gradient-boosting reference; tuned on validation PR-AUC |

## Protocol
- Splits: chronological quantiles on `TransactionDT` (AG-004).
- Features: AG-004 preprocessor + rolling aggregates + `has_identity`.
- Imbalance: `scale_pos_weight = neg/pos` on train only (LightGBM);
  `class_weight="balanced"` (LogReg).
- Tuning: validation PR-AUC over a bounded grid (<= 6 fits for LightGBM).
- Test fold: scored once via AG-005 harness.
- Seeds: fixed (`seed + trial_index`).

## Artifacts per run
- `baseline_table.json`: one row per model with headline metrics + CIs.
- `frozen_reference.json`: folds, feature list, preprocessor report,
  hyperparameter search log, parameter counts.
- `preds/<model>/{train,val,test}.csv`: frozen predictions for later
  comparison (Tier 2 of the three-tier comparison).
- `evaluation/<model>/evaluation_report.{json,md}`.

## Interpretation guide
- `lightgbm` is the frozen tabular reference for AG-010.
- If the unified Adapt-G does not beat LightGBM on PR-AUC (with CI),
  that is a finding, not a failure — report it.
- Parameter counts are in `baseline_table.json` so gains cannot be
  attributed solely to capacity.