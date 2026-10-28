# Split Report (AG-004)

## Rule
Chronological quantiles on `TransactionDT`:
- train: `TransactionDT <= Q(0.80)`
- val:   `Q(0.80) < TransactionDT <= Q(0.90)`
- test:  `TransactionDT > Q(0.90)`

Ties broken by `TransactionID` ascending. Boundaries are absolute
`TransactionDT` values, recorded in every run manifest.

## What the report contains
- `bounds`: max train/val dt, quantiles, fold sizes, fold fraud rates.
- `entity_overlap`: for each entity column (card1, card4, P_emaildomain,
  has_identity), counts and fractions of val/test entities seen in earlier
  folds.
- `forbidden_columns_flagged`: name-based heuristic for post-outcome or
  target-derived columns.
- `preprocessor`: fitted numeric medians, category sizes, dropped columns.

## Interpretation guide
- High `val_seen_frac` / `test_seen_frac` on `card1` means strong entity
  persistence across time — leakage risk if entity identifiers are used as
  features, but expected in fraud data. Documented, not silently accepted.
- Low `test_seen_frac` on `has_identity` means test set is identity-sparse;
  graph component must handle cold-start.
- Any column in `forbidden_columns_flagged` requires human sign-off before
  use as a feature.