# Adapt-G Research Specification (AG-001)

## Prediction unit
One transaction at authorisation time.

## Fraud label
`isFraud` (binary 0/1). IEEE-CIS derives labels from reported chargebacks
plus linked-entity propagation within a 120-day window.

## Allowed knowledge at prediction time
Only fields observable before or at the authorisation decision.
Excluded: post-transaction outcomes, chargeback results arriving later,
future aggregates, target-derived fields.

## Target deployment context
Offline research. No live blocking, no real financial decisions.

## Success criteria
Rigorous, reproducible answer to RQ1–RQ6 (Brief Section 2). Success is not
defined by using every component; negative findings are valid outcomes.

## Research questions
See Brief Section 2 (RQ1–RQ6).