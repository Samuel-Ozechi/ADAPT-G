# IEEE-CIS Data Dictionary (initial; expanded in AG-004)

## Transaction table (`train_transaction.csv`)
| Column | Meaning |
|--------|---------|
| TransactionID | Unique transaction identifier (join key) |
| isFraud | Target (0/1) |
| TransactionDT | Time delta in seconds from a reference point (not calendar time) |
| TransactionAmt | Transaction amount (payment currency) |
| ProductCD | Product code (W, H, C, S, R) |
| card1–card6 | Payment card attributes |
| addr1, addr2 | Address |
| dist1, dist2 | Distances |
| P_emaildomain, R_emaildomain | Purchaser / recipient email domains |
| C1–C14 | Counting features (Vesta) |
| D1–D15 | Timedelta features (Vesta) |
| M1–M9 | Match features (Vesta) |
| V1–V339 | Engineered features (Vesta) |

## Identity table (`train_identity.csv`)
| Column | Meaning |
|--------|---------|
| TransactionID | Join key |
| id_01–id_38 | Identity features |
| DeviceType | Desktop / mobile |
| DeviceInfo | Device string |

## Join semantics
Left-join identity on TransactionID. Add `has_identity ∈ {0,1}`.
Do NOT impute identity for transactions with no identity record in AG-004
without an explicit, documented policy.

## Known issues
- `TransactionDT` is a delta, not calendar time; interpret shifts as relative ordering.
- Identity table covers only a subset of transactions.
- V-columns have substantial missingness and variable width; parse defensively.