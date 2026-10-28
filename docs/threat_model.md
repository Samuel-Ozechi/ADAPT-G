# Threat Model (AG-001)

| Dimension | Specification |
|-----------|---------------|
| Attacker knowledge | Black-box. No weights, architecture, or training data. |
| Capabilities | Modify mutable transaction attributes within plausible bounds; create new accounts/entities; coordinate across entities. |
| Mutable features | Amount, timestamp (bounded), merchant/category, device identifiers (synthetic), channel. |
| Immutable features | Existing account IDs, historical patterns, inherent entity relationships. |
| Constraints | Domain validity; no impossible feature combinations; no obvious label flips. |
| Objective | Maximise undetected fraud while minimising false-positive harm to legitimate activity. |

## Realistic adversarial example
Preserves domain constraints, maintains plausible feature distributions,
uses no future information, and represents an executable fraud strategy.

## Out of scope
Formal security guarantees; attacks requiring model internals; arbitrary
perturbations that violate domain constraints.