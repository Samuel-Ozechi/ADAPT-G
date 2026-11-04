# Temporal Graph Schema (AG-008)

## Node types

| Kind | Source | Node id format | Notes |
|------|--------|----------------|-------|
| Transaction | `TransactionID` | `txn:<id>` | Target; no outgoing edges used at inference. |
| Card | `card1` | `card:<v>` | Primary entity. Isolated if null. |
| Address | `addr1` | `addr:<v>` | Geographic co-location. |
| Email domain | `P_emaildomain` | `email:<v>` | Purchaser-side identity proxy. |
| Device | `DeviceInfo` | `device:<v>` | Identity table; cold-start heavy. |

`card4`, `card6`, `ProductCD` are **not** nodes. They are transaction attributes. Adding them as nodes would fabricate graph complexity, which the brief forbids (Section 4).

## Edge types

| Edge | Direction | Attributes | Temporal semantics |
|------|-----------|------------|---------------------|
| card_card | undirected | first_seen_dt, last_seen_dt, n_obs | Visible at dt_t iff last_seen_dt < dt_t. |
| card_addr | undirected | same | same |
| addr_email | undirected | same | same |
| card_device | undirected | same | same; only when has_identity == 1 |

Direction is stored normalised: `src < dst` lexicographically to avoid duplicate edges.

## Temporal cut-off rule

For target transaction at `dt_t`, only edges with `last_seen_dt < dt_t` are eligible. **Strict `<`, ties excluded.** This matches AG-004 rolling aggregator and AG-007 sequence builder.

## Sampling

- Start node: `card:<card1>`, or `__ISOLATED__` if card1 null/never seen.
- BFS up to `k=2` hops, fan-out `F=10` per node.
- Candidates ranked by `last_seen_dt` descending; deterministic tie-break by `md5(node_id | dt_t) mod 2^32`.
- All candidates filtered by strict-`<` dt_t **before** ranking.

## Cold-start handling

- Null `card1`: isolated token.
- Unseen `card1`: isolated token.
- Sparse `DeviceInfo`: `card_device` edges simply absent; no imputation.

## Leakage controls

- Strict-`<` dt_t cutoff (tested).
- Deleting future rows does not change earlier snapshots (tested).
- Every edge in every sampled snapshot satisfies `last_seen_dt < dt_t` (tested).