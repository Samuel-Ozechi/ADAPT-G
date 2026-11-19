"""Deterministic temporal neighbourhood sampler.

For a target (transaction, dt_t, card1) triple:
1. Start from "card:<card1>" (or an isolated token if null).
2. Apply strict-< dt_t cutoff to every candidate edge.
3. BFS up to k hops with fixed fan-out F, deterministic tie-break.
4. Return the sampled subgraph as node list + typed edges + per-node
   last-active scalar.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from adapt_g.graph.builder import EntityEdge, GraphBuild


@dataclass
class SamplerConfig:
    k_hops: int = 2
    fan_out: int = 10
    isolated_token: str = "__ISOLATED__"


@dataclass
class SampledSubgraph:
    root: str
    nodes: list[str]                # root first, then BFS order
    edges: list[EntityEdge]         # edges among nodes in `nodes`
    node_last_active: dict[str, float]
    is_isolated: bool = False
    depth: dict[str, int] = field(default_factory=dict)
    node_kinds: list[int] | None = None  # pre-computed NodeKind ints aligned with nodes


def _node_kind_idx(node_id: str) -> int:
    if node_id == "__ISOLATED__":
        return 4  # NodeKind.ISOLATED
    prefix = node_id.split(":", 1)[0]
    return {"card": 0, "addr": 1, "email": 2, "device": 3}.get(prefix, 5)


def _stable_tiebreak(node_id: str, dt_t: float) -> int:
    return hash((node_id, dt_t)) & 0xFFFFFFFF


def build_adjacency(
    build: GraphBuild,
) -> dict[str, list[tuple[str, EntityEdge]]]:
    adj: dict[str, list[tuple[str, EntityEdge]]] = defaultdict(list)
    for e in build.edges:
        adj[e.src].append((e.dst, e))
        adj[e.dst].append((e.src, e))
    return adj


def sample_neighbourhood(
    build: GraphBuild,
    card_id: int | None,
    dt_t: float,
    cfg: SamplerConfig | None = None,
    adjacency: dict[str, list[tuple[str, EntityEdge]]] | None = None,
) -> SampledSubgraph:
    cfg = cfg or SamplerConfig()
    if card_id is None:
        return SampledSubgraph(
            root=cfg.isolated_token, nodes=[cfg.isolated_token],
            node_kinds=[4], edges=[],
            node_last_active={cfg.isolated_token: -1.0},
            is_isolated=True, depth={cfg.isolated_token: 0},
        )
    root = f"card:{card_id}"
    if root not in build.node_last_active:
        return SampledSubgraph(
            root=cfg.isolated_token, nodes=[cfg.isolated_token],
            node_kinds=[4], edges=[],
            node_last_active={cfg.isolated_token: -1.0},
            is_isolated=True, depth={cfg.isolated_token: 0},
        )

    adj = adjacency if adjacency is not None else build_adjacency(build)
    visited: set[str] = {root}
    depth = {root: 0}
    queue = deque([(root, 0)])
    chosen_edges: dict[tuple[str, str, str], EntityEdge] = {}

    while queue:
        node, d = queue.popleft()
        if d >= cfg.k_hops:
            continue
        # Collect valid candidates as (neighbour_id, last_seen_dt, edge) tuples.
        candidates: list[tuple[str, float, EntityEdge]] = []
        for nbr, e in adj.get(node, []):
            if e.first_seen_dt >= dt_t:
                continue
            if nbr in visited:
                continue
            effective_last_seen = min(float(e.last_seen_dt), float(dt_t) - 1e-9)
            candidates.append((nbr, effective_last_seen, e))
        candidates.sort(
            key=lambda x: (-x[1], _stable_tiebreak(x[0], dt_t))
        )
        for nbr, effective_last_seen, e in candidates[:cfg.fan_out]:
            visited.add(nbr)
            depth[nbr] = d + 1
            queue.append((nbr, d + 1))
            snapshot_e = EntityEdge(
                src=e.src, dst=e.dst, edge_type=e.edge_type,
                first_seen_dt=float(e.first_seen_dt),
                last_seen_dt=effective_last_seen,
                n_observations=e.n_observations,
            )
            key = (min(e.src, e.dst), max(e.src, e.dst), e.edge_type)
            chosen_edges[key] = snapshot_e

    # Order nodes for stable output: root first, then BFS order.
    nodes_ordered = [root] + [n for n in visited if n != root]
    node_kinds_list = [_node_kind_idx(n) for n in nodes_ordered]
    node_last = {n: build.node_last_active.get(n, -1.0) for n in nodes_ordered}
    return SampledSubgraph(
        root=root, nodes=nodes_ordered,
        edges=list(chosen_edges.values()),
        node_last_active=node_last, is_isolated=False, depth=depth,
        node_kinds=node_kinds_list,
    )