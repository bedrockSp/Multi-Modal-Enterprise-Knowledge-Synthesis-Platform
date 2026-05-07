"""
Phase 6: prune low-signal nodes and edges.

- Drop entities below GRAPH_MIN_ENTITY_FREQ unless they're an endpoint of a
  retained edge (preserves connectivity for low-frequency entities that are
  nevertheless part of explicit relations).
- Drop edges below GRAPH_MIN_EDGE_CONFIDENCE.
- Drop edges referencing entities that no longer exist.
- Truncate entity set to GRAPH_MAX_NODES, keeping highest-frequency first.
"""

from typing import Dict, Set

from core.constants import (
    GRAPH_MAX_NODES,
    GRAPH_MIN_EDGE_CONFIDENCE,
    GRAPH_MIN_ENTITY_FREQ,
)
from core.document_graph.models import Entity, GraphBuildContext


async def run(ctx: GraphBuildContext) -> None:
    if not ctx.entities:
        ctx.relations = []
        return

    edge_endpoints: Set[str] = set()
    surviving_edges = []
    for r in ctx.relations:
        if r.confidence < GRAPH_MIN_EDGE_CONFIDENCE:
            continue
        edge_endpoints.add(r.source_id)
        edge_endpoints.add(r.target_id)
        surviving_edges.append(r)

    keep: Dict[str, Entity] = {}
    for eid, ent in ctx.entities.items():
        if ent.frequency >= GRAPH_MIN_ENTITY_FREQ or eid in edge_endpoints:
            keep[eid] = ent

    # Truncate by frequency if we exceed the ceiling
    if len(keep) > GRAPH_MAX_NODES:
        ranked = sorted(keep.values(), key=lambda e: e.frequency, reverse=True)
        keep = {e.id: e for e in ranked[:GRAPH_MAX_NODES]}

    # Drop dangling edges
    surviving_edges = [
        r for r in surviving_edges if r.source_id in keep and r.target_id in keep
    ]

    dropped_nodes = len(ctx.entities) - len(keep)
    dropped_edges = len(ctx.relations) - len(surviving_edges)
    ctx.entities = keep
    ctx.relations = surviving_edges
    print(
        f"[DocGraph][prune] dropped {dropped_nodes} nodes / {dropped_edges} edges; "
        f"kept {len(keep)} nodes / {len(surviving_edges)} edges"
    )
