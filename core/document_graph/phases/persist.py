"""
Phase 9: assemble the final DocumentGraph and write it via the active GraphStore.
"""

from datetime import datetime, timezone

from core.document_graph.models import DocumentGraph, GraphBuildContext, GraphMetadata
from core.document_graph.store import get_store


async def run(ctx: GraphBuildContext) -> DocumentGraph:
    nodes = list(ctx.entities.values())
    edges = list(ctx.relations)
    communities = list(ctx.communities)

    # doc_count: distinct documents represented across all entity provenance
    doc_ids = set()
    for e in nodes:
        for p in e.provenance:
            if p.document_id:
                doc_ids.add(p.document_id)

    metadata = GraphMetadata(
        user_id=ctx.user_id,
        thread_id=ctx.thread_id,
        version=1,
        status="ready",
        built_at=datetime.now(timezone.utc),
        doc_count=len(doc_ids),
        node_count=len(nodes),
        edge_count=len(edges),
        community_count=len(communities),
    )
    graph = DocumentGraph(
        metadata=metadata, nodes=nodes, edges=edges, communities=communities
    )

    store = get_store()
    store.write_graph(graph)
    print(
        f"[DocGraph][persist] wrote graph via {store.backend} "
        f"({len(nodes)} nodes / {len(edges)} edges / {len(communities)} communities)"
    )
    return graph
