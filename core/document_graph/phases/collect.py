"""
Phase 1: collect_chunks.

Pull every child chunk for the thread out of ChromaDB and stash it on the
build context. We use the ChromaDB collection's `get(where=...)` API directly
rather than `as_retriever()` because we want the full corpus, not similarity
search.

We deliberately pull 'child' chunks plus any chunk carrying NER metadata
(entity_profile, document_summary). Anything else can be ignored — those are
the only chunks that contribute to graph signal.
"""

from typing import Dict, List

from core.document_graph.models import GraphBuildContext
from core.embeddings.vectorstore import get_vectorstore


async def run(ctx: GraphBuildContext) -> None:
    import asyncio

    def _fetch():
        vs = get_vectorstore(ctx.user_id, thread_id=ctx.thread_id)
        raw = vs._collection.get(
            where={
                "$and": [
                    {"user_id": {"$eq": ctx.user_id}},
                    {"thread_id": {"$eq": ctx.thread_id}},
                ]
            },
            include=["documents", "metadatas"],
        )
        out: List[Dict] = []
        ids = raw.get("ids", []) or []
        docs = raw.get("documents", []) or []
        metas = raw.get("metadatas", []) or []
        for i, cid in enumerate(ids):
            out.append(
                {
                    "id": cid,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                }
            )
        return out

    chunks = await asyncio.to_thread(_fetch)
    ctx.chunks = chunks
    print(f"[DocGraph][collect] {len(chunks)} chunks loaded for {ctx.thread_id}")
