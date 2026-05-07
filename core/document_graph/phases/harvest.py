"""
Phases 2 + 3: harvest entities and triples from existing extractions.

Phase 2 pulls NER entities from chunk metadata (the 'entities' / 'entity_types'
fields populated during ingest by core.embeddings.context_enrichment).

Phase 3 pulls SPO triples from the per-thread TripleStore SQLite that was
populated during ingest by core.embeddings.context_enrichment.extract_entity_triples.

Both phases attach provenance to every node/edge candidate.
"""

import asyncio
import re
import sqlite3
from typing import Dict, List

from core.document_graph.models import (
    ChunkRef,
    Entity,
    GraphBuildContext,
)
from core.services.triple_store import TripleStore


def _slug(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return s or "entity"


def _make_chunk_ref(meta: Dict, chunk_id: str = "") -> ChunkRef:
    return ChunkRef(
        document_id=meta.get("document_id", ""),
        page_no=int(meta.get("page_no", 0) or 0),
        chunk_index=int(meta.get("chunk_index", 0) or 0),
        file_name=meta.get("file_name", ""),
        title=meta.get("title", ""),
    )


async def run_entities(ctx: GraphBuildContext) -> None:
    """Phase 2 — gather entity candidates from chunk metadata."""
    raw: Dict[str, Entity] = {}

    for chunk in ctx.chunks:
        meta = chunk.get("metadata", {}) or {}
        ents_str = meta.get("entities", "") or ""
        types_str = meta.get("entity_types", "") or ""
        if not ents_str:
            # Profile chunks store entity_name/entity_type instead.
            ents_str = meta.get("entity_name", "") or ""
            types_str = meta.get("entity_type", "") or ""
            if not ents_str:
                continue

        names = [n.strip() for n in ents_str.split("|") if n.strip()]
        types = [t.strip() for t in types_str.split("|") if t.strip()]
        # Pad types to match names length
        while len(types) < len(names):
            types.append("OTHER")

        cref = _make_chunk_ref(meta)
        seen_in_chunk = set()
        for name, etype in zip(names, types):
            key = name.lower()
            if key in seen_in_chunk:
                continue
            seen_in_chunk.add(key)
            ent = raw.get(key)
            if ent is None:
                ent = Entity(
                    id=_slug(name),
                    label=name,
                    type=etype or "OTHER",
                    aliases=[name],
                    frequency=0,
                    doc_count=0,
                    provenance=[],
                )
                raw[key] = ent
            ent.frequency += 1
            # provenance capped to keep payload small
            if len(ent.provenance) < 8:
                ent.provenance.append(cref)
            # doc_count tracked separately below

    # Compute distinct doc_count per entity
    doc_seen: Dict[str, set] = {}
    for chunk in ctx.chunks:
        meta = chunk.get("metadata", {}) or {}
        ents_str = meta.get("entities", "") or ""
        if not ents_str:
            continue
        doc_id = meta.get("document_id", "")
        for name in ents_str.split("|"):
            key = name.strip().lower()
            if not key:
                continue
            doc_seen.setdefault(key, set()).add(doc_id)
    for key, ent in raw.items():
        ent.doc_count = len(doc_seen.get(key, set()))

    # Pull short profiles from entity_profile chunks
    for chunk in ctx.chunks:
        meta = chunk.get("metadata", {}) or {}
        if (meta.get("chunk_index") == -2) and meta.get("entity_name"):
            key = meta["entity_name"].lower()
            if key in raw and not raw[key].profile:
                # First ~400 chars of the profile body (without the search prefix)
                text = chunk.get("text", "") or ""
                raw[key].profile = text[:400]

    ctx.raw_entities = raw
    print(
        f"[DocGraph][harvest_entities] {len(raw)} candidate entities "
        f"({sum(e.frequency for e in raw.values())} mentions)"
    )


async def run_triples(ctx: GraphBuildContext) -> None:
    """Phase 3 — pull existing triples from the per-thread TripleStore."""

    def _fetch() -> List[Dict]:
        db_path = TripleStore._get_db_path(ctx.user_id, ctx.thread_id)
        try:
            conn = sqlite3.connect(db_path)
        except Exception:
            return []
        try:
            cur = conn.execute(
                "SELECT subject, predicate, object, document_id, page_no FROM triples"
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            rows = []
        finally:
            conn.close()
        out = []
        for s, p, o, did, pno in rows:
            out.append(
                {
                    "subject": s,
                    "predicate": p,
                    "object": o,
                    "document_id": did,
                    "page_no": pno or 0,
                }
            )
        return out

    triples = await asyncio.to_thread(_fetch)
    ctx.raw_triples = triples
    print(f"[DocGraph][harvest_triples] {len(triples)} triples loaded")
