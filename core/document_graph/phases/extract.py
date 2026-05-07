"""
Phase 5: relation harvest + LLM gap-fill.

Two passes:

(a) Map every triple in ctx.raw_triples to canonical entity ids using
    ctx.alias_to_id and emit Relation records with extractor='triple_store'.

(b) For chunks that have ≥2 known entities but produced no triples, batch them
    and ask the LLM to extract relations via RelationExtractionBatch. Each
    extracted relation is mapped through alias_to_id; relations referencing
    unknown entities are dropped silently.
"""

import asyncio
from typing import Dict, List, Set, Tuple

from core.constants import (
    GPU_GRAPH_RELATION_LLM,
    GRAPH_RELATION_BATCH_CHUNKS,
)
from core.document_graph.models import (
    ChunkRef,
    GraphBuildContext,
    Relation,
)
from core.llm.client import invoke_llm
from core.llm.output_schemas.document_graph_outputs import RelationExtractionBatch
from core.llm.prompts.document_graph_prompts import build_relation_extraction_prompt


_LLM_PARALLEL = 2          # how many extraction batches to run concurrently
_MAX_LLM_CHUNKS = 240      # safety cap on number of chunks sent to the LLM


def _make_chunk_ref(meta: Dict) -> ChunkRef:
    return ChunkRef(
        document_id=meta.get("document_id", ""),
        page_no=int(meta.get("page_no", 0) or 0),
        chunk_index=int(meta.get("chunk_index", 0) or 0),
        file_name=meta.get("file_name", ""),
        title=meta.get("title", ""),
    )


def _resolve(label: str, alias_to_id: Dict[str, str]) -> str:
    return alias_to_id.get(label.lower(), "")


async def _harvest_existing_triples(ctx: GraphBuildContext) -> Set[Tuple[str, int]]:
    """
    Convert ctx.raw_triples into Relations and return the set of
    (document_id, page_no) pairs already covered by triples — used downstream
    to avoid re-extracting from chunks the existing extractor handled.
    """
    covered: Set[Tuple[str, int]] = set()
    for t in ctx.raw_triples:
        s_id = _resolve(t["subject"], ctx.alias_to_id)
        o_id = _resolve(t["object"], ctx.alias_to_id)
        if not s_id or not o_id or s_id == o_id:
            continue
        ctx.relations.append(
            Relation(
                source_id=s_id,
                target_id=o_id,
                predicate=t["predicate"],
                confidence=0.7,
                extractor="triple_store",
                provenance=ChunkRef(
                    document_id=t.get("document_id", ""),
                    page_no=int(t.get("page_no", 0) or 0),
                ),
            )
        )
        covered.add((t.get("document_id", ""), int(t.get("page_no", 0) or 0)))
    return covered


def _select_llm_chunks(
    ctx: GraphBuildContext, covered: Set[Tuple[str, int]]
) -> List[Tuple[str, str, List[str]]]:
    """Pick chunks worth sending to the LLM: ≥2 known entities, not already covered."""
    selected: List[Tuple[str, str, List[str]]] = []
    for chunk in ctx.chunks:
        meta = chunk.get("metadata", {}) or {}
        # Only child chunks contribute textual relations
        if meta.get("chunk_type") and meta["chunk_type"] != "child":
            continue
        ents_str = meta.get("entities", "") or ""
        if not ents_str:
            continue

        ent_labels = [n.strip() for n in ents_str.split("|") if n.strip()]
        known = [
            ctx.entities[ctx.alias_to_id[lbl.lower()]].label
            for lbl in ent_labels
            if lbl.lower() in ctx.alias_to_id
        ]
        # Dedup while preserving order
        seen = set()
        known_dedup = []
        for k in known:
            if k not in seen:
                seen.add(k)
                known_dedup.append(k)
        if len(known_dedup) < 2:
            continue

        key = (meta.get("document_id", ""), int(meta.get("page_no", 0) or 0))
        if key in covered:
            continue

        text = chunk.get("text", "") or ""
        # Strip the indexing prefix that was prepended at ingest time
        if text.startswith("search_document:"):
            text = text.split("\n", 1)[-1]
        selected.append((chunk["id"], text[:1500], known_dedup))
        if len(selected) >= _MAX_LLM_CHUNKS:
            break
    return selected


async def _run_one_batch(
    ctx: GraphBuildContext,
    batch: List[Tuple[str, str, List[str]]],
) -> List[Relation]:
    """Run the LLM on a single batch, return resolved Relations."""
    if not batch:
        return []
    chunk_ref_by_id = {c["id"]: c.get("metadata", {}) or {} for c in ctx.chunks}

    prompt = build_relation_extraction_prompt(batch)
    try:
        resp: RelationExtractionBatch = await invoke_llm(
            response_schema=RelationExtractionBatch,
            contents=prompt,
            gpu_model=GPU_GRAPH_RELATION_LLM.model,
            port=GPU_GRAPH_RELATION_LLM.port,
        )
    except Exception as e:
        print(f"[DocGraph][extract] LLM batch failed: {e}")
        return []

    out: List[Relation] = []
    # Use the first chunk's metadata as provenance — good enough for most cases
    fallback_meta = chunk_ref_by_id.get(batch[0][0], {})
    fallback_ref = _make_chunk_ref(fallback_meta)

    for er in resp.relations:
        s_id = _resolve(er.subject, ctx.alias_to_id)
        o_id = _resolve(er.obj, ctx.alias_to_id)
        if not s_id or not o_id or s_id == o_id:
            continue
        out.append(
            Relation(
                source_id=s_id,
                target_id=o_id,
                predicate=(er.predicate or "related_to")[:80],
                confidence=max(0.0, min(1.0, er.confidence)),
                extractor="llm",
                provenance=fallback_ref,
                evidence=(er.evidence or "")[:200],
            )
        )
    return out


async def run(ctx: GraphBuildContext) -> None:
    covered = await _harvest_existing_triples(ctx)
    candidates = _select_llm_chunks(ctx, covered)
    print(
        f"[DocGraph][extract] {len(ctx.relations)} relations from triple_store, "
        f"{len(candidates)} chunks queued for LLM gap-fill"
    )

    # Batch and run with bounded parallelism
    batches = [
        candidates[i : i + GRAPH_RELATION_BATCH_CHUNKS]
        for i in range(0, len(candidates), GRAPH_RELATION_BATCH_CHUNKS)
    ]

    sem = asyncio.Semaphore(_LLM_PARALLEL)

    async def _bounded(batch):
        async with sem:
            return await _run_one_batch(ctx, batch)

    results = await asyncio.gather(*[_bounded(b) for b in batches])
    added = 0
    for batch_relations in results:
        ctx.relations.extend(batch_relations)
        added += len(batch_relations)
    print(f"[DocGraph][extract] LLM gap-fill added {added} relations")
