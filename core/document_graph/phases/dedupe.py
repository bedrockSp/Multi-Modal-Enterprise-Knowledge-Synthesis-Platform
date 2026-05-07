"""
Phase 4: dedupe entities.

Three-pass merge:
1. Normalize-and-collapse: case-fold, strip non-alphanumerics, merge exact matches.
2. Embedding cosine similarity: pairs above GRAPH_DEDUPE_SIM_THRESHOLD auto-merge.
3. Ambiguous range [GRAPH_DEDUPE_AMBIGUOUS_LO, GRAPH_DEDUPE_SIM_THRESHOLD) →
   batched LLM judgment via EntityMergeBatch schema.

After this phase, ctx.entities contains canonical entities (deduped) and
ctx.alias_to_id maps every surface form (lowercased) to its canonical id.
"""

import asyncio
import re
from typing import Dict, List, Tuple

from core.constants import (
    GPU_GRAPH_DEDUPE_LLM,
    GRAPH_DEDUPE_AMBIGUOUS_LO,
    GRAPH_DEDUPE_SIM_THRESHOLD,
)
from core.document_graph.models import Entity, GraphBuildContext
from core.llm.client import invoke_llm
from core.llm.output_schemas.document_graph_outputs import EntityMergeBatch
from core.llm.prompts.document_graph_prompts import build_entity_merge_prompt


_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DEDUPE_BATCH_PAIRS = 12       # ambiguous pairs per LLM call
_DEDUPE_TOP_N_FOR_EMB = 800    # only embed top-N entities by frequency


def _normalize(label: str) -> str:
    return _NON_ALNUM.sub("", label.lower()).strip()


def _merge_into(target: Entity, src: Entity) -> None:
    """Fold src into target — accumulate frequency, aliases, and provenance."""
    target.frequency += src.frequency
    target.doc_count = max(target.doc_count, src.doc_count)
    for a in src.aliases:
        if a not in target.aliases:
            target.aliases.append(a)
    if src.label not in target.aliases:
        target.aliases.append(src.label)
    cap = 16
    for p in src.provenance:
        if len(target.provenance) < cap:
            target.provenance.append(p)
    if src.profile and not target.profile:
        target.profile = src.profile


def _pass1_normalize(raw: Dict[str, Entity]) -> Dict[str, Entity]:
    by_norm: Dict[str, Entity] = {}
    for ent in raw.values():
        norm = _normalize(ent.label)
        if not norm:
            continue
        existing = by_norm.get(norm)
        if existing is None:
            by_norm[norm] = ent
        else:
            # Keep the more-frequent label as canonical
            if ent.frequency > existing.frequency:
                _merge_into(ent, existing)
                by_norm[norm] = ent
            else:
                _merge_into(existing, ent)
    return by_norm


def _cosine(a, b) -> float:
    import numpy as np

    va = np.asarray(a, dtype="float32")
    vb = np.asarray(b, dtype="float32")
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


async def _embed_labels(labels: List[str]) -> List:
    """Embed a list of entity labels using the project's embedding function."""

    def _do():
        from core.embeddings.embeddings import get_embedding_function

        ef = get_embedding_function()
        # Use the document prefix used at index time so query/document space matches
        prefixed = [f"search_document: {lbl}" for lbl in labels]
        return ef.embed_documents(prefixed)

    return await asyncio.to_thread(_do)


def _disjoint_set_factory(keys):
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    return find, union


async def _llm_judge_pairs(pairs: List[Tuple[Entity, Entity]]) -> Dict[str, Tuple[bool, str]]:
    """Batched LLM merge judgment. Returns pair_id -> (same, canonical_label)."""
    if not pairs:
        return {}

    indexed = []
    for i, (a, b) in enumerate(pairs):
        sample_a = a.provenance[0].title if a.provenance else ""
        sample_b = b.provenance[0].title if b.provenance else ""
        indexed.append(
            {
                "pair_id": f"p{i}",
                "label_a": a.label,
                "type_a": a.type,
                "sample_a": sample_a,
                "label_b": b.label,
                "type_b": b.type,
                "sample_b": sample_b,
            }
        )

    out: Dict[str, Tuple[bool, str]] = {}
    for start in range(0, len(indexed), _DEDUPE_BATCH_PAIRS):
        batch = indexed[start : start + _DEDUPE_BATCH_PAIRS]
        prompt = build_entity_merge_prompt(batch)
        try:
            resp: EntityMergeBatch = await invoke_llm(
                response_schema=EntityMergeBatch,
                contents=prompt,
                gpu_model=GPU_GRAPH_DEDUPE_LLM.model,
                port=GPU_GRAPH_DEDUPE_LLM.port,
            )
            for j in resp.judgments:
                out[j.pair_id] = (bool(j.same), j.canonical or "")
        except Exception as e:
            print(f"[DocGraph][dedupe] LLM judge batch failed: {e}; treating pairs as 'not same'")
    return out


async def run(ctx: GraphBuildContext) -> None:
    raw = ctx.raw_entities
    if not raw:
        ctx.entities = {}
        ctx.alias_to_id = {}
        print("[DocGraph][dedupe] no entities to dedupe")
        return

    # Pass 1: collapse trivially-identical surface forms
    by_norm = _pass1_normalize(raw)
    print(f"[DocGraph][dedupe] pass1 normalize: {len(raw)} -> {len(by_norm)}")

    # Pass 2 + 3: only on top-N by frequency to bound cost
    items: List[Entity] = list(by_norm.values())
    items.sort(key=lambda e: e.frequency, reverse=True)
    embed_pool = items[:_DEDUPE_TOP_N_FOR_EMB]
    skipped = items[_DEDUPE_TOP_N_FOR_EMB:]

    if len(embed_pool) > 1:
        embeddings = await _embed_labels([e.label for e in embed_pool])
    else:
        embeddings = []

    auto_merge_pairs: List[Tuple[int, int]] = []
    ambiguous_pairs: List[Tuple[Entity, Entity]] = []
    ambiguous_idx: List[Tuple[int, int]] = []

    n = len(embed_pool)
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine(embeddings[i], embeddings[j]) if embeddings else 0.0
            if sim >= GRAPH_DEDUPE_SIM_THRESHOLD:
                auto_merge_pairs.append((i, j))
            elif sim >= GRAPH_DEDUPE_AMBIGUOUS_LO:
                ambiguous_pairs.append((embed_pool[i], embed_pool[j]))
                ambiguous_idx.append((i, j))

    # LLM judgment on ambiguous pairs
    judgments = await _llm_judge_pairs(ambiguous_pairs)
    for k, (i, j) in enumerate(ambiguous_idx):
        decision = judgments.get(f"p{k}")
        if decision and decision[0]:
            auto_merge_pairs.append((i, j))

    # Apply union-find
    keys = list(range(n))
    find, union = _disjoint_set_factory(keys)
    for i, j in auto_merge_pairs:
        union(i, j)

    # Group members and merge
    groups: Dict[int, List[int]] = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)

    canonical: Dict[str, Entity] = {}
    alias_to_id: Dict[str, str] = {}

    for root, members in groups.items():
        # Pick the highest-frequency entity as canonical seed
        seed_idx = max(members, key=lambda m: embed_pool[m].frequency)
        seed = embed_pool[seed_idx].model_copy(deep=True)
        # If LLM suggested a canonical label, prefer it when it appears in the cluster
        for m in members:
            if m == seed_idx:
                continue
            other = embed_pool[m]
            _merge_into(seed, other)
        canonical[seed.id] = seed
        for m in members:
            alias_to_id[embed_pool[m].label.lower()] = seed.id
            for a in embed_pool[m].aliases:
                alias_to_id[a.lower()] = seed.id

    # Skipped (long-tail) entities pass through unmerged
    for e in skipped:
        canonical[e.id] = e
        alias_to_id[e.label.lower()] = e.id
        for a in e.aliases:
            alias_to_id[a.lower()] = e.id

    ctx.entities = canonical
    ctx.alias_to_id = alias_to_id
    print(
        f"[DocGraph][dedupe] {len(raw)} -> {len(canonical)} entities "
        f"(auto-merge pairs: {len(auto_merge_pairs)}, llm-judged: {len(ambiguous_pairs)})"
    )
