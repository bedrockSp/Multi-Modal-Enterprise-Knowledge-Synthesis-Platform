"""
Community-summary global search (rag-refactor step 8).

Loads the DocumentGraph for the active thread, scores its Leiden communities
against the user query by term overlap, and converts the top-K communities
into EvidenceRecord entries that feed the schema-first synthesizer (step 6).

Graceful degradation by design — every failure path returns [] and the caller
falls back to chunk-only retrieval:
    * No DocumentGraph for this thread          -> []
    * Graph exists but has zero communities      -> []
    * answer_class is focused (factoid/ranking)  -> []
    * No community shares any terms with query   -> []
    * Any unexpected error                       -> []  (logged)
"""

import asyncio
import re
from typing import List, Optional

from core.document_graph.store import get_store
from core.llm.output_schemas.synthesis_outputs import EvidenceRecord


# Answer classes for which community summaries help. Focused classes
# (factoid, comparison, ranking) skip community search because synthetic
# summaries dilute the precise evidence those queries need.
_BROAD_ANSWER_CLASSES = frozenset({
    "narrative",
    "enumeration",
    "achievements_by_period",
    "timeline",
    "multi_entity_summary",
})

_TOP_K_DEFAULT = 5
_MIN_OVERLAP = 1


def _tokenize(s: str) -> set:
    return {t for t in re.findall(r"\w+", s.lower()) if len(t) > 2}


def _score_community(query_tokens: set, name: str, summary: str) -> int:
    text_tokens = _tokenize(f"{name} {summary}")
    return len(query_tokens & text_tokens)


async def community_search(
    user_id: str,
    thread_id: str,
    query: str,
    answer_class: Optional[str],
    top_k: int = _TOP_K_DEFAULT,
) -> List[EvidenceRecord]:
    """
    Return a list of pre-structured EvidenceRecords harvested from the
    DocumentGraph's community summaries. Safe to call unconditionally — every
    "no graph / no relevance" path returns an empty list with a log line.
    """
    ac = (answer_class or "").lower()
    if ac and ac not in _BROAD_ANSWER_CLASSES:
        print(f"[Community Search] answer_class={ac!r} is focused, skipping")
        return []

    try:
        store = get_store()
    except Exception as e:
        print(f"[Community Search] store init failed: {e}")
        return []

    if not store.exists(user_id, thread_id):
        print(f"[Community Search] no DocumentGraph for thread {thread_id}, skipping")
        return []

    try:
        graph = await asyncio.to_thread(store.read_graph, user_id, thread_id)
    except Exception as e:
        print(f"[Community Search] read_graph failed: {e}")
        return []

    if not graph or not graph.communities:
        print(f"[Community Search] graph has no communities, skipping")
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    # Rank communities by term overlap; ties broken by community size.
    scored = []
    for c in graph.communities:
        overlap = _score_community(query_tokens, c.name or "", c.summary or "")
        if overlap < _MIN_OVERLAP:
            continue
        scored.append((overlap, c.size, c))

    if not scored:
        print(
            f"[Community Search] {len(graph.communities)} communities exist but none "
            f"overlap with query terms, skipping"
        )
        return []

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    selected = [c for _, _, c in scored[:top_k]]

    # Build provenance lookup once. Entity provenance carries the doc title and
    # page number harvested during graph construction so we can cite concretely
    # rather than "the document graph said so".
    entity_by_id = {e.id: e for e in graph.nodes}

    records: List[EvidenceRecord] = []
    for community in selected:
        member_entities = [
            entity_by_id[mid] for mid in community.member_ids if mid in entity_by_id
        ]
        if not member_entities:
            continue
        # Pick the most-mentioned member as the representative for citation.
        representative = max(member_entities, key=lambda e: e.frequency)
        provenance = representative.provenance[0] if representative.provenance else None

        sample_labels = ", ".join(e.label for e in member_entities[:5])
        records.append(
            EvidenceRecord(
                claim=(community.summary or f"Topic cluster: {community.name}").strip(),
                entity=(community.name or f"Cluster {community.id}").strip(),
                source_doc=(provenance.title if provenance and provenance.title else "DocumentGraph"),
                source_page=(provenance.page_no if provenance else None),
                evidence_span=(
                    f"Community of {community.size} entities including: {sample_labels}"
                ),
                # Community summaries are LLM-derived (step 8 of graph build),
                # not direct quotes — moderate confidence is appropriate.
                confidence=0.65,
            )
        )

    print(
        f"[Community Search] selected {len(records)}/{len(graph.communities)} communities "
        f"for query (top overlap={scored[0][0]} terms)"
    )
    return records
