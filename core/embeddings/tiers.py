"""
Evidence-tier routing (rag-refactor step 7).

Today three chunk types — `child` (raw document fragments), `document_summary`
(synthetic per-doc summaries), and `entity_profile` (synthetic per-entity
profiles) — share the same ChromaDB collection and compete in the same RRF
pool. Result: a doc-level summary mentioning every topic broadly can outrank
the actual evidence chunk for a precise factoid question.

This module classifies a query into focused vs broad retrieval and returns
the chunk_type set the retriever should consider.

    focused: ["child"]               — primary evidence only
    broad:   ["child", "document_summary", "entity_profile"]
"""

from typing import List, Optional


PRIMARY_TIER: List[str] = ["child"]
BROAD_TIERS: List[str] = ["child", "document_summary", "entity_profile"]


# Answer classes that benefit from synthetic tiers (summaries, profiles)
_BROAD_CLASSES = frozenset(
    {
        "narrative",
        "enumeration",
        "achievements_by_period",
        "timeline",
        "multi_entity_summary",
    }
)

# Answer classes that should restrict to primary chunks — synthetic tiers
# tend to dilute precise lookups.
_FOCUSED_CLASSES = frozenset({"factoid", "comparison", "ranking"})


def select_tiers_for_answer_class(answer_class: Optional[str]) -> List[str]:
    """
    Pick the chunk_type vocabulary appropriate for the answer class.
    Default to BROAD when answer_class is missing or unrecognized — better
    to let synthetic tiers compete than to silently exclude them.
    """
    if not answer_class:
        return BROAD_TIERS
    ac = answer_class.lower()
    if ac in _FOCUSED_CLASSES:
        return PRIMARY_TIER
    if ac in _BROAD_CLASSES:
        return BROAD_TIERS
    return BROAD_TIERS


def tiers_to_chroma_condition(tiers: List[str]) -> Optional[dict]:
    """Return the ChromaDB where condition that restricts to the given tiers."""
    if not tiers:
        return None
    if len(tiers) == 1:
        return {"chunk_type": {"$eq": tiers[0]}}
    return {"chunk_type": {"$in": tiers}}
