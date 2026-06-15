"""
Per-sub-query record extraction.

Takes the existing prose sub-answer plus the chunks that grounded it and
turns them into a list of typed EvidenceRecords via one LLM call per
sub-query. Calls run in parallel across sub-queries.
"""

import asyncio
import re
from typing import Dict, List

from core.constants import GPU_COMBINATION_LLM
from core.llm.client import invoke_llm
from core.llm.output_schemas.synthesis_outputs import EvidenceRecord, SubAnswerRecords
from core.llm.prompts.synthesis_prompts import build_extraction_prompt


# Sub-query chunks may not be available per-sub-query (the route dedupes chunks
# at thread level before combination). When per-sub-query chunks are unknown
# we fall back to the global deduped chunk set, capped to top-N.
_MAX_CHUNKS_PER_SUBQUERY = 10
_PARALLEL_EXTRACTIONS = 4


def _filter_chunks_for_subquery(
    sub_query: str, all_chunks: List[Dict], cap: int = _MAX_CHUNKS_PER_SUBQUERY
) -> List[Dict]:
    """
    Cheap term-overlap scoring to pick the most relevant chunks for this
    sub-query when per-sub-query chunk lists aren't available. Falls back
    to top-by-rerank-score if the sub_query has no usable terms.
    """
    if not all_chunks:
        return []

    tokens = {t.lower() for t in re.findall(r"\w+", sub_query) if len(t) > 2}
    if not tokens:
        return sorted(all_chunks, key=lambda c: c.get("rerank_score", 0.0), reverse=True)[:cap]

    scored = []
    for c in all_chunks:
        content = (c.get("content", "") or "").lower()
        if not content:
            continue
        overlap = sum(1 for t in tokens if t in content)
        if overlap == 0:
            continue
        base = c.get("rerank_score", 0.0)
        scored.append((overlap + base, c))

    if not scored:
        return sorted(all_chunks, key=lambda c: c.get("rerank_score", 0.0), reverse=True)[:cap]

    scored.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scored[:cap]]


async def _extract_one(
    sub_query: str, sub_answer: str, chunks: List[Dict]
) -> SubAnswerRecords:
    if not sub_answer or not sub_answer.strip():
        return SubAnswerRecords(sub_query=sub_query, records=[])

    prompt = build_extraction_prompt(sub_query, sub_answer, chunks)
    try:
        result: SubAnswerRecords = await invoke_llm(
            response_schema=SubAnswerRecords,
            contents=prompt,
            gpu_model=GPU_COMBINATION_LLM.model,
            port=GPU_COMBINATION_LLM.port,
            remove_thinking=True,
        )
        return result
    except Exception as e:
        print(f"[Synthesis][extract] sub_query extraction failed: {e}")
        return SubAnswerRecords(sub_query=sub_query, records=[])


async def extract_records(
    sub_answers: List[Dict], chunks: List[Dict]
) -> List[SubAnswerRecords]:
    """
    Run record extraction over all sub-answers with bounded concurrency.

    Args:
        sub_answers: list of dicts with keys {sub_query, sub_answer} from the
            existing query route.
        chunks: globally-deduped chunk set produced by the route before
            combination. Used to choose the most relevant subset per sub-query.

    Returns:
        List[SubAnswerRecords] in the same order as `sub_answers`.
    """
    if not sub_answers:
        return []

    sem = asyncio.Semaphore(_PARALLEL_EXTRACTIONS)

    async def _bounded(item):
        async with sem:
            sub_query = item.get("sub_query", "")
            sub_answer = item.get("sub_answer", "") or ""
            relevant = _filter_chunks_for_subquery(sub_query, chunks)
            return await _extract_one(sub_query, sub_answer, relevant)

    return await asyncio.gather(*[_bounded(item) for item in sub_answers])
