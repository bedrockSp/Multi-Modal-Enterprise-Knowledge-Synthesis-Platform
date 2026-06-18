"""
Contextual Retrieval (rag-refactor step 10).

Per Anthropic's "Contextual Retrieval in AI Systems" (Sept 2024): prepending
a short LLM-generated context that situates each chunk within its parent
document measurably improves retrieval — Anthropic reported a 35% reduction
in top-20 retrieval failure rate with contextual embeddings alone, and 49%
with contextual BM25 added.

This module produces per-chunk context blurbs at ingest time, with bounded
concurrency so a 100-chunk PDF adds ~25 seconds of ingest latency rather
than 100. Failures degrade gracefully: a single chunk that fails to get a
context blurb falls back to programmatic-only enrichment.

Gated by SWITCHES["CONTEXTUAL_RETRIEVAL"] so existing deployments keep their
current ingest behavior unchanged.
"""

import asyncio
from typing import List, Optional, Tuple

from pydantic import Field

from core.constants import GPU_DOC_SUMMARIZER_LLM
from core.llm.client import invoke_llm
from core.llm.output_schemas.base import LLMOutputBase


class ContextualPrefix(LLMOutputBase):
    """Output schema for a single chunk's contextual prefix."""

    context: str = Field(
        description=(
            "1-3 sentence context situating the chunk within the document. Should "
            "name the document, the section/topic, and any antecedents (e.g. acronyms, "
            "subjects referenced by pronouns) needed to make the chunk self-standing. "
            "Do NOT repeat the chunk text — only the surrounding context."
        )
    )


_PARALLEL_CONTEXT_CALLS = 4
_MAX_PARENT_CHARS = 1500  # parent text rarely exceeds this; trim defensively
_MAX_CHUNK_CHARS = 1200


def _build_prompt(
    doc_title: str,
    doc_summary: Optional[str],
    parent_text: str,
    chunk_text: str,
) -> str:
    parent_block = parent_text[:_MAX_PARENT_CHARS]
    chunk_block = chunk_text[:_MAX_CHUNK_CHARS]
    summary_block = (
        f"\n<document_summary>\n{doc_summary[:1500]}\n</document_summary>\n"
        if doc_summary
        else ""
    )
    return f"""You produce a short context blurb that will be PREPENDED to a chunk of
text so that the chunk can be retrieved correctly even when read in isolation.

<document_title>
{doc_title}
</document_title>
{summary_block}
<section>
{parent_block}
</section>

Here is the specific chunk you must situate within the document:
<chunk>
{chunk_block}
</chunk>

RULES
1. Produce 1-3 sentences. No more.
2. Name the document title and the section/topic the chunk sits inside.
3. Resolve antecedents — if the chunk uses pronouns, acronyms, or shorthand,
   spell them out in the context (e.g. "Refers to the FY24 budget proposal,"
   "Acme Corp's Q3 earnings call," etc.).
4. Do NOT repeat the chunk text. The context is SEPARATE from the chunk.
5. Do NOT add preamble like "Here is the context:" — output only the context.
"""


async def _generate_one(
    doc_title: str,
    doc_summary: Optional[str],
    parent_text: str,
    chunk_text: str,
) -> str:
    try:
        prompt = _build_prompt(doc_title, doc_summary, parent_text, chunk_text)
        result: ContextualPrefix = await invoke_llm(
            response_schema=ContextualPrefix,
            contents=prompt,
            gpu_model=GPU_DOC_SUMMARIZER_LLM.model,
            port=GPU_DOC_SUMMARIZER_LLM.port,
            remove_thinking=True,
        )
        return (result.context or "").strip()
    except Exception as e:
        # Single-chunk failures are non-fatal — caller falls back to programmatic
        # enrichment for that chunk. Log lightly to avoid noise on every retry.
        print(f"[ContextualRetrieval] context generation failed: {e}")
        return ""


async def generate_chunk_contexts(
    doc_title: str,
    doc_summary: Optional[str],
    chunk_pairs: List[Tuple[str, str]],
) -> List[str]:
    """
    Generate one context blurb per chunk.

    Args:
        doc_title: Document title (used in every prompt).
        doc_summary: Optional doc-level summary for broader framing.
        chunk_pairs: List of (parent_text, child_text) tuples in the same
            order the caller wants the contexts back.

    Returns:
        List[str] aligned with `chunk_pairs`. Empty strings for chunks where
        the LLM call failed — caller should fall back to programmatic context.
    """
    if not chunk_pairs:
        return []

    sem = asyncio.Semaphore(_PARALLEL_CONTEXT_CALLS)

    async def _bounded(pair):
        async with sem:
            parent, child = pair
            return await _generate_one(doc_title, doc_summary, parent, child)

    return await asyncio.gather(*[_bounded(p) for p in chunk_pairs])
