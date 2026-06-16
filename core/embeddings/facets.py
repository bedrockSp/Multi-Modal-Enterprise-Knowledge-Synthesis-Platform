"""
Facet extraction — both ingest-time (per document) and query-time (per query).

Strategy:
    1. Run a cheap regex pre-pass for the high-volume temporal facets
       (fiscal_year, quarter, calendar_year). These are deterministic and
       cost nothing.
    2. Call the LLM only to fill the harder facets (doc_type) and to
       confirm/override the regex when text looks ambiguous.

The query path always uses the LLM because the user's phrasing varies more
than document text. Regex fallback there still helps when the LLM is
unavailable.
"""

import re
from typing import Optional

from core.constants import GPU_DECOMPOSITION_LLM
from core.llm.client import invoke_llm
from core.llm.output_schemas.facet_outputs import (
    DOC_TYPES,
    DocumentFacets,
    QueryFacets,
)
from core.llm.prompts.facet_prompts import (
    build_doc_facet_prompt,
    build_query_facet_prompt,
)


_FY_RE = re.compile(
    r"\b(?:FY|fiscal\s+(?:year\s+)?)\s*[' ]?\s*(\d{2}|\d{4})\b",
    re.IGNORECASE,
)
_QUARTER_RE = re.compile(r"\b(Q[1-4])\b", re.IGNORECASE)
_CAL_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

_DOC_SAMPLE_CHARS = 3000


def _normalize_fy(raw: str) -> str:
    n = raw.strip().lstrip("'")
    if len(n) == 2:
        # FY24 -> FY2024 ; treat 00-69 as 2000s, 70-99 as 1900s
        i = int(n)
        n = f"20{i:02d}" if i < 70 else f"19{i:02d}"
    return f"FY{n}"


def regex_fiscal_year(text: str) -> Optional[str]:
    m = _FY_RE.search(text)
    if m:
        return _normalize_fy(m.group(1))
    return None


def regex_quarter(text: str) -> Optional[str]:
    m = _QUARTER_RE.search(text)
    if m:
        return m.group(1).upper()
    return None


def regex_calendar_year(text: str) -> Optional[int]:
    m = _CAL_YEAR_RE.search(text)
    if m:
        try:
            return int(m.group(0))
        except ValueError:
            return None
    return None


def _empty_doc_facets() -> DocumentFacets:
    return DocumentFacets()


def _empty_query_facets() -> QueryFacets:
    return QueryFacets()


async def extract_document_facets(
    title: str,
    text: str,
    summary: Optional[str] = None,
) -> DocumentFacets:
    """
    Extract document-level facets. Combines a regex pre-pass for temporal
    fields with an LLM call (uses the sample = title + summary + head text).
    LLM may override regex when text disambiguates.
    """
    sample_parts = [f"Title: {title}"]
    if summary:
        sample_parts.append(f"Summary: {summary[:1500]}")
    sample_parts.append(f"Body sample:\n{text[:_DOC_SAMPLE_CHARS]}")
    sample = "\n\n".join(sample_parts)

    # Regex pre-pass on the sample so we have a baseline if the LLM bails.
    pre_fy = regex_fiscal_year(sample)
    pre_q = regex_quarter(sample)
    pre_cy = regex_calendar_year(sample)

    try:
        prompt = build_doc_facet_prompt(title=title, sample_text=sample)
        facets: DocumentFacets = await invoke_llm(
            response_schema=DocumentFacets,
            contents=prompt,
            gpu_model=GPU_DECOMPOSITION_LLM.model,
            port=GPU_DECOMPOSITION_LLM.port,
            remove_thinking=True,
        )
    except Exception as e:
        print(f"[Facets][doc] LLM failed for {title!r}: {e}; using regex-only fallback")
        return DocumentFacets(
            fiscal_year=pre_fy,
            quarter=pre_q,
            calendar_year=pre_cy,
            doc_type="other",
        )

    # Backfill any field the LLM left null with the regex pre-pass.
    if not facets.fiscal_year:
        facets.fiscal_year = pre_fy
    if not facets.quarter:
        facets.quarter = pre_q
    if not facets.calendar_year:
        facets.calendar_year = pre_cy
    if facets.doc_type not in DOC_TYPES:
        facets.doc_type = "other"

    return facets


async def extract_query_facets(query: str) -> QueryFacets:
    """Extract locked facets from a user query (single LLM call)."""
    try:
        prompt = build_query_facet_prompt(query=query)
        facets: QueryFacets = await invoke_llm(
            response_schema=QueryFacets,
            contents=prompt,
            gpu_model=GPU_DECOMPOSITION_LLM.model,
            port=GPU_DECOMPOSITION_LLM.port,
            remove_thinking=True,
        )
    except Exception as e:
        print(f"[Facets][query] LLM failed: {e}; using regex-only fallback")
        return QueryFacets(
            fiscal_year=regex_fiscal_year(query),
            quarter=regex_quarter(query),
            calendar_year=regex_calendar_year(query),
            doc_type=None,
        )

    # Sanity-check doc_type against the vocabulary so a wandering LLM doesn't
    # produce filter-breaking values like 'whitepaper'.
    if facets.doc_type and facets.doc_type not in DOC_TYPES:
        facets.doc_type = None

    return facets


def facets_to_chroma_metadata(facets: DocumentFacets) -> dict:
    """Flatten DocumentFacets into Chroma-compatible primitive metadata."""
    out: dict = {}
    if facets.fiscal_year:
        out["fiscal_year"] = facets.fiscal_year
    if facets.quarter:
        out["quarter"] = facets.quarter
    if facets.calendar_year is not None:
        out["calendar_year"] = int(facets.calendar_year)
    if facets.doc_type:
        out["doc_type"] = facets.doc_type
    return out


def query_facets_to_chroma_where(facets: QueryFacets) -> list:
    """
    Convert QueryFacets into ChromaDB `where` conditions to append to the
    existing user_id/thread_id filter. Returns a list of single-clause dicts
    that can be combined under $and at the call site.
    """
    conditions = []
    if facets.fiscal_year:
        conditions.append({"fiscal_year": {"$eq": facets.fiscal_year}})
    if facets.quarter:
        conditions.append({"quarter": {"$eq": facets.quarter}})
    if facets.calendar_year is not None:
        conditions.append({"calendar_year": {"$eq": int(facets.calendar_year)}})
    if facets.doc_type:
        conditions.append({"doc_type": {"$eq": facets.doc_type}})
    return conditions
