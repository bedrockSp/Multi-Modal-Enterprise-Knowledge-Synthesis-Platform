"""
Locked-facet schemas (rag-refactor step 5).

DocumentFacets — extracted once per document at ingest time and stamped onto
every chunk's ChromaDB metadata.
QueryFacets — extracted once per user query and applied as HARD filters at
the ChromaDB where-clause layer. Locked facets pass through CRAG re-retrieval
unchanged so a corrective refinement cannot silently relax e.g. fiscal_year.
"""

from typing import Optional

from pydantic import BaseModel, Field

from core.llm.output_schemas.base import LLMOutputBase


# Canonical doc type vocabulary — kept tight so filtering is reliable.
# Extend in a follow-up once we see real coverage gaps.
DOC_TYPES = (
    "report",
    "policy",
    "contract",
    "sow",
    "deck",
    "memo",
    "earnings",
    "manual",
    "spec",
    "minutes",
    "other",
)


class DocumentFacets(LLMOutputBase):
    fiscal_year: Optional[str] = Field(
        default=None,
        description=(
            "Primary fiscal year covered by this document, normalized as 'FYYYYY' (e.g. 'FY2024'). "
            "Use null when the document is not scoped to a fiscal year (e.g. evergreen policy, "
            "reference manual)."
        ),
    )
    quarter: Optional[str] = Field(
        default=None,
        description="Primary quarter as 'Q1', 'Q2', 'Q3', or 'Q4'. Null if the document covers a whole year or no quarter is identifiable.",
    )
    calendar_year: Optional[int] = Field(
        default=None,
        description="Primary calendar year (4-digit). Use null when no specific year applies.",
    )
    doc_type: str = Field(
        default="other",
        description=(
            "One of: report, policy, contract, sow, deck, memo, earnings, manual, spec, minutes, other. "
            "Default to 'other' when uncertain."
        ),
    )


class QueryFacets(LLMOutputBase):
    """
    Locked facets extracted from a user query. Empty values mean the user did
    not constrain that dimension — they are NOT 'no documents match'.
    """

    fiscal_year: Optional[str] = Field(
        default=None,
        description="If the query names a fiscal year (FY24, fiscal 2024), normalize to 'FYYYYY'. Else null.",
    )
    quarter: Optional[str] = Field(
        default=None,
        description="If the query names a quarter, normalize to 'Q1'/'Q2'/'Q3'/'Q4'. Else null.",
    )
    calendar_year: Optional[int] = Field(
        default=None,
        description="If the query names a calendar year (2023, 2024) and NOT a fiscal year, use the int. Else null.",
    )
    doc_type: Optional[str] = Field(
        default=None,
        description=(
            "If the query restricts to a document type (e.g. 'in the contract', 'per the policy', "
            "'from the SOW'), set it to one of: report, policy, contract, sow, deck, memo, "
            "earnings, manual, spec, minutes. Else null."
        ),
    )
