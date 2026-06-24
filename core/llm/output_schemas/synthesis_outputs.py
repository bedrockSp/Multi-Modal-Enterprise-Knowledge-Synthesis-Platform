"""
Schema-first synthesis outputs (rag-refactor step 6).

Replaces free-form prose-then-combine with typed evidence records that are
extracted per sub-query, reconciled globally, and then rendered via a
canonical template selected by `AnswerClass`. Eliminates the style drift
between independently-generated sub-answers.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from core.llm.output_schemas.base import LLMOutputBase


class AnswerClass(str, Enum):
    """Coarse answer shape — drives the renderer template."""

    FACTOID = "factoid"                          # single fact: who/what/when/where lookups
    COMPARISON = "comparison"                    # 2+ entities side-by-side table
    TABULAR = "tabular"                          # generic HTML table — explicit "as a table" request that isn't entity-vs-entity
    TIMELINE = "timeline"                        # events ordered by time
    ENUMERATION = "enumeration"                  # list of items (pros/cons, modules, features)
    ACHIEVEMENTS_BY_PERIOD = "achievements_by_period"  # accomplishments grouped by entity x period
    MULTI_ENTITY_SUMMARY = "multi_entity_summary"      # multiple entities summarized
    RANKING = "ranking"                          # sorted by metric/value
    NARRATIVE = "narrative"                      # free-form prose answer (default)


class EvidenceRecord(BaseModel):
    """A single typed claim extracted from a sub-answer + its supporting chunks."""

    claim: str = Field(
        description="The factual claim in 1-2 sentences. Self-contained — do NOT use pronouns or 'this/that'."
    )
    entity: Optional[str] = Field(
        default=None,
        description="The subject entity if applicable — team name, product, person, org, etc.",
    )
    value: Optional[str] = Field(
        default=None,
        description="Specific value/data point if the claim contains one (e.g. '30%', '$2.4B', 'Q3 2024').",
    )
    metric: Optional[str] = Field(
        default=None,
        description="The metric or attribute if applicable (e.g. 'revenue', 'processing time reduction').",
    )
    time_period: Optional[str] = Field(
        default=None,
        description="Time period this claim applies to (e.g. 'FY24', 'Q3 2024', '2023-2024').",
    )
    source_doc: str = Field(
        description="Document title or file name this claim came from. Use exact title from chunks."
    )
    source_page: Optional[int] = Field(
        default=None, description="Page number in the source document if known."
    )
    evidence_span: str = Field(
        default="",
        description="Short quote (<= 30 words) from a chunk that grounds this claim. Empty if no direct quote available.",
    )
    confidence: float = Field(
        default=0.7,
        description="Your confidence in this claim being correctly supported by the source [0,1]. 0.5 = guess, 0.9 = explicit quote.",
    )


class SubAnswerRecords(LLMOutputBase):
    """One LLM extraction pass turns a (sub_query + sub_answer + chunks) into records."""

    sub_query: str = Field(description="Echo the sub_query verbatim.")
    records: List[EvidenceRecord] = Field(
        default_factory=list,
        description="One record per distinct claim. Empty list if the sub-answer reports no evidence.",
    )


class ReconciledRecord(BaseModel):
    """Output of the reconcile step — same shape as EvidenceRecord plus group metadata."""

    record: EvidenceRecord
    duplicates: List[EvidenceRecord] = Field(
        default_factory=list,
        description="Other records merged into this one (same claim/entity/time).",
    )
    conflicting: List[EvidenceRecord] = Field(
        default_factory=list,
        description="Records that share entity+metric+time but disagree on value.",
    )


class RenderedAnswer(LLMOutputBase):
    """Final renderer output."""

    answer: str = Field(
        description="The canonical markdown answer rendered from reconciled records."
    )
