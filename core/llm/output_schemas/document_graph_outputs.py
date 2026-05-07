"""
LLM output schemas for the DocumentGraph pipeline.

- ExtractedRelation / RelationExtractionBatch — Phase 5 (relation gap-fill)
- EntityMergeJudgment / EntityMergeBatch — Phase 4 (ambiguous-pair LLM judge)
- CommunityNaming / CommunityNamingBatch — Phase 8 (cluster name + summary)
"""

from typing import List

from pydantic import BaseModel, Field

from core.llm.output_schemas.base import LLMOutputBase


# ─── Phase 5: relation extraction ───────────────────────────────────────────


class ExtractedRelation(BaseModel):
    subject: str = Field(description="Source entity surface form, exactly as it appears in the text.")
    predicate: str = Field(description="Verb-phrase describing the relation (3-6 words).")
    obj: str = Field(description="Target entity surface form, exactly as it appears in the text.")
    confidence: float = Field(
        default=0.6,
        description="Your confidence in this relation in [0,1]. Use 0.4 if unsure, 0.9 if explicit.",
    )
    evidence: str = Field(
        default="",
        description="Short quote (≤25 words) from the text that supports this relation.",
    )


class RelationExtractionBatch(LLMOutputBase):
    relations: List[ExtractedRelation] = Field(
        default_factory=list,
        description="Relations extracted from the chunk(s). Empty list if no clear relations exist.",
    )


# ─── Phase 4: entity merge judgment ─────────────────────────────────────────


class EntityMergeJudgment(BaseModel):
    pair_id: str = Field(description="Echo the pair_id from the input prompt.")
    same: bool = Field(description="True if the two surface forms refer to the same real-world entity.")
    canonical: str = Field(
        default="",
        description="If same=true, the preferred display label. Empty if same=false.",
    )


class EntityMergeBatch(LLMOutputBase):
    judgments: List[EntityMergeJudgment] = Field(
        default_factory=list,
        description="One judgment per input pair; preserve order and pair_id values.",
    )


# ─── Phase 8: community naming ──────────────────────────────────────────────


class CommunityNaming(BaseModel):
    community_id: int = Field(description="Echo the community id from the input prompt.")
    name: str = Field(description="Concise 2-5 word cluster name.")
    summary: str = Field(description="One- or two-sentence summary of the cluster's theme.")


class CommunityNamingBatch(LLMOutputBase):
    communities: List[CommunityNaming] = Field(
        default_factory=list,
        description="One naming per input community; preserve order and community_id values.",
    )
