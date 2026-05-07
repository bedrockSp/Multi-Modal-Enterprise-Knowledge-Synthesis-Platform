"""
Pydantic data models for the DocumentGraph feature.

These describe the shape of an in-memory knowledge graph built from a thread's
documents. Persistence (Kuzu / NetworkX-on-disk) lives in store.py and converts
to/from these types at the boundary.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ChunkRef(BaseModel):
    """Provenance pointer back to a single source chunk."""

    document_id: str
    page_no: int = 0
    chunk_index: int = 0
    file_name: str = ""
    title: str = ""


class Entity(BaseModel):
    """A node in the document graph — person, org, concept, product, location, etc."""

    id: str = Field(description="Stable canonical id (slug of normalized label).")
    label: str = Field(description="Display name (canonical alias after dedupe).")
    type: str = Field(default="OTHER", description="Coarse type tag from NER.")
    aliases: List[str] = Field(
        default_factory=list,
        description="Other surface forms that were merged into this entity.",
    )
    frequency: int = Field(
        default=0, description="Total chunk-mention count across the corpus."
    )
    doc_count: int = Field(
        default=0, description="Number of distinct documents this entity appears in."
    )
    community_id: Optional[int] = Field(
        default=None, description="Leiden community id, set after Phase 7."
    )
    profile: str = Field(
        default="",
        description="Optional short profile text harvested from entity-profile chunks.",
    )
    provenance: List[ChunkRef] = Field(
        default_factory=list,
        description="Up to N chunk refs where this entity is mentioned.",
    )


class Relation(BaseModel):
    """A directed edge between two entities."""

    source_id: str
    target_id: str
    predicate: str = Field(description="Verb-phrase describing the relation.")
    confidence: float = Field(
        default=0.5, description="Extractor confidence in [0,1]."
    )
    extractor: str = Field(
        default="triple_store",
        description="Origin: 'triple_store' | 'llm' | 'ner_cooccurrence'.",
    )
    provenance: ChunkRef = Field(
        default_factory=ChunkRef,
        description="The chunk this edge was extracted from.",
    )
    evidence: str = Field(
        default="",
        description="Optional 1-2 sentence quote or paraphrase supporting the edge.",
    )


class Community(BaseModel):
    """A Leiden cluster of related entities."""

    id: int
    name: str = Field(default="", description="LLM-generated short name.")
    summary: str = Field(default="", description="LLM-generated 2-line summary.")
    member_ids: List[str] = Field(default_factory=list)
    size: int = 0


class GraphMetadata(BaseModel):
    """Header info stored alongside the graph for fast listing."""

    user_id: str
    thread_id: str
    version: int = 1
    status: str = Field(default="pending", description="pending|building|ready|failed")
    built_at: Optional[datetime] = None
    doc_count: int = 0
    node_count: int = 0
    edge_count: int = 0
    community_count: int = 0
    error: str = ""


class DocumentGraph(BaseModel):
    """The full graph payload returned to the frontend."""

    metadata: GraphMetadata
    nodes: List[Entity] = Field(default_factory=list)
    edges: List[Relation] = Field(default_factory=list)
    communities: List[Community] = Field(default_factory=list)


class GraphBuildContext(BaseModel):
    """
    Mutable accumulator passed between pipeline phases. Borrowed from
    GitNexus's single-graph-accumulator pattern — each phase reads what
    upstream phases produced and writes its own additions.
    """

    user_id: str
    thread_id: str
    # Phase 1 — raw chunks
    chunks: List[Dict] = Field(default_factory=list)
    # Phase 2 — entity candidates keyed by surface form (lowercased)
    raw_entities: Dict[str, Entity] = Field(default_factory=dict)
    # Phase 3 — raw triples from existing TripleStore
    raw_triples: List[Dict] = Field(default_factory=list)
    # Phase 4 — entities after dedupe; alias_to_id maps every surface form to canonical id
    entities: Dict[str, Entity] = Field(default_factory=dict)
    alias_to_id: Dict[str, str] = Field(default_factory=dict)
    # Phase 5+ — accumulated relations
    relations: List[Relation] = Field(default_factory=list)
    # Phase 7 — communities
    communities: List[Community] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
