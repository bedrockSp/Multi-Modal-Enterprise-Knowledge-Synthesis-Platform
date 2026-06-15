from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from core.constants import *
from core.llm.outputs import ChunksUsed
from core.models.gpu_config import GPULLMConfig


class AgentState(BaseModel):
    user_id: str
    thread_id: str
    query: str
    resolved_query: str
    original_query: str
    messages: List[BaseMessage]

    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    web_search: bool = False
    web_search_queries: List[str] = Field(default_factory=list)
    web_search_results: List[Dict[str, Any]] = Field(default_factory=list)
    document_id: Optional[str] = None  # if using document_summarizer
    after_summary: Optional[Literal[f"{ANSWER}", f"{GENERATE}"]] = Field(
        default=f"{GENERATE}", description="The action to be taken after summarization."
    )

    summary: Optional[str] = None

    answer: Optional[str] = None
    chunks_used: List[ChunksUsed] = Field(default_factory=list)
    confidence_score: Optional[str] = Field(
        default=None,
        description="Confidence level of the answer: 'high', 'medium', or 'low'.",
    )
    attempts: int = 0
    web_search_attempts: int = 0

    # SQL query fields for spreadsheet analysis
    sql_query: Optional[str] = None
    sql_result: Optional[str] = None
    sql_last_executed_query: Optional[str] = None  # tracks last executed query for loop detection
    sql_nlp_summary: Optional[str] = None  # pre-extracted NLP theme summary from chunked analysis
    sql_batched_answer: Optional[str] = None  # combined answer from batched SQL processing
    doc_batched_answer: Optional[str] = None  # combined answer from batched multi-doc MapReduce
    vlm_visual_answer: Optional[str] = None  # query-time VLM answer for page/slide/figure references
    requires_full_data: Optional[bool] = None  # LLM-classified: query needs full-data NLP analysis
    answer_class: Optional[str] = Field(
        default="narrative",
        description="Coarse answer shape from decomposition; drives the schema-first synthesis renderer.",
    )
    sql_attempts: int = 0
    has_spreadsheet_data: bool = False
    spreadsheet_only: bool = False  # True when ALL thread documents are spreadsheets
    spreadsheet_schema: Optional[str] = None

    # Excel Skill fields
    excel_request: Optional[str] = None  # User's Excel creation request extracted by LLM
    excel_result: Optional[str] = None  # Download URL after Excel generation

    action: Optional[
        Literal[
            f"{ANSWER}",
            f"{WEB_SEARCH}",
            f"{DOCUMENT_SUMMARIZER}",
            f"{GLOBAL_SUMMARIZER}",
            f"{FAILURE}",
            f"{SQL_QUERY}",
            f"{EXCEL_CREATE}",
        ]
    ] = Field(
        default=None,
        description="The action to be taken by the agent. Can be 'answer', 'web_search', 'document_summarizer', 'global_summarizer', 'sql_query', 'excel_create', or 'failure'.",
    )

    # Used to determine the next step in the state graph
    next: Optional[str] = None
    llm: Optional[GPULLMConfig] = Field(
        default=None, description="The model to be used for the agent."
    )  # add more validation
    mode: Literal[f"{INTERNAL}", f"{EXTERNAL}"] = Field(
        description="The mode of the agent, either 'Internal' or 'External'."
    )
    initial_search_answer: Optional[str] = None  # to store initial web search answer
    initial_search_results: List[Dict[str, Any]] = Field(
        default_factory=list
    )  # to store initial web search results
    use_self_knowledge: bool = False
    thread_instructions: List[str] = Field(
        default_factory=list,
        description="User-defined instructions that apply to every message in this thread.",
    )

    # Full-document retrieval mode — retrieves ALL chunks from the target document
    # instead of relevance-filtered top-k.  Triggered by "each slide", "every page", etc.
    full_document_mode: bool = False

    # Phase 2.1: CRAG Corrective Retrieval
    retrieval_attempts: int = 0  # How many times retrieval has been performed
    retrieval_verdict: Optional[str] = Field(
        default=None,
        description="Evaluator verdict: 'sufficient', 'ambiguous', or 'insufficient'.",
    )

    # Phase 2.2: Enhanced Decomposition — sub-queries for parallel retrieval
    sub_queries: List[str] = Field(
        default_factory=list,
        description="Sub-queries from decomposition for parallel retrieval within the agent.",
    )

    # Semantic retrieval expansion — alternative phrasings for broader vector search
    requires_retrieval_expansion: bool = Field(
        default=False,
        description="True only when the decomposition LLM judges the query benefits from semantic expansion. Narrow scoped queries (specific entity + facet) should set this False so locked constraints survive.",
    )
    retrieval_queries: List[str] = Field(
        default_factory=list,
        description="LLM-generated alternative query phrasings with synonyms and related terminology for wider retrieval coverage. Only used when requires_retrieval_expansion is True AND SWITCHES['QUERY_EXPANSION'] is on.",
    )

    # Phase 3.2: Triple context — entity relationships injected at retrieval time
    triple_context: Optional[str] = Field(
        default=None,
        description="Formatted entity relationship triples relevant to the query.",
    )
