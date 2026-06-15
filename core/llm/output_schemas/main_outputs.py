from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from core.llm.output_schemas.base import LLMOutputBase


class ChunksUsed(BaseModel):
    document_id: str = Field(
        description="The ID of the document used to which the chunk belongs."
    )
    title: Optional[str] = Field(
        default=None,
        description="The title/name of the document used.",
    )
    page_no: int = Field(description="The page_no of the document used.")


class MainLLMOutputInternal(LLMOutputBase):
    answer: str = Field(default="", description="The answer to the user's question.")
    action: Literal[
        "answer",
        "document_summarizer",  # requires document id of the document to summarize
        "global_summarizer",
        "failure",
        "sql_query",  # query spreadsheet data via SQL - use for ANY spreadsheet-related question
        "excel_create",  # create a downloadable Excel file from the data
    ] = Field(
        description="The action to take based on the answer. Use 'sql_query' for ANY question that can be answered from spreadsheet/CSV data. Use 'excel_create' when the user wants to create/export/download an Excel file."
    )
    chunks_used: Optional[List[ChunksUsed]] = Field(
        default=None,
        description="List of chunks used to generate the answer, if applicable.",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="The ID of the document to summarize if using document_summarizer, if applicable.",
    )
    sql_query: Optional[str] = Field(
        default=None,
        description="The SQL SELECT query to execute against the spreadsheet data. Required when action is 'sql_query'.",
    )
    excel_request: Optional[str] = Field(
        default=None,
        description="Natural-language description of the Excel file to create. Required when action is 'excel_create'. E.g., 'Create a pivot table of sales by region with a bar chart'.",
    )
    @field_validator("action", mode="before")
    @classmethod
    def normalize_action(cls, v):
        if hasattr(v, "lower"):
            val = v.lower().strip()
            # Map common hallucinations or partial matches
            if val in ["failure", "fail", "error"]:
                return "failure"
            if val in ["search", "google", "web"]:
                return "web_search"
            if val in ["query", "database", "sql"]:
                return "sql_query"
            if val in ["summarize", "summary"]:
                return "document_summarizer"
            if val in ["excel", "spreadsheet", "export", "download_excel", "create_excel"]:
                return "excel_create"
            return val
        return v


class MainLLMOutputInternalWithFailure(LLMOutputBase):
    answer: str = Field(default="", description="The answer to the user's question.")
    action: Literal[
        "answer",
        "document_summarizer",  # requires document id of the document to summarize
        "global_summarizer",
        "failure",
        "sql_query",  # query spreadsheet data via SQL - use for ANY spreadsheet-related question
        "excel_create",  # create a downloadable Excel file from the data
    ] = Field(
        description="The action to take based on the answer. Use 'sql_query' for ANY question that can be answered from spreadsheet/CSV data. Use 'excel_create' when the user wants to create/export/download an Excel file."
    )
    chunks_used: Optional[List[ChunksUsed]] = Field(
        default=None,
        description="List of chunks used to generate the answer, if applicable.",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="The ID of the document to summarize if using document_summarizer, if applicable.",
    )
    sql_query: Optional[str] = Field(
        default=None,
        description="The SQL SELECT query to execute against the spreadsheet data. Required when action is 'sql_query'.",
    )
    excel_request: Optional[str] = Field(
        default=None,
        description="Natural-language description of the Excel file to create. Required when action is 'excel_create'.",
    )
    @field_validator("action", mode="before")
    @classmethod
    def normalize_action(cls, v):
        if hasattr(v, "lower"):
            val = v.lower().strip()
            # Map common hallucinations or partial matches
            if val in ["failure", "fail", "error"]:
                return "failure"
            if val in ["search", "google", "web"]:
                return "web_search"
            if val in ["query", "database", "sql"]:
                return "sql_query"
            if val in ["summarize", "summary"]:
                return "document_summarizer"
            if val in ["excel", "spreadsheet", "export", "download_excel", "create_excel"]:
                return "excel_create"
            return val
        return v


class MainLLMOutputExternal(LLMOutputBase):
    answer: str = Field(default="", description="The answer to the user's question.")
    action: Literal[
        "answer",
        "web_search",
        "document_summarizer",  # requires document id of the document to summarize
        "global_summarizer",
        "failure",
        "sql_query",  # query spreadsheet data via SQL - use for ANY spreadsheet-related question
        "excel_create",  # create a downloadable Excel file from the data
    ] = Field(
        description="The action to take based on the answer. Use 'sql_query' for ANY question that can be answered from spreadsheet/CSV data. Use 'excel_create' when the user wants to create/export/download an Excel file."
    )
    chunks_used: Optional[List[ChunksUsed]] = Field(
        default=None,
        description="List of chunks used to generate the answer, if applicable.",
    )
    web_search_queries: Optional[List[str]] = Field(
        default=None,
        description="List of 2-3 web search queries used to generate the answer, if applicable.",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="The ID of the document to summarize if using document_summarizer, if applicable.",
    )
    sql_query: Optional[str] = Field(
        default=None,
        description="The SQL SELECT query to execute against the spreadsheet data. Required when action is 'sql_query'.",
    )
    excel_request: Optional[str] = Field(
        default=None,
        description="Natural-language description of the Excel file to create. Required when action is 'excel_create'.",
    )
    @field_validator("action", mode="before")
    @classmethod
    def normalize_action(cls, v):
        if hasattr(v, "lower"):
            val = v.lower().strip()
            if val in ["search", "google", "web"]:
                return "web_search"
            if val in ["query", "database", "sql"]:
                return "sql_query"
            if val in ["summarize", "summary"]:
                return "document_summarizer"
            if val in ["excel", "spreadsheet", "export", "download_excel", "create_excel"]:
                return "excel_create"
            return val
        return v


class SelfKnowledgeLLMOutput(LLMOutputBase):
    answer: str = Field(description="The answer to the user's question.")


class DecompositionLLMOutput(LLMOutputBase):
    requires_decomposition: bool = Field(
        description="Indicates whether the query requires decomposition."
    )
    resolved_query: str = Field(
        description="The resolved query after context resolution."
    )
    sub_queries: List[str] = Field(
        description="List of standalone sub-queries generated from the original query."
    )
    requires_retrieval_expansion: bool = Field(
        default=False,
        description=(
            "Set true ONLY when the query is broad/conceptual, acronym-heavy, or uses "
            "vocabulary that the documents are unlikely to use directly. Set false for "
            "narrow scoped queries with specific entities/time/facets (e.g., 'Achievements "
            "of FY24'), direct factual lookups, or spreadsheet-routable questions — these "
            "must not be broadened or the system will return off-topic chunks."
        ),
    )
    retrieval_queries: List[str] = Field(
        default_factory=list,
        description=(
            "Used only when requires_retrieval_expansion is true. 2-3 alternative phrasings "
            "of the resolved query using synonyms, related terminology, and different "
            "vocabulary that documents might use — e.g., 'timelines' → 'milestones and "
            "schedule', 'SoW' → 'Statement of Work scope and deliverables'. MUST be empty "
            "when requires_retrieval_expansion is false."
        ),
    )
    answer_class: str = Field(
        default="narrative",
        description=(
            "Coarse shape of the expected answer — drives the canonical renderer. One of: "
            "'factoid' (single fact lookup), 'comparison' (2+ entities side-by-side), "
            "'timeline' (events over time), 'enumeration' (list of items like pros/cons/modules), "
            "'achievements_by_period' (accomplishments grouped by entity x time period), "
            "'multi_entity_summary' (summarise each of several entities), "
            "'ranking' (sorted by metric), 'narrative' (free-form prose, default)."
        ),
    )
    requires_full_data: bool = Field(
        default=False,
        description=(
            "Set to true when the question requires reading/understanding the text content "
            "of ALL rows in a spreadsheet — e.g., theme extraction, sentiment analysis, "
            "categorization of text, identifying patterns, qualitative analysis. "
            "False for counts, filters, lookups, aggregations, or non-spreadsheet questions."
        ),
    )


class CombinationLLMOutput(LLMOutputBase):
    answer: str = Field(description="The combined answer from multiple sub-answers.")
