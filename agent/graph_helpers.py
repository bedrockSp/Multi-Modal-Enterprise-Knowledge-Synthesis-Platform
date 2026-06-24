import asyncio
from typing import Dict, List

from agent.state import AgentState
from core.llm.prompts.grounded_inference_prompt import grounded_inference_prompt
from core.llm.prompts.main_prompt import main_prompt
from core.llm.prompts.self_knowledge_prompt import self_knowledge_prompt


def get_recent_history(
    full_history: List[Dict[str, str]], turns: int = 2
) -> List[Dict[str, str]]:
    """
    Returns the most recent conversation turns from the full history.
    Each turn consists of a user message and an assistant response.
    """
    if len(full_history) < turns * 2:
        return full_history

    # Get the last 'turns' pairs of user and AI messages
    recent_history = full_history[-(turns * 2) :]
    return recent_history


async def parallel_search(queries, tool):
    tasks = [tool(query) for query in queries]
    results = await asyncio.gather(*tasks)
    return results


def build_main_prompt(state: AgentState):
    """
    Builds the main prompt for the agent based on the current state.
    """

    return main_prompt(
        messages=state.prior_chat_context,
        prior_answer_mode=state.prior_answer_mode,
        chunks=state.chunks,
        question=state.query or state.resolved_query or state.original_query,
        summary=state.summary,
        web_search_results=state.web_search_results or None,
        initial_search_answer=state.initial_search_answer or None,
        initial_search_results=state.initial_search_results or None,
        mode=state.mode,
        use_self_knowledge=state.use_self_knowledge or False,
        spreadsheet_schema=state.spreadsheet_schema or None,
        sql_result=state.sql_result or None,
        sql_query=state.sql_query or None,
        original_query=state.original_query or None,
        thread_instructions=state.thread_instructions or [],
        triple_context=state.triple_context or None,
        sql_nlp_summary=state.sql_nlp_summary or None,
        sql_batched_answer=state.sql_batched_answer or None,
        doc_batched_answer=state.doc_batched_answer or None,
        vlm_visual_answer=state.vlm_visual_answer or None,
    )


def build_self_knowledge_prompt(
    state: AgentState,
):
    """
    Builds the self-knowledge prompt for the agent based on the current state.
    """

    return self_knowledge_prompt(
        messages=[],
        question=state.query or state.resolved_query or state.original_query,
        thread_instructions=state.thread_instructions or [],
    )


def build_grounded_inference_prompt(state: AgentState):
    """
    Builds the grounded inference prompt.

    Used when the LLM could not answer directly from the retrieved chunks
    (action='failure') but chunks ARE available as a reasoning foundation.
    Allows the LLM to reason analytically on top of the document content.
    """
    return grounded_inference_prompt(
        chunks=state.chunks,
        question=state.query or state.resolved_query or state.original_query,
        thread_instructions=state.thread_instructions or [],
    )
