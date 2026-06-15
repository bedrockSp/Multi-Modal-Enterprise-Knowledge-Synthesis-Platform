"""
Canonical markdown rendering of reconciled records.

One LLM call. The renderer prompt branches on `answer_class` so the SAME
reconciled record set produces a comparison table for comparison queries,
a chronological list for timeline queries, etc. The renderer has only the
records — it cannot invent facts because nothing else is in context.
"""

from typing import List

from core.constants import GPU_COMBINATION_LLM
from core.llm.client import invoke_llm
from core.llm.output_schemas.synthesis_outputs import RenderedAnswer
from core.llm.prompts.synthesis_prompts import build_render_prompt


async def render(
    answer_class: str,
    original_query: str,
    resolved_query: str,
    records_payload: List[dict],
) -> str:
    """
    Returns the markdown answer string. Falls through to "No information
    found" message when there are no records to render.
    """
    if not records_payload:
        return (
            "I could not find supporting information in the uploaded documents "
            "to answer this question."
        )

    prompt = build_render_prompt(
        answer_class=answer_class or "narrative",
        original_query=original_query,
        resolved_query=resolved_query,
        records=records_payload,
    )
    result: RenderedAnswer = await invoke_llm(
        response_schema=RenderedAnswer,
        contents=prompt,
        gpu_model=GPU_COMBINATION_LLM.model,
        port=GPU_COMBINATION_LLM.port,
        remove_thinking=True,
    )
    return result.answer
