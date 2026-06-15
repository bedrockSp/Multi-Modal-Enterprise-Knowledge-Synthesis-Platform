from core.constants import GPU_COMBINATION_LLM, SWITCHES
from core.llm.client import invoke_llm
from core.llm.outputs import CombinationLLMOutput
from core.llm.prompts.combination_prompt import combination_prompt


async def combination_node(
    sub_answers: list, resolved_query: str, original_query: str,
    chunks: list | None = None,
    generative_mode: bool = False,
    answer_class: str | None = None,
) -> str:
    """
    Synthesize multiple sub-answers into one final answer.

    When SWITCHES["SCHEMA_SYNTHESIS"] is on and we're not in generative_mode,
    runs the schema-first pipeline: extract typed records per sub-answer ->
    reconcile globally -> render via canonical template selected by
    answer_class. Falls back to the legacy prose combination on extraction
    failure or when generative_mode is set (scripts/decks etc.).
    """
    use_schema = (
        SWITCHES.get("SCHEMA_SYNTHESIS", True)
        and not generative_mode
        and sub_answers
    )
    if use_schema:
        try:
            return await _schema_synthesis(
                sub_answers=sub_answers,
                resolved_query=resolved_query or original_query,
                original_query=original_query,
                chunks=chunks or [],
                answer_class=answer_class or "narrative",
            )
        except Exception as e:
            print(f"[Synthesis] schema-first path failed: {e}; falling back to prose combine")

    combined_prompt = combination_prompt(
        resolved_query=resolved_query or original_query,
        original_query=original_query,
        sub_answers=sub_answers,
        chunks=chunks,
        generative_mode=generative_mode,
    )

    result: CombinationLLMOutput = await invoke_llm(
        contents=combined_prompt,
        gpu_model=GPU_COMBINATION_LLM.model,
        port=GPU_COMBINATION_LLM.port,
        response_schema=CombinationLLMOutput,
    )

    return result.answer


async def _schema_synthesis(
    sub_answers: list,
    resolved_query: str,
    original_query: str,
    chunks: list,
    answer_class: str,
) -> str:
    """Run extract -> reconcile -> render."""
    from agent.synthesis.extract import extract_records
    from agent.synthesis.reconcile import reconcile, to_render_payload
    from agent.synthesis.render import render

    extractions = await extract_records(sub_answers, chunks)
    total_records = sum(len(e.records) for e in extractions)
    print(f"[Synthesis] extracted {total_records} records from {len(extractions)} sub-answers")

    if total_records == 0:
        # Nothing to render -> fall back to prose combination to avoid an
        # empty/canned answer when there's still useful prose to surface.
        raise RuntimeError("no records extracted; fallback to prose")

    reconciled = reconcile(extractions)
    print(
        f"[Synthesis] reconciled {total_records} -> {len(reconciled)} records "
        f"({sum(1 for r in reconciled if r.conflicting)} conflicts flagged)"
    )

    payload = to_render_payload(reconciled)
    answer = await render(
        answer_class=answer_class,
        original_query=original_query,
        resolved_query=resolved_query,
        records_payload=payload,
    )
    print(f"[Synthesis] rendered answer ({len(answer)} chars, class={answer_class})")
    return answer
