"""Prompts for schema-first synthesis (rag-refactor step 6)."""

import json
from typing import List


def build_extraction_prompt(
    sub_query: str,
    sub_answer: str,
    chunks: List[dict],
) -> str:
    """
    Per-sub-query extraction: turn a prose sub-answer + its supporting chunks
    into a list of typed EvidenceRecords. The LLM should not invent facts —
    everything must be grounded in the sub_answer text or chunk text.
    """
    chunk_blocks = []
    for c in chunks[:10]:
        title = c.get("title", "Unknown")
        page = c.get("page_no", "?")
        text = c.get("content", "")[:1200]
        chunk_blocks.append(f"[{title}, Page {page}]\n{text}")
    chunks_str = "\n\n---\n\n".join(chunk_blocks) if chunk_blocks else "(no chunks)"

    return f"""You are an information-extraction system. Given a sub-question, an
existing prose answer to that sub-question, and supporting document chunks,
emit one EvidenceRecord per distinct factual claim.

RULES
1. Every record must be grounded in the prose answer OR the chunks. Do NOT
   add information from your prior knowledge.
2. Each record's `claim` must be self-contained — no pronouns ('this', 'that'),
   no shorthand. Re-state the subject explicitly so the claim makes sense
   read in isolation.
3. Use the EXACT document title from the chunks for `source_doc`.
4. Populate `entity`, `value`, `metric`, `time_period` when the claim contains
   them. Leave null otherwise — do NOT invent values.
5. `evidence_span` should be a <= 30 word quote from a chunk that grounds
   the claim. Empty string if no clean quote available.
6. `confidence` reflects how directly the source supports the claim:
   0.5 = inference, 0.7 = strong inference, 0.9 = explicit verbatim support.
7. If the sub-answer says "information not found" or contains no factual
   claims, return an empty records list.
8. Echo `sub_query` verbatim.

SUB-QUESTION:
{sub_query}

PROSE SUB-ANSWER:
{sub_answer}

SUPPORTING CHUNKS:
{chunks_str}
"""


def _records_to_json(records: List[dict]) -> str:
    return json.dumps(records, indent=2, ensure_ascii=False, default=str)


def build_render_prompt(
    answer_class: str,
    original_query: str,
    resolved_query: str,
    records: List[dict],
) -> str:
    """
    Render the reconciled record set into a canonical markdown answer. The
    template branches on `answer_class` — same records render differently
    when the user asked for a comparison vs a timeline vs a summary.
    """
    records_json = _records_to_json(records)

    base = f"""You are a renderer. Convert the reconciled evidence records below
into a single, well-structured markdown answer for the user's question.

USER QUESTION (original): {original_query}
RESOLVED QUERY:           {resolved_query}
ANSWER CLASS:             {answer_class}

ABSOLUTE RULES (apply to every answer class):
1. Use ONLY the records below. Do NOT add external knowledge or invented data.
2. Cite EVERY claim inline using the record's source_doc and source_page in
   the form `[<source_doc>, Page <source_page>]`. Use the exact document title
   from the record — never 'Document 1', 'the first source', etc.
3. If a record has `confidence` < 0.4, mark it as tentative ("reportedly", "appears to") in the answer.
4. If two records share entity+metric+time but disagree on value, surface the
   conflict explicitly — do not silently pick one.
5. Output ONLY the rendered markdown answer in the `answer` field. No
   preamble like "Here is your answer", no schema definitions.

RECORDS (JSON):
{records_json}
"""

    template_guidance = {
        "factoid": (
            "Render as a single, direct sentence (or two) answering the question. "
            "Inline-cite the source. No headings, no lists."
        ),
        "comparison": (
            "Render as an HTML <table> with the compared entities as columns and "
            "the metrics/aspects as rows. Group records by entity. Below the table, "
            "add a 1-2 sentence summary highlighting the most important differences."
        ),
        "timeline": (
            "Render as a chronological list grouped by `time_period`. Use `## "
            "<time_period>` headings sorted oldest-to-newest. Within each period, "
            "bullet the records in priority order. End with a 1-2 sentence summary."
        ),
        "enumeration": (
            "Render as a clean bulleted markdown list. One item per record. Keep "
            "items parallel in structure (same grammatical shape, similar length)."
        ),
        "achievements_by_period": (
            "Render grouped first by `entity`, then by `time_period`. Use `## "
            "<entity>` headings, then `### <time_period>` subheadings, then "
            "bulleted achievements with inline citations. Sort entities by record "
            "count (most-cited first), time periods chronologically."
        ),
        "multi_entity_summary": (
            "Render one short paragraph per entity, with `## <entity>` headings. "
            "Each paragraph synthesizes that entity's records into 2-4 sentences."
        ),
        "ranking": (
            "Render as an HTML <table> sorted by `value` descending (numeric where "
            "possible). Columns: rank, entity, metric, value, source. End with a "
            "1-sentence note on the top entries."
        ),
        "narrative": (
            "Render as flowing paragraphs grouped thematically. Use `##` headings "
            "only when the answer has multiple distinct sub-topics. Maintain "
            "consistent paragraph length and citation style throughout."
        ),
    }
    guidance = template_guidance.get(answer_class, template_guidance["narrative"])

    return base + "\n\nTEMPLATE FOR THIS ANSWER CLASS:\n" + guidance + "\n"
