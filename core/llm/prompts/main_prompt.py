from typing import Any, Dict, List, Optional

from core.constants import INTERNAL, EXTERNAL
from core.llm.prompts.thread_context import build_thread_context_block


def detect_full_document_query(question: str) -> bool:
    """
    Detect whether the query requires complete document coverage
    (all slides, all pages, etc.) rather than relevance-filtered top-k retrieval.
    """
    question_lower = question.lower()
    full_doc_patterns = [
        "each slide", "every slide", "all slides", "all the slides",
        "each page", "every page", "all pages", "all the pages",
        "entire document", "whole document", "full document",
        "slide by slide", "page by page",
        "each section", "every section", "all sections",
        "for each chapter", "every chapter",
    ]
    return any(p in question_lower for p in full_doc_patterns)


def detect_answer_style(question: str) -> str:
    """
    Detect the desired answer style based on keywords in the question.

    Returns:
        'generative' - User wants content CREATED (scripts, speeches, talking points)
        'brief'      - User wants a concise answer
        'compare'    - User wants cross-document comparison
        'analyst'    - User wants strategic analysis with recommendations
        'detailed'   - User wants a detailed answer (default)
    """
    question_lower = question.lower()

    # Generative / content-creation keywords — checked FIRST
    generative_keywords = [
        "create a script",
        "write a script",
        "draft a script",
        "prepare a script",
        "generate a script",
        "script for each slide",
        "script for each page",
        "talking points for each",
        "talking points for every",
        "narration for each",
        "narration for every",
        "create a presentation",
        "prepare a presentation",
        "write a speech",
        "draft a speech",
        "create content for",
        "write content for",
        "generate content for",
        "write for each slide",
        "draft for each slide",
        "speaking notes for",
        "presenter notes for",
    ]
    for keyword in generative_keywords:
        if keyword in question_lower:
            return "generative"

    # Brief answer keywords
    brief_keywords = [
        "3 bullet points",
        "summarize",
        "brief",
        "short",
        "concise",
        "in short",
        "quick summary",
    ]
    for keyword in brief_keywords:
        if keyword in question_lower:
            return "brief"

    # Comparison keywords — detect BEFORE analyst to avoid analyst absorbing these
    compare_keywords = [
        "compare",
        "comparison",
        "contrast",
        "difference between",
        "differences between",
        "how do they differ",
        "how does it differ",
        "versus",
        " vs ",
        " vs.",
        "side by side",
        "side-by-side",
        "which is better",
        "which one",
        "similarities and differences",
        "common and different",
    ]
    for keyword in compare_keywords:
        if keyword in question_lower:
            return "compare"

    # Analyst / strategic answer keywords
    analyst_keywords = [
        "recommend",
        "recommendation",
        "what should",
        "implications",
        "strategy",
        "strategic",
        "suggest",
        "advise",
        "action items",
        "what are the risks",
        "swot",
        "pros and cons",
        "trade-off",
        "tradeoff",
        "evaluate",
        "assessment",
        "analysis",
        "analyze",
        "analyse",
        "what can we learn",
        "what does this mean",
        "insights",
        "common trend",
        "common themes",
        "gaps",
        "opportunities",
        "trends",
        "trend analysis",
    ]
    for keyword in analyst_keywords:
        if keyword in question_lower:
            return "analyst"

    # Detailed answer keywords
    detailed_keywords = [
        "detailed",
        "elaborate",
        "explain in detail",
        "comprehensive",
        "in depth",
        "thorough",
    ]
    for keyword in detailed_keywords:
        if keyword in question_lower:
            return "detailed"

    # Default to detailed answers
    return "detailed"


def _build_system_prompt(
    mode: str,
    answer_style: str,
    has_spreadsheet: bool = False,
    use_self_knowledge: bool = False,
) -> str:
    """
    Build the system prompt from shared components.
    Eliminates the 4-way duplication (INTERNAL×brief, INTERNAL×detailed, EXTERNAL×brief, EXTERNAL×detailed).
    """

    is_brief = answer_style == "brief"
    is_analyst = answer_style == "analyst"
    is_compare = answer_style == "compare"
    is_generative = answer_style == "generative"
    is_external = mode == EXTERNAL

    # ── Role ──
    if is_external:
        role = (
            "You are an expert assistant that answers questions using the provided **documents** "
            "and any supplied **external data** (such as web search results).\n"
        )
    elif is_compare:
        role = (
            "You are a **cross-document comparison specialist** that identifies similarities, differences, "
            "patterns, and contradictions across multiple documents. "
            "You present findings in a structured, side-by-side format grounded in the source data.\n"
        )
    elif is_analyst:
        role = (
            "You are a **senior strategic analyst** that provides data-driven insights, "
            "recommendations, and risk assessments based on the provided **documents**. "
            "Your analysis must always be grounded in the data — never speculate beyond what the evidence supports.\n"
        )
    elif is_generative:
        role = (
            "You are an expert **content creator** that generates original content "
            "(scripts, talking points, narratives, speeches) grounded in the provided **documents**. "
            "You CREATE content — you do not just describe or summarize what documents contain.\n"
        )
    else:
        role = "You are an expert assistant that answers questions based on the provided **documents**.\n"

    if has_spreadsheet:
        role += (
            "\n**IMPORTANT: SQL ENGINE AVAILABLE — MANDATORY FIRST STEP**\n"
            "You have access to a **SQL engine** that can query the uploaded spreadsheet data directly. "
            "For ANY question regarding data in the spreadsheets (counting, filtering, aggregating, listing, "
            "finding values, or retrieving records), you **MUST** use the `sql_query` action FIRST before "
            "attempting to answer. Do NOT answer spreadsheet questions from memory, text chunks, or prior context. "
            "Always query the actual data to ensure accuracy and completeness.\n"
        )

    # ── Task ──
    if is_brief:
        task = "Your job is to give clear, concise, and brief answers using Markdown formatting.\n\n"
    elif is_compare:
        task = (
            "Your job is to provide a **structured cross-document comparison** using Markdown formatting. "
            "Analyze each document's perspective and present a clear side-by-side analysis.\n\n"
        )
    elif is_analyst:
        task = (
            "Your job is to provide **strategic analysis with actionable recommendations** using Markdown formatting. "
            "Every claim must be backed by evidence from the documents.\n\n"
        )
    elif is_generative:
        task = (
            "Your job is to **create the requested content** using the information from the documents as source material. "
            "The output should be the actual deliverable the user asked for (e.g., a speaking script, talking points), "
            "NOT a description or summary of the documents. Write the content the user would directly use.\n\n"
        )
    else:
        task = "Your job is to create **clear, structured, and comprehensive answers** using Markdown formatting.\n\n"

    # ── Formatting Guidelines ──
    if is_brief:
        guidelines = (
            "### Answer Guidelines\n"
            "- Use **headings** (##, ###) for major sections.\n"
            "- Use **bullet points** and **numbered lists** to organize ideas concisely.\n"
            "- Keep explanations **short and to the point**.\n"
            "- Focus on the most important information only.\n"
            "- Avoid unnecessary details or elaboration.\n"
            "- Merge overlapping ideas and remove redundancy.\n"
        )
    else:
        guidelines = (
            "### Answer Guidelines\n"
            "- Use **headings (##, ###)** for major sections.\n"
            "- Use **bullet points** and **numbered lists** to organize ideas.\n"
            "- Highlight important terms in **bold** and examples in *italics*.\n"
            "- Provide **detailed explanations** for each point.\n"
            "- Include relevant examples, comparisons, and clarifications.\n"
            "- Extract and use as much relevant information as possible from the documents.\n"
            "- Provide context and background where helpful.\n"
            "- Merge overlapping ideas but maintain comprehensive coverage.\n"
        )

    # ── Grounding Rules (single authoritative block) ──
    grounding = (
        "\n### Grounding Rules\n"
        "- Rely on the supplied data (documents, summaries, conversation history, "
        "and any user-provided context for this thread).\n"
        "- Do NOT fabricate or infer beyond the supplied information.\n"
        "- **User-provided context** (such as names, preferences, background info) is first-class supplied data. "
        "Treat it as if the user stated it directly in their question.\n"
    )
    if use_self_knowledge:
        grounding += (
            "- **Self-knowledge mode is ON.** You MAY use your own general knowledge to complement the supplied data. "
            "Combine user-provided context with your knowledge and documents to answer as fully as possible. "
            "Still prioritize document content when it is available and relevant.\n"
        )
    else:
        grounding += (
            "- Do NOT use your own world knowledge or unstated assumptions. "
            "Only use information explicitly present in the supplied data.\n"
        )
    grounding += (
        "- If the provided data is insufficient to answer, clearly state: "
        "*I cannot answer based on the provided data.*\n"
    )
    if is_external:
        grounding += (
            "- Always **prioritize information from documents** over web results.\n"
            "- If conflicting data exists between document and web sources, "
            "state clearly: *Some sources provide conflicting information...*\n"
        )
    else:
        grounding += (
            "- If multiple sources contradict, mention it clearly using a note block.\n"
        )

    # ── Document References & Citations (applies to ALL answer styles) ──
    doc_refs = (
        "\n### Document References & Inline Citations\n"
        "- **MANDATORY**: When referencing documents, you MUST use the **exact document name/title** "
        "as shown in the `[Document: <name>]` prefix of each chunk.\n"
        "- **STRICTLY FORBIDDEN**: NEVER use ANY of these patterns:\n"
        "  - ❌ 'Document 1', 'Document 2', 'Document 3'\n"
        "  - ❌ 'the first document', 'the second document'\n"
        "  - ❌ 'the uploaded file', 'your document'\n"
        "  - ❌ Document IDs (internal tracking codes)\n"
        "- **CORRECT examples**:\n"
        '  - ✅ "According to **Annual Report 2025**, revenue increased by 15%."\n'
        '  - ✅ "The **Q3 Financial Summary** highlights a declining trend."\n'
        "- To find the correct name: look for `[Document: XYZ]` at the start of each chunk — use `XYZ` as the name.\n"
        "- **INLINE CITATIONS**: For every factual claim or data point, include a citation in the format "
        "`[Document Title, Page X]` at the end of the sentence or paragraph.\n"
        '- Example: "Revenue grew by 15% year-over-year [Annual Report 2025, Page 12]."\n'
        "- If a claim is supported by multiple documents, cite all of them: "
        '"Both reports confirm the trend [Report A, Page 3] [Report B, Page 7]."\n'
    )

    # ── Table & Data Formatting ──
    table_fmt = (
        "\n### Table & Data Formatting\n"
        "- When presenting **comparative data** across documents, features, or items, use **HTML tables** for reliability.\n"
        "- Use tables when you have **3 or more comparable items** with shared attributes.\n"
        "- Use tables for any **numerical data, metrics, or statistics** that can be organized in rows/columns.\n"
        "- **IMPORTANT**: Since your answer goes inside a JSON string, use HTML tables instead of Markdown pipe tables to avoid JSON parsing issues:\n"
        "  <table><tr><th>Metric</th><th>2024</th><th>2025</th><th>Change</th></tr>"
        "<tr><td>Revenue</td><td>$10M</td><td>$12M</td><td>+20%</td></tr></table>\n"
        "- For simple inline comparisons (2-3 items), bullet points are also acceptable.\n"
    )

    # ── Output Structure Example ──
    if is_brief:
        structure = (
            "\n### Output Structure\n"
            "```\n"
            "## Overview\n"
            "(Brief explanation)\n\n"
            "## Key Points\n"
            "- **Point 1:** Brief explanation...\n"
            "- **Point 2:** Brief explanation...\n"
            "- **Point 3:** Brief explanation...\n"
            "```\n"
        )
    if is_compare:
        structure = (
            "\n### Output Structure (Cross-Document Comparison)\n"
            "```\n"
            "## Overview\n"
            "(Brief context: what is being compared and why)\n\n"
            "## Per-Document Findings\n"
            "### [Exact Document Name A]\n"
            "- Key points from this document [Document A, Page X]\n\n"
            "### [Exact Document Name B]\n"
            "- Key points from this document [Document B, Page X]\n\n"
            "## Comparative Analysis\n"
            "<table><tr><th>Aspect</th><th>Document A</th><th>Document B</th></tr>"
            "<tr><td>Topic 1</td><td>Position/Data</td><td>Position/Data</td></tr>"
            "<tr><td>Topic 2</td><td>Position/Data</td><td>Position/Data</td></tr></table>\n\n"
            "## Agreements\n"
            "- Points where documents align\n\n"
            "## Contradictions & Gaps\n"
            "- Points where documents differ or information is missing\n\n"
            "## Synthesis\n"
            "(What can we conclude from both documents together?)\n"
            "```\n"
        )
    elif is_generative:
        structure = (
            "\n### Output Structure (Content Creation)\n"
            "```\n"
            "## [Content Title]\n\n"
            "### [Section/Slide 1 Title]\n"
            "(Generated content for this section — the actual script/text the user asked for)\n\n"
            "### [Section/Slide 2 Title]\n"
            "(Generated content — continues in document order)\n\n"
            "...(continue for ALL sections/slides)...\n"
            "```\n"
            "\n**CRITICAL**: Write the actual content the user requested. "
            "Do NOT describe what each slide contains — write what the presenter should SAY. "
            "Cover ALL slides/sections in document order. Include transitions between sections.\n"
        )
    elif is_analyst:
        structure = (
            "\n### Output Structure (Analyst Mode)\n"
            "```\n"
            "## Key Findings\n"
            "- **Finding 1:** Evidence-based insight [Document, Page X]\n"
            "- **Finding 2:** Evidence-based insight [Document, Page X]\n\n"
            "## Implications\n"
            "- What do these findings mean for the organization?\n"
            "- What patterns or trends emerge from the data?\n\n"
            "## Recommendations\n"
            "1. **Action Item 1:** Specific recommendation with rationale...\n"
            "2. **Action Item 2:** Specific recommendation with rationale...\n\n"
            "## Risks & Considerations\n"
            "- Potential risks or caveats associated with the recommendations\n\n"
            "## Summary\n"
            "(Concise conclusion with key takeaway)\n"
            "```\n"
        )
    elif not is_brief:
        if is_external:
            structure = (
                "\n### Output Structure\n"
                "```\n"
                "## Overview\n"
                "(Comprehensive explanation)\n\n"
                "## Key Information\n"
                "- **Document Insight:** Detailed explanation with context [Document, Page X]...\n"
                "- **Web Insight:** Detailed explanation with examples...\n\n"
                "## Additional Insights\n"
                "- Examples, comparisons, or clarifications.\n"
                "- Related information from sources.\n\n"
                "## Conflicts or Gaps\n"
                "- *Some sources differ on...*\n\n"
                "## Summary\n"
                "(Comprehensive conclusion)\n"
                "```\n"
            )
        else:
            structure = (
                "\n### Output Structure\n"
                "```\n"
                "## Overview\n"
                "(Comprehensive explanation)\n\n"
                "## Key Details\n"
                "- **Point 1:** Detailed explanation with context [Document, Page X]...\n"
                "- **Point 2:** Detailed explanation with examples [Document, Page X]...\n"
                "- **Point 3:** Detailed explanation with clarifications [Document, Page X]...\n\n"
                "## Additional Insights\n"
                "- *Examples, comparisons, or clarifications.*\n"
                "- *Related information from documents.*\n\n"
                "## Summary\n"
                "(Comprehensive conclusion)\n"
                "```\n"
            )

    return role + task + guidelines + grounding + doc_refs + table_fmt + structure


def main_prompt(
    messages: list,
    chunks: str,
    question: str,
    summary: str,
    mode: str,
    web_search_results: List[Dict[str, Any]] = None,
    initial_search_answer: str = None,
    initial_search_results: List[Dict[str, Any]] = None,
    use_self_knowledge: bool = False,
    spreadsheet_schema: Optional[str] = None,
    sql_result: Optional[str] = None,
    sql_query: Optional[str] = None,
    original_query: Optional[str] = None,
    thread_instructions: Optional[List[str]] = None,
    triple_context: Optional[str] = None,
    sql_nlp_summary: Optional[str] = None,
    sql_batched_answer: Optional[str] = None,
    doc_batched_answer: Optional[str] = None,
    vlm_visual_answer: Optional[str] = None,
    prior_answer_mode: str = "none",
):
    contents = []

    # Detect answer style based on question
    answer_style = detect_answer_style(question)

    if mode not in (INTERNAL, EXTERNAL):
        raise ValueError("Invalid mode. Mode must be either 'INTERNAL' or 'EXTERNAL'.")

    # ── System prompt (built from shared components) ──
    system_prompt = _build_system_prompt(
        mode,
        answer_style,
        has_spreadsheet=(spreadsheet_schema is not None),
        use_self_knowledge=use_self_knowledge,
    )
    contents.append({"role": "system", "parts": system_prompt})

    # ── Thread-level context (user-provided guidance, always included) ──
    thread_ctx = build_thread_context_block(thread_instructions)
    if thread_ctx:
        contents.append(thread_ctx)

    # ── Spreadsheet SQL schema (prioritized BEFORE chunks) ──
    if spreadsheet_schema:
        contents.append(
            {
                "role": "system",
                "parts": (
                    "### Spreadsheet Data (SQL Queryable)\n"
                    "The user has uploaded spreadsheet files (Excel/CSV) that have been loaded into a SQL database. "
                    "You can query this data using SQL SELECT statements.\n\n"
                    "**Available Tables and Columns:**\n"
                    f"```\n{spreadsheet_schema}\n```\n\n"
                    "**SQL Query Guidelines:**\n"
                    "- Use the `sql_query` action to run a SQL SELECT query against the spreadsheet data.\n"
                    "- Write standard SQLite-compatible SQL queries.\n"
                    "- Use aggregate functions like COUNT(), SUM(), AVG(), MIN(), MAX() for calculations.\n"
                    "- Use GROUP BY and ORDER BY for grouping and sorting.\n"
                    "- Use WHERE clauses to filter data.\n"
                    "- Use LIKE with wildcards for **factual** text matching (e.g., WHERE column LIKE '%keyword%'). "
                    "Only use LIKE for searching specific names, places, IDs, or concrete keywords.\n"
                    "- Column names and table names are case-sensitive and use underscores instead of spaces.\n"
                    "- Only SELECT queries are allowed (no INSERT, UPDATE, DELETE).\n"
                    "- **DO NOT add LIMIT unless the user explicitly asks for a specific number** "
                    "(e.g., 'top 10', 'first 5', 'show me 20'). When the user asks for analysis, "
                    "categorization, listing, or 'all' data, you MUST fetch ALL rows — do NOT "
                    "add LIMIT on your own. Incomplete data leads to wrong analysis.\n"
                    "\n"
                    "**CRITICAL — NLP / SEMANTIC ANALYSIS QUESTIONS:**\n"
                    "When the user asks a question that requires **understanding meaning, sentiment, tone, opinion, "
                    "or subjective judgment** of text data (e.g., 'which comments are negative', 'find positive feedback', "
                    "'identify critical remarks', 'what is the sentiment', 'which reviews are harsh', "
                    "'classify the responses', 'which entries are complaints'), you MUST follow this approach:\n"
                    "1. **DO NOT** use SQL keyword matching (LIKE '%bad%', '%not%', '%poor%', etc.) to filter subjective text. "
                    "Keyword matching CANNOT understand sentiment — a comment like 'This is not bad, great work!' "
                    "contains 'not' and 'bad' but is POSITIVE.\n"
                    "2. Instead, write a SQL query that **fetches ALL rows** from the relevant columns WITHOUT keyword filtering. "
                    "For example: `SELECT id, name, comment FROM table` or `SELECT * FROM table`.\n"
                    "3. Once you receive the full data in the SQL result, use YOUR language understanding to "
                    "analyze each entry's sentiment/tone/meaning and present a properly classified answer.\n"
                    "4. This applies to ANY question involving: sentiment, tone, opinion, satisfaction, positivity, "
                    "negativity, complaints, praise, criticism, quality assessment, mood, attitude, or subjective classification.\n"
                    "5. Fetch ALL rows for NLP analysis — do NOT add LIMIT. You need the complete dataset "
                    "to provide accurate sentiment/classification results. NEVER filter by keywords "
                    "when the question is about meaning or sentiment.\n"
                    "- **CRITICAL — SQL-FIRST RULE**: For ANY question whose answer could exist in the spreadsheet tables above, "
                    "you MUST use the `sql_query` action. This includes but is NOT limited to:\n"
                    "  * Looking up a specific person's details (address, email, phone, etc.)\n"
                    "  * Finding or listing records that match a condition (e.g., students from a state, employees in a department)\n"
                    "  * Searching for a name, value, or keyword in the data\n"
                    "  * Counting, summing, averaging, ranking, or any aggregation\n"
                    "  * Filtering, sorting, or comparing rows\n"
                    "  * ANY data retrieval from tabular/spreadsheet content\n"
                    "  NEVER answer from text chunks when the question relates to spreadsheet data — "
                    "text chunks are incomplete fragments and WILL give wrong or partial results. "
                    "The SQL database contains ALL rows and ALL columns and will give exact, complete results.\n"
                    "- Always provide the `sql_query` field in your response when choosing the `sql_query` action.\n"
                    "- **VALIDATION RULE**: If you choose `action='sql_query'`, you **MUST** provide the `sql_query` field with the valid SQL statement. Failing to do so will cause a system error.\n"
                    "- Even if you see some spreadsheet data in the document chunks, ALWAYS use `sql_query` instead. "
                    "The document chunks are only text previews and do NOT contain the full dataset.\n"
                ),
            }
        )

    # ── Visual Reference Context (query-time VLM analysis of referenced page/figure) ──
    # For generative mode, VLM answer is deferred to AFTER chunks for recency bias.
    _deferred_vlm_section = None
    if vlm_visual_answer and answer_style == "generative":
        # Defer — will be inserted after chunks with priority framing
        _deferred_vlm_section = {
            "role": "system",
            "parts": (
                "### Primary Source: Visual Analysis (VLM)\n"
                f"{vlm_visual_answer}\n\n"
                "**CRITICAL INSTRUCTION**: The Visual Analysis above is your PRIMARY source. "
                "It was generated by analyzing the actual slide/page images and captures visual "
                "content that text chunks may miss. Use it as the FOUNDATION for your answer. "
                "The Document Chunks above provide supplementary text detail only.\n"
                "Do NOT rewrite the visual analysis as a description — "
                "use it to CREATE the content the user requested.\n"
            ),
        }
    elif vlm_visual_answer:
        contents.append(
            {
                "role": "system",
                "parts": (
                    "### Visual Context (VLM analysis of the referenced page/figure)\n"
                    f"{vlm_visual_answer}\n\n"
                    "The above was extracted by a Vision Language Model that analyzed the image of "
                    "the referenced page/slide/figure. Treat it as additional context alongside "
                    "the Document Chunks below. Use it to inform your answer — especially for "
                    "data points, table contents, chart values, or diagram structures that may "
                    "not appear in the text chunks. Synthesize all available sources into a "
                    "single coherent answer.\n"
                ),
            }
        )

    # ── Pre-analyzed document batch answer (MapReduce result) ──
    if doc_batched_answer:
        contents.append(
            {
                "role": "system",
                "parts": (
                    "### Pre-Analyzed Document Context (complete multi-document analysis)\n"
                    f"{doc_batched_answer}\n\n"
                    "**INSTRUCTIONS:**\n"
                    "The **Pre-Analyzed Document Context** above was generated from the COMPLETE retrieved "
                    "document set (all chunks processed in batches). A raw sample follows for reference.\n\n"
                    "You MUST:\n"
                    '1. Set `action` to `"answer"`.\n'
                    "2. Write your answer based primarily on the **Pre-Analyzed Document Context**.\n"
                    "3. Enhance and format it using Markdown (tables, headings, bullet points).\n"
                    "4. Use the raw document chunks below only to quote specific excerpts if helpful.\n\n"
                    "You MUST NOT:\n"
                    "- Ignore the pre-analyzed context in favour of only the raw chunks.\n"
                    "- Return a blank or empty answer.\n"
                ),
            }
        )

    # ── Retrieved context ──
    if chunks:
        contents.append(
            {"role": "system", "parts": f"**Document Chunks (Context):**\n{chunks}\n"}
        )

    # ── Deferred VLM section for generative mode (placed AFTER chunks for recency bias) ──
    if _deferred_vlm_section:
        contents.append(_deferred_vlm_section)

    # ── Phase 3.2: Entity relationship triples ──
    if triple_context:
        contents.append(
            {
                "role": "system",
                "parts": (
                    f"**{triple_context}**\n"
                    "Use these relationships to understand connections between entities "
                    "mentioned in the documents."
                ),
            }
        )

    # ── External-only sources ──
    if mode == EXTERNAL:
        if initial_search_results:
            contents.append(
                {
                    "role": "system",
                    "parts": f"**Initial External Knowledge Sources:**\n{initial_search_results}\n",
                }
            )

    # ── Conversation history — Layer 3 segregation framing ──
    # `messages` here is the curated prior_chat_context from AgentState — at
    # most one user+assistant pair, populated ONLY when the decomposition LLM
    # decided this is a follow-up that refers to the prior assistant answer.
    # The framing depends on prior_answer_mode — each mode tells the LLM a
    # different thing to do with the prior content and how to use the docs.
    if messages:
        framings = {
            "reasoning": (
                "PRIOR EXCHANGE (for topical context only — the current question MUST "
                "be answered from the document evidence below, NOT from the prior "
                "assistant message). The prior assistant message may contain stale "
                "facts; verify against the documents before reusing any value.",
                "END OF PRIOR EXCHANGE. Now answer the CURRENT question using the "
                "documents below as your primary source.",
            ),
            "reformat": (
                "PRIOR EXCHANGE (this is the SOURCE CONTENT the user wants transformed). "
                "The user's current question requests a REFORMATTING or RESTATEMENT of the "
                "prior assistant message — preserve its substance and apply the requested "
                "transformation (table / bullets / shorter / translation / etc.). The "
                "documents below are available to FILL GAPS or VERIFY SPECIFIC FACTS, but "
                "the prior assistant message is the primary content to work from. Do NOT "
                "substitute different content from the documents that wasn't in the prior "
                "answer just because it's also relevant.",
                "END OF PRIOR EXCHANGE. Now apply the user's requested transformation "
                "to the prior assistant content.",
            ),
            "correction": (
                "PRIOR EXCHANGE (the user is CORRECTING a claim in the prior assistant "
                "message). The user's current input identifies an error — treat the "
                "user's correction as AUTHORITATIVE for the corrected fact and produce "
                "a revised answer that respects it. Acknowledge the correction explicitly "
                "in your reply ('You are right — ...'). Use the documents below to expand "
                "or verify the rest of the answer; if the documents disagree with the "
                "user's correction, surface the disagreement explicitly and cite the "
                "document evidence rather than silently overriding the user.",
                "END OF PRIOR EXCHANGE. Apply the correction and produce a revised answer.",
            ),
            "expansion": (
                "PRIOR EXCHANGE (this content stays — the user wants the SAME analysis "
                "extended to additional scope). Keep the prior content intact and ADD "
                "parallel content for the new scope using the documents below. Do not "
                "rewrite, summarize, or compress what was already said unless the user "
                "explicitly asked for that. Present the result as an integrated whole "
                "covering both the original scope (from the prior message) and the new "
                "scope (from the documents).",
                "END OF PRIOR EXCHANGE. Preserve the prior content and add the new scope.",
            ),
            "comparison": (
                "PRIOR EXCHANGE (this is ONE SIDE of the comparison the user wants). "
                "The user is asking you to RELATE the prior assistant content to a "
                "different entity, period, or scope. The documents below contain the "
                "OTHER SIDE — retrieve and structure them alongside the prior content "
                "as a side-by-side comparison (HTML table when 2+ dimensions are "
                "compared). Preserve specific values from the prior message verbatim; "
                "do not paraphrase them.",
                "END OF PRIOR EXCHANGE. Now produce a structured comparison between the "
                "prior content and the new scope from the documents.",
            ),
        }
        opener, closer = framings.get(prior_answer_mode, framings["reasoning"])

        contents.append({"role": "system", "parts": opener})
        for m in messages:
            if m.type == "human":
                contents.append({"role": "user", "parts": f"[prior] {m.content}"})
            elif m.type == "ai":
                contents.append({"role": "assistant", "parts": f"[prior] {m.content}"})
        contents.append({"role": "system", "parts": closer})

    # ── Summary context ──
    if summary:
        contents.append(
            {"role": "system", "parts": f"**Summary Reference:**\n{summary}\n"}
        )

    # ── External-only: web search results ──
    if mode == EXTERNAL:
        if web_search_results:
            contents.append(
                {
                    "role": "system",
                    "parts": f"**Web Search Results:**\n{web_search_results}\n",
                }
            )
        if initial_search_answer:
            contents.append(
                {
                    "role": "system",
                    "parts": f"**Initial Web Search Answer:**\n{initial_search_answer}\n",
                }
            )
        contents.append(
            {
                "role": "system",
                "parts": (
                    "If conflicting information exists, always **prioritize document content over web sources.**\n"
                    "If no provided data resolves the question, respond that you cannot answer based on the provided data."
                ),
            }
        )

    # ── Title caveat ──
    contents.append(
        {
            "role": "system",
            "parts": (
                "Titles shown in the document chunks are filenames and may not accurately reflect the document content. "
                "Use them for reference attribution but do not rely on them as indicators of what the document covers."
            ),
        }
    )

    # ── Spreadsheet SQL schema (Moved to TOP) ──
    # (Removed from here, placed before chunks)

    # ── SQL query result from a previous iteration ──
    if sql_result:
        display_question = original_query or question

        # When batched analysis or NLP theme summary exists, use simplified instructions —
        # the pre-analyzed content is the primary answer source, raw data is just a sample.
        if sql_batched_answer:
            contents.append(
                {
                    "role": "system",
                    "parts": (
                        "### Pre-Analyzed SQL Result (complete dataset analysis)\n"
                        f"{sql_batched_answer}\n\n"
                        "### Raw Data Sample (for reference only)\n"
                        + (f"**SQL Query:** `{sql_query}`\n\n" if sql_query else "")
                        + f"{sql_result}\n\n"
                        f"**Original User Question:** {display_question}\n\n"
                        "**INSTRUCTIONS:**\n"
                        "The **Pre-Analyzed SQL Result** above was generated from the COMPLETE dataset "
                        "(all rows were processed in batches). The raw data sample is a small subset for reference.\n\n"
                        "You MUST:\n"
                        '1. Set `action` to `"answer"`.\n'
                        "2. Write your answer based on the **Pre-Analyzed SQL Result**.\n"
                        "3. Enhance and format it using Markdown (tables, headings, bullet points).\n"
                        "4. Use the raw data sample to quote specific examples if helpful.\n\n"
                        "You MUST NOT:\n"
                        "- Request another SQL query. The data has already been fully analyzed.\n"
                        "- Return a blank or empty answer.\n"
                    ),
                }
            )
        elif sql_nlp_summary:
            contents.append(
                {
                    "role": "system",
                    "parts": (
                        "### Pre-Analyzed Themes (complete dataset analysis)\n"
                        f"{sql_nlp_summary}\n\n"
                        "### Raw Data Sample (for reference only)\n"
                        + (f"**SQL Query:** `{sql_query}`\n\n" if sql_query else "")
                        + f"{sql_result}\n\n"
                        f"**Original User Question:** {display_question}\n\n"
                        "**INSTRUCTIONS — READ CAREFULLY:**\n"
                        "The **Pre-Analyzed Themes** above were extracted from the COMPLETE dataset "
                        "(every single row). The raw data sample is a small subset for reference only.\n\n"
                        "You MUST:\n"
                        '1. Set `action` to `"answer"`.\n'
                        "2. Write a comprehensive answer based on the **Pre-Analyzed Themes**.\n"
                        "3. Present the themes with their counts, percentages, and examples.\n"
                        "4. Use the raw data sample only to quote specific examples if helpful.\n\n"
                        "You MUST NOT:\n"
                        "- Request another SQL query. The data has already been fully analyzed.\n"
                        "- Re-derive themes from the small sample. The themes above cover ALL rows.\n"
                        "- Return a blank or empty answer.\n"
                    ),
                }
            )
        else:
            contents.append(
                {
                    "role": "system",
                    "parts": (
                        "### SQL Query Result\n"
                        "A SQL query was already executed on the spreadsheet data.\n\n"
                        + (f"**SQL Query Executed:** `{sql_query}`\n\n" if sql_query else "")
                        + f"**Result:**\n{sql_result}\n\n"
                        f"**Original User Question:** {display_question}\n\n"
                        "**CRITICAL — STOP AND EVALUATE BEFORE CHOOSING AN ACTION:**\n"
                        "You have ALREADY received a SQL query result above. Follow these rules strictly:\n"
                        "1. Compare the SQL result against the **Original User Question**.\n"
                        "2. If the SQL result contains the data needed to answer the question (even partially), "
                        'you **MUST** set `action` to `"answer"` and use the result to write your final answer. '
                        "Do NOT request another SQL query.\n"
                        '3. You should ONLY set `action` to `"sql_query"` again if ALL of these conditions are true:\n'
                        "   - The SQL result above is an ERROR message (e.g., 'SQL execution error: ...'), OR\n"
                        "   - The SQL query was clearly WRONG (e.g., queried the wrong column/table), OR\n"
                        "   - The original question explicitly requires MULTIPLE SEPARATE pieces of data that cannot "
                        "be retrieved in a single query and the current result only covers part of it.\n"
                        "4. If the result is a valid number, table, or dataset — even if small or unexpected — "
                        "that IS your answer. Present it clearly. Do NOT re-query to 'verify' or 'get more details'.\n"
                        "5. An empty result set (0 rows) is still a valid answer (it means 'none found'). "
                        "Do NOT re-query for empty results unless the query itself was incorrect.\n"
                        "6. **SEMANTIC ANALYSIS**: If the original question is about sentiment, tone, opinion, "
                        "or subjective classification and the SQL result contains the raw text data, you MUST now "
                        "analyze EACH entry using your language understanding. Read each text entry carefully, "
                        "understand its full meaning in context, and classify it appropriately. "
                        "Do NOT rely on the presence of individual words like 'not', 'bad', 'poor' — "
                        "understand the COMPLETE sentence meaning (e.g., 'not bad' = positive, "
                        "'could not be better' = positive, 'not what I expected' = negative).\n"
                        "7. **LARGE DATASET OUTPUT RULE**: If the SQL result has many rows (more than ~20) "
                        "and the question requires listing, categorizing, or detailing each row:\n"
                        "   - Use `excel_create` action to generate a downloadable Excel file with the full analysis.\n"
                        "   - Set `excel_request` to describe what to create (e.g., 'categorize all worklet titles "
                        "into groups with counts and details').\n"
                        "   - Also set `answer` to a brief summary: total counts, key categories/patterns, "
                        "top/bottom items. Do NOT try to list every row in the answer.\n"
                        "   - If the question only needs aggregates (counts, averages, totals), use `answer` directly "
                        "with the summarized data — no Excel needed.\n"
                ),
            }
        )

    # ── Available actions ──
    sql_action_text = ""
    if spreadsheet_schema:
        if sql_result:
            # SQL result already available — discourage re-querying
            sql_action_text = (
                "- **sql_query**: Execute a SQL SELECT query against the spreadsheet data. "
                "**A SQL query has ALREADY been executed and the result is shown above. "
                "Only use this action again if the previous query returned an ERROR or was clearly wrong. "
                "Otherwise, you MUST use `answer` to present the result.**\n"
            )
        else:
            sql_action_text = (
                "- **sql_query**: Execute a SQL SELECT query against the spreadsheet data. Use this for ANY question "
                "that can be answered from the uploaded spreadsheet/CSV files — including lookups, searches, filters, "
                "aggregations, listings, and data retrieval. Requires the `sql_query` field with a valid SQLite SELECT statement. "
                "**You have NOT queried the data yet. You MUST use `sql_query` as your action — do NOT choose `answer` "
                "for spreadsheet questions without first running a SQL query. Even if you think you know the answer, "
                "always verify by querying the actual data.**\n"
            )

    excel_action_text = (
        "- **excel_create**: Create a downloadable Excel (.xlsx) file from the data. "
        "Use in TWO scenarios:\n"
        "  1. **User explicitly requests an export**: 'create an Excel', 'export to spreadsheet', "
        "'download as Excel', 'create a pivot table', 'generate a report in Excel', etc.\n"
        "  2. **Your analysis output would be too long**: If answering the question requires listing, "
        "categorizing, or detailing MORE than ~20 rows of data (e.g., categorizing hundreds of items, "
        "listing all records with details, building a full breakdown), you MUST use `excel_create` "
        "instead of `answer`. Put the detailed data in the Excel file and provide a brief summary "
        "in the `answer` field (counts, key patterns, top items).\n"
        "Requires the `excel_request` field with a natural-language description of what to create "
        "(e.g., 'categorize all 2500 worklet titles into groups with counts' or "
        "'export filtered employee data with department breakdown'). "
        "When using this for large output, also set `answer` to a short summary of the analysis.\n"
    )

    # When spreadsheet data is available but no SQL has been run yet,
    # reorder actions to put sql_query first and restrict "answer"
    sql_not_yet_run = spreadsheet_schema and not sql_result
    if sql_not_yet_run:
        answer_action_text = (
            "- **answer**: Directly answer the question. "
            "**ONLY use this for greetings, clarification requests, or questions clearly unrelated to the spreadsheet data. "
            "For ANY data question, you MUST use `sql_query` first.**\n"
        )
    else:
        answer_action_text = "- **answer**: Directly answer the question using available information.\n"

    # Build action list — sql_query comes first when it should be the default
    if sql_not_yet_run:
        action_list = (
            "You can perform the following actions:\n"
            + sql_action_text
            + excel_action_text
            + answer_action_text
        )
    else:
        action_list = (
            "You can perform the following actions:\n"
            + answer_action_text
            + (
                "- **web_search**: Search for recent or external information not in the documents.\n"
                if mode == EXTERNAL
                else ""
            )
            + sql_action_text
            + excel_action_text
        )

    action_list += (
        "- **document_summarizer**: Request a summary of a specific document (requires `document_id`).\n"
        "- **global_summarizer**: Request a collective summary of all documents.\n"
        "- **failure**: Indicate inability to answer with available information.\n"
        "Do not choose an action lightly; only use 'failure' when absolutely necessary.\n"
        "Do not choose any other action other than the ones mentioned above.\n"
    )

    contents.append({"role": "system", "parts": action_list})

    # Final reminder for SQL-first enforcement (recency bias — last instruction is strongest)
    if sql_not_yet_run:
        import time as _time

        contents.append(
            {
                "role": "system",
                "parts": (
                    f"[Request ID: {int(_time.time() * 1000)}] "
                    "⚠️ CRITICAL INSTRUCTION — READ CAREFULLY:\n"
                    "No SQL query has been executed yet for this request. "
                    "You MUST choose `sql_query` as your action and write a SQL SELECT query. "
                    "Choosing `answer` without first running a SQL query is WRONG and will produce "
                    "inaccurate results. The spreadsheet data can ONLY be accessed through SQL.\n\n"
                    "Your response MUST have:\n"
                    '  "action": "sql_query"\n'
                    '  "sql_query": "SELECT ... FROM ..."\n\n'
                    "Any other action for a data question is incorrect."
                ),
            }
        )

    contents.append(
        {
            "role": "user",
            "parts": "Please use all the provided information to answer the question.",
        }
    )

    # Final user question
    if sql_not_yet_run:
        contents.append(
            {
                "role": "user",
                "parts": (
                    f"**Question:** {question}\n\n"
                    "Remember: You MUST use `sql_query` action to query the spreadsheet data first. "
                    "Write a SQL SELECT statement to retrieve the data needed to answer this question."
                ),
            }
        )
    else:
        contents.append({"role": "user", "parts": f"**Question:** {question}\n"})

    # JSON formatting requirement
    contents.append(
        {
            "role": "user",
            "parts": (
                "Return ONLY a valid JSON object matching the required schema. No markdown fencing, no commentary, no text before or after the JSON.\n"
                "CRITICAL JSON RULES:\n"
                "- All string values MUST use double quotes and properly escape special characters.\n"
                "- Newlines inside string values MUST be written as \\n (escaped), NOT as actual line breaks.\n"
                '- Double quotes inside string values MUST be escaped as \\".\n'
                "- Backslashes inside string values MUST be escaped as \\\\.\n"
                "- Do NOT use trailing commas after the last item in arrays or objects.\n"
                "- For tables inside the answer field, use HTML <table> tags, NOT Markdown pipe tables."
            ),
        }
    )

    return contents
