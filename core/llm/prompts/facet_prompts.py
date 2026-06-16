"""Prompts for locked-facet extraction (rag-refactor step 5)."""


def build_doc_facet_prompt(title: str, sample_text: str) -> str:
    """
    Document-level facet extraction prompt. Caller supplies a short sample
    (title + first ~2000 chars + summary if available) so cost is bounded.
    """
    return f"""You extract structured facets from a document so that downstream
retrieval can filter precisely on temporal scope and document type.

DOCUMENT TITLE:
{title}

DOCUMENT SAMPLE (title + opening pages, possibly truncated):
{sample_text}

RULES
1. Output facets that are EXPLICITLY supported by the title or sample. Do NOT
   guess.
2. Normalize fiscal year to 'FYYYYY' (e.g. 'FY2024'). If the document only
   names a quarter (e.g. 'Q3 results'), leave fiscal_year null unless the
   year is otherwise stated.
3. Normalize quarter to 'Q1' / 'Q2' / 'Q3' / 'Q4'. If multiple quarters are
   covered with no primary focus, leave null.
4. calendar_year is the 4-digit year (e.g. 2024). Use it when a calendar year
   is named but no fiscal year is — when both are present, set both.
5. doc_type must be one of: report, policy, contract, sow, deck, memo,
   earnings, manual, spec, minutes, other. Default 'other' when unclear.
6. Prefer FALSE NEGATIVES over false positives: when uncertain, leave a
   facet null. Null means 'no constraint' downstream — incorrect values
   would actively filter out correct documents.
"""


def build_query_facet_prompt(query: str) -> str:
    return f"""You extract locked retrieval facets from a user query. These
facets are applied as HARD filters at the vector DB level — they MUST survive
any later query rewrite. Be conservative: only set a facet when the query
explicitly names that constraint.

QUERY:
{query}

RULES
1. fiscal_year: set to 'FYYYYY' format (e.g. 'FY2024') ONLY when the query
   names a fiscal year ('FY24', 'fiscal 2024', 'fiscal year 2024').
   - 'this year', 'last year' etc. -> null (ambiguous).
   - A bare calendar year ('2024 revenue') -> set calendar_year, NOT fiscal_year.
2. quarter: set to 'Q1' / 'Q2' / 'Q3' / 'Q4' ONLY when the query names a
   quarter. 'Q3 2024' -> quarter='Q3', calendar_year=2024.
3. calendar_year: set the 4-digit year ONLY when a calendar year is named
   and no fiscal year is. Default null.
4. doc_type: set to one of report/policy/contract/sow/deck/memo/earnings/
   manual/spec/minutes ONLY when the query explicitly restricts to that doc
   type (e.g. 'per the contract', 'in the policy', 'from the SOW'). Else null.
5. Default everything to null. Better to under-filter than to filter out the
   correct evidence.
"""
