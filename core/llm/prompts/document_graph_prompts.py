"""Prompt builders for the DocumentGraph pipeline."""

from typing import List, Tuple


def build_relation_extraction_prompt(
    chunks: List[Tuple[str, str, List[str]]]
) -> str:
    """
    Build a Phase 5 prompt asking the LLM to extract entity relations.

    Args:
        chunks: list of (chunk_id, text, known_entity_labels) tuples.
    """
    parts = [
        "You are an information extraction system. From the text below, extract "
        "directed relations of the form (subject, predicate, object) between named "
        "entities. Use the provided ENTITIES list as the canonical surface forms — "
        "if a relation involves an entity not in that list, do not invent one.",
        "",
        "RULES:",
        "1. Only extract relations that are explicitly stated or clearly implied by the text.",
        "2. predicate must be a 3-6 word verb-phrase (e.g. 'is_a_subsidiary_of', 'announced', 'depends_on').",
        "3. Use the EXACT surface forms from ENTITIES for subject and obj fields.",
        "4. Skip generic / trivial relations like 'is_a_thing' or 'mentioned_in'.",
        "5. Skip relations that are pure speculation or your own inference.",
        "6. Include short evidence quotes (≤25 words) when possible.",
        "",
    ]
    for cid, text, ents in chunks:
        parts.append(f"--- CHUNK {cid} ---")
        if ents:
            parts.append(f"ENTITIES: {', '.join(ents)}")
        parts.append("TEXT:")
        parts.append(text.strip())
        parts.append("")
    parts.append(
        "Return at most 12 high-quality relations across all chunks. "
        "If no clear relations exist, return an empty list."
    )
    return "\n".join(parts)


def build_entity_merge_prompt(pairs: List[dict]) -> str:
    """
    Build a Phase 4 prompt asking the LLM to judge ambiguous entity merges.

    Args:
        pairs: list of {pair_id, label_a, type_a, sample_a, label_b, type_b, sample_b}.
    """
    parts = [
        "You are an entity-resolution assistant. For each PAIR below, decide whether "
        "the two surface forms refer to the SAME real-world entity (e.g. 'Acme Corp' "
        "and 'Acme Corporation' usually do; 'Apple' the company and 'apple' the fruit "
        "do not).",
        "",
        "RULES:",
        "1. same=true only when you are confident it's the same entity.",
        "2. If same=true, choose the more complete/formal label as `canonical`.",
        "3. If types disagree (e.g. PERSON vs ORG), default to same=false unless context overrides.",
        "4. Echo each pair_id exactly as given.",
        "",
    ]
    for p in pairs:
        parts.append(f"PAIR {p['pair_id']}:")
        parts.append(f"  A: label='{p['label_a']}', type='{p['type_a']}', sample='{p['sample_a']}'")
        parts.append(f"  B: label='{p['label_b']}', type='{p['type_b']}', sample='{p['sample_b']}'")
        parts.append("")
    return "\n".join(parts)


def build_community_naming_prompt(communities: List[dict]) -> str:
    """
    Build a Phase 8 prompt asking the LLM to name each cluster.

    Args:
        communities: list of {id, members: [labels], top_relations: [str]}.
    """
    parts = [
        "You are summarizing clusters of related entities from a document corpus. "
        "For each COMMUNITY below, produce a concise name (2-5 words) and a "
        "1-2 sentence summary of what unifies these entities.",
        "",
        "RULES:",
        "1. The name should be specific (e.g. 'Cloud Cost Optimization', not 'Things').",
        "2. The summary should describe the cluster's theme — not list every member.",
        "3. Echo community_id exactly as given.",
        "",
    ]
    for c in communities:
        parts.append(f"COMMUNITY {c['id']}:")
        parts.append(f"  Members ({len(c['members'])}): {', '.join(c['members'][:25])}")
        if c.get("top_relations"):
            parts.append(f"  Sample relations: {'; '.join(c['top_relations'][:6])}")
        parts.append("")
    return "\n".join(parts)
