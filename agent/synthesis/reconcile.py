"""
Reconciliation step — collapse duplicate records and flag conflicting ones
across all sub-query extractions. Pure Python, no LLM calls.

Dedup key: (entity, metric, time_period) when all three are present, else
                 (claim_normalized, source_doc, source_page).

Conflict: same (entity, metric, time_period) but different `value`.
"""

import re
from typing import Dict, List, Optional, Tuple

from core.llm.output_schemas.synthesis_outputs import (
    EvidenceRecord,
    ReconciledRecord,
    SubAnswerRecords,
)


def _normalize_claim(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())[:200]


def _emt_key(r: EvidenceRecord) -> Optional[Tuple[str, str, str]]:
    if r.entity and r.metric and r.time_period:
        return (r.entity.strip().lower(), r.metric.strip().lower(), r.time_period.strip().lower())
    return None


def _fallback_key(r: EvidenceRecord) -> Tuple[str, str, str]:
    return (_normalize_claim(r.claim), r.source_doc.lower(), str(r.source_page or ""))


def reconcile(extractions: List[SubAnswerRecords]) -> List[ReconciledRecord]:
    """
    Returns deduped + conflict-tagged records. Order is by best-confidence
    representative.

    Groups records by EMT key (entity, metric, time_period). Within a group:
      - if all `value`s agree (or are None) -> single deduped record, others
        recorded as `duplicates`
      - if `value`s disagree -> kept the highest-confidence record as primary,
        the disagreeing records go in `conflicting`
    Records that lack EMT identity fall back to claim-text grouping.
    """
    by_emt: Dict[Tuple[str, str, str], List[EvidenceRecord]] = {}
    by_claim: Dict[Tuple[str, str, str], List[EvidenceRecord]] = {}

    for sub in extractions:
        for rec in sub.records:
            emt = _emt_key(rec)
            if emt:
                by_emt.setdefault(emt, []).append(rec)
            else:
                by_claim.setdefault(_fallback_key(rec), []).append(rec)

    reconciled: List[ReconciledRecord] = []

    for _, recs in by_emt.items():
        recs_sorted = sorted(recs, key=lambda r: r.confidence, reverse=True)
        primary = recs_sorted[0]
        rest = recs_sorted[1:]
        values = {(r.value or "").strip().lower() for r in recs if r.value}
        if len(values) > 1:
            conflicting = [r for r in rest if (r.value or "").strip().lower() != (primary.value or "").strip().lower()]
            duplicates = [r for r in rest if r not in conflicting]
            reconciled.append(
                ReconciledRecord(record=primary, duplicates=duplicates, conflicting=conflicting)
            )
        else:
            reconciled.append(ReconciledRecord(record=primary, duplicates=rest))

    for _, recs in by_claim.items():
        recs_sorted = sorted(recs, key=lambda r: r.confidence, reverse=True)
        reconciled.append(ReconciledRecord(record=recs_sorted[0], duplicates=recs_sorted[1:]))

    reconciled.sort(key=lambda rr: rr.record.confidence, reverse=True)
    return reconciled


def to_render_payload(reconciled: List[ReconciledRecord]) -> List[dict]:
    """Flatten reconciled records into the JSON payload for the renderer."""
    out = []
    for rr in reconciled:
        d = rr.record.model_dump()
        if rr.conflicting:
            d["conflicting_values"] = [c.value for c in rr.conflicting if c.value]
            d["conflict_sources"] = [
                {"value": c.value, "source_doc": c.source_doc, "source_page": c.source_page}
                for c in rr.conflicting
            ]
        out.append(d)
    return out
