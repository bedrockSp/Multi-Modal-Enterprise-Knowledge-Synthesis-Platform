"""
Backfill locked-facet metadata onto every existing ChromaDB chunk AND the
per-thread BM25 pickle.

Walks data/{user_id}/threads/{thread_id}/parsed/{doc_id}.json, extracts
document-level facets (fiscal_year, quarter, calendar_year, doc_type), and
merges them into each existing chunk's metadata in place — NO re-embedding
needed since vectors are unchanged.

Usage:
    python -m scripts.backfill_facets                      # all users, all threads
    python -m scripts.backfill_facets --user-id <uid>      # one user
    python -m scripts.backfill_facets --user-id <uid> --thread-id <tid>
    python -m scripts.backfill_facets --dry-run            # show what would change
    python -m scripts.backfill_facets --force              # overwrite existing facets

Iterates thread-by-thread so the BM25 pickle is loaded and saved only once
per thread regardless of how many documents it contains.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple


def _iter_threads(
    user_id: Optional[str], thread_id: Optional[str]
) -> Iterable[Tuple[str, str]]:
    """Yield (user_id, thread_id) pairs for every thread with a parsed/ dir."""
    base = "data"
    if not os.path.isdir(base):
        return

    user_dirs = [user_id] if user_id else os.listdir(base)
    for uid in user_dirs:
        threads_dir = os.path.join(base, uid, "threads")
        if not os.path.isdir(threads_dir):
            continue
        thread_dirs = [thread_id] if thread_id else os.listdir(threads_dir)
        for tid in thread_dirs:
            parsed_dir = os.path.join(threads_dir, tid, "parsed")
            if os.path.isdir(parsed_dir):
                yield uid, tid


def _doc_ids_in_thread(uid: str, tid: str) -> List[str]:
    parsed_dir = os.path.join("data", uid, "threads", tid, "parsed")
    return [
        os.path.splitext(f)[0]
        for f in sorted(os.listdir(parsed_dir))
        if f.endswith(".json")
    ]


def _load_parsed(uid: str, tid: str, doc_id: str) -> Optional[Dict]:
    path = os.path.join("data", uid, "threads", tid, "parsed", f"{doc_id}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ! could not load {path}: {e}")
        return None


async def _facets_for_doc(uid: str, tid: str, doc_id: str) -> Optional[dict]:
    from core.embeddings.facets import (
        extract_document_facets,
        facets_to_chroma_metadata,
    )

    parsed = _load_parsed(uid, tid, doc_id)
    if not parsed:
        return None

    title = parsed.get("title") or parsed.get("file_name") or doc_id
    full_text = parsed.get("full_text") or ""
    summary = parsed.get("summary") or None

    try:
        facets = await extract_document_facets(title=title, text=full_text, summary=summary)
    except Exception as e:
        print(f"  ! facet extraction failed for {doc_id}: {e}")
        return None

    facet_meta = facets_to_chroma_metadata(facets)
    if not facet_meta:
        return None
    return {"facets": facet_meta, "title": title}


def _patch_bm25(uid: str, tid: str, doc_facets: Dict[str, dict], dry_run: bool) -> int:
    """Merge per-doc facets into the thread's BM25 pickle. Returns chunks patched."""
    from core.embeddings.vectorstore import load_bm25, _get_bm25_path

    bm25_data = load_bm25(uid, tid)
    if not bm25_data:
        return 0

    metadatas = bm25_data.get("chunk_metadatas") or []
    patched = 0
    for m in metadatas:
        doc_id = (m or {}).get("document_id")
        if doc_id and doc_id in doc_facets:
            for k, v in doc_facets[doc_id].items():
                if m.get(k) != v:
                    m[k] = v
                    patched += 1

    if patched and not dry_run:
        import pickle

        path = _get_bm25_path(uid, tid)
        with open(path, "wb") as f:
            pickle.dump(bm25_data, f)
    return patched


async def _process_thread(uid: str, tid: str, dry_run: bool, force: bool) -> Tuple[int, int, int]:
    """Returns (chroma_updates, bm25_updates, docs_processed)."""
    from core.embeddings.vectorstore import get_vectorstore

    doc_ids = _doc_ids_in_thread(uid, tid)
    if not doc_ids:
        return (0, 0, 0)

    print(f"[{uid}/{tid}] {len(doc_ids)} docs")
    vs = get_vectorstore(uid, thread_id=tid)
    doc_facets: Dict[str, dict] = {}
    chroma_updates = 0

    for doc_id in doc_ids:
        # Skip docs whose chunks already carry facet metadata, unless --force.
        try:
            existing = vs._collection.get(
                where={
                    "$and": [
                        {"user_id": {"$eq": uid}},
                        {"thread_id": {"$eq": tid}},
                        {"document_id": {"$eq": doc_id}},
                    ]
                },
                include=["metadatas"],
            )
        except Exception as e:
            print(f"  ! Chroma get failed for {doc_id}: {e}")
            continue

        ids = existing.get("ids", []) or []
        metas = existing.get("metadatas", []) or []
        if not ids:
            continue
        if not force:
            facet_keys = ("fiscal_year", "quarter", "doc_type")
            already = sum(
                1 for m in metas if any(((m or {}).get(k) for k in facet_keys))
            )
            if already == len(ids):
                print(f"  - {doc_id}: chunks already carry facets, skipping")
                continue

        result = await _facets_for_doc(uid, tid, doc_id)
        if not result:
            continue

        facet_meta = result["facets"]
        title = result["title"]
        doc_facets[doc_id] = facet_meta
        print(f"  > {title!r}: {facet_meta}  (chunks: {len(ids)})")

        if dry_run:
            chroma_updates += len(ids)
            continue

        new_metas = []
        for m in metas:
            merged = dict(m or {})
            merged.update(facet_meta)
            new_metas.append(merged)

        try:
            await asyncio.to_thread(
                vs._collection.update, ids=ids, metadatas=new_metas
            )
            chroma_updates += len(ids)
        except Exception as e:
            print(f"  ! Chroma update failed for {doc_id}: {e}")

    bm25_updates = 0
    if doc_facets:
        try:
            bm25_updates = _patch_bm25(uid, tid, doc_facets, dry_run)
            print(f"  BM25: patched {bm25_updates} chunk metadata entries")
        except Exception as e:
            print(f"  ! BM25 patch failed: {e}")

    return chroma_updates, bm25_updates, len(doc_facets)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", type=str, default=None)
    parser.add_argument("--thread-id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="overwrite existing facet metadata")
    args = parser.parse_args()

    if args.thread_id and not args.user_id:
        parser.error("--thread-id requires --user-id")

    started = time.time()
    total_chroma = 0
    total_bm25 = 0
    total_docs = 0

    for uid, tid in _iter_threads(args.user_id, args.thread_id):
        try:
            chroma, bm25, docs = await _process_thread(uid, tid, args.dry_run, args.force)
        except Exception as e:
            print(f"[{uid}/{tid}] ! unexpected error: {e}")
            continue
        total_chroma += chroma
        total_bm25 += bm25
        total_docs += docs

    elapsed = time.time() - started
    print(
        f"\nDone in {elapsed:.1f}s: {total_docs} docs processed, "
        f"{total_chroma} Chroma chunks updated, {total_bm25} BM25 metadata entries patched"
        f"{' (dry-run)' if args.dry_run else ''}"
    )


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    asyncio.run(main())
