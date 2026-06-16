import asyncio
import gc
import math
import os
import pickle
import re
import time
from typing import Any, Dict, List

import torch

# ── FUSE Filesystem Compatibility ──
# Must be set BEFORE any chromadb/sqlite3 imports.
# WAL mode fails on FUSE because it requires mmap() and shared memory (-shm file).
# DELETE mode uses simple file-based rollback journals that work on any filesystem.
os.environ.setdefault("CHROMA_SQLITE_JOURNAL_MODE", "DELETE")

# Turn off synchronous writes — avoids extra fsync() calls that block on FUSE
os.environ.setdefault("CHROMA_SQLITE_SYNCHRONOUS", "OFF")

# Use normal locking mode (not exclusive) so file locks are released between transactions
os.environ.setdefault("CHROMA_SQLITE_LOCKING_MODE", "NORMAL")

# Disable memory-mapped I/O — FUSE doesn't support mmap reliably
os.environ.setdefault("CHROMA_SQLITE_MMAP_SIZE", "0")


from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.embeddings.embeddings import get_embedding_function
from core.embeddings.context_enrichment import (
    build_enriched_chunk,
    build_entity_profiles,
    build_extractive_summary,
    detect_page_heading,
    extract_document_keywords,
    extract_entities,
    extract_entities_for_metadata,
    extract_entity_triples,
    get_adjacent_sentences,
)
from core.models.document import Documents
from core.services.triple_store import TripleStore

try:
    import nltk
    from nltk.tokenize import sent_tokenize

    nltk.download("punkt_tab", quiet=True)
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False

print("Loading embedding model...")
embedding_function = get_embedding_function()
print("Embedding model loaded.")

# Hierarchical chunking parameters.
# Parent chunks: larger, section-level context sent to the LLM.
# Child chunks: smaller, precise units indexed in Chroma and retrieved/reranked.
PARENT_CHUNK_SIZE = 1500   # characters — large enough that most slides fit in 1 parent
PARENT_CHUNK_OVERLAP = 150
CHILD_CHUNK_SIZE = 500     # characters — better semantic units for retrieval
CHILD_CHUNK_OVERLAP = 75

# nomic-embed-text-v1.5 task prefix for document embeddings.
# Queries use "search_query: " (configured in embeddings.py via query_instruction).
SEARCH_DOCUMENT_PREFIX = "search_document: "

# Separators ordered from coarsest to finest for structural awareness.
_CHUNK_SEPARATORS = [
    "\n## ", "\n### ", "\n#### ",  # Markdown section headings
    "\n\n",
    "\n",
    ". ", "! ", "? ",              # Sentence boundaries
    "|---|",                       # Markdown table rows
    " ",
    "",
]


def chunk_page_text_hierarchical(page_text: str) -> List[Dict[str, Any]]:
    """
    Split page text into a two-level hierarchy of parent/child chunks.

    Each child is derived from exactly one parent.  Child chunks are indexed
    in Chroma for precise retrieval; parent text is stored in chunk metadata
    so callers can expand results to richer context before prompting the LLM.

    Returns:
        List of dicts with keys:
            parent_text  — the larger context block
            child_text   — the precise snippet to embed and retrieve
            parent_idx   — index of the parent within this page
            child_idx    — index of the child within its parent
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=_CHUNK_SEPARATORS,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=_CHUNK_SEPARATORS,
    )

    parents = parent_splitter.split_text(page_text)
    result: List[Dict[str, Any]] = []
    for p_idx, parent in enumerate(parents):
        children = child_splitter.split_text(parent)
        for c_idx, child in enumerate(children):
            result.append(
                {
                    "parent_text": parent,
                    "child_text": child,
                    "parent_idx": p_idx,
                    "child_idx": c_idx,
                }
            )
    return result


def chunk_page_text(page_text: str) -> List[str]:
    """
    Backward-compatible wrapper: returns only child chunk texts.
    Use chunk_page_text_hierarchical() when parent context is needed.
    """
    return [item["child_text"] for item in chunk_page_text_hierarchical(page_text)]


# Expected embedding dimension for the current model
_EXPECTED_DIM = None


def _get_expected_dim() -> int:
    """Get the expected embedding dimension from the current model (cached)."""
    global _EXPECTED_DIM
    if _EXPECTED_DIM is None:
        test_emb = embedding_function.embed_query("test")
        _EXPECTED_DIM = len(test_emb)
        print(f"Embedding model dimension: {_EXPECTED_DIM}")
    return _EXPECTED_DIM


def _check_and_migrate_chroma(persist_path: str, user_id: str):
    """
    Check if existing ChromaDB data has mismatched embedding dimensions.
    If so, delete the entire persist directory to force re-creation.

    IMPORTANT: Does NOT nuke the directory on harmless errors like
    'collection does not exist' — that's normal for first-time users.
    """
    import shutil
    import chromadb

    client = None
    needs_reset = False
    try:
        client = chromadb.PersistentClient(path=persist_path)
        try:
            collection = client.get_collection("user_docs")
            if collection.count() > 0:
                sample = collection.get(limit=1, include=["embeddings"])
                if (
                    sample
                    and sample.get("embeddings")
                    and len(sample["embeddings"]) > 0
                ):
                    existing_dim = len(sample["embeddings"][0])
                    expected_dim = _get_expected_dim()
                    if existing_dim != expected_dim:
                        print(
                            f"[MIGRATION] Embedding dimension changed: {existing_dim} → {expected_dim}. "
                            f"Resetting ChromaDB for user {user_id}."
                        )
                        needs_reset = True
        except (ValueError, Exception) as e:
            # Collection doesn't exist — this is NORMAL for new users, not an error.
            # Do NOT nuke the directory for this.
            print(f"[MIGRATION CHECK] Collection check: {e} (OK for new users)")
    except Exception as e:
        # Client creation itself failed — the DB may be corrupted
        print(f"[MIGRATION CHECK] ChromaDB client creation failed: {e}")
        needs_reset = True
    finally:
        # Always clean up the client before LangChain creates its own
        if client is not None:
            del client
            client = None
        gc.collect()
        time.sleep(0.5)  # Allow FUSE to release file locks

    # Reset OUTSIDE the try/finally so client is fully released first
    if needs_reset:
        print(f"[MIGRATION CHECK] Resetting ChromaDB directory for user {user_id}")
        shutil.rmtree(persist_path, ignore_errors=True)
        os.makedirs(persist_path, exist_ok=True)


# Get Chroma vector store instance (with auto-migration for dimension changes)
def get_vectorstore(user_id: str, thread_id: str) -> Chroma:
    persist_path = os.path.join("data", user_id, "chroma")
    os.makedirs(persist_path, exist_ok=True)

    # Check for dimension mismatch before creating the LangChain Chroma wrapper
    _check_and_migrate_chroma(persist_path, user_id)

    return Chroma(
        collection_name="user_docs",
        persist_directory=persist_path,
        embedding_function=embedding_function,
    )


def _get_bm25_path(user_id: str, thread_id: str) -> str:
    """Get path for BM25 index storage."""
    bm25_dir = os.path.join("data", user_id, "bm25")
    os.makedirs(bm25_dir, exist_ok=True)
    return os.path.join(bm25_dir, f"{thread_id}.pkl")


def _build_and_save_bm25(chunk_data: list, user_id: str, thread_id: str):
    """Build and persist a BM25 index from chunk data."""
    if not chunk_data:
        print(f"No chunks to index for BM25 (thread {thread_id}), skipping.")
        return

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("rank_bm25 not installed, skipping BM25 index creation.")
        return

    # Tokenize documents for BM25 (with punctuation stripping for better matching)
    tokenized_docs = [_tokenize_for_bm25(text) for (_, text, _) in chunk_data]
    bm25 = BM25Okapi(tokenized_docs)

    bm25_data = {
        "bm25": bm25,
        "chunk_ids": [cid for (cid, _, _) in chunk_data],
        "chunk_texts": [text for (_, text, _) in chunk_data],
        "chunk_metadatas": [meta for (_, _, meta) in chunk_data],
    }

    bm25_path = _get_bm25_path(user_id, thread_id)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_data, f)
    print(f"BM25 index saved to {bm25_path} ({len(chunk_data)} documents)")


def load_bm25(user_id: str, thread_id: str):
    """Load BM25 index from disk. Returns None if not found."""
    bm25_path = _get_bm25_path(user_id, thread_id)
    if not os.path.exists(bm25_path):
        return None
    with open(bm25_path, "rb") as f:
        return pickle.load(f)


def _tokenize_for_bm25(text: str) -> list:
    """Tokenize text for BM25: lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # replace punctuation with space
    return text.split()


def search_bm25(user_id: str, thread_id: str, query: str, top_k: int = 20):
    """Search the BM25 index for the given query."""
    bm25_data = load_bm25(user_id, thread_id)
    if bm25_data is None:
        return []

    tokenized_query = _tokenize_for_bm25(query)
    scores = bm25_data["bm25"].get_scores(tokenized_query)

    # Get top_k indices sorted by score
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
        :top_k
    ]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # Only include results with positive BM25 score
            results.append(
                {
                    "page_content": bm25_data["chunk_texts"][idx],
                    "metadata": bm25_data["chunk_metadatas"][idx],
                    "bm25_score": float(scores[idx]),
                }
            )
    return results


async def delete_document_from_chroma(user_id: str, thread_id: str, document_id: str):
    """Delete all chunks belonging to a document in a specific thread from ChromaDB."""
    vectorstore = await asyncio.to_thread(get_vectorstore, user_id, thread_id)
    try:
        await asyncio.to_thread(
            vectorstore._collection.delete,
            where={
                "$and": [
                    {"document_id": document_id},
                    {"thread_id": thread_id},
                ]
            },
        )
        print(
            f"Deleted ChromaDB chunks for document {document_id} in thread {thread_id}"
        )
    except Exception as e:
        print(f"Error deleting ChromaDB chunks for document {document_id}: {e}")
        raise


def rebuild_bm25_after_deletion(user_id: str, thread_id: str, document_id: str):
    """Rebuild BM25 index excluding chunks from a deleted document."""
    bm25_data = load_bm25(user_id, thread_id)
    if bm25_data is None:
        return

    remaining = [
        (cid, text, meta)
        for cid, text, meta in zip(
            bm25_data["chunk_ids"],
            bm25_data["chunk_texts"],
            bm25_data["chunk_metadatas"],
        )
        if meta.get("document_id") != document_id
    ]

    if not remaining:
        bm25_path = _get_bm25_path(user_id, thread_id)
        if os.path.exists(bm25_path):
            os.remove(bm25_path)
        print(f"BM25 index removed (no chunks remain) for thread {thread_id}")
        return

    _build_and_save_bm25(remaining, user_id, thread_id)
    print(
        f"BM25 index rebuilt for thread {thread_id} after removing document {document_id}"
    )


async def save_documents_to_store(docs: Documents, user_id: str, thread_id: str):
    start_time = time.time()
    vectorstore = await asyncio.to_thread(get_vectorstore, user_id, thread_id)
    end_time = time.time()
    print(
        f"Initialized Chroma vector store in {end_time - start_time:.2f} seconds for user {user_id}"
    )

    chunk_data = []

    # Chunking with contextual enrichment (Phase 1.1 + 1.2 + 1.3 + 3.1)
    start_time = time.time()
    for doc in docs.documents:
        if doc.type == "spreadsheet":
            print(f"[VectorStore] Skipping vector chunking for spreadsheet {doc.title}. SQL engine will be used exclusively.")
            continue

        total_pages = len(doc.content)
        # Phase 1.1: Extract document-level keywords once per document
        doc_keywords = extract_document_keywords(doc.full_text)
        # Phase 3.1: Collect entity mentions for entity profile building
        entity_mentions = []

        # Step 5 (rag-refactor): document-level facets stamped onto every chunk
        # so the retriever can hard-filter by fiscal_year/quarter/doc_type.
        from core.constants import SWITCHES
        from core.embeddings.facets import (
            extract_document_facets,
            facets_to_chroma_metadata,
        )

        doc_facet_meta: dict = {}
        if SWITCHES.get("FACET_FILTERING", True):
            try:
                facets = await extract_document_facets(
                    title=doc.title,
                    text=doc.full_text or "",
                    summary=getattr(doc, "summary", None),
                )
                doc_facet_meta = facets_to_chroma_metadata(facets)
                if doc_facet_meta:
                    print(f"[Facets][ingest] {doc.title!r}: {doc_facet_meta}")
            except Exception as e:
                print(f"[Facets][ingest] failed for {doc.title!r}: {e}")

        for page in doc.content:
            # Hierarchical chunking: child chunks are indexed; parent text is stored
            # in metadata for later expansion before LLM prompting.
            hier_chunks = await asyncio.to_thread(
                chunk_page_text_hierarchical, page.text
            )
            # Phase 1.1: Detect section heading for this page
            heading = detect_page_heading(page.text)
            # Extract child texts for adjacent-sentence context building
            child_texts = [item["child_text"] for item in hier_chunks]

            for flat_idx, item in enumerate(hier_chunks):
                p_idx = item["parent_idx"]
                c_idx = item["child_idx"]
                child_text = item["child_text"]
                parent_text = item["parent_text"]

                child_id = f"{doc.id}_page{page.number}_p{p_idx}_c{c_idx}"
                parent_id = f"{doc.id}_page{page.number}_p{p_idx}"

                # Phase 1.1: Build enriched child chunk with programmatic context
                adjacent_ctx = get_adjacent_sentences(child_texts, flat_idx, _HAS_NLTK)
                enriched_chunk = build_enriched_chunk(
                    chunk_text=child_text,
                    doc_title=doc.title,
                    page_no=page.number,
                    total_pages=total_pages,
                    heading=heading,
                    keywords=doc_keywords,
                    adjacent_ctx=adjacent_ctx,
                    search_prefix=SEARCH_DOCUMENT_PREFIX,
                )

                metadata = {
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "document_id": doc.id,
                    "page_no": page.number,
                    "chunk_index": flat_idx,
                    "chunk_type": "child",
                    "parent_chunk_id": parent_id,
                    "parent_text": parent_text,
                    "file_name": doc.file_name,
                    "title": doc.title,
                }
                if doc_facet_meta:
                    metadata.update(doc_facet_meta)

                # Phase 1.2: Extract entities — used for both metadata and profiles
                names, types = extract_entities(child_text)
                if names:
                    metadata["entities"] = "|".join(names[:15])
                    metadata["entity_types"] = "|".join(types[:15])
                    # Phase 3.1: Collect for entity profiles
                    for name, etype in zip(names, types):
                        entity_mentions.append(
                            {
                                "name": name,
                                "type": etype,
                                "chunk_text": child_text,
                                "page_no": page.number,
                            }
                        )

                chunk_data.append((child_id, enriched_chunk, metadata))

        # Phase 1.3: Add document summary as a special indexed chunk
        summary_text = doc.summary if doc.summary else None
        if not summary_text:
            summary_text = build_extractive_summary(doc.full_text, doc.title)
        if summary_text:
            summary_id = f"{doc.id}_summary"
            summary_enriched = (
                f"{SEARCH_DOCUMENT_PREFIX}Document: {doc.title} | Summary\n"
                f"Topics: {', '.join(doc_keywords)}\n\n{summary_text}"
            )
            summary_metadata = {
                "user_id": user_id,
                "thread_id": thread_id,
                "document_id": doc.id,
                "page_no": 0,
                "chunk_index": -1,
                "file_name": doc.file_name,
                "title": doc.title,
                "chunk_type": "document_summary",
            }
            if doc_facet_meta:
                summary_metadata.update(doc_facet_meta)
            ner_meta = extract_entities_for_metadata(summary_text)
            summary_metadata.update(ner_meta)
            chunk_data.append((summary_id, summary_enriched, summary_metadata))

        # Phase 3.1: Build entity profile chunks for frequently mentioned entities
        if entity_mentions:
            profiles = build_entity_profiles(
                entity_mentions, doc.title, search_prefix=SEARCH_DOCUMENT_PREFIX
            )
            for profile_text, extra_meta in profiles:
                safe_name = re.sub(r"[^a-z0-9_]", "_", extra_meta["entity_name"].lower())
                profile_id = f"{doc.id}_entity_{safe_name}"
                profile_metadata = {
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "document_id": doc.id,
                    "page_no": 0,
                    "chunk_index": -2,
                    "file_name": doc.file_name,
                    "title": doc.title,
                }
                if doc_facet_meta:
                    profile_metadata.update(doc_facet_meta)
                profile_metadata.update(extra_meta)
                chunk_data.append((profile_id, profile_text, profile_metadata))
            print(
                f"[Entity Profiles] Created {len(profiles)} profiles for doc {doc.title}"
            )

        # Phase 3.2: Extract and store entity-relation triples
        try:
            doc_triples = []
            for page in doc.content:
                page_triples = extract_entity_triples(page.text)
                for t in page_triples:
                    t["page_no"] = page.number
                doc_triples.extend(page_triples)

            if doc_triples:
                stored = await asyncio.to_thread(
                    TripleStore.store_triples,
                    user_id,
                    thread_id,
                    doc.id,
                    doc_triples,
                )
                print(
                    f"[Triples] Stored {stored} triples for doc {doc.title}"
                )
        except Exception as e:
            print(f"[Triples] Error extracting triples for {doc.title}: {e}")

    end_time = time.time()
    print(
        f"Processed {len(chunk_data)} chunks in {end_time - start_time:.2f} seconds for user {user_id}"
    )

    # Batch embedding and upsert to Chroma FIRST — BM25 is saved after so both
    # indexes stay in sync even if a Chroma batch fails partway through.
    batch_size = 1000  # Reduced to avoid VRAM hoarding on 48GB GPU
    total_batches = math.ceil(len(chunk_data) / batch_size)

    for batch_idx in range(total_batches):
        batch = chunk_data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_ids, batch_texts, batch_metadatas = zip(*batch)

        start_time = time.time()
        embeddings = await asyncio.to_thread(
            vectorstore.embeddings.embed_documents, list(batch_texts)
        )
        end_time = time.time()
        print(
            f"Generated embeddings for batch {batch_idx + 1} in {end_time - start_time:.2f} seconds"
        )

        # Release cached GPU memory after batch embedding to prevent VRAM hoarding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Upsert to Chroma (with retry for FUSE lock issues)
        print(f"Upserting batch {batch_idx + 1} to Chroma")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                await asyncio.to_thread(
                    vectorstore._collection.upsert,
                    embeddings=embeddings,
                    documents=list(batch_texts),
                    metadatas=list(batch_metadatas),
                    ids=list(batch_ids),
                )
                end_time = time.time()
                print(
                    f"Upserted batch {batch_idx + 1} in {end_time - start_time:.2f} seconds"
                )
                break
            except Exception as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    wait = 2**attempt  # 1s, 2s, 4s
                    print(
                        f"[ChromaDB] Lock detected on batch {batch_idx + 1}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    await asyncio.sleep(wait)
                else:
                    raise

    print(f"Saved {len(chunk_data)} chunks to Chroma for user {user_id}")

    # Build and save BM25 index for hybrid search AFTER Chroma succeeds.
    # Merge with existing index so that incremental uploads (second doc, third doc, …)
    # do not evict chunks from previously indexed documents in the same thread.
    existing_bm25 = load_bm25(user_id, thread_id)
    if existing_bm25:
        existing_chunks = list(zip(
            existing_bm25["chunk_ids"],
            existing_bm25["chunk_texts"],
            existing_bm25["chunk_metadatas"],
        ))
        # Deduplicate: new chunks override old ones with the same ID (upsert semantics)
        merged_by_id = {cid: (cid, txt, meta) for cid, txt, meta in existing_chunks}
        for cid, txt, meta in chunk_data:
            merged_by_id[cid] = (cid, txt, meta)
        all_bm25_chunks = list(merged_by_id.values())
    else:
        all_bm25_chunks = chunk_data

    await asyncio.to_thread(_build_and_save_bm25, all_bm25_chunks, user_id, thread_id)


async def add_existing_document_to_store(doc, user_id: str, thread_id: str):
    """
    Add an already-parsed document to a thread's vector/BM25 stores.
    Skips OCR/parsing entirely — only re-chunks, re-embeds, and indexes.

    Uses thread-prefixed chunk IDs to avoid collisions when the same document
    exists in multiple threads within the same user's ChromaDB collection.
    """
    if doc.type == "spreadsheet":
        print(f"[VectorStore] Skipping vector chunking for existing spreadsheet {doc.title}. SQL engine will be used exclusively.")
        return

    start_time = time.time()
    vectorstore = await asyncio.to_thread(get_vectorstore, user_id, thread_id)

    chunk_data = []
    total_pages = len(doc.content)
    doc_keywords = extract_document_keywords(getattr(doc, "full_text", ""))
    # Phase 3.1: Collect entity mentions for entity profile building
    entity_mentions = []

    for page in doc.content:
        # Hierarchical chunking: child chunks are indexed; parent text is stored
        # in metadata for later expansion before LLM prompting.
        hier_chunks = await asyncio.to_thread(
            chunk_page_text_hierarchical, page.text
        )
        heading = detect_page_heading(page.text)
        child_texts = [item["child_text"] for item in hier_chunks]

        for flat_idx, item in enumerate(hier_chunks):
            p_idx = item["parent_idx"]
            c_idx = item["child_idx"]
            child_text = item["child_text"]
            parent_text = item["parent_text"]

            # Thread-prefixed ID prevents collision with same doc in other threads
            child_id = f"{thread_id}_{doc.id}_page{page.number}_p{p_idx}_c{c_idx}"
            parent_id = f"{thread_id}_{doc.id}_page{page.number}_p{p_idx}"

            # Phase 1.1: Build enriched child chunk with programmatic context
            adjacent_ctx = get_adjacent_sentences(child_texts, flat_idx, _HAS_NLTK)
            enriched_chunk = build_enriched_chunk(
                chunk_text=child_text,
                doc_title=doc.title,
                page_no=page.number,
                total_pages=total_pages,
                heading=heading,
                keywords=doc_keywords,
                adjacent_ctx=adjacent_ctx,
                search_prefix=SEARCH_DOCUMENT_PREFIX,
            )

            metadata = {
                "user_id": user_id,
                "thread_id": thread_id,
                "document_id": doc.id,
                "page_no": page.number,
                "chunk_index": flat_idx,
                "chunk_type": "child",
                "parent_chunk_id": parent_id,
                "parent_text": parent_text,
                "file_name": doc.file_name,
                "title": doc.title,
            }
            # Phase 1.2: Extract entities — used for both metadata and profiles
            names, types = extract_entities(child_text)
            if names:
                metadata["entities"] = "|".join(names[:15])
                metadata["entity_types"] = "|".join(types[:15])
                # Phase 3.1: Collect for entity profiles
                for name, etype in zip(names, types):
                    entity_mentions.append(
                        {
                            "name": name,
                            "type": etype,
                            "chunk_text": child_text,
                            "page_no": page.number,
                        }
                    )

            chunk_data.append((child_id, enriched_chunk, metadata))

    # Phase 1.3: Document summary chunk
    full_text = getattr(doc, "full_text", "")
    summary_text = getattr(doc, "summary", None) or build_extractive_summary(full_text, doc.title)
    if summary_text:
        summary_id = f"{thread_id}_{doc.id}_summary"
        summary_enriched = (
            f"{SEARCH_DOCUMENT_PREFIX}Document: {doc.title} | Summary\n"
            f"Topics: {', '.join(doc_keywords)}\n\n{summary_text}"
        )
        summary_metadata = {
            "user_id": user_id,
            "thread_id": thread_id,
            "document_id": doc.id,
            "page_no": 0,
            "chunk_index": -1,
            "file_name": doc.file_name,
            "title": doc.title,
            "chunk_type": "document_summary",
        }
        ner_meta = extract_entities_for_metadata(summary_text)
        summary_metadata.update(ner_meta)
        chunk_data.append((summary_id, summary_enriched, summary_metadata))

    # Phase 3.1: Build entity profile chunks
    if entity_mentions:
        profiles = build_entity_profiles(
            entity_mentions, doc.title, search_prefix=SEARCH_DOCUMENT_PREFIX
        )
        for profile_text, extra_meta in profiles:
            safe_name = re.sub(r"[^a-z0-9_]", "_", extra_meta["entity_name"].lower())
            profile_id = f"{thread_id}_{doc.id}_entity_{safe_name}"
            profile_metadata = {
                "user_id": user_id,
                "thread_id": thread_id,
                "document_id": doc.id,
                "page_no": 0,
                "chunk_index": -2,
                "file_name": doc.file_name,
                "title": doc.title,
            }
            profile_metadata.update(extra_meta)
            chunk_data.append((profile_id, profile_text, profile_metadata))
        print(
            f"[Entity Profiles] Created {len(profiles)} profiles for doc {doc.title}"
        )

    # Phase 3.2: Extract and store entity-relation triples
    try:
        doc_triples = []
        for page in doc.content:
            page_triples = extract_entity_triples(page.text)
            for t in page_triples:
                t["page_no"] = page.number
            doc_triples.extend(page_triples)

        if doc_triples:
            stored = await asyncio.to_thread(
                TripleStore.store_triples,
                user_id,
                thread_id,
                doc.id,
                doc_triples,
            )
            print(f"[Triples] Stored {stored} triples for doc {doc.title}")
    except Exception as e:
        print(f"[Triples] Error extracting triples for doc {doc.id}: {e}")

    if not chunk_data:
        print(f"No chunks to store for document {doc.id} in thread {thread_id}")
        return

    print(f"Adding {len(chunk_data)} chunks for doc {doc.id} to thread {thread_id}")

    # Merge with existing BM25 data (BM25 is per-thread, rebuild includes all docs)
    existing_bm25 = load_bm25(user_id, thread_id)
    if existing_bm25:
        existing_chunks = list(
            zip(
                existing_bm25["chunk_ids"],
                existing_bm25["chunk_texts"],
                existing_bm25["chunk_metadatas"],
            )
        )
        all_bm25_chunks = existing_chunks + chunk_data
    else:
        all_bm25_chunks = chunk_data

    await asyncio.to_thread(_build_and_save_bm25, all_bm25_chunks, user_id, thread_id)

    # Batch embed and upsert to ChromaDB
    batch_size = 1000
    total_batches = math.ceil(len(chunk_data) / batch_size)

    for batch_idx in range(total_batches):
        batch = chunk_data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_ids, batch_texts, batch_metadatas = zip(*batch)

        embeddings = await asyncio.to_thread(
            vectorstore.embeddings.embed_documents, list(batch_texts)
        )

        # Release cached GPU memory after batch embedding to prevent VRAM hoarding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                await asyncio.to_thread(
                    vectorstore._collection.upsert,
                    embeddings=embeddings,
                    documents=list(batch_texts),
                    metadatas=list(batch_metadatas),
                    ids=list(batch_ids),
                )
                break
            except Exception as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    wait = 2**attempt
                    await asyncio.sleep(wait)
                else:
                    raise

    elapsed = time.time() - start_time
    print(
        f"Added {len(chunk_data)} chunks for doc {doc.id} to thread {thread_id} in {elapsed:.2f}s"
    )
