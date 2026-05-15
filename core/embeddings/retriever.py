import math
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import CrossEncoder

from core.embeddings.context_enrichment import (
    expand_keywords_with_synonyms,
    extract_query_entities,
    extract_query_keywords,
)
from core.embeddings.vectorstore import get_vectorstore, search_bm25

# Initialize cross-encoder for re-ranking (lazy loading)
_cross_encoder = None


def get_cross_encoder():
    """Lazy load the cross-encoder model. FP16 applied on CUDA for ~4-9x speedup."""
    global _cross_encoder
    if _cross_encoder is None:
        from core.config import settings
        from core.utils.device import resolve_device

        device = resolve_device(settings.CROSS_ENCODER_DEVICE)
        print(f"Loading cross-encoder model for re-ranking ({device})...")
        _cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,  # Truncate inputs to model's max position embeddings
            device=device,
        )
        if device == "cuda":
            _cross_encoder.model.half()
        print(f"Cross-encoder model loaded on {device}.")
    return _cross_encoder


def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]], k: int = 60
) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF assigns a score to each document based on its rank in each list:
        score(d) = sum(1 / (k + rank_i)) for each list i that contains d

    Args:
        result_lists: List of ranked result lists (each is a list of dicts with metadata)
        k: RRF constant (default 60, standard in literature)

    Returns:
        Merged and deduplicated results sorted by RRF score
    """
    scores = {}
    doc_map = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            # Create a unique key for deduplication
            doc_id = doc.get("metadata", {}).get("document_id", "")
            page_no = doc.get("metadata", {}).get("page_no", 0)
            chunk_idx = doc.get("metadata", {}).get("chunk_index", rank)
            key = f"{doc_id}_p{page_no}_c{chunk_idx}"

            rrf_score = 1.0 / (k + rank + 1)
            scores[key] = scores.get(key, 0) + rrf_score
            if key not in doc_map:
                doc_map[key] = doc

    # Sort by RRF score descending
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for key in sorted_keys:
        doc = doc_map[key].copy()
        doc["rrf_score"] = scores[key]
        results.append(doc)

    return results


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = None,
    diversity_lambda: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Re-rank retrieved chunks using cross-encoder and ensure diversity.

    This function:
    1. Re-ranks chunks based on query relevance using cross-encoder
    2. Ensures diversity across documents using embedding-based MMR
    3. Removes redundant chunks
    4. Balances representation across documents

    Args:
        query: The user's query
        chunks: List of retrieved chunks with metadata
        top_k: Number of top chunks to return (None for all)
        diversity_lambda: Trade-off between relevance and diversity (0-1)
                         Higher values prioritize diversity

    Returns:
        Re-ranked and diversified list of chunks
    """
    if not chunks:
        return []

    if top_k is None:
        top_k = len(chunks)

    print(f"Re-ranking {len(chunks)} chunks for query...")

    # Step 1: Cross-encoder re-ranking for relevance
    try:
        cross_encoder = get_cross_encoder()

        # Prepare query-chunk pairs
        pairs = [(query, chunk.get("page_content", "")) for chunk in chunks]

        # Get relevance scores (raw logits, typically -10 to +10)
        # CrossEncoder is initialized with max_length=512 to truncate long inputs.
        # GLM-OCR Markdown (tables, formulas) tokenizes into far more tokens per char
        # than plain text and can exceed the model's 512-token position embedding limit.
        scores = cross_encoder.predict(pairs)

        # Release cached GPU memory after cross-encoder inference
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Normalize to 0-1 range using sigmoid (clamped to prevent overflow)
        def _sigmoid(x):
            x = max(-500.0, min(500.0, float(x)))
            return 1.0 / (1.0 + math.exp(-x))

        for i, chunk in enumerate(chunks):
            chunk["relevance_score"] = _sigmoid(scores[i])

        print(f"Cross-encoder re-ranking completed.")

    except Exception as e:
        print(f"Cross-encoder re-ranking failed: {e}. Using original order.")
        # Fallback: use original order with default scores
        for i, chunk in enumerate(chunks):
            chunk["relevance_score"] = 1.0 - (i / len(chunks))  # Decreasing scores


    # Step 2: MMR with cosine similarity for diversity
    reranked_chunks = []
    selected_indices = set()

    # Sort by relevance score initially
    sorted_indices = sorted(
        range(len(chunks)), key=lambda i: chunks[i]["relevance_score"], reverse=True
    )

    # Pre-compute TF-IDF-like vectors for cosine similarity (lightweight)
    chunk_vectors = _compute_tfidf_vectors(chunks)

    # Select chunks using MMR
    for _ in range(min(top_k, len(chunks))):
        best_idx = None
        best_score = -float("inf")

        for idx in sorted_indices:
            if idx in selected_indices:
                continue

            # Relevance score
            relevance = chunks[idx]["relevance_score"]

            # Diversity penalty using cosine similarity
            diversity_penalty = 0.0
            if reranked_chunks:
                max_similarity = 0.0
                for sel_idx in selected_indices:
                    sim = _cosine_similarity(chunk_vectors[idx], chunk_vectors[sel_idx])
                    max_similarity = max(max_similarity, sim)
                diversity_penalty = max_similarity

            # MMR score: balance relevance and diversity
            mmr_score = (
                1 - diversity_lambda
            ) * relevance - diversity_lambda * diversity_penalty

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.add(best_idx)
            chunk = chunks[best_idx]
            chunk["rerank_score"] = chunks[best_idx]["relevance_score"]
            reranked_chunks.append(chunk)

    print(f"Re-ranking complete. Selected {len(reranked_chunks)} chunks.")

    # Step 4: Log document diversity
    doc_counts = {}
    for chunk in reranked_chunks:
        doc_id = chunk.get("metadata", {}).get("document_id", "unknown")
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

    print(f"Document distribution after re-ranking:")
    for doc_id, count in doc_counts.items():
        print(f"  Document {doc_id}: {count} chunks")

    return reranked_chunks


def _compute_tfidf_vectors(chunks: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Compute simple TF-IDF-like word frequency vectors for cosine similarity."""
    from collections import Counter

    # Build vocabulary from all chunks
    all_words = set()
    chunk_word_counts = []
    for chunk in chunks:
        words = chunk.get("page_content", "").lower().split()
        word_counts = Counter(words)
        chunk_word_counts.append(word_counts)
        all_words.update(words)

    # Compute document frequency
    doc_freq = Counter()
    for wc in chunk_word_counts:
        for word in wc:
            doc_freq[word] += 1

    n_docs = len(chunks)
    vectors = []
    for wc in chunk_word_counts:
        vec = {}
        for word, count in wc.items():
            tf = count / max(sum(wc.values()), 1)
            idf = math.log((n_docs + 1) / (doc_freq[word] + 1))
            vec[word] = tf * idf
        vectors.append(vec)

    return vectors


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Compute cosine similarity between two sparse TF-IDF vectors."""
    # Find common words
    common_words = set(vec_a.keys()) & set(vec_b.keys())
    if not common_words:
        return 0.0

    dot_product = sum(vec_a[w] * vec_b[w] for w in common_words)
    norm_a = math.sqrt(sum(v**2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v**2 for v in vec_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def get_user_retriever(
    user_id: str, thread_id: str, document_id: str = None, k: int = 5
):
    """
    Get a retriever for a specific user, thread, and optionally document.

    Args:
        user_id: User identifier
        thread_id: Thread identifier
        document_id: Optional document identifier to filter by
        k: Number of chunks to retrieve

    Returns:
        LangChain retriever object
    """
    vectorstore = get_vectorstore(user_id, thread_id=thread_id)
    filter_conditions = []

    if user_id is not None:
        filter_conditions.append({"user_id": {"$eq": user_id}})
    if thread_id is not None:
        filter_conditions.append({"thread_id": {"$eq": thread_id}})
    if document_id is not None:
        filter_conditions.append({"document_id": {"$eq": document_id}})

    search_kwargs = {
        "k": k,
        "filter": {"$and": filter_conditions},
    }

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    return retriever


async def hybrid_retrieve(
    user_id: str,
    thread_id: str,
    query: str,
    additional_queries: List[str] = None,
    vector_k: int = 30,
    bm25_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval combining vector search (ChromaDB) and BM25 keyword search.

    Uses Reciprocal Rank Fusion (RRF) to merge results from both retrievers,
    providing both semantic understanding and keyword matching.

    Phase 2.2: Supports multi-query retrieval — when additional_queries are provided,
    retrieves for each query variant in parallel and merges all result lists via RRF.
    This gives broader coverage for rewritten/decomposed queries.

    Args:
        user_id: User identifier
        thread_id: Thread identifier
        query: The primary search query
        additional_queries: Optional extra query variants for multi-query retrieval
        vector_k: Number of results from vector search (per query)
        bm25_k: Number of results from BM25 search (per query)

    Returns:
        Merged and deduplicated results sorted by RRF score
    """
    import asyncio

    all_queries = [query]
    if additional_queries:
        all_queries.extend(additional_queries)

    # Scale k per query to keep total retrieval volume manageable
    num_queries = len(all_queries)
    if num_queries > 1:
        per_query_vector_k = max(15, vector_k // num_queries)
        per_query_bm25_k = max(10, bm25_k // num_queries)
        print(
            f"[Multi-query retrieval] {num_queries} queries, "
            f"vector_k={per_query_vector_k}/query, bm25_k={per_query_bm25_k}/query"
        )
    else:
        per_query_vector_k = vector_k
        per_query_bm25_k = bm25_k

    vector_retriever = get_user_retriever(user_id, thread_id, k=per_query_vector_k)

    async def get_vector_results(q: str):
        try:
            docs = await vector_retriever.ainvoke(q)
            return [doc.model_dump() for doc in docs]
        except Exception as e:
            print(f"[Retrieval] Vector search failed for query '{q[:80]}': {e}")
            return []

    async def get_bm25_results(q: str):
        try:
            return await asyncio.to_thread(
                search_bm25, user_id, thread_id, q, per_query_bm25_k
            )
        except Exception as e:
            print(f"[Retrieval] BM25 search failed for query '{q[:80]}': {e}")
            return []

    # Run vector + BM25 for every query variant in parallel
    tasks = []
    for q in all_queries:
        tasks.append(get_vector_results(q))
        tasks.append(get_bm25_results(q))

    all_results = await asyncio.gather(*tasks)

    # Collect all non-empty result lists for RRF fusion
    result_lists = [res for res in all_results if res]

    total_retrieved = sum(len(r) for r in result_lists)
    print(
        f"Hybrid search: {total_retrieved} total results from {num_queries} "
        f"query variant(s), {len(result_lists)} result lists"
    )

    if not result_lists:
        fused = []
    elif len(result_lists) == 1:
        fused = result_lists[0]
    else:
        # Merge all result lists using Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(result_lists)
        print(f"RRF fusion produced {len(fused)} unique results")

    # Phase 1.2: Entity + keyword boosting with synonym expansion
    # 1) Named entity boost — only from PRIMARY query (not HyDE/variants) to avoid noise
    query_entities = extract_query_entities(query)

    # 2) Keyword boost with synonyms — extract nouns and expand with synonyms
    query_keywords = extract_query_keywords(query)
    expanded_keywords = expand_keywords_with_synonyms(query_keywords)

    has_boost_terms = bool(query_entities) or bool(expanded_keywords)

    if has_boost_terms:
        entity_lower = {e.lower() for e in query_entities}
        for doc in fused:
            doc_entities_str = doc.get("metadata", {}).get("entities", "")
            doc_content = doc.get("page_content", "").lower()
            boost = 1.0

            # Entity boost: match on pipe-delimited boundaries for precision
            if doc_entities_str and entity_lower:
                doc_entity_list = [e.strip().lower() for e in doc_entities_str.split("|")]
                entity_matches = sum(1 for e in entity_lower if e in doc_entity_list)
                if entity_matches > 0:
                    boost += 0.25 * entity_matches  # 25% per entity match

            # Keyword + synonym boost: check document content for expanded terms
            if expanded_keywords:
                kw_matches = sum(1 for kw in expanded_keywords if kw in doc_content)
                if kw_matches > 0:
                    # Smaller boost per keyword, capped to avoid runaway boosting
                    boost += min(0.3, 0.1 * kw_matches)

            if boost > 1.0:
                doc["rrf_score"] = doc.get("rrf_score", 0.0) * boost

        # Re-sort after boosting
        fused.sort(key=lambda d: d.get("rrf_score", 0.0), reverse=True)
        print(
            f"Boost applied — entities: {query_entities}, "
            f"keywords: {query_keywords}, expanded: {expanded_keywords}"
        )

    return fused


async def get_multi_document_retriever(
    user_id: str,
    thread_id: str,
    document_ids: List[str],
    query: str = "",
    k_per_document: int = 6,
    total_k: int = 12,
) -> List[Dict[str, Any]]:
    """
    Robust retrieval for multiple documents with balanced representation.

    This function ensures that:
    1. Each document gets a minimum number of chunks (k_per_document)
    2. Total chunks don't exceed total_k
    3. Documents are represented proportionally

    Args:
        user_id: User identifier
        thread_id: Thread identifier
        document_ids: List of document IDs to retrieve from
        query: The user query for semantic similarity search
        k_per_document: Minimum chunks to retrieve per document
        total_k: Maximum total chunks to return

    Returns:
        List of retrieved document chunks with metadata
    """
    if not document_ids:
        # Fallback to hybrid retrieval if no documents specified
        return await hybrid_retrieve(user_id, thread_id, query, vector_k=total_k)

    num_documents = len(document_ids)

    # Calculate chunks per document
    # Strategy: Ensure minimum chunks per document, then distribute remaining
    if num_documents == 1:
        chunks_per_doc = total_k
    else:
        # Calculate balanced distribution
        chunks_per_doc = min(k_per_document, math.ceil(total_k / num_documents))

    print(
        f"Retrieving {chunks_per_doc} chunks per document from {num_documents} documents"
    )

    all_retrieved_docs = []

    # Retrieve chunks from each document separately
    for doc_id in document_ids:
        retriever = get_user_retriever(
            user_id, thread_id, document_id=doc_id, k=chunks_per_doc
        )

        try:
            retrieved_docs = await retriever.ainvoke(query)
            all_retrieved_docs.extend([doc.model_dump() for doc in retrieved_docs])
            print(f"Retrieved {len(retrieved_docs)} chunks from document {doc_id}")
        except Exception as e:
            print(f"Error retrieving from document {doc_id}: {e}")
            continue

    # If we have fewer chunks than total_k, try to get more from all documents
    if len(all_retrieved_docs) < total_k:
        additional_chunks_needed = total_k - len(all_retrieved_docs)
        print(
            f"Retrieving {additional_chunks_needed} additional chunks from all documents"
        )

        # Get additional chunks without document filter
        retriever = get_user_retriever(user_id, thread_id, k=additional_chunks_needed)
        additional_docs = await retriever.ainvoke(query)

        # Filter out documents we already have enough chunks from
        existing_doc_ids = set(
            doc.get("metadata", {}).get("document_id") for doc in all_retrieved_docs
        )
        for doc in additional_docs:
            doc_data = doc.model_dump()
            doc_id = doc_data.get("metadata", {}).get("document_id")
            if doc_id not in existing_doc_ids or len(all_retrieved_docs) < total_k:
                all_retrieved_docs.append(doc_data)

    # Ensure we don't exceed total_k
    all_retrieved_docs = all_retrieved_docs[:total_k]

    print(f"Total retrieved chunks: {len(all_retrieved_docs)}")
    return all_retrieved_docs


async def get_thread_documents_retriever(
    user_id: str,
    thread_id: str,
    query: str = "",
    additional_queries: List[str] = None,
    k: int = None,
    max_total_chunks: int = 50,
    full_document_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Get retriever for all documents in a thread with score-aware document diversity.
    Uses hybrid search (vector + BM25) for improved recall.

    Phase 2.2: Supports multi-query retrieval via additional_queries parameter.

    When ``full_document_mode`` is True, the function still runs a lightweight
    hybrid retrieval to identify the most-relevant document, then fetches ALL
    chunks from that document via ChromaDB ``get()`` — bypassing the quality
    gate, adaptive budget, and k-limit.  Chunks are returned sorted by page
    number so the LLM receives them in slide/page order.

    Strategy (normal mode):
    1. Fetch a broad candidate pool via hybrid retrieval.
    2. Quality-gate: keep only documents whose best RRF score is ≥ 25% of the
       top document's score (weak/irrelevant documents are excluded).
    3. Guarantee top-1 child chunk per kept document so every qualifying doc
       has at least one representative chunk.
    4. Fill remaining budget by descending RRF score with a per-doc cap.
    5. Deduplicate by content_hash when available.

    Args:
        user_id: User identifier
        thread_id: Thread identifier
        query: The user query for semantic similarity search
        additional_queries: Optional extra query variants for multi-query retrieval
        k: Total number of chunks to retrieve (None for adaptive)
        max_total_chunks: Hard upper bound on returned chunks
        full_document_mode: When True, fetch ALL chunks from the top document
    """
    # Use hybrid retrieval (vector + BM25) for better recall
    retrieved_docs = await hybrid_retrieve(
        user_id,
        thread_id,
        query,
        additional_queries=additional_queries,
        vector_k=max_total_chunks * 2,
        bm25_k=max_total_chunks,
    )

    if not retrieved_docs:
        return []

    # Group by document_id
    docs_by_document: Dict[str, List[Dict[str, Any]]] = {}
    for doc in retrieved_docs:
        doc_id = doc.get("metadata", {}).get("document_id", "unknown")
        docs_by_document.setdefault(doc_id, []).append(doc)

    num_documents = len(docs_by_document)
    if num_documents == 0:
        return []

    # --- Quality gate ---
    # Keep only documents whose best fused score is at least 25% of the top doc.
    best_score_per_doc = {
        doc_id: max(d.get("rrf_score", 0.0) for d in docs)
        for doc_id, docs in docs_by_document.items()
    }
    top_score = max(best_score_per_doc.values(), default=0.0)
    threshold = top_score * 0.25

    kept_docs = {
        doc_id: docs
        for doc_id, docs in docs_by_document.items()
        if best_score_per_doc[doc_id] >= threshold
    }

    num_kept = len(kept_docs)
    if num_kept == 0:
        return []

    # ── Full-document mode: fetch ALL chunks from the top-scoring document ──
    if full_document_mode:
        target_doc_id = max(best_score_per_doc, key=best_score_per_doc.get)
        print(
            f"[FullDocMode] Fetching ALL chunks for document {target_doc_id} "
            f"(score={best_score_per_doc[target_doc_id]:.4f})"
        )
        try:
            vectorstore = get_vectorstore(user_id, thread_id=thread_id)
            raw = vectorstore._collection.get(
                where={
                    "$and": [
                        {"document_id": {"$eq": target_doc_id}},
                        {"chunk_type": {"$eq": "child"}},
                    ]
                },
                include=["documents", "metadatas"],
            )
            all_chunks: List[Dict[str, Any]] = []
            ids = raw.get("ids", [])
            docs_list = raw.get("documents", [])
            metas = raw.get("metadatas", [])
            for i, cid in enumerate(ids):
                meta = metas[i] if i < len(metas) else {}
                text = docs_list[i] if i < len(docs_list) else ""
                all_chunks.append({
                    "id": cid,
                    "page_content": text,
                    "metadata": meta,
                    "rrf_score": 1.0,  # placeholder — all chunks are equally relevant
                })
            # Sort by page_no then chunk_index for slide-sequential order
            all_chunks.sort(
                key=lambda c: (
                    c.get("metadata", {}).get("page_no", 0),
                    c.get("metadata", {}).get("chunk_index", 0),
                )
            )
            print(f"[FullDocMode] Retrieved {len(all_chunks)} child chunks for document")
            return all_chunks
        except Exception as e:
            print(f"[FullDocMode] ChromaDB get() failed: {e}, falling back to normal retrieval")

    # --- Normal mode: Adaptive total budget ---
    if k is None:
        if num_kept <= 2:
            k = 40
        elif num_kept <= 5:
            k = 80
        elif num_kept <= 10:
            k = 150
        else:
            k = min(max_total_chunks, num_kept * 15)

    k = min(k, max_total_chunks)

    print(
        f"[MultiDoc] {num_kept}/{num_documents} docs passed quality gate "
        f"(threshold={threshold:.4f}), budget k={k}"
    )

    # --- Guarantee top-1 per doc, then fill remaining budget by score (no per-doc cap) ---
    guaranteed: List[Dict[str, Any]] = []
    remainder_pool: List[Dict[str, Any]] = []

    for doc_id in sorted(
        kept_docs, key=lambda d: best_score_per_doc[d], reverse=True
    ):
        docs_sorted = sorted(
            kept_docs[doc_id],
            key=lambda d: d.get("rrf_score", 0.0),
            reverse=True,
        )
        guaranteed.append(docs_sorted[0])
        remainder_pool.extend(docs_sorted[1:])

    remainder_pool.sort(key=lambda d: d.get("rrf_score", 0.0), reverse=True)
    result = guaranteed + remainder_pool
    result = result[:k]

    # --- Deduplicate by content_hash if available ---
    seen_hashes: set = set()
    deduped: List[Dict[str, Any]] = []
    for doc in result:
        ch = doc.get("metadata", {}).get("content_hash")
        if ch:
            if ch in seen_hashes:
                continue
            seen_hashes.add(ch)
        deduped.append(doc)

    print(f"[MultiDoc] Final: {len(deduped)} chunks from {num_kept} documents")
    for doc_id in kept_docs:
        count = sum(
            1
            for d in deduped
            if d.get("metadata", {}).get("document_id") == doc_id
        )
        print(f"  Document {doc_id}: {count} chunks")

    return deduped


def expand_to_parent_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Expand retrieved child chunks to their parent-level context for LLM prompting.

    After retrieval and reranking on small child chunks, this helper replaces
    each chunk's page_content with the larger parent text stored in metadata.

    Deduplication is **page-level**: when multiple parent chunks belong to the
    same (document_id, page_no) they are merged into a single chunk whose
    page_content is the concatenation of all distinct parent texts for that
    page, ordered by parent_idx.  This preserves complete slide/page context
    and prevents the same page from appearing as separate fragments.

    Chunks without a parent_chunk_id (summaries, entity profiles) are passed
    through unchanged.

    Args:
        chunks: Reranked list of child chunk dicts (with page_content + metadata)

    Returns:
        Deduplicated list of page-level chunks ready for LLM context assembly
    """
    # Pass 1 — group parent texts by (document_id, page_no), dedup by parent_chunk_id
    # page_key → {best_chunk, parent_parts: {parent_idx: parent_text}, best_score}
    page_groups: Dict[tuple, dict] = {}
    seen_parent_ids: set = set()
    non_parent_chunks: List[Dict[str, Any]] = []

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        parent_id = meta.get("parent_chunk_id")
        parent_text = meta.get("parent_text")

        if not (parent_id and parent_text):
            # Summary / entity-profile / legacy chunk — use as-is
            non_parent_chunks.append(chunk)
            continue

        if parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)

        doc_id = meta.get("document_id", "")
        page_no = meta.get("page_no", 0)
        page_key = (doc_id, page_no)
        score = chunk.get("rerank_score", 0.0)
        parent_idx = meta.get("parent_chunk_id", "").rsplit("_p", 1)[-1]
        try:
            parent_idx = int(parent_idx)
        except (ValueError, TypeError):
            parent_idx = 0

        if page_key not in page_groups:
            page_groups[page_key] = {
                "best_chunk": chunk,
                "best_score": score,
                "parent_parts": {parent_idx: parent_text},
            }
        else:
            grp = page_groups[page_key]
            grp["parent_parts"][parent_idx] = parent_text
            if score > grp["best_score"]:
                grp["best_chunk"] = chunk
                grp["best_score"] = score

    # Pass 2 — build merged page-level chunks
    expanded: List[Dict[str, Any]] = []
    for page_key, grp in page_groups.items():
        merged_chunk = grp["best_chunk"].copy()
        # Concatenate parent texts in document order
        ordered_parts = [
            text for _, text in sorted(grp["parent_parts"].items())
        ]
        merged_chunk["page_content"] = "\n\n".join(ordered_parts)
        merged_chunk["rerank_score"] = grp["best_score"]
        expanded.append(merged_chunk)

    if any(len(g["parent_parts"]) > 1 for g in page_groups.values()):
        merged_count = sum(1 for g in page_groups.values() if len(g["parent_parts"]) > 1)
        print(f"[Parent Expand] Merged parent chunks on {merged_count} page(s) into page-level context")

    # Non-parent chunks go after parent-expanded ones
    expanded.extend(non_parent_chunks)
    return expanded
