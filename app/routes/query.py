import asyncio
from datetime import datetime, timezone
import json
import time
from typing import Literal

from fastapi import APIRouter, Request
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from agent.builder import Agent, AgentState
from agent.combination import combination_node
from agent.decomposition import decomposition_node
from agent.tools.search import search_tavily as search_tool
from agent.tools.sql_query import get_sql_schema
from core.constants import EXTERNAL, GPU_QUERY_LLM, GPU_QUERY_LLM2, INTERNAL, SWITCHES
from core.database import db
from core.llm.outputs import DecompositionLLMOutput
from core.services.sqlite_manager import SQLiteManager
from core.llm.prompts.main_prompt import detect_answer_style
from core.utils.extra_done_check import is_extra_done

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    thread_id: str
    question: str
    mode: Literal[f"{INTERNAL}", f"{EXTERNAL}"] = EXTERNAL
    use_self_knowledge: bool = False
    use_context: bool = False


@router.post("/")
async def query(request: Request, body: QueryRequest):
    payload = request.state.user

    if not payload:
        return {"error": "User not authenticated"}

    thread_id = body.thread_id
    question = body.question
    mode = body.mode
    use_self_knowledge = body.use_self_knowledge
    use_context = body.use_context

    print(
        f"Received query for thread_id: {thread_id} with question: {question} and mode: {mode} (use_self_knowledge={use_self_knowledge}, use_context={use_context})"
    )

    user_id = payload.userId
    user = db.users.find_one({"userId": user_id}, {"_id": 0, "password": 0})
    if not user:
        return {"error": "User not found"}

    thread = user["threads"].get(thread_id)
    if not thread:
        return {"error": "Thread not found"}

    # Collect selected thread instructions
    thread_instructions = [
        ins["text"]
        for ins in thread.get("instructions", [])
        if ins.get("selected", False)
    ]

    messages = []
    if use_context:
        # Build messages from previous chat history in the thread
        for chat in thread.get("chats", []):
            if chat.get("type") == "user":
                messages.append(HumanMessage(content=chat.get("content", "")))
            elif chat.get("type") == "agent":
                messages.append(AIMessage(content=chat.get("content", "")))

    # Cap thread history threaded into the agent so follow-up grounding works
    # (main_prompt and other nodes consume state.messages) without inflating
    # every LLM call with the entire conversation history.
    AGENT_HISTORY_LIMIT = 6  # last 3 user/agent turns
    agent_history = messages[-AGENT_HISTORY_LIMIT:] if messages else []

    def _resolve_prior_chat_context(decomp_result, history):
        """
        Layer 1 + 3 chat-context refactor: include the prior user+assistant
        turn ONLY when decomposition judged the user is asking about the
        prior assistant message (not just the prior topic). Always honours
        the user-facing use_context override — if that is off, we never
        include prior context regardless of the classifier.
        """
        if not use_context or not history:
            return []
        mode = getattr(decomp_result, "prior_answer_mode", "none")
        if mode not in ("reasoning", "reformat"):
            return []
        return list(history[-2:])

    def _resolve_prior_answer_mode(decomp_result):
        if not use_context:
            return "none"
        mode = getattr(decomp_result, "prior_answer_mode", "none")
        return mode if mode in ("reasoning", "reformat") else "none"
    chunks = []
    chunks_used = []
    confidence_scores = []

    # Reload spreadsheet data if needed (handling server restarts/older chats)
    try:
        # Check if thread has documents that might be spreadsheets
        thread_docs = thread.get("documents", [])
        spreadsheet_files = []
        for doc in thread_docs:
            # Check file extension/type
            fname = doc.get("file_name", "").lower()
            if (
                fname.endswith(".xlsx")
                or fname.endswith(".xls")
                or fname.endswith(".csv")
            ):
                # Construct path based on upload logic: data/{user_id}/threads/{thread_id}/uploads/{file_name}
                file_path = (
                    f"data/{user_id}/threads/{thread_id}/uploads/{doc['file_name']}"
                )
                spreadsheet_files.append(
                    {
                        "path": file_path,
                        "file_name": doc["file_name"],
                        "doc_id": doc.get("docId"),  # Use docId from DB
                    }
                )

        if spreadsheet_files:
            SQLiteManager.reload_from_files(user_id, thread_id, spreadsheet_files)
    except Exception as e:
        print(f"Error reloading spreadsheets: {e}")

    # Check if spreadsheet data is available for this thread
    has_spreadsheet = SQLiteManager.has_spreadsheet_data(user_id, thread_id)
    spreadsheet_schema = None
    if has_spreadsheet:
        spreadsheet_schema = get_sql_schema(user_id, thread_id)
        print(f"[SQL] Spreadsheet data available for thread {thread_id}")

    # Determine if the thread contains ONLY spreadsheet documents (no PDFs, PPTX, etc.)
    # When True, the retriever will skip RAG to avoid wasted latency and LLM confusion
    thread_docs = thread.get("documents", [])
    spreadsheet_extensions = {".xlsx", ".xls", ".csv"}
    spreadsheet_only = has_spreadsheet and len(thread_docs) > 0 and all(
        any(doc.get("file_name", "").lower().endswith(ext) for ext in spreadsheet_extensions)
        for doc in thread_docs
    )

    ds = time.time()
    if SWITCHES["DECOMPOSITION"]:
        decomposition_result: DecompositionLLMOutput = await decomposition_node(
            question,
            messages,
            has_spreadsheet_data=has_spreadsheet,
            spreadsheet_schema=spreadsheet_schema,
        )
    else:
        decomposition_result = DecompositionLLMOutput(
            requires_decomposition=False, resolved_query=question, sub_queries=[]
        )

    de = time.time() - ds
    print(f"Rewrite query time: {de:.2f} seconds")
    decomposed = decomposition_result.requires_decomposition
    all_favicons = []
    start_time = time.time()
    if decomposed:
        can_use_second_model = is_extra_done(user_id, thread_id)
        print("Query to be decomposed")
        print("No of sub-queries:", len(decomposition_result.sub_queries))

        async def run_worker(model, task_queue, results):
            while True:
                try:
                    idx, query_data = task_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                qs = time.time()
                state = await Agent.ainvoke(
                    AgentState(
                        user_id=user_id,
                        thread_id=thread_id,
                        query=query_data["query"],
                        resolved_query=decomposition_result.resolved_query,
                        original_query=question,
                        messages=list(agent_history),
                        web_search=False,
                        llm=model,
                        initial_search_answer=query_data["answer"] or "",
                        initial_search_results=query_data["results"] or [],
                        mode=mode,
                        use_self_knowledge=use_self_knowledge,
                        has_spreadsheet_data=has_spreadsheet,
                        spreadsheet_only=spreadsheet_only,
                        spreadsheet_schema=spreadsheet_schema,
                        thread_instructions=thread_instructions,
                        requires_full_data=getattr(decomposition_result, "requires_full_data", False),
                        requires_retrieval_expansion=getattr(decomposition_result, "requires_retrieval_expansion", False),
                        retrieval_queries=getattr(decomposition_result, "retrieval_queries", []),
                        prior_answer_mode=_resolve_prior_answer_mode(decomposition_result),
                        prior_chat_context=_resolve_prior_chat_context(decomposition_result, agent_history),
                    )
                )

                state = AgentState(**state)

                if getattr(state, "web_search_results", None):
                    for res in state.web_search_results:
                        favicons = [
                            r.get("favicon") for r in res["results"] if r.get("favicon")
                        ]
                        all_favicons.extend(favicons)

                qe = time.time() - qs
                chunks.extend(state.chunks)
                chunks_used.extend(state.chunks_used)
                if state.confidence_score:
                    confidence_scores.append(state.confidence_score)
                print(
                    f"Sub-query '{idx}. {query_data['query']}' processed in {qe:.2f} seconds using {model}"
                )

                results[idx] = {
                    "sub_query": query_data["query"],
                    "sub_answer": state.answer,
                    "excel_result": getattr(state, "excel_result", None),
                }

        # Prepare a queue of sub-queries
        if mode == EXTERNAL:
            search_results = await asyncio.gather(
                *(
                    search_tool(sub_query)
                    for sub_query in decomposition_result.sub_queries
                )
            )

            cleaned_results = []

            for idx, sub_query in enumerate(decomposition_result.sub_queries):
                if idx < len(search_results) and search_results[idx]:
                    res = search_results[idx]

                    favicons = [
                        {
                            "favicon": r.get("favicon", None),
                            "url": r.get("url", None),
                            "title": r.get("title", None),
                        }
                        for r in res.get("results", [])
                    ]
                    all_favicons.extend(favicons)

                    # Strip unwanted keys
                    for r in res.get("results", []):
                        r.pop("raw_content", None)
                        r.pop("score", None)
                        r.pop("favicon", None)

                    cleaned_results.append(
                        {
                            "query": res.get("query", sub_query),
                            "answer": res.get("answer", None),
                            "results": res.get("results", None),
                        }
                    )
                else:
                    # No search result → keep subquery with None values
                    cleaned_results.append(
                        {
                            "query": sub_query,
                            "answer": None,
                            "results": None,
                        }
                    )

        else:
            cleaned_results = [
                {
                    "query": sub_query,
                    "answer": None,
                    "results": None,
                }
                for sub_query in decomposition_result.sub_queries
            ]

        task_queue = asyncio.Queue()
        for idx, query_data in enumerate(cleaned_results):
            task_queue.put_nowait((idx, query_data))

        # Results stored in index order
        results = [None] * len(cleaned_results)

        # Start with the first model
        workers = [asyncio.create_task(run_worker(GPU_QUERY_LLM, task_queue, results))]

        # Add the second model only if allowed
        if (
            len(thread.get("documents", [])) == 0
            or not SWITCHES["MIND_MAP"]
            or can_use_second_model
        ):
            print("Using second model for parallel execution")
            workers.append(
                asyncio.create_task(run_worker(GPU_QUERY_LLM2, task_queue, results))
            )
        else:
            print("Second model disabled, running only on first model")

        # if can_use_second_model:
        #     print("Using second model for parallel execution")
        #     workers.append(asyncio.create_task(run_worker(GPU_QUERY_LLM2, task_queue, results)))
        # else:
        #     print("Second model disabled, running only on first model")

        await asyncio.gather(*workers)

        # Deduplicate chunks across sub-queries (same content from different
        # sub-queries wastes LLM context). Keep highest rerank_score per chunk.
        seen_chunk_keys = {}
        for c in chunks:
            key = (c.get("document_id", ""), c.get("page_no", 0), c.get("content", "")[:100])
            existing_score = seen_chunk_keys.get(key, {}).get("rerank_score", -1)
            if c.get("rerank_score", 0.0) > existing_score:
                seen_chunk_keys[key] = c
        deduped_chunks = sorted(
            seen_chunk_keys.values(),
            key=lambda c: c.get("rerank_score", 0.0),
            reverse=True,
        )
        if len(deduped_chunks) < len(chunks):
            print(f"[Combination] Deduplicated chunks: {len(chunks)} → {len(deduped_chunks)}")
        chunks = deduped_chunks

        # Step 8 (rag-refactor): pull community summaries from the DocumentGraph
        # if one exists for this thread. self-skips on focused answer classes,
        # missing graph, or zero-overlap with the query.
        community_records: list = []
        if SWITCHES.get("COMMUNITY_SEARCH", True):
            try:
                from agent.synthesis.community_search import community_search

                records = await community_search(
                    user_id=user_id,
                    thread_id=thread_id,
                    query=question,
                    answer_class=getattr(decomposition_result, "answer_class", "narrative"),
                )
                community_records = [r.model_dump() for r in records]
            except Exception as e:
                print(f"[Community Search] failed: {e}")

        cs = time.time()
        is_generative = detect_answer_style(question) == "generative"
        answer = await combination_node(
            results, decomposition_result.resolved_query, question,
            chunks=chunks,
            generative_mode=is_generative,
            answer_class=getattr(decomposition_result, "answer_class", "narrative"),
            community_records=community_records,
        )
        ce = time.time() - cs
        print(f"Subqueries combination time: {ce:.2f} seconds (generative={is_generative})")

        # Append Excel download links that may have been generated by sub-queries.
        # The combination LLM only synthesizes text — it drops download URLs.
        excel_links = []
        for r in results:
            if r and r.get("excel_result"):
                excel_links.append(r["excel_result"])
        if excel_links:
            links_md = "\n".join(
                f"- [Download Excel File]({url})" for url in excel_links
            )
            answer += f"\n\n---\n\n**Generated Files:**\n{links_md}"
    else:
        print("Query not being decomposed")

        if mode == EXTERNAL:
            search_result = await search_tool(
                decomposition_result.resolved_query or question
            )
        else:
            search_result = {}

        all_favicons.extend(
            [
                {
                    "favicon": r.get("favicon", None),
                    "url": r.get("url", None),
                    "title": r.get("title", None),
                }
                for r in search_result.get("results", [])
            ]
        )

        for r in search_result.get("results", []):
            r.pop("raw_content", None)
            r.pop("score", None)
            r.pop("favicon", None)

        resolved_query = (
            getattr(decomposition_result, "resolved_query", None) or question
        )
        state = await Agent.ainvoke(
            AgentState(
                user_id=user_id,
                thread_id=thread_id,
                query=resolved_query,
                resolved_query=resolved_query,
                original_query=question,
                messages=list(agent_history),
                web_search=False,
                llm=GPU_QUERY_LLM,
                mode=mode,
                initial_search_answer=search_result.get("answer", ""),
                initial_search_results=search_result.get("results", []),
                use_self_knowledge=use_self_knowledge,
                has_spreadsheet_data=has_spreadsheet,
                spreadsheet_only=spreadsheet_only,
                spreadsheet_schema=spreadsheet_schema,
                thread_instructions=thread_instructions,
                requires_full_data=getattr(decomposition_result, "requires_full_data", False),
                requires_retrieval_expansion=getattr(decomposition_result, "requires_retrieval_expansion", False),
                retrieval_queries=getattr(decomposition_result, "retrieval_queries", []),
            )
        )

        state = AgentState(**state)
        if getattr(state, "web_search_results", None):
            for res in state.web_search_results:
                favicons = [
                    {
                        "favicon": r.get("favicon", None),
                        "url": r.get("url", None),
                        "title": r.get("title", None),
                    }
                    for r in res["results"]
                ]
                all_favicons.extend(favicons)

        answer = state.answer
        chunks.extend(state.chunks)
        chunks_used.extend(state.chunks_used)
        if state.confidence_score:
            confidence_scores.append(state.confidence_score)
    end_time = time.time()

    print(f"Total Agent response time: {end_time - start_time:.2f} seconds")

    documents_used = []
    if chunks_used:
        print(f"Processing {len(chunks_used)} citations...")

        for doc_i in chunks_used:
            for doc_j in chunks:
                if doc_i.document_id == doc_j.get(
                    "document_id"
                ) and doc_i.page_no == doc_j.get("page_no"):
                    documents_used.append(doc_j)
                    break

    modified_used = []
    for doc in documents_used:
        modified_used.append(
            {
                "title": doc.get("title", "Untitled Document"),
                "document_id": doc.get("document_id", "unknown"),
                "page_no": doc.get("page_no", 1),
                "content": doc.get("content", ""),
            }
        )

    print(f"Found {len(documents_used)} citation matches")

    with open("debug_agent_response.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "thread_id": thread_id,
                "user_id": user_id,
                "question": question,
                "answer": answer,
                "documents_used": documents_used,
                "all_favicons": all_favicons,
                "decomposed": decomposed,
                "decomposition_result": decomposition_result.dict(),
                "chunks": chunks,
                "chunks_used": [doc.dict() for doc in chunks_used],
                "modified_used": modified_used,
                "use_self_knowledge": use_self_knowledge,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )

    # Build the final response metadata early so we can persist it in MongoDB
    # Resolve final confidence score — worst-case across all sub-queries
    confidence_priority = {"low": 0, "medium": 1, "high": 2}
    final_confidence = "low"
    if confidence_scores:
        final_confidence = min(
            confidence_scores, key=lambda c: confidence_priority.get(c, 0)
        )

    # Update the thread with the new messages (including metadata for persistence)
    now = datetime.now(timezone.utc)
    new_messages = [
        {"type": "user", "content": question, "timestamp": now},
        {
            "type": "agent",
            "content": answer,
            "timestamp": now,
            "sources": {"documents_used": modified_used, "web_used": all_favicons},
            "confidence_score": final_confidence,
        },
    ]

    db.users.update_one(
        {"userId": user_id},
        {
            "$push": {f"threads.{thread_id}.chats": {"$each": new_messages}},
            "$set": {f"threads.{thread_id}.updatedAt": now},
        },
    )

    response = {
        "thread_id": thread_id,
        "user_id": user_id,
        "question": question,
        "answer": answer,
        "sources": {
            "documents_used": modified_used,
            "web_used": all_favicons,
        },
        "confidence_score": final_confidence,
        "use_self_knowledge": use_self_knowledge,
    }

    return response
