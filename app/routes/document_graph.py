"""
DocumentGraph API.

Endpoints:
  POST   /graph/build/{thread_id}    Kick off async graph construction
  GET    /graph/{thread_id}          Fetch the current graph for a thread
  GET    /graph/{thread_id}/status   Quick status probe (pending|building|ready|failed)
  GET    /graphs                     List graphs across all of the user's threads
  DELETE /graph/{thread_id}          Delete the graph + DB rows
"""

import asyncio
import datetime
import traceback
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from app.socket_handler import sio
from core.constants import SWITCHES
from core.database import db
from core.document_graph.pipeline import build_graph
from core.document_graph.store import get_store


router = APIRouter(prefix="/graph", tags=["DocumentGraph"])


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def _require_feature():
    if not SWITCHES.get("DOCUMENT_GRAPH", False):
        raise HTTPException(
            status_code=503, detail="DocumentGraph feature is disabled"
        )


def _user_id(request: Request) -> str:
    payload = getattr(request.state, "user", None)
    if not payload:
        raise HTTPException(status_code=401, detail="User not authenticated")
    return payload.userId


def _thread_exists(user_id: str, thread_id: str) -> bool:
    user = db.users.find_one({"userId": user_id}, {"threads": 1, "_id": 0})
    if not user:
        return False
    return thread_id in (user.get("threads", {}) or {})


def _set_meta(user_id: str, thread_id: str, fields: dict) -> None:
    fields["updated_at"] = _utc_now()
    db.document_graphs.update_one(
        {"user_id": user_id, "thread_id": thread_id},
        {"$set": fields, "$setOnInsert": {"created_at": _utc_now()}},
        upsert=True,
    )


def _read_meta(user_id: str, thread_id: str) -> Optional[dict]:
    return db.document_graphs.find_one(
        {"user_id": user_id, "thread_id": thread_id}, {"_id": 0}
    )


@router.post("/build/{thread_id}")
async def build_thread_graph(thread_id: str, request: Request):
    """Start async graph construction for the given thread."""
    _require_feature()
    user_id = _user_id(request)
    if not _thread_exists(user_id, thread_id):
        raise HTTPException(status_code=404, detail="Thread not found for the user")

    # If a build is already running for this thread, return its current status
    # rather than spawning a duplicate worker.
    existing = _read_meta(user_id, thread_id)
    if existing and existing.get("status") == "building":
        return {"status": "building", "thread_id": thread_id, "message": "Build already in progress"}

    _set_meta(
        user_id,
        thread_id,
        {
            "user_id": user_id,
            "thread_id": thread_id,
            "status": "building",
            "error": "",
        },
    )
    topic = f"{user_id}/{thread_id}/graph/progress"

    async def _progress(stage: str, pct: int, msg: str):
        await sio.emit(
            topic,
            {"stage": stage, "progress": pct, "message": msg},
        )

    async def _run():
        try:
            graph = await build_graph(user_id, thread_id, progress_cb=_progress)
            _set_meta(
                user_id,
                thread_id,
                {
                    "status": "ready",
                    "node_count": graph.metadata.node_count,
                    "edge_count": graph.metadata.edge_count,
                    "community_count": graph.metadata.community_count,
                    "doc_count": graph.metadata.doc_count,
                    "built_at": graph.metadata.built_at,
                    "error": "",
                },
            )
            await sio.emit(
                topic,
                {"stage": "complete", "progress": 100, "message": "Graph ready"},
            )
        except Exception:
            err = traceback.format_exc()
            print(f"[DocGraph][route] build failed: {err}")
            _set_meta(
                user_id,
                thread_id,
                {"status": "failed", "error": err[-2000:]},
            )
            await sio.emit(
                topic,
                {"stage": "error", "progress": 0, "message": "Build failed"},
            )

    asyncio.create_task(_run())
    return {"status": "started", "thread_id": thread_id}


@router.get("/{thread_id}/status")
async def get_status(thread_id: str, request: Request):
    _require_feature()
    user_id = _user_id(request)
    meta = _read_meta(user_id, thread_id)
    if not meta:
        return {"status": "absent", "thread_id": thread_id}
    # Strip mongo-specific datetime types so JSON is serializable
    if isinstance(meta.get("built_at"), datetime.datetime):
        meta["built_at"] = meta["built_at"].isoformat()
    if isinstance(meta.get("created_at"), datetime.datetime):
        meta["created_at"] = meta["created_at"].isoformat()
    if isinstance(meta.get("updated_at"), datetime.datetime):
        meta["updated_at"] = meta["updated_at"].isoformat()
    return meta


@router.get("/{thread_id}")
async def get_graph(thread_id: str, request: Request):
    _require_feature()
    user_id = _user_id(request)
    if not _thread_exists(user_id, thread_id):
        raise HTTPException(status_code=404, detail="Thread not found for the user")

    store = get_store()
    graph = store.read_graph(user_id, thread_id)
    if graph is None:
        raise HTTPException(
            status_code=404,
            detail="No graph for this thread yet. POST /graph/build/{thread_id} first.",
        )
    return graph.model_dump(mode="json")


@router.get("")
async def list_graphs(request: Request):
    """List all graphs across the user's threads (for the standalone picker)."""
    _require_feature()
    user_id = _user_id(request)
    cursor = db.document_graphs.find({"user_id": user_id}, {"_id": 0})

    user = db.users.find_one({"userId": user_id}, {"threads": 1, "_id": 0}) or {}
    thread_names = {
        tid: (t.get("thread_name") or "Untitled Thread")
        for tid, t in (user.get("threads", {}) or {}).items()
    }

    out = []
    for meta in cursor:
        for key in ("built_at", "created_at", "updated_at"):
            if isinstance(meta.get(key), datetime.datetime):
                meta[key] = meta[key].isoformat()
        meta["thread_name"] = thread_names.get(meta.get("thread_id", ""), "")
        out.append(meta)
    return {"graphs": out}


@router.delete("/{thread_id}")
async def delete_graph(thread_id: str, request: Request):
    _require_feature()
    user_id = _user_id(request)
    store = get_store()
    store.delete(user_id, thread_id)
    db.document_graphs.delete_one({"user_id": user_id, "thread_id": thread_id})
    return {"status": "deleted", "thread_id": thread_id}
