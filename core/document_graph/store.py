"""
GraphStore — pluggable persistence layer for DocumentGraph.

Two implementations:

- KuzuGraphStore: embedded graph DB at data/{user_id}/graphs/{thread_id}.kuzu/.
  Default when `kuzu` is importable.

- NetworkXGraphStore: pickled NetworkX DiGraph at
  data/{user_id}/graphs/{thread_id}.nx.json. Fallback when Kuzu is unavailable
  (older Python builds, missing wheels).

`get_store()` picks Kuzu if available, NetworkX otherwise.  Both implementations
share the same write_graph / read_graph / exists / delete contract, so callers
never need to know which backend is in use.

Imports of `kuzu` and `networkx` are lazy so a missing dep only breaks this
module, not the rest of the app.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Optional, Protocol

from core.document_graph.models import DocumentGraph


def _graph_dir(user_id: str) -> str:
    return os.path.join("data", user_id, "graphs")


class GraphStore(Protocol):
    """Minimal interface every backend must satisfy."""

    backend: str

    def write_graph(self, graph: DocumentGraph) -> None: ...

    def read_graph(self, user_id: str, thread_id: str) -> Optional[DocumentGraph]: ...

    def exists(self, user_id: str, thread_id: str) -> bool: ...

    def delete(self, user_id: str, thread_id: str) -> None: ...


# ─── Kuzu backend ───────────────────────────────────────────────────────────


class KuzuGraphStore:
    backend = "kuzu"

    @staticmethod
    def _db_path(user_id: str, thread_id: str) -> str:
        d = _graph_dir(user_id)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{thread_id}.kuzu")

    @staticmethod
    def _meta_path(user_id: str, thread_id: str) -> str:
        return os.path.join(_graph_dir(user_id), f"{thread_id}.meta.json")

    @classmethod
    def _connect(cls, user_id: str, thread_id: str):
        import kuzu

        db_path = cls._db_path(user_id, thread_id)
        db = kuzu.Database(db_path)
        conn = kuzu.Connection(db)
        # Create schema if absent. Kuzu raises on duplicate table creation, so
        # swallow and continue — schema is idempotent in practice.
        # NOTE: Kuzu reserves 'profile' (used by Cypher's PROFILE statement) so we
        # store the entity bio under the column name `bio`. The JSON sidecar
        # still uses `profile` for frontend compatibility.
        ddl = [
            (
                "CREATE NODE TABLE IF NOT EXISTS Entity("
                "id STRING, label STRING, type STRING, aliases STRING[], "
                "frequency INT64, doc_count INT64, community_id INT64, "
                "bio STRING, provenance STRING, "
                "PRIMARY KEY (id))"
            ),
            (
                "CREATE NODE TABLE IF NOT EXISTS Community("
                "id INT64, name STRING, summary STRING, "
                "member_ids STRING[], size INT64, "
                "PRIMARY KEY (id))"
            ),
            (
                "CREATE REL TABLE IF NOT EXISTS RELATES_TO("
                "FROM Entity TO Entity, "
                "predicate STRING, confidence DOUBLE, extractor STRING, "
                "evidence STRING, provenance STRING)"
            ),
            (
                "CREATE REL TABLE IF NOT EXISTS MEMBER_OF("
                "FROM Entity TO Community)"
            ),
        ]
        for stmt in ddl:
            try:
                conn.execute(stmt)
            except Exception:
                # Older Kuzu versions may not support IF NOT EXISTS — try plain
                # CREATE and ignore "already exists" errors.
                try:
                    conn.execute(stmt.replace(" IF NOT EXISTS", ""))
                except Exception:
                    pass
        return db, conn

    @classmethod
    def write_graph(cls, graph: DocumentGraph) -> None:
        # Wipe any prior graph at this path so writes are deterministic.
        cls.delete(graph.metadata.user_id, graph.metadata.thread_id)
        db, conn = cls._connect(graph.metadata.user_id, graph.metadata.thread_id)

        for e in graph.nodes:
            try:
                conn.execute(
                    "CREATE (n:Entity {id: $id, label: $label, type: $type, "
                    "aliases: $aliases, frequency: $frequency, doc_count: $doc_count, "
                    "community_id: $community_id, bio: $bio, provenance: $provenance})",
                    {
                        "id": e.id,
                        "label": e.label,
                        "type": e.type,
                        "aliases": e.aliases,
                        "frequency": e.frequency,
                        "doc_count": e.doc_count,
                        "community_id": e.community_id if e.community_id is not None else -1,
                        "bio": e.profile,
                        "provenance": json.dumps([p.model_dump() for p in e.provenance]),
                    },
                )
            except Exception as ex:
                # One bad row shouldn't kill the whole graph — log and continue.
                # The JSON sidecar still gets written below so the frontend works.
                print(f"[DocumentGraph][kuzu] entity insert failed for {e.id}: {ex}")

        for c in graph.communities:
            conn.execute(
                "CREATE (c:Community {id: $id, name: $name, summary: $summary, "
                "member_ids: $member_ids, size: $size})",
                {
                    "id": c.id,
                    "name": c.name,
                    "summary": c.summary,
                    "member_ids": c.member_ids,
                    "size": c.size,
                },
            )
            for mid in c.member_ids:
                try:
                    conn.execute(
                        "MATCH (e:Entity {id: $eid}), (c:Community {id: $cid}) "
                        "CREATE (e)-[:MEMBER_OF]->(c)",
                        {"eid": mid, "cid": c.id},
                    )
                except Exception:
                    # Skip dangling member references silently — they were pruned upstream
                    pass

        for r in graph.edges:
            try:
                conn.execute(
                    "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                    "CREATE (a)-[:RELATES_TO {predicate: $predicate, confidence: $confidence, "
                    "extractor: $extractor, evidence: $evidence, provenance: $provenance}]->(b)",
                    {
                        "src": r.source_id,
                        "tgt": r.target_id,
                        "predicate": r.predicate,
                        "confidence": r.confidence,
                        "extractor": r.extractor,
                        "evidence": r.evidence,
                        "provenance": json.dumps(r.provenance.model_dump()),
                    },
                )
            except Exception:
                pass

        # Sidecar JSON for fast metadata reads + format-stable frontend payload.
        # Reading via Cypher and reconstructing on every GET is wasteful when the
        # graph is small enough to stream as JSON.
        cls._write_sidecar(graph)

    @classmethod
    def _write_sidecar(cls, graph: DocumentGraph) -> None:
        path = cls._meta_path(graph.metadata.user_id, graph.metadata.thread_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(graph.model_dump(mode="json"), f, ensure_ascii=False)

    @classmethod
    def read_graph(cls, user_id: str, thread_id: str) -> Optional[DocumentGraph]:
        path = cls._meta_path(user_id, thread_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DocumentGraph.model_validate(data)

    @classmethod
    def exists(cls, user_id: str, thread_id: str) -> bool:
        return os.path.exists(cls._meta_path(user_id, thread_id))

    @classmethod
    def delete(cls, user_id: str, thread_id: str) -> None:
        db_path = cls._db_path(user_id, thread_id)
        if os.path.isdir(db_path):
            shutil.rmtree(db_path, ignore_errors=True)
        meta = cls._meta_path(user_id, thread_id)
        if os.path.exists(meta):
            try:
                os.remove(meta)
            except OSError:
                pass


# ─── NetworkX fallback ──────────────────────────────────────────────────────


class NetworkXGraphStore:
    backend = "networkx"

    @staticmethod
    def _path(user_id: str, thread_id: str) -> str:
        d = _graph_dir(user_id)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{thread_id}.nx.json")

    @classmethod
    def write_graph(cls, graph: DocumentGraph) -> None:
        with open(cls._path(graph.metadata.user_id, graph.metadata.thread_id), "w", encoding="utf-8") as f:
            json.dump(graph.model_dump(mode="json"), f, ensure_ascii=False)

    @classmethod
    def read_graph(cls, user_id: str, thread_id: str) -> Optional[DocumentGraph]:
        p = cls._path(user_id, thread_id)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DocumentGraph.model_validate(data)

    @classmethod
    def exists(cls, user_id: str, thread_id: str) -> bool:
        return os.path.exists(cls._path(user_id, thread_id))

    @classmethod
    def delete(cls, user_id: str, thread_id: str) -> None:
        p = cls._path(user_id, thread_id)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


# ─── Selector ───────────────────────────────────────────────────────────────


_BACKEND_CACHE: Optional[str] = None


def get_store() -> GraphStore:
    """
    Return the active backend. Kuzu wins if importable, else NetworkX.
    Result is cached so we don't re-attempt the import on every call.
    """
    global _BACKEND_CACHE
    if _BACKEND_CACHE == "kuzu":
        return KuzuGraphStore  # type: ignore[return-value]
    if _BACKEND_CACHE == "networkx":
        return NetworkXGraphStore  # type: ignore[return-value]

    try:
        import kuzu  # noqa: F401

        _BACKEND_CACHE = "kuzu"
        print("[DocumentGraph] Using Kuzu backend")
        return KuzuGraphStore  # type: ignore[return-value]
    except Exception as e:
        _BACKEND_CACHE = "networkx"
        print(f"[DocumentGraph] Kuzu unavailable ({e}); falling back to NetworkX-on-disk")
        return NetworkXGraphStore  # type: ignore[return-value]
