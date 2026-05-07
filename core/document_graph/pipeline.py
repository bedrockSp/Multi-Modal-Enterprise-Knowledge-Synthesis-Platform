"""
DocumentGraph pipeline orchestrator.

Runs the typed phase DAG end-to-end and emits Socket.IO progress events at
every stage transition. Mirrors the GitNexus phase orchestrator pattern: a
declared list of phases with input/output signature, executed in order, with
percentage progress derived from phase index.

Public entrypoint: `build_graph(user_id, thread_id, progress_cb)`.
"""

import time
import traceback
from typing import Awaitable, Callable, Optional

from core.document_graph.models import DocumentGraph, GraphBuildContext
from core.document_graph.phases import (
    collect as p_collect,
    communities as p_communities,
    dedupe as p_dedupe,
    extract as p_extract,
    harvest as p_harvest,
    persist as p_persist,
    prune as p_prune,
)


ProgressCB = Callable[[str, int, str], Awaitable[None]]


# Each entry: (stage_key, label, async_fn). stage_key is what frontend listens for.
_PHASES = [
    ("collect", "Loading chunks", p_collect.run),
    ("harvest_entities", "Harvesting entities", p_harvest.run_entities),
    ("harvest_triples", "Harvesting triples", p_harvest.run_triples),
    ("dedupe", "De-duplicating entities", p_dedupe.run),
    ("extract", "Extracting relations", p_extract.run),
    ("prune", "Pruning low-signal nodes", p_prune.run),
    ("detect_communities", "Detecting communities", p_communities.run_detect),
    ("name_communities", "Naming communities", p_communities.run_name),
    ("persist", "Saving graph", p_persist.run),
]


async def build_graph(
    user_id: str,
    thread_id: str,
    progress_cb: Optional[ProgressCB] = None,
) -> DocumentGraph:
    """
    Run the full pipeline. Returns the assembled DocumentGraph.
    Raises on unrecoverable failure (caller should mark MongoDB metadata 'failed').
    """
    ctx = GraphBuildContext(user_id=user_id, thread_id=thread_id)
    total = len(_PHASES)
    start = time.time()

    async def emit(stage: str, idx: int, msg: str):
        if progress_cb is None:
            return
        pct = int((idx / total) * 100)
        try:
            await progress_cb(stage, pct, msg)
        except Exception as e:
            print(f"[DocGraph][pipeline] progress_cb failed: {e}")

    await emit("start", 0, "Starting graph build")

    graph: Optional[DocumentGraph] = None
    for idx, (stage, label, fn) in enumerate(_PHASES):
        await emit(stage, idx, label)
        phase_start = time.time()
        try:
            result = await fn(ctx)
            if stage == "persist":
                graph = result  # persist.run returns the DocumentGraph
        except Exception:
            tb = traceback.format_exc()
            print(f"[DocGraph][pipeline] phase '{stage}' failed:\n{tb}")
            await emit("error", idx, f"Phase '{stage}' failed")
            raise
        print(f"[DocGraph][pipeline] phase '{stage}' done in {time.time() - phase_start:.2f}s")

    await emit("complete", total, f"Graph ready in {time.time() - start:.1f}s")

    if graph is None:
        # Persist phase didn't return a graph — synthesize from ctx as a defensive fallback
        from core.document_graph.phases.persist import run as persist_run

        graph = await persist_run(ctx)
    return graph
