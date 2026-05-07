"""
Phase 7 + 8: community detection and naming.

Phase 7 — Leiden community detection over the entity graph using python-igraph.
If python-igraph is unavailable we fall back to NetworkX connected components,
which is coarser but always works.

Phase 8 — batched LLM call to name + summarize each community.
"""

import asyncio
from collections import defaultdict
from typing import Dict, List, Tuple

from core.constants import (
    GPU_GRAPH_COMMUNITY_LLM,
    GRAPH_COMMUNITY_BATCH_SIZE,
)
from core.document_graph.models import Community, GraphBuildContext
from core.llm.client import invoke_llm
from core.llm.output_schemas.document_graph_outputs import CommunityNamingBatch
from core.llm.prompts.document_graph_prompts import build_community_naming_prompt


def _detect_with_igraph(ctx: GraphBuildContext) -> List[List[str]]:
    """Leiden via python-igraph. Returns list of clusters (each a list of entity ids)."""
    import igraph as ig

    id_list = list(ctx.entities.keys())
    idx_of = {eid: i for i, eid in enumerate(id_list)}

    edges_seen = set()
    edge_idx = []
    weights = []
    for r in ctx.relations:
        if r.source_id not in idx_of or r.target_id not in idx_of:
            continue
        a, b = idx_of[r.source_id], idx_of[r.target_id]
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in edges_seen:
            continue
        edges_seen.add(key)
        edge_idx.append(key)
        weights.append(max(r.confidence, 0.1))

    g = ig.Graph(n=len(id_list), edges=edge_idx, directed=False)
    g.es["weight"] = weights

    # Leiden modularity. Resolution=1.0 gives default-tightness clusters.
    partition = g.community_leiden(
        objective_function="modularity", weights="weight", n_iterations=4
    )
    clusters: List[List[str]] = [[] for _ in range(len(partition))]
    for i, cid in enumerate(partition.membership):
        clusters[cid].append(id_list[i])
    # Filter out empty
    return [c for c in clusters if c]


def _detect_with_networkx(ctx: GraphBuildContext) -> List[List[str]]:
    """Fallback: connected components on an undirected projection."""
    import networkx as nx

    g = nx.Graph()
    for eid in ctx.entities:
        g.add_node(eid)
    for r in ctx.relations:
        if r.source_id in g and r.target_id in g and r.source_id != r.target_id:
            g.add_edge(r.source_id, r.target_id, weight=max(r.confidence, 0.1))
    return [list(c) for c in nx.connected_components(g)]


async def run_detect(ctx: GraphBuildContext) -> None:
    if not ctx.entities:
        ctx.communities = []
        return

    def _do() -> List[List[str]]:
        try:
            return _detect_with_igraph(ctx)
        except Exception as e:
            print(f"[DocGraph][communities] igraph unavailable ({e}); using NetworkX components")
            try:
                return _detect_with_networkx(ctx)
            except Exception as e2:
                print(f"[DocGraph][communities] NetworkX also failed ({e2}); single cluster fallback")
                return [list(ctx.entities.keys())]

    clusters = await asyncio.to_thread(_do)

    # Sort clusters by size desc, then assign stable ids 0..N
    clusters.sort(key=len, reverse=True)
    communities: List[Community] = []
    for idx, members in enumerate(clusters):
        for mid in members:
            ent = ctx.entities.get(mid)
            if ent:
                ent.community_id = idx
        communities.append(
            Community(id=idx, name="", summary="", member_ids=members, size=len(members))
        )

    ctx.communities = communities
    print(f"[DocGraph][communities] detected {len(communities)} clusters")


def _top_relations_for(ctx: GraphBuildContext, member_ids: List[str], cap: int = 6) -> List[str]:
    members = set(member_ids)
    out = []
    for r in ctx.relations:
        if r.source_id in members and r.target_id in members:
            s = ctx.entities.get(r.source_id)
            t = ctx.entities.get(r.target_id)
            if s and t:
                out.append(f"{s.label} {r.predicate} {t.label}")
        if len(out) >= cap:
            break
    return out


async def run_name(ctx: GraphBuildContext) -> None:
    """Phase 8 — LLM names + summaries. Skips singleton clusters."""
    if not ctx.communities:
        return

    nameable = [c for c in ctx.communities if c.size >= 3]
    if not nameable:
        # Give singletons a default label so the UI isn't empty
        for c in ctx.communities:
            members_labels = [ctx.entities[m].label for m in c.member_ids if m in ctx.entities]
            c.name = members_labels[0] if members_labels else f"Cluster {c.id}"
            c.summary = ""
        return

    batches = [
        nameable[i : i + GRAPH_COMMUNITY_BATCH_SIZE]
        for i in range(0, len(nameable), GRAPH_COMMUNITY_BATCH_SIZE)
    ]

    by_id: Dict[int, Community] = {c.id: c for c in ctx.communities}

    for batch in batches:
        payload = []
        for c in batch:
            labels = [ctx.entities[m].label for m in c.member_ids if m in ctx.entities]
            payload.append(
                {
                    "id": c.id,
                    "members": labels,
                    "top_relations": _top_relations_for(ctx, c.member_ids),
                }
            )
        prompt = build_community_naming_prompt(payload)
        try:
            resp: CommunityNamingBatch = await invoke_llm(
                response_schema=CommunityNamingBatch,
                contents=prompt,
                gpu_model=GPU_GRAPH_COMMUNITY_LLM.model,
                port=GPU_GRAPH_COMMUNITY_LLM.port,
            )
            for cn in resp.communities:
                target = by_id.get(cn.community_id)
                if target:
                    target.name = cn.name[:80]
                    target.summary = cn.summary[:400]
        except Exception as e:
            print(f"[DocGraph][communities] naming batch failed: {e}")

    # Default-fill any community the LLM didn't name
    for c in ctx.communities:
        if not c.name:
            members_labels = [
                ctx.entities[m].label for m in c.member_ids if m in ctx.entities
            ]
            c.name = members_labels[0] if members_labels else f"Cluster {c.id}"
