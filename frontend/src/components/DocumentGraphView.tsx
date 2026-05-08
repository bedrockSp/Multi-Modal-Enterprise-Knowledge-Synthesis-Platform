/**
 * DocumentGraphView — Sigma.js + Graphology visualization of a thread's
 * knowledge graph. Renders entities as nodes coloured by community, edges
 * as relations, with click-through to provenance and a search box for
 * jump-to-entity.
 *
 * Sigma is initialized via @react-sigma/core; we build the Graphology
 * graph in a useMemo and pass it through SigmaContainer.
 */

import { SigmaContainer, useLoadGraph, useRegisterEvents, useSigma } from '@react-sigma/core';
import '@react-sigma/core/lib/react-sigma.min.css';
import Graph from 'graphology';
import forceAtlas2 from 'graphology-layout-forceatlas2';
import { useEffect, useMemo, useState } from 'react';

import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import type { DocumentGraph, GraphCommunity, GraphEntity, GraphRelation } from '@/lib/api';

// Distinct, perceptually different colours for community ids 0..15.
// Beyond 16 communities we cycle — that's deliberate and good enough for
// the typical thread-scoped graph (rarely exceeds a dozen communities).
const COMMUNITY_PALETTE = [
  '#4f46e5', '#0891b2', '#16a34a', '#ca8a04',
  '#dc2626', '#db2777', '#7c3aed', '#0d9488',
  '#65a30d', '#ea580c', '#be123c', '#9333ea',
  '#0369a1', '#15803d', '#a16207', '#6b21a8',
];

const colorFor = (communityId: number | null): string => {
  if (communityId === null || communityId < 0) return '#6b7280';
  return COMMUNITY_PALETTE[communityId % COMMUNITY_PALETTE.length];
};

interface DocumentGraphViewProps {
  graph: DocumentGraph;
  onSelectEntity?: (entity: GraphEntity) => void;
  onSelectEdge?: (edge: GraphRelation) => void;
}

const buildGraphology = (graph: DocumentGraph): Graph => {
  const g = new Graph({ multi: true, type: 'directed' });

  graph.nodes.forEach((entity) => {
    // Node size scales with frequency, log-bounded so high-frequency entities
    // don't dwarf the rest.
    const size = Math.max(4, Math.min(22, 4 + Math.log2(entity.frequency + 1) * 3));
    g.addNode(entity.id, {
      label: entity.label,
      size,
      color: colorFor(entity.community_id),
      // Random initial position; ForceAtlas2 will lay things out.
      x: Math.random(),
      y: Math.random(),
      entity,
    });
  });

  graph.edges.forEach((edge, i) => {
    if (!g.hasNode(edge.source_id) || !g.hasNode(edge.target_id)) return;
    g.addEdgeWithKey(`e${i}`, edge.source_id, edge.target_id, {
      label: edge.predicate,
      size: Math.max(0.5, edge.confidence * 1.6),
      color: '#cbd5e1',
      type: 'arrow',
      edge,
    });
  });

  // Run ForceAtlas2 to give the graph an initial layout. Iteration count
  // scales with node count but is capped — we'd rather have a slightly
  // imperfect layout than freeze the UI on a 5k-node graph.
  const iterations = Math.min(200, 30 + Math.round(g.order / 10));
  forceAtlas2.assign(g, { iterations, settings: { gravity: 1, scalingRatio: 8 } });
  return g;
};

const GraphLoader = ({ graph }: { graph: DocumentGraph }) => {
  const loadGraph = useLoadGraph();
  const built = useMemo(() => buildGraphology(graph), [graph]);
  useEffect(() => {
    loadGraph(built);
  }, [built, loadGraph]);
  return null;
};

const GraphEvents = ({ onSelectEntity, onSelectEdge }: {
  onSelectEntity?: (e: GraphEntity) => void;
  onSelectEdge?: (e: GraphRelation) => void;
}) => {
  const sigma = useSigma();
  const registerEvents = useRegisterEvents();

  useEffect(() => {
    registerEvents({
      clickNode: ({ node }) => {
        const attrs = sigma.getGraph().getNodeAttributes(node);
        if (attrs.entity && onSelectEntity) onSelectEntity(attrs.entity as GraphEntity);
      },
      clickEdge: ({ edge }) => {
        const attrs = sigma.getGraph().getEdgeAttributes(edge);
        if (attrs.edge && onSelectEdge) onSelectEdge(attrs.edge as GraphRelation);
      },
      enterNode: ({ node }) => {
        sigma.getGraph().setNodeAttribute(node, 'highlighted', true);
        sigma.refresh();
      },
      leaveNode: ({ node }) => {
        sigma.getGraph().removeNodeAttribute(node, 'highlighted');
        sigma.refresh();
      },
    });
  }, [registerEvents, sigma, onSelectEntity, onSelectEdge]);
  return null;
};

const SearchBox = ({ nodes }: { nodes: GraphEntity[] }) => {
  const sigma = useSigma();
  const [q, setQ] = useState('');
  const matches = useMemo(() => {
    if (!q.trim()) return [];
    const needle = q.toLowerCase();
    return nodes
      .filter((n) => n.label.toLowerCase().includes(needle))
      .slice(0, 8);
  }, [q, nodes]);

  const goto = (id: string) => {
    const g = sigma.getGraph();
    if (!g.hasNode(id)) return;
    const attrs = g.getNodeAttributes(id);
    sigma.getCamera().animate({ x: attrs.x, y: attrs.y, ratio: 0.3 }, { duration: 600 });
    setQ('');
  };

  return (
    <div className="absolute top-3 left-3 z-10 w-72">
      <Input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Search entity…"
        className="bg-background/90 backdrop-blur"
      />
      {matches.length > 0 && (
        <div className="mt-1 rounded-md border bg-popover shadow-md">
          {matches.map((m) => (
            <button
              key={m.id}
              onClick={() => goto(m.id)}
              className="block w-full px-3 py-1.5 text-left text-sm hover:bg-accent"
            >
              {m.label}
              <span className="ml-2 text-xs text-muted-foreground">{m.type}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

const CommunityLegend = ({ communities }: { communities: GraphCommunity[] }) => {
  if (communities.length === 0) return null;
  // Show top 12 by size; rest are summarized as "…and N more"
  const top = [...communities].sort((a, b) => b.size - a.size).slice(0, 12);
  const rest = communities.length - top.length;
  return (
    <div className="absolute top-3 right-3 z-10 w-72 rounded-md border bg-background/90 backdrop-blur p-3">
      <div className="text-xs font-semibold text-muted-foreground mb-2">Communities</div>
      <ScrollArea className="max-h-72">
        <ul className="space-y-1.5 pr-2">
          {top.map((c) => (
            <li key={c.id} className="flex items-start gap-2 text-sm">
              <span
                className="mt-1 h-3 w-3 shrink-0 rounded-sm"
                style={{ background: colorFor(c.id) }}
              />
              <div className="min-w-0 flex-1">
                <div className="truncate font-medium">{c.name || `Cluster ${c.id}`}</div>
                {c.summary && (
                  <div className="text-xs text-muted-foreground line-clamp-2">{c.summary}</div>
                )}
                <div className="text-xs text-muted-foreground">{c.size} entities</div>
              </div>
            </li>
          ))}
        </ul>
      </ScrollArea>
      {rest > 0 && <div className="text-xs text-muted-foreground mt-2">…and {rest} more</div>}
    </div>
  );
};

const DocumentGraphView = ({ graph, onSelectEntity, onSelectEdge }: DocumentGraphViewProps) => {
  return (
    <div className="relative h-full w-full">
      <SigmaContainer
        style={{ height: '100%', width: '100%', background: 'transparent' }}
        settings={{
          renderEdgeLabels: false,
          defaultEdgeType: 'arrow',
          labelDensity: 0.07,
          labelGridCellSize: 60,
          labelRenderedSizeThreshold: 8,
          minCameraRatio: 0.05,
          maxCameraRatio: 4,
          // Sigma throws if it mounts before the parent has a resolved pixel
          // height. The page now uses h-screen so this should not fire, but
          // keep it on as a guard against transient first-render races.
          allowInvalidContainer: true,
        }}
      >
        <GraphLoader graph={graph} />
        <GraphEvents onSelectEntity={onSelectEntity} onSelectEdge={onSelectEdge} />
        <SearchBox nodes={graph.nodes} />
      </SigmaContainer>
      <div className="absolute bottom-3 left-3 z-10 flex gap-1.5 pointer-events-none">
        <Badge variant="secondary">{graph.nodes.length} nodes</Badge>
        <Badge variant="secondary">{graph.edges.length} edges</Badge>
        <Badge variant="secondary">{graph.communities.length} clusters</Badge>
      </div>
      <CommunityLegend communities={graph.communities} />
    </div>
  );
};

export default DocumentGraphView;
