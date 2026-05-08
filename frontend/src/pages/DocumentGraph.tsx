/**
 * DocumentGraph page — corpus-level knowledge graph for a thread.
 *
 * URL forms:
 *   /document-graph                — thread picker
 *   /document-graph/:threadId      — direct view of a single thread's graph
 *
 * Streams build progress over Socket.IO topic `${userId}/${threadId}/graph/progress`
 * and re-fetches the graph when the build completes.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { io, Socket } from 'socket.io-client';
import { Loader2, Network, RefreshCw, Trash2, FileText, AlertCircle } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { toast } from '@/components/ui/use-toast';

import AppNavbar from '@/components/AppNavbar';
import DocumentGraphView from '@/components/DocumentGraphView';
import { useAuth } from '@/lib/auth-context';
import {
  api,
  getAuthToken,
  type DocumentGraph as DocumentGraphData,
  type GraphEntity,
  type GraphRelation,
  type GraphStatus,
} from '@/lib/api';
import { API_URL } from '../../config';

type ProgressEvent = { stage: string; progress: number; message: string };

const formatTimestamp = (s: string | null | undefined) => {
  if (!s) return '—';
  try {
    return new Date(s).toLocaleString();
  } catch {
    return s;
  }
};

const ThreadPicker = ({
  threads,
  graphs,
  onPick,
}: {
  threads: { thread_id: string; thread_name: string; doc_count: number }[];
  graphs: Record<string, GraphStatus>;
  onPick: (id: string) => void;
}) => {
  if (threads.length === 0) {
    return (
      <Card className="max-w-2xl mx-auto mt-12">
        <CardContent className="py-10 text-center text-muted-foreground">
          You don't have any threads yet. Upload some documents from the Knowledge Forge
          dashboard, then come back to build a knowledge graph.
        </CardContent>
      </Card>
    );
  }
  return (
    <div className="max-w-3xl mx-auto mt-8 space-y-3">
      <div className="text-sm text-muted-foreground mb-2">
        Pick a thread to build or view its document graph.
      </div>
      {threads.map((t) => {
        const status = graphs[t.thread_id];
        return (
          <button
            key={t.thread_id}
            onClick={() => onPick(t.thread_id)}
            className="w-full text-left rounded-md border bg-card hover:bg-accent transition p-4 flex items-start gap-3"
          >
            <FileText className="w-5 h-5 mt-0.5 text-muted-foreground shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="font-medium truncate">{t.thread_name || 'Untitled thread'}</div>
              <div className="text-xs text-muted-foreground mt-0.5">
                {t.doc_count} document{t.doc_count === 1 ? '' : 's'}
              </div>
            </div>
            <div className="shrink-0">
              {status?.status === 'ready' ? (
                <Badge variant="secondary">{status.node_count ?? 0} nodes</Badge>
              ) : status?.status === 'building' ? (
                <Badge>building</Badge>
              ) : status?.status === 'failed' ? (
                <Badge variant="destructive">failed</Badge>
              ) : (
                <Badge variant="outline">no graph</Badge>
              )}
            </div>
          </button>
        );
      })}
    </div>
  );
};

const ProvenancePanel = ({
  entity,
  edge,
  onClose,
}: {
  entity: GraphEntity | null;
  edge: GraphRelation | null;
  onClose: () => void;
}) => {
  if (!entity && !edge) return null;
  return (
    <div className="absolute right-3 bottom-3 z-10 w-80 rounded-md border bg-background/95 backdrop-blur shadow-lg">
      <div className="flex items-center justify-between px-3 py-2 border-b">
        <div className="text-sm font-semibold">{entity ? 'Entity' : 'Relation'}</div>
        <button onClick={onClose} className="text-muted-foreground hover:text-foreground text-xs">
          close
        </button>
      </div>
      <ScrollArea className="max-h-72 p-3">
        {entity && (
          <div className="space-y-2 text-sm">
            <div className="font-medium text-base">{entity.label}</div>
            <div className="flex flex-wrap gap-1.5">
              <Badge variant="outline">{entity.type}</Badge>
              <Badge variant="secondary">freq {entity.frequency}</Badge>
              <Badge variant="secondary">{entity.doc_count} docs</Badge>
            </div>
            {entity.profile && (
              <p className="text-xs text-muted-foreground whitespace-pre-line">{entity.profile}</p>
            )}
            {entity.aliases.length > 0 && (
              <div className="text-xs text-muted-foreground">
                Aliases: {entity.aliases.slice(0, 6).join(', ')}
              </div>
            )}
            {entity.provenance.length > 0 && (
              <div className="pt-1">
                <div className="text-xs font-semibold mb-1">Mentions</div>
                <ul className="space-y-1">
                  {entity.provenance.slice(0, 6).map((p, i) => (
                    <li key={i} className="text-xs text-muted-foreground">
                      {p.title || p.file_name || p.document_id} · p. {p.page_no}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
        {edge && (
          <div className="space-y-2 text-sm">
            <div className="font-mono text-xs">
              {edge.source_id} <span className="text-primary">→</span> {edge.target_id}
            </div>
            <div className="font-medium">{edge.predicate}</div>
            <div className="flex flex-wrap gap-1.5">
              <Badge variant="outline">{edge.extractor}</Badge>
              <Badge variant="secondary">conf {edge.confidence.toFixed(2)}</Badge>
            </div>
            {edge.evidence && (
              <p className="text-xs text-muted-foreground italic">"{edge.evidence}"</p>
            )}
            {edge.provenance && (
              <div className="text-xs text-muted-foreground">
                {edge.provenance.title || edge.provenance.file_name || edge.provenance.document_id}
                {edge.provenance.page_no ? ` · p. ${edge.provenance.page_no}` : ''}
              </div>
            )}
          </div>
        )}
      </ScrollArea>
    </div>
  );
};

const DocumentGraphPage = () => {
  const { user } = useAuth();
  const { threadId: routeThreadId } = useParams<{ threadId?: string }>();
  const navigate = useNavigate();

  const [graph, setGraph] = useState<DocumentGraphData | null>(null);
  const [status, setStatus] = useState<GraphStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [building, setBuilding] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [graphList, setGraphList] = useState<Record<string, GraphStatus>>({});
  const [selectedEntity, setSelectedEntity] = useState<GraphEntity | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<GraphRelation | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const socketRef = useRef<Socket | null>(null);

  const threads = useMemo(() => {
    if (!user?.threads) return [];
    return Object.entries(user.threads)
      .map(([id, t]) => ({
        thread_id: id,
        thread_name: t.thread_name || 'Untitled',
        doc_count: t.documents?.length ?? 0,
      }))
      .sort((a, b) => a.thread_name.localeCompare(b.thread_name));
  }, [user]);

  const refreshGraphList = useCallback(async () => {
    try {
      const resp = await api.listDocumentGraphs();
      const map: Record<string, GraphStatus> = {};
      resp.graphs.forEach((g) => {
        map[g.thread_id] = g;
      });
      setGraphList(map);
    } catch (e) {
      // Don't toast — list is non-critical
      console.error('[DocumentGraph] list failed', e);
    }
  }, []);

  const loadGraph = useCallback(async (tid: string) => {
    setLoading(true);
    try {
      const data = await api.getDocumentGraph(tid);
      setGraph(data);
    } catch (e: any) {
      // 404 means no graph yet — that's fine, we just stay in "no graph" state
      setGraph(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshStatus = useCallback(async (tid: string) => {
    try {
      const s = await api.getDocumentGraphStatus(tid);
      setStatus(s);
      return s;
    } catch (e) {
      setStatus(null);
      return null;
    }
  }, []);

  const startBuild = useCallback(async () => {
    if (!routeThreadId) return;
    setBuilding(true);
    setProgress(0);
    setProgressMessage('Starting…');
    try {
      await api.buildDocumentGraph(routeThreadId);
      toast({ title: 'Graph build started' });
    } catch (e: any) {
      setBuilding(false);
      toast({ variant: 'destructive', title: 'Failed to start build', description: e?.message });
    }
  }, [routeThreadId]);

  const deleteGraph = useCallback(async () => {
    if (!routeThreadId) return;
    try {
      await api.deleteDocumentGraph(routeThreadId);
      setGraph(null);
      setStatus(null);
      toast({ title: 'Graph deleted' });
      refreshGraphList();
    } catch (e: any) {
      toast({ variant: 'destructive', title: 'Delete failed', description: e?.message });
    } finally {
      setConfirmDelete(false);
    }
  }, [routeThreadId, refreshGraphList]);

  // Socket.IO subscription scoped to the active thread
  useEffect(() => {
    if (!routeThreadId || !user?.userId) return;
    const token = getAuthToken();
    const sock = io(API_URL, { auth: { token } });
    socketRef.current = sock;
    const topic = `${user.userId}/${routeThreadId}/graph/progress`;

    sock.on(topic, (msg: ProgressEvent) => {
      setProgress(msg.progress ?? 0);
      setProgressMessage(msg.message || msg.stage);
      if (msg.stage === 'complete') {
        setBuilding(false);
        loadGraph(routeThreadId);
        refreshStatus(routeThreadId);
        refreshGraphList();
        toast({ title: 'Graph built' });
      } else if (msg.stage === 'error') {
        setBuilding(false);
        toast({ variant: 'destructive', title: 'Build failed', description: msg.message });
      }
    });

    return () => {
      sock.off(topic);
      sock.disconnect();
      socketRef.current = null;
    };
  }, [routeThreadId, user?.userId, loadGraph, refreshStatus, refreshGraphList]);

  // Load graph + status when route thread changes
  useEffect(() => {
    if (!routeThreadId) return;
    refreshStatus(routeThreadId).then((s) => {
      if (s?.status === 'ready') loadGraph(routeThreadId);
      if (s?.status === 'building') {
        setBuilding(true);
        setProgressMessage('Build in progress…');
      }
    });
  }, [routeThreadId, refreshStatus, loadGraph]);

  // Initial graph list load (for the picker)
  useEffect(() => {
    refreshGraphList();
  }, [refreshGraphList]);

  // ── Render branches ──

  if (!routeThreadId) {
    return (
      <div className="h-screen flex flex-col overflow-hidden">
        <AppNavbar />
        <div className="px-6 py-6 flex-1 overflow-auto">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-center gap-2 mb-2">
              <Network className="w-5 h-5" />
              <h1 className="text-xl font-semibold">Document Graph</h1>
            </div>
            <p className="text-sm text-muted-foreground mb-4">
              Build a corpus-level knowledge graph from any thread's documents — entities,
              relations, and topical clusters. Click a thread to view or build its graph.
            </p>
          </div>
          <ThreadPicker
            threads={threads}
            graphs={graphList}
            onPick={(id) => navigate(`/document-graph/${id}`)}
          />
        </div>
      </div>
    );
  }

  const threadName = user?.threads?.[routeThreadId]?.thread_name ?? routeThreadId;
  const hasGraph = graph && graph.nodes.length > 0;

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <AppNavbar />
      <div className="px-4 py-3 border-b flex items-center gap-3">
        <Network className="w-4 h-4 text-muted-foreground" />
        <div className="font-medium truncate">{threadName}</div>
        {status?.status && status.status !== 'absent' && (
          <Badge variant={status.status === 'ready' ? 'secondary' : status.status === 'failed' ? 'destructive' : 'default'}>
            {status.status}
          </Badge>
        )}
        {status?.built_at && (
          <span className="text-xs text-muted-foreground">built {formatTimestamp(status.built_at)}</span>
        )}
        <div className="flex-1" />
        <Button
          size="sm"
          onClick={startBuild}
          disabled={building}
        >
          {building ? <Loader2 className="w-4 h-4 mr-1.5 animate-spin" /> : <RefreshCw className="w-4 h-4 mr-1.5" />}
          {hasGraph ? 'Rebuild' : 'Build graph'}
        </Button>
        {hasGraph && (
          <Button size="sm" variant="ghost" onClick={() => setConfirmDelete(true)}>
            <Trash2 className="w-4 h-4" />
          </Button>
        )}
        <Button size="sm" variant="ghost" onClick={() => navigate('/document-graph')}>
          All graphs
        </Button>
      </div>

      {building && (
        <div className="px-4 py-2 border-b bg-muted/40">
          <div className="text-xs text-muted-foreground mb-1">{progressMessage || 'Working…'}</div>
          <Progress value={progress} className="h-1.5" />
        </div>
      )}

      <div className="flex-1 relative overflow-hidden">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
            <Loader2 className="w-5 h-5 animate-spin mr-2" /> Loading graph…
          </div>
        )}
        {!loading && !hasGraph && !building && (
          <div className="absolute inset-0 flex items-center justify-center">
            <Card className="max-w-md">
              <CardContent className="py-8 text-center space-y-3">
                <AlertCircle className="w-8 h-8 text-muted-foreground mx-auto" />
                <div className="font-medium">No graph yet</div>
                <p className="text-sm text-muted-foreground">
                  Click <span className="font-medium">Build graph</span> to extract entities,
                  relations, and communities from this thread's documents.
                </p>
                {status?.error && (
                  <p className="text-xs text-destructive font-mono whitespace-pre-wrap text-left max-h-32 overflow-auto">
                    {status.error}
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
        )}
        {hasGraph && (
          <DocumentGraphView
            graph={graph!}
            onSelectEntity={(e) => {
              setSelectedEntity(e);
              setSelectedEdge(null);
            }}
            onSelectEdge={(e) => {
              setSelectedEdge(e);
              setSelectedEntity(null);
            }}
          />
        )}
        <ProvenancePanel
          entity={selectedEntity}
          edge={selectedEdge}
          onClose={() => {
            setSelectedEntity(null);
            setSelectedEdge(null);
          }}
        />
      </div>

      <AlertDialog open={confirmDelete} onOpenChange={setConfirmDelete}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this graph?</AlertDialogTitle>
            <AlertDialogDescription>
              The underlying documents won't be touched, but you'll need to rebuild the graph
              from scratch if you want to view it again.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={deleteGraph}>Delete</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default DocumentGraphPage;
