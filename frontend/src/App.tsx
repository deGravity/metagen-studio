import { useEffect, useRef, useState } from 'react';
import { CodeEditor } from './components/Editor';
import { Viewer3D } from './components/Viewer3D';
import { SettingsPanel } from './components/Settings';
import { ResultsPanel } from './components/Results';
import { ChatPanel } from './components/Chat';
import { SessionBar } from './components/SessionBar';
import { CleanupModal } from './components/CleanupModal';
import {
  simulate, getInfo, decodeMesh, streamExecute, cancelJob, getCachedResults,
  createSession, listSessions, getSession, renameSession, getNode, logSessionEvent,
} from './api';
import type { SessionInfo, NodeRestore } from './api';
import type {
  ExecuteResponse, SimulateResponse, InfoResponse,
  TpmsMode, SimBackend, MeshData, ChatStateContext,
} from './types';

const LS_SESSION = 'studio.sessionId';

const DEFAULT_CODE = `from metagen import *


def make_structure(beamRadius: float = 0.04) -> Structure:
    """A simple BCC beam lattice — beams from the cube center to all corners."""
    side = 1.0
    embedding = cuboid.embed(
        side, side, side,
        cornerAtAABBMin=cuboid.corners.FRONT_BOTTOM_LEFT,
    )
    pat = CuboidFullMirror()

    center = vertex(cuboid.INTERIOR)
    lines = []
    for cornerEntity in cuboid.corners.getAll():
        lines.append(Polyline([center, vertex(cornerEntity)]))
    skel = skeleton(lines)
    beams = UniformBeams(skel, beamRadius)

    tile = Tile([beams], embedding)
    return Structure(tile, pat)
`;

async function hashCode(code: string): Promise<string> {
  const enc = new TextEncoder().encode(code);
  const buf = await crypto.subtle.digest('SHA-256', enc);
  return Array.from(new Uint8Array(buf)).slice(0, 6)
    .map((b) => b.toString(16).padStart(2, '0')).join('');
}

export default function App() {
  const [code, setCode] = useState(DEFAULT_CODE);
  const [currentHash, setCurrentHash] = useState<string>('');
  const [resolution, setResolution] = useState(65);
  const [tpmsMode, setTpmsMode] = useState<TpmsMode>('current');
  const [simBackend, setSimBackend] = useState<SimBackend>('auto');
  const [geometry, setGeometry] = useState<ExecuteResponse | null>(null);
  const [sim, setSim] = useState<SimulateResponse | null>(null);
  const [mesh, setMesh] = useState<MeshData | null>(null);
  const [info, setInfo] = useState<InfoResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<
    { phase: string; attempt?: number; elapsed?: number; detail?: string } | null
  >(null);
  const [chatHeight, setChatHeight] = useState(320);
  const jobIdRef = useRef<string | null>(null);
  // sessions
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionName, setSessionName] = useState('Untitled session');
  const [sessionNameSource, setSessionNameSource] = useState('auto');
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [thinking, setThinking] = useState(true);
  const [cleanupOpen, setCleanupOpen] = useState(false);

  // refresh current session name/source + the picker list (e.g. after auto-naming)
  async function refreshSessionMeta() {
    if (!sessionId) return;
    try {
      const t = await getSession(sessionId);
      setSessionName(t.name); setSessionNameSource(t.name_source);
    } catch { /* ignore */ }
    listSessions().then(setSessions).catch(() => {});
  }

  // after cleanup deletions: if the current session is gone, start fresh
  async function afterCleanup() {
    try {
      const list = await listSessions();
      setSessions(list);
      if (sessionId && !list.some((s) => s.id === sessionId)) await newSession();
    } catch { /* ignore */ }
  }

  // Restore the editor/viewer/results from a session node snapshot.
  function applyRestore(r: NodeRestore) {
    const snap = r.snapshot;
    if (snap.code != null) setCode(snap.code);
    if (snap.geometry) { setGeometry(snap.geometry); setMesh(decodeMesh(snap.geometry)); setResolution(snap.geometry.resolution); }
    else { setGeometry(null); setMesh(null); }
    setSim(snap.sim ?? null);
  }

  async function adoptSession(tree: { session_id: string; name: string; name_source: string; head: string }) {
    setSessionId(tree.session_id);
    setSessionName(tree.name);
    setSessionNameSource(tree.name_source);
    localStorage.setItem(LS_SESSION, tree.session_id);
    listSessions().then(setSessions).catch(() => {});
  }

  // init: resume stored session or create a new one (guard StrictMode double-run)
  const sessionInitRef = useRef(false);
  useEffect(() => {
    if (sessionInitRef.current) return;
    sessionInitRef.current = true;
    (async () => {
      const stored = localStorage.getItem(LS_SESSION);
      if (stored) {
        try {
          const tree = await getSession(stored);
          await adoptSession(tree);
          // restore HEAD state if it carries a snapshot
          const head = tree.nodes[tree.head];
          if (head?.snapshot?.code != null) {
            const r = await getNode(stored, tree.head);
            applyRestore(r);
          }
          return;
        } catch { /* stored session gone — fall through to create */ }
      }
      try { await adoptSession(await createSession('claude-opus-4-7')); } catch (e: any) { setError(`session init: ${e.message}`); }
    })();
  }, []);

  async function newSession() {
    try {
      const tree = await createSession('claude-opus-4-7');
      await adoptSession(tree);
      setGeometry(null); setSim(null); setMesh(null);
    } catch (e: any) { setError(`new session: ${e.message}`); }
  }

  async function pickSession(id: string) {
    try {
      const tree = await getSession(id);
      await adoptSession(tree);
      const head = tree.nodes[tree.head];
      if (head?.snapshot?.code != null) applyRestore(await getNode(id, tree.head));
      else { setGeometry(null); setSim(null); setMesh(null); }
    } catch (e: any) { setError(`open session: ${e.message}`); }
  }

  // when the explorer rewinds HEAD (separate tab), restore that node here
  useEffect(() => {
    if (!sessionId) return;
    let bc: BroadcastChannel | null = null;
    try {
      bc = new BroadcastChannel('studio-session');
      bc.onmessage = (e) => {
        const m: any = e.data;
        if (m?.type === 'checkout' && m.session_id === sessionId && m.node_id) {
          getNode(sessionId, m.node_id).then(applyRestore).catch(() => {});
        }
      };
    } catch { /* BroadcastChannel unsupported */ }
    return () => { try { bc?.close(); } catch { /* noop */ } };
  }, [sessionId]);

  async function doRename(name: string) {
    if (!sessionId) return;
    try { const t = await renameSession(sessionId, name); setSessionName(t.name); setSessionNameSource(t.name_source); listSessions().then(setSessions).catch(() => {}); }
    catch (e: any) { setError(`rename: ${e.message}`); }
  }

  // Drag the splitter between the editor and the chat dock to resize the chat.
  function startChatDrag(e: React.MouseEvent) {
    e.preventDefault();
    const startY = e.clientY;
    const startH = chatHeight;
    const onMove = (ev: MouseEvent) => {
      const h = startH + (startY - ev.clientY);
      setChatHeight(Math.max(120, Math.min(h, window.innerHeight - 240)));
    };
    const onUp = () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      document.body.style.cursor = '';
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    document.body.style.cursor = 'row-resize';
  }

  useEffect(() => {
    getInfo().then(setInfo).catch((e) => setError(`info: ${e.message}`));
  }, []);

  useEffect(() => {
    hashCode(code).then(setCurrentHash);
  }, [code]);

  async function runGeometry() {
    setBusy(true); setError(null); setProgress({ phase: 'starting' });
    jobIdRef.current = null;
    try {
      for await (const ev of streamExecute(code, resolution, tpmsMode, undefined, sessionId ?? undefined)) {
        if (ev.kind === 'job') {
          jobIdRef.current = ev.job_id;
        } else if (ev.kind === 'progress') {
          setProgress({ phase: ev.phase, attempt: ev.attempt, elapsed: ev.elapsed, detail: ev.detail });
        } else if (ev.kind === 'result') {
          setGeometry(ev.resp);
          setMesh(decodeMesh(ev.resp));
        } else if (ev.kind === 'cancelled') {
          setError('geometry run cancelled');
        } else if (ev.kind === 'error') {
          setError(ev.message);
        }
      }
    } catch (e: any) {
      setError(e.message ?? String(e));
    } finally {
      setBusy(false); setProgress(null); jobIdRef.current = null;
    }
  }

  async function cancelGeometry() {
    if (jobIdRef.current) await cancelJob(jobIdRef.current);
  }

  async function runSim() {
    setBusy(true); setError(null);
    try {
      const r = await simulate(code, resolution, tpmsMode, simBackend, 1.0, 0.45, sessionId ?? undefined);
      setSim(r);
    } catch (e: any) {
      setError(e.message ?? String(e));
    } finally {
      setBusy(false);
    }
  }

  async function applyProposal(newCode: string, _proposalId?: string, summary?: string) {
    setCode(newCode);
    if (sessionId) {
      const h = await hashCode(newCode);
      logSessionEvent(sessionId, 'proposal_decision',
        { status: 'accepted', summary: summary ?? '', code_hash: h },
        { kind: 'edit', label: `accepted: ${(summary ?? 'edit').slice(0, 50)}`,
          snapshot: { code: newCode, code_hash: h, geometry_ref: null, sim_ref: null, chat_len: 0 } },
      ).catch(() => {});
    }
    // If the copilot already ran geometry/sim on this exact code, reuse those
    // results so the viewer and results panel jump straight to up-to-date
    // instead of showing stale data until a manual re-run.
    try {
      const cached = await getCachedResults(newCode);
      if (cached.geometry) {
        setGeometry(cached.geometry);
        setMesh(decodeMesh(cached.geometry));
        setResolution(cached.geometry.resolution);
        if (cached.geometry.tpms_optimizer_mode) setTpmsMode(cached.geometry.tpms_optimizer_mode);
      }
      if (cached.sim) setSim(cached.sim);
    } catch {
      /* no cached results — leave panels as-is (staleness tag will show) */
    }
  }

  // Chat's run_geometry tool now returns the mesh in its UI event, so update
  // the viewer directly — no second (blocking) refetch.
  function chatGeometryDone(summary: any) {
    setResolution(summary.resolution);
    if (summary.tpms_optimizer_mode) setTpmsMode(summary.tpms_optimizer_mode);
    if (!summary.vertices_b64 || !summary.triangles_b64) return;
    const r: ExecuteResponse = {
      code_hash: summary.code_hash,
      resolution: summary.resolution,
      tpms_optimizer_mode: summary.tpms_optimizer_mode ?? 'current',
      stats: {
        cell_resolution: summary.cell_resolution,
        volume_fraction: summary.volume_fraction,
        n_vertices: summary.n_vertices,
        n_triangles: summary.n_triangles,
        n_active_voxels: summary.n_active_voxels,
        n_total_voxels: summary.n_total_voxels,
      },
      vertices_b64: summary.vertices_b64,
      triangles_b64: summary.triangles_b64,
      elapsed_geometry_s: summary.elapsed_s ?? 0,
      cached: false,
    };
    setGeometry(r);
    setMesh(decodeMesh(r));
  }

  function chatSimDone(summary: any) {
    setSim({
      code_hash: summary.code_hash,
      resolution: summary.resolution,
      tpms_optimizer_mode: summary.tpms_optimizer_mode ?? 'current',
      backend_used: summary.backend_used,
      C_matrix: summary.C_matrix,
      properties: summary.properties,
      elapsed_sim_s: summary.elapsed_s,
      cached: false,
    });
    setResolution(summary.resolution);
  }

  const geomStale = geometry !== null && geometry.code_hash !== currentHash;
  const simStale = sim !== null && sim.code_hash !== currentHash;

  const chatState: ChatStateContext = {
    code,
    geometry_code_hash: geometry?.code_hash ?? null,
    geometry_summary: geometry ? {
      resolution: geometry.resolution,
      cell_resolution: geometry.stats.cell_resolution,
      volume_fraction: geometry.stats.volume_fraction,
      n_active_voxels: geometry.stats.n_active_voxels,
      n_total_voxels: geometry.stats.n_total_voxels,
      n_vertices: geometry.stats.n_vertices,
      n_triangles: geometry.stats.n_triangles,
    } : undefined,
    sim_code_hash: sim?.code_hash ?? null,
    sim_summary: sim ? {
      resolution: sim.resolution,
      backend_used: sim.backend_used,
      properties: sim.properties,
    } : undefined,
    last_error: error,
  };

  return (
    <div className="app">
      <header>
        <h1>metaDSL Studio</h1>
        <SessionBar
          name={sessionName}
          nameSource={sessionNameSource}
          sessions={sessions}
          currentId={sessionId}
          onNew={newSession}
          onPick={pickSession}
          onRename={doRename}
          onManage={() => setCleanupOpen(true)}
        />
        <button
          className="logs-btn"
          disabled={!sessionId}
          title="open the full session log explorer in a new tab"
          onClick={() => sessionId && window.open(`/explorer?session=${sessionId}`, '_blank')}
        >
          logs ↗
        </button>
        <div className="hash">code: <code>{currentHash || '…'}</code></div>
      </header>
      <div className="layout">
        <div className="pane editor-pane">
          <div className="editor-wrap">
            <CodeEditor value={code} onChange={setCode} />
          </div>
          <div className="vsplit" onMouseDown={startChatDrag}
               title="drag to resize the copilot panel" />
          <div className="chat-dock" style={{ height: chatHeight }}>
            <div className="chat-dock-title">
              Copilot
              <label className="think-toggle" title="extended thinking (chain-of-thought)">
                <input type="checkbox" checked={thinking}
                       onChange={(e) => setThinking(e.target.checked)} />
                thinking
              </label>
            </div>
            <ChatPanel
              state={chatState}
              available={info?.chat_available ?? false}
              sessionId={sessionId ?? undefined}
              thinking={thinking}
              onApplyProposal={applyProposal}
              onGeometryDone={chatGeometryDone}
              onSimDone={chatSimDone}
              onTurnDone={refreshSessionMeta}
            />
          </div>
        </div>
        <div className="pane viewer-pane">
          <Viewer3D mesh={mesh} />
        </div>
        <div className="pane right-pane">
          <SettingsPanel
            resolution={resolution}
            setResolution={setResolution}
            tpmsMode={tpmsMode}
            setTpmsMode={setTpmsMode}
            simBackend={simBackend}
            setSimBackend={setSimBackend}
            gpuAvailable={info?.gpu_available ?? false}
            validGpuResolutions={info?.valid_gpu_resolutions ?? []}
            onRunGeometry={runGeometry}
            onRunSim={runSim}
            onCancel={cancelGeometry}
            busy={busy}
            progress={progress}
          />
          <div className="right-body">
            <ResultsPanel
              geometry={geometry}
              sim={sim}
              geomStaleVsCode={geomStale}
              simStaleVsCode={simStale}
              error={error}
            />
          </div>
        </div>
      </div>
      {cleanupOpen && (
        <CleanupModal
          currentId={sessionId}
          onClose={() => setCleanupOpen(false)}
          onChanged={afterCleanup}
        />
      )}
    </div>
  );
}
