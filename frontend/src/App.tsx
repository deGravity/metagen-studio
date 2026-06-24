import { useEffect, useRef, useState } from 'react';
import { CodeEditor } from './components/Editor';
import { Viewer3D } from './components/Viewer3D';
import { SettingsPanel } from './components/Settings';
import { ResultsPanel } from './components/Results';
import { ChatPanel } from './components/Chat';
import { SessionBar } from './components/SessionBar';
import { CleanupModal } from './components/CleanupModal';
import { LlmSelector } from './components/LlmSelector';
import {
  simulate, getInfo, decodeMesh, streamExecute, cancelJob, getCachedResults,
  createSession, listSessions, getSession, renameSession, getNode, logSessionEvent,
  getProviders,
} from './api';
import type { SessionInfo, NodeRestore } from './api';
import type {
  ExecuteResponse, SimulateResponse, InfoResponse,
  TpmsMode, SimBackend, MeshData, ChatStateContext, ProviderInfo, LlmCreds,
} from './types';
import {
  Selection, loadCreds, loadCustomModels, loadSelection, saveSelection, credsToSend,
  baseUrlFor, isAvailable,
} from './llm';
import { discoverModels } from './api';

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
  const [sessionName, setSessionName] = useState('New session');
  const [sessionNameSource, setSessionNameSource] = useState('none');
  // mirror sessionId in a ref so ensureSession() sees the latest value across
  // concurrent async callers (state updates lag).
  const sessionIdRef = useRef<string | null>(null);
  useEffect(() => { sessionIdRef.current = sessionId; }, [sessionId]);
  const creatingRef = useRef<Promise<string | null> | null>(null);
  // bump to ask ChatPanel to rehydrate the transcript; node = which prefix
  // (undefined = HEAD). Reset on session switch, set to the node on checkout.
  const [chatRestore, setChatRestore] = useState<{ token: number; node?: string }>({ token: 0 });
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [thinking, setThinking] = useState(true);
  const [cleanupOpen, setCleanupOpen] = useState(false);
  // LLM provider/model selection (per-session, persisted in localStorage)
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [llmCreds, setLlmCreds] = useState<LlmCreds>(() => loadCreds());
  const [customModels, setCustomModels] = useState<Record<string, string[]>>(() => loadCustomModels());
  const [discovered, setDiscovered] = useState<Record<string, string[]>>({});
  const [discErr, setDiscErr] = useState<Record<string, string>>({});
  const [llmSel, setLlmSel] = useState<Selection>({ provider: 'anthropic', model: 'claude-opus-4-7' });

  // fetch provider availability once; seed the default selection
  useEffect(() => {
    getProviders().then(({ providers: ps, default_provider }) => {
      setProviders(ps);
      setLlmSel((cur) => {
        if (cur.model) return cur;   // already set (e.g. from a session)
        const p = ps.find((x) => x.name === default_provider) || ps[0];
        return p ? { provider: p.name, model: p.default_model || cur.model } : cur;
      });
    }).catch(() => {});
  }, []);

  function selectLlm(sel: Selection) {
    setLlmSel(sel);
    saveSelection(sessionId ?? undefined, sel);
  }

  // discover models for one provider; capture any error so the UI can show
  // *why* the list is empty instead of an opaque "no models".
  async function runDiscovery(p: ProviderInfo) {
    const base = baseUrlFor(p, llmCreds);
    if (!base) { setDiscErr((e) => ({ ...e, [p.name]: 'no base URL configured' })); return; }
    try {
      const r = await discoverModels(p.name, base, llmCreds[p.name]?.api_key);
      setDiscovered((d) => ({ ...d, [p.name]: r.models || [] }));
      setDiscErr((e) => ({ ...e, [p.name]: r.error || (r.models?.length ? '' : 'server returned no models') }));
    } catch (err: any) {
      setDiscErr((e) => ({ ...e, [p.name]: `request failed: ${err?.message || err}` }));
    }
  }

  // live-discover models for discoverable providers (vLLM) whenever the set of
  // providers or the stored creds/base_url change (and on provider switch).
  useEffect(() => {
    for (const p of providers) if (p.discover) runDiscovery(p);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [providers, llmCreds, llmSel.provider]);

  // For a discovery provider (vLLM), the selectable models come only from the
  // live list — so if the current selection holds a stale/profile-key model
  // (e.g. 'qwen3.6' from config or an old session), snap it to a real served id.
  useEffect(() => {
    const p = providers.find((x) => x.name === llmSel.provider);
    if (!p?.discover) return;
    const disc = discovered[llmSel.provider] || [];
    if (disc.length && !disc.includes(llmSel.model)) selectLlm({ provider: llmSel.provider, model: disc[0] });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [discovered, llmSel.provider, providers]);

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
    // rehydrate chat at HEAD for the adopted session (clears any stale node)
    setChatRestore((s) => ({ token: s.token + 1, node: undefined }));
    // restore this session's saved provider/model choice, if any
    const savedSel = loadSelection(tree.session_id);
    if (savedSel) setLlmSel(savedSel);
    listSessions().then(setSessions).catch(() => {});
  }

  // init: NO session is created on load — a page load is a clean, unsaved slate.
  // A session is created lazily on the first real action (edit / run / sim /
  // chat) via ensureSession(). Prior sessions live on the server and are
  // reachable from the picker. We just load the picker list here.
  useEffect(() => {
    listSessions().then(setSessions).catch(() => {});
  }, []);

  // Create-and-adopt a session iff none exists yet; idempotent + race-safe.
  // Returns the session id (or null on failure). Callers that log to a session
  // must await this and use the returned id (state updates lag).
  async function ensureSession(): Promise<string | null> {
    if (sessionIdRef.current) return sessionIdRef.current;
    if (creatingRef.current) return creatingRef.current;
    const p = (async () => {
      try {
        const tree = await createSession('claude-opus-4-7');
        sessionIdRef.current = tree.session_id;   // visible to concurrent callers now
        await adoptSession(tree);
        return tree.session_id;
      } catch (e: any) {
        setError(`session: ${e.message}`);
        return null;
      } finally {
        creatingRef.current = null;
      }
    })();
    creatingRef.current = p;
    return p;
  }

  // "New session" = back to an unsaved clean slate (no server session until the
  // next real action). Old sessions remain in the picker.
  function newSession() {
    sessionIdRef.current = null;
    setSessionId(null);
    setSessionName('New session'); setSessionNameSource('none');
    setCode(DEFAULT_CODE); setGeometry(null); setSim(null); setMesh(null);
    setChatRestore((s) => ({ token: s.token + 1, node: undefined }));   // clears chat
  }

  async function pickSession(id: string) {
    try {
      const tree = await getSession(id);
      await adoptSession(tree);
      const head = tree.nodes[tree.head];
      if (head?.snapshot?.code != null) applyRestore(await getNode(id, tree.head));
      else { setCode(DEFAULT_CODE); setGeometry(null); setSim(null); setMesh(null); }
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
          getNode(sessionId, m.node_id).then((r) => {
            applyRestore(r);
            // rehydrate chat to the rewound node's conversation prefix
            setChatRestore((s) => ({ token: s.token + 1, node: m.node_id }));
          }).catch(() => {});
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
    const sid = await ensureSession();
    try {
      for await (const ev of streamExecute(code, resolution, tpmsMode, undefined, sid ?? undefined)) {
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
    const sid = await ensureSession();
    try {
      const r = await simulate(code, resolution, tpmsMode, simBackend, 1.0, 0.45, sid ?? undefined);
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

  // Chat is enabled when the *selected* provider is usable — either the backend
  // has its creds, or the browser supplied them (key, or base_url for vLLM).
  // NOT the global info.chat_available, which only reflects the server-side
  // Anthropic key and ignores per-session provider choice + client creds.
  const selProvider = providers.find((p) => p.name === llmSel.provider);
  const chatAvailable = selProvider ? isAvailable(selProvider, llmCreds) : false;

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
            <CodeEditor value={code} onChange={(v) => { setCode(v); if (!sessionIdRef.current) ensureSession(); }} />
          </div>
          <div className="vsplit" onMouseDown={startChatDrag}
               title="drag to resize the copilot panel" />
          <div className="chat-dock" style={{ height: chatHeight }}>
            <div className="chat-dock-title">
              Copilot
              {providers.length > 0 && (
                <LlmSelector providers={providers} creds={llmCreds} setCreds={setLlmCreds}
                             customModels={customModels} setCustomModels={setCustomModels}
                             discovered={discovered} discErr={discErr} selection={llmSel}
                             onSelect={selectLlm}
                             onRefresh={(name) => { const p = providers.find((x) => x.name === name); if (p) runDiscovery(p); }} />
              )}
              <label className="think-toggle" title="extended thinking (chain-of-thought)">
                <input type="checkbox" checked={thinking}
                       onChange={(e) => setThinking(e.target.checked)} />
                thinking
              </label>
            </div>
            <ChatPanel
              state={chatState}
              available={chatAvailable}
              sessionId={sessionId ?? undefined}
              ensureSession={ensureSession}
              thinking={thinking}
              provider={llmSel.provider}
              model={llmSel.model}
              chatCreds={credsToSend(llmSel.provider, providers, llmCreds)}
              restoreToken={chatRestore.token}
              restoreNode={chatRestore.node}
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
