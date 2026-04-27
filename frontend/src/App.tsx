import { useEffect, useState } from 'react';
import { CodeEditor } from './components/Editor';
import { Viewer3D } from './components/Viewer3D';
import { SettingsPanel } from './components/Settings';
import { ResultsPanel } from './components/Results';
import { ChatPanel } from './components/Chat';
import { executeCode, simulate, getInfo, decodeMesh } from './api';
import type {
  ExecuteResponse, SimulateResponse, InfoResponse,
  TpmsMode, SimBackend, MeshData, ChatStateContext,
} from './types';

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

type RightTab = 'results' | 'chat';

export default function App() {
  const [code, setCode] = useState(DEFAULT_CODE);
  const [currentHash, setCurrentHash] = useState<string>('');
  const [resolution, setResolution] = useState(33);
  const [tpmsMode, setTpmsMode] = useState<TpmsMode>('current');
  const [simBackend, setSimBackend] = useState<SimBackend>('auto');
  const [geometry, setGeometry] = useState<ExecuteResponse | null>(null);
  const [sim, setSim] = useState<SimulateResponse | null>(null);
  const [mesh, setMesh] = useState<MeshData | null>(null);
  const [info, setInfo] = useState<InfoResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<RightTab>('results');

  useEffect(() => {
    getInfo().then(setInfo).catch((e) => setError(`info: ${e.message}`));
  }, []);

  useEffect(() => {
    hashCode(code).then(setCurrentHash);
  }, [code]);

  async function runGeometry() {
    setBusy(true); setError(null);
    try {
      const r = await executeCode(code, resolution, tpmsMode);
      setGeometry(r);
      setMesh(decodeMesh(r));
    } catch (e: any) {
      setError(e.message ?? String(e));
    } finally {
      setBusy(false);
    }
  }

  async function runSim() {
    setBusy(true); setError(null);
    try {
      const r = await simulate(code, resolution, tpmsMode, simBackend);
      setSim(r);
    } catch (e: any) {
      setError(e.message ?? String(e));
    } finally {
      setBusy(false);
    }
  }

  function applyProposal(newCode: string) {
    setCode(newCode);
    // Geometry/sim viewing data is now stale by definition; the staleness
    // tag will surface that. Don't auto-rerun.
  }

  // Refetch fresh mesh when chat triggers run_geometry on the server.
  async function chatGeometryDone(summary: any) {
    try {
      const r = await executeCode(code, summary.resolution, summary.tpms_optimizer_mode ?? 'current');
      setGeometry(r);
      setMesh(decodeMesh(r));
      setResolution(summary.resolution);
      if (summary.tpms_optimizer_mode) setTpmsMode(summary.tpms_optimizer_mode);
    } catch (e: any) {
      setError(`refresh after chat run_geometry: ${e.message}`);
    }
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
        <div className="hash">code: <code>{currentHash || '…'}</code></div>
      </header>
      <div className="layout">
        <div className="pane editor-pane">
          <CodeEditor value={code} onChange={setCode} />
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
            busy={busy}
          />
          <div className="tabs">
            <button
              className={tab === 'results' ? 'tab active' : 'tab'}
              onClick={() => setTab('results')}
            >
              Results
            </button>
            <button
              className={tab === 'chat' ? 'tab active' : 'tab'}
              onClick={() => setTab('chat')}
            >
              Copilot
            </button>
          </div>
          <div className="tab-body">
            {tab === 'results' ? (
              <ResultsPanel
                geometry={geometry}
                sim={sim}
                geomStaleVsCode={geomStale}
                simStaleVsCode={simStale}
                error={error}
              />
            ) : (
              <ChatPanel
                state={chatState}
                available={info?.chat_available ?? false}
                onApplyProposal={applyProposal}
                onGeometryDone={chatGeometryDone}
                onSimDone={chatSimDone}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
