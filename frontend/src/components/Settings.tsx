import type { TpmsMode, SimBackend } from '../types';

interface Props {
  resolution: number;
  setResolution: (n: number) => void;
  tpmsMode: TpmsMode;
  setTpmsMode: (m: TpmsMode) => void;
  simBackend: SimBackend;
  setSimBackend: (b: SimBackend) => void;
  gpuAvailable: boolean;
  validGpuResolutions: number[];
  onRunGeometry: () => void;
  onRunSim: () => void;
  busy: boolean;
}

const COMMON_RES = [17, 33, 49, 65, 97, 100, 129];

export function SettingsPanel(props: Props) {
  const gpuValid = props.validGpuResolutions.includes(props.resolution);
  return (
    <div className="settings">
      <h3>Settings</h3>

      <label>
        <span>Resolution</span>
        <select value={props.resolution} onChange={(e) => props.setResolution(parseInt(e.target.value))}>
          {COMMON_RES.map((r) => (
            <option key={r} value={r}>
              {r}{props.validGpuResolutions.includes(r) ? '' : ' (CPU only)'}
            </option>
          ))}
        </select>
      </label>

      <label>
        <span>TPMS optimizer</span>
        <select value={props.tpmsMode} onChange={(e) => props.setTpmsMode(e.target.value as TpmsMode)}>
          <option value="current">current (BOBYQA, fast)</option>
          <option value="global">global (ESCH, deterministic, ~10× slower)</option>
          <option value="experimental">experimental</option>
        </select>
      </label>

      <label>
        <span>Sim backend</span>
        <select value={props.simBackend} onChange={(e) => props.setSimBackend(e.target.value as SimBackend)}>
          <option value="auto">auto (GPU if valid, else CPU)</option>
          <option value="gpu" disabled={!props.gpuAvailable || !gpuValid}>gpu</option>
          <option value="cpu">cpu</option>
        </select>
      </label>

      <div className="status-line">
        GPU: {props.gpuAvailable ? <span className="ok">available</span> : <span className="warn">unavailable</span>}
        {!gpuValid && <span className="warn"> · current res not multigrid-valid</span>}
      </div>

      <div className="actions">
        <button onClick={props.onRunGeometry} disabled={props.busy}>
          {props.busy ? '…' : 'Run geometry'}
        </button>
        <button onClick={props.onRunSim} disabled={props.busy}>
          {props.busy ? '…' : 'Simulate'}
        </button>
      </div>
    </div>
  );
}
