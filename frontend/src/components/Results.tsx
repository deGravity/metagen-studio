import type { ExecuteResponse, SimulateResponse } from '../types';

interface Props {
  geometry: ExecuteResponse | null;
  sim: SimulateResponse | null;
  geomStaleVsCode: boolean;
  simStaleVsCode: boolean;
  error: string | null;
}

function fmtNum(x: number, sig = 4): string {
  if (Math.abs(x) < 1e-12) return '0';
  if (Math.abs(x) > 1e4 || Math.abs(x) < 1e-3) return x.toExponential(sig - 1);
  return x.toPrecision(sig);
}

export function ResultsPanel(props: Props) {
  return (
    <div className="results">
      {props.error && (
        <div className="error">
          <h4>Error</h4>
          <pre>{props.error}</pre>
        </div>
      )}

      {props.geometry && (
        <section className={props.geomStaleVsCode ? 'stale' : ''}>
          <h4>Geometry {props.geomStaleVsCode && <span className="stale-tag">(stale)</span>}</h4>
          <table>
            <tbody>
              <tr><td>code</td><td><code>{props.geometry.code_hash}</code></td></tr>
              <tr><td>resolution</td><td>{props.geometry.resolution} (cell_dim {props.geometry.stats.cell_resolution})</td></tr>
              <tr><td>fill</td><td>{(props.geometry.stats.n_active_voxels / props.geometry.stats.n_total_voxels).toFixed(3)}</td></tr>
              <tr><td>volume frac</td><td>{props.geometry.stats.volume_fraction.toFixed(4)}</td></tr>
              <tr><td>vertices</td><td>{props.geometry.stats.n_vertices.toLocaleString()}</td></tr>
              <tr><td>triangles</td><td>{props.geometry.stats.n_triangles.toLocaleString()}</td></tr>
              <tr><td>elapsed</td><td>{props.geometry.elapsed_geometry_s.toFixed(2)}s {props.geometry.cached && <span className="cached">(cached)</span>}</td></tr>
            </tbody>
          </table>
        </section>
      )}

      {props.sim && (
        <section className={props.simStaleVsCode ? 'stale' : ''}>
          <h4>Simulation {props.simStaleVsCode && <span className="stale-tag">(stale)</span>}</h4>
          <table>
            <tbody>
              <tr><td>code</td><td><code>{props.sim.code_hash}</code></td></tr>
              <tr><td>backend</td><td>{props.sim.backend_used}</td></tr>
              <tr><td>elapsed</td><td>{props.sim.elapsed_sim_s.toFixed(2)}s {props.sim.cached && <span className="cached">(cached)</span>}</td></tr>
            </tbody>
          </table>

          <h5>C matrix (6×6)</h5>
          <table className="matrix">
            <tbody>
              {props.sim.C_matrix.map((row, i) => (
                <tr key={i}>
                  {row.map((v, j) => <td key={j}>{fmtNum(v)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>

          <h5>Properties</h5>
          <table>
            <tbody>
              {Object.entries(props.sim.properties).map(([k, v]) => (
                <tr key={k}><td>{k}</td><td>{fmtNum(v)}</td></tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {!props.geometry && !props.sim && !props.error && (
        <div className="empty">No results yet — edit code and click <em>Run geometry</em>.</div>
      )}
    </div>
  );
}
