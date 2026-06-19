import { useEffect, useState } from 'react';
import { sessionsUsage, deleteSession, pruneSession, deleteOlder } from '../api';
import type { SessionUsage } from '../api';

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

interface Props {
  currentId: string | null;
  onClose: () => void;
  onChanged: () => void;   // refresh session list / current after deletions
}

export function CleanupModal(props: Props) {
  const [usage, setUsage] = useState<SessionUsage | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  async function refresh() {
    try { setUsage(await sessionsUsage()); } catch { /* ignore */ }
  }
  useEffect(() => { refresh(); }, []);

  async function act(label: string, fn: () => Promise<any>, confirmMsg?: string) {
    if (confirmMsg && !window.confirm(confirmMsg)) return;
    setBusy(label);
    try { await fn(); await refresh(); props.onChanged(); }
    finally { setBusy(null); }
  }

  const rows = usage?.sessions ?? [];

  return (
    <div className="modal-overlay" onClick={props.onClose}>
      <div className="modal cleanup" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h2>Session storage</h2>
          <button className="modal-close" onClick={props.onClose}>✕</button>
        </div>
        <div className="cleanup-total">
          {rows.length} sessions · total {usage ? fmtBytes(usage.total_bytes) : '…'}
        </div>
        <div className="cleanup-list">
          {rows.map((s) => (
            <div key={s.id} className={'cleanup-row' + (s.id === props.currentId ? ' current' : '')}>
              <div className="cr-main">
                <div className="cr-name">{s.name}{s.id === props.currentId && <span className="cr-cur"> (current)</span>}</div>
                <div className="cr-meta">{s.n_nodes} nodes · {fmtBytes(s.size_bytes)} · {new Date(s.updated).toLocaleString()}</div>
              </div>
              <div className="cr-actions">
                {s.id === props.currentId && (
                  <button disabled={!!busy} title="keep only the current branch (drop other branches + their blobs)"
                    onClick={() => act('prune', () => pruneSession(s.id),
                      'Delete everything off the current branch of this session? This cannot be undone.')}>
                    prune branches
                  </button>
                )}
                <button disabled={!!busy} title="delete this session and every session older than it"
                  onClick={() => act('older', () => deleteOlder(s.id),
                    `Delete "${s.name}" AND all sessions older than it? This cannot be undone.`)}>
                  delete ≤ this
                </button>
                <button className="danger" disabled={!!busy} title="delete this session"
                  onClick={() => act('del', () => deleteSession(s.id),
                    `Delete session "${s.name}"? This cannot be undone.`)}>
                  delete
                </button>
              </div>
            </div>
          ))}
          {rows.length === 0 && <div className="cleanup-empty">no sessions</div>}
        </div>
      </div>
    </div>
  );
}
