import { useState } from 'react';
import type { SessionInfo } from '../api';

interface Props {
  name: string;
  nameSource: string;
  sessions: SessionInfo[];
  currentId: string | null;
  onNew: () => void;
  onPick: (id: string) => void;
  onRename: (name: string) => void;
}

export function SessionBar(props: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(props.name);
  const [open, setOpen] = useState(false);

  return (
    <div className="session-bar">
      <span className="session-label">session:</span>
      {editing ? (
        <input
          className="session-name-input"
          value={draft}
          autoFocus
          onChange={(e) => setDraft(e.target.value)}
          onBlur={() => {
            setEditing(false);
            if (draft.trim() && draft.trim() !== props.name) props.onRename(draft.trim());
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
            if (e.key === 'Escape') { setDraft(props.name); setEditing(false); }
          }}
        />
      ) : (
        <button
          className="session-name"
          title="rename session"
          onClick={() => { setDraft(props.name); setEditing(true); }}
        >
          {props.name || 'Untitled'}
          {props.nameSource === 'auto' && <span className="auto-badge">auto</span>}
          <span className="edit-pencil"> ✎</span>
        </button>
      )}
      <div className="session-menu">
        <button className="session-toggle" onClick={() => setOpen((o) => !o)}>▾</button>
        {open && (
          <div className="session-dropdown" onMouseLeave={() => setOpen(false)}>
            <button className="session-new" onClick={() => { setOpen(false); props.onNew(); }}>
              + New session
            </button>
            {props.sessions.map((s) => (
              <button
                key={s.id}
                className={s.id === props.currentId ? 'session-item active' : 'session-item'}
                onClick={() => { setOpen(false); props.onPick(s.id); }}
              >
                <span className="si-name">{s.name}</span>
                <span className="si-meta">{s.n_nodes} nodes · {new Date(s.updated).toLocaleString()}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
