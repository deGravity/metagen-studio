import { useEffect, useMemo, useState } from 'react';
import { getSession, getNode, checkoutNode } from './api';
import type { SessionTree, SessionNode, NodeRestore } from './api';
import './explorer.css';

const KIND_ICON: Record<string, string> = {
  root: '●', geometry: '◆', sim: '∿', assistant_turn: '✦', edit: '✎',
};

// flatten the DAG into a DFS-ordered list with depth, for an indented tree
function dfsOrder(tree: SessionTree): { node: SessionNode; depth: number }[] {
  const out: { node: SessionNode; depth: number }[] = [];
  const root = Object.values(tree.nodes).find((n) => n.parent === null);
  if (!root) return out;
  const walk = (id: string, depth: number) => {
    const n = tree.nodes[id];
    if (!n) return;
    out.push({ node: n, depth });
    const kids = [...n.children].sort(
      (a, b) => (tree.nodes[a]?.ts ?? '').localeCompare(tree.nodes[b]?.ts ?? ''));
    for (const c of kids) walk(c, depth + (n.children.length > 1 ? 1 : 0));
  };
  walk(root.id, 0);
  return out;
}

function Raw({ label, data }: { label: string; data: any }) {
  return (
    <details className="raw">
      <summary>{label}</summary>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </details>
  );
}

function EventView({ ev }: { ev: any }) {
  const p = ev.payload || {};
  const t = ev.type;
  if (t === 'user_message') {
    const c = p.content;
    const text = typeof c === 'string' ? c
      : Array.isArray(c) ? c.filter((b: any) => b.type === 'text').map((b: any) => b.text).join('\n') : '';
    return <div className="ev ev-user"><div className="ev-h">user</div><div className="ev-body">{text}</div></div>;
  }
  if (t === 'copilot_request') {
    return (
      <div className="ev ev-req">
        <div className="ev-h">request · call {p.call_index} · {p.model}
          {p.thinking ? ` · thinking ${p.thinking.budget_tokens}` : ' · no thinking'}</div>
        <Raw label={`messages (${(p.messages || []).length})`} data={p.messages} />
        <Raw label="system" data={p.system} />
      </div>
    );
  }
  if (t === 'copilot_response') {
    const blocks = p.content_blocks || [];
    return (
      <div className="ev ev-resp">
        <div className="ev-h">response · call {p.call_index} · stop: {p.stop_reason}
          {p.usage ? ` · ${p.usage.input_tokens}→${p.usage.output_tokens} tok` : ''}</div>
        {blocks.map((b: any, i: number) => {
          if (b.type === 'thinking')
            return <details key={i} className="thinking-ev"><summary>💭 thinking</summary><pre>{b.thinking}</pre></details>;
          if (b.type === 'text')
            return <div key={i} className="ev-text">{b.text}</div>;
          if (b.type === 'tool_use')
            return <div key={i} className="ev-tooluse">→ {b.name}({JSON.stringify(b.input).slice(0, 200)})</div>;
          if (b.type === 'redacted_thinking')
            return <div key={i} className="ev-text dim">[redacted thinking]</div>;
          return null;
        })}
      </div>
    );
  }
  if (t === 'tool_exec') {
    return (
      <div className="ev ev-tool">
        <div className="ev-h">tool · {p.name} · {p.elapsed_s}s</div>
        <Raw label="args" data={p.args} />
        <Raw label="result" data={p.result} />
      </div>
    );
  }
  if (t === 'proposal') {
    return (
      <div className="ev ev-prop">
        <div className="ev-h">proposal · {p.summary}</div>
        <details className="raw"><summary>new_code</summary><pre className="code">{p.new_code}</pre></details>
      </div>
    );
  }
  if (t === 'geometry_run')
    return <div className="ev ev-geo"><div className="ev-h">geometry @{p.resolution} · {p.origin}</div>
      <div className="ev-body">vf {p.stats?.volume_fraction?.toFixed?.(4)} · {p.stats?.n_triangles} tris · {p.elapsed_s}s</div></div>;
  if (t === 'sim_run')
    return <div className="ev ev-simr"><div className="ev-h">sim @{p.resolution} · {p.backend} · {p.origin}</div>
      <div className="ev-body">C[0][0] {p.C_matrix?.[0]?.[0]?.toFixed?.(4)} · {p.elapsed_s}s</div></div>;
  if (t === 'editor_snapshot')
    return <div className="ev ev-snap"><div className="ev-h">editor snapshot · {p.reason} · {p.code_hash}</div></div>;
  if (t === 'proposal_decision')
    return <div className="ev ev-snap"><div className="ev-h">proposal {p.status}</div></div>;
  return <div className="ev"><div className="ev-h">{t}</div><Raw label="payload" data={p} /></div>;
}

export function Explorer() {
  const sid = new URLSearchParams(window.location.search).get('session');
  const [tree, setTree] = useState<SessionTree | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [detail, setDetail] = useState<NodeRestore | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function refresh() {
    if (!sid) return;
    try {
      const t = await getSession(sid);
      setTree(t);
      setSel((s) => s ?? t.head);
    } catch (e: any) { setErr(e.message); }
  }
  useEffect(() => { refresh(); /* eslint-disable-next-line */ }, []);
  useEffect(() => {
    if (!sid || !sel) return;
    getNode(sid, sel).then(setDetail).catch((e) => setErr(e.message));
  }, [sid, sel]);

  const rows = useMemo(() => (tree ? dfsOrder(tree) : []), [tree]);

  async function rewind() {
    if (!sid || !sel) return;
    try {
      await checkoutNode(sid, sel);
      try { new BroadcastChannel('studio-session')
        .postMessage({ type: 'checkout', session_id: sid, node_id: sel }); } catch { /* no BC */ }
      await refresh();
    } catch (e: any) { setErr(e.message); }
  }

  if (!sid) return <div className="explorer-empty">No session — open via /explorer?session=&lt;id&gt;</div>;

  return (
    <div className="explorer">
      <header className="exp-header">
        <h1>Session log · {tree?.name ?? sid}</h1>
        {err && <span className="exp-err">{err}</span>}
      </header>
      <div className="exp-body">
        <div className="exp-tree">
          {rows.map(({ node, depth }) => (
            <button
              key={node.id}
              className={'tree-node' + (node.id === sel ? ' sel' : '') + (node.id === tree?.head ? ' head' : '')}
              style={{ paddingLeft: 8 + depth * 16 }}
              onClick={() => setSel(node.id)}
              title={node.id}
            >
              <span className="tn-icon">{KIND_ICON[node.kind] ?? '•'}</span>
              <span className="tn-label">{node.label}</span>
              {node.children.length > 1 && <span className="tn-branch">⑂{node.children.length}</span>}
              {node.id === tree?.head && <span className="tn-head">HEAD</span>}
            </button>
          ))}
        </div>
        <div className="exp-detail">
          {detail && (
            <>
              <div className="detail-head">
                <div>
                  <span className="dh-kind">{detail.node.kind}</span> · {detail.node.label}
                  <div className="dh-meta">{new Date(detail.node.ts).toLocaleString()} · {detail.node.id}</div>
                </div>
                <button className="rewind-btn" onClick={rewind}
                  title="Move HEAD here. Continuing in the studio will branch from this point.">
                  ⟲ Rewind here
                </button>
              </div>
              {detail.snapshot.code != null && (
                <details className="raw"><summary>snapshot code ({detail.snapshot.code_hash})</summary>
                  <pre className="code">{detail.snapshot.code}</pre></details>
              )}
              <div className="events">
                {(detail.events || []).map((ev: any) => <EventView key={ev.id} ev={ev} />)}
                {(!detail.events || detail.events.length === 0) &&
                  <div className="explorer-empty">no events on this node</div>}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
