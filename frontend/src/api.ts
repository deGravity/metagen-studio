import type {
  ExecuteResponse, SimulateResponse, InfoResponse,
  TpmsMode, SimBackend, MeshData,
  ChatMessage, ChatStateContext, ChatEvent, UploadResponse,
} from './types';

const API = '/api';

async function postJson<TIn, TOut>(path: string, body: TIn): Promise<TOut> {
  const r = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail ?? `HTTP ${r.status}`);
  }
  return r.json();
}

export async function getInfo(): Promise<InfoResponse> {
  const r = await fetch(`${API}/info`);
  if (!r.ok) throw new Error('info failed');
  return r.json();
}

async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(`${API}${path}`);
  if (!r.ok) throw new Error(`GET ${path} failed: ${r.status}`);
  return r.json();
}

// --- sessions -----------------------------------------------------------

export interface SessionInfo {
  id: string; name: string; name_source: string;
  created: string; updated: string; n_nodes: number;
}
export interface SessionNode {
  id: string; parent: string | null; children: string[]; ts: string;
  kind: string; label: string; event_ids: string[];
  snapshot: { code: string | null; code_hash: string | null;
              geometry_ref: string | null; sim_ref: string | null; chat_len: number };
}
export interface SessionTree {
  session_id: string; name: string; name_source: string;
  created: string; updated: string; head: string; model?: string;
  nodes: Record<string, SessionNode>;
}
export interface NodeRestore {
  node: SessionNode;
  snapshot: SessionNode['snapshot'] & {
    geometry?: ExecuteResponse | null; sim?: SimulateResponse | null;
  };
  events?: any[];
}

export async function createSession(model?: string): Promise<SessionTree> {
  return postJson('/sessions', { model });
}
export async function listSessions(): Promise<SessionInfo[]> {
  return (await getJson<{ sessions: SessionInfo[] }>('/sessions')).sessions;
}
export async function getSession(id: string): Promise<SessionTree> {
  return getJson(`/sessions/${id}`);
}
export async function renameSession(id: string, name: string): Promise<SessionTree> {
  const r = await fetch(`${API}/sessions/${id}`, {
    method: 'PATCH', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  if (!r.ok) throw new Error(`rename failed: ${r.status}`);
  return r.json();
}
export async function deleteSession(id: string): Promise<void> {
  await fetch(`${API}/sessions/${id}`, { method: 'DELETE' });
}
export async function checkoutNode(id: string, nodeId: string): Promise<NodeRestore> {
  return postJson(`/sessions/${id}/checkout`, { node_id: nodeId });
}
export async function getNode(id: string, nodeId: string): Promise<NodeRestore> {
  return getJson(`/sessions/${id}/node/${nodeId}`);
}
export async function logSessionEvent(
  id: string, type: string, payload: any,
  node?: { kind: string; label: string; snapshot: any },
): Promise<any> {
  return postJson(`/sessions/${id}/event`, {
    type, payload,
    make_node: !!node, kind: node?.kind, label: node?.label, snapshot: node?.snapshot,
  });
}

export async function executeCode(
  code: string, resolution: number, tpms_optimizer_mode: TpmsMode,
): Promise<ExecuteResponse> {
  return postJson('/execute', { code, resolution, tpms_optimizer_mode });
}

export async function simulate(
  code: string, resolution: number, tpms_optimizer_mode: TpmsMode,
  backend: SimBackend, E = 1.0, nu = 0.45, session_id?: string,
): Promise<SimulateResponse> {
  return postJson('/simulate', { code, resolution, tpms_optimizer_mode, backend, E, nu, session_id });
}

// --- streaming geometry (SSE) with live progress + cancellation ----------

export type ExecEvent =
  | { kind: 'job'; job_id: string }
  | { kind: 'progress'; phase: string; attempt?: number; elapsed?: number; detail?: string }
  | { kind: 'result'; resp: ExecuteResponse }
  | { kind: 'error'; message: string }
  | { kind: 'cancelled' }
  | { kind: 'done' };

export async function* streamExecute(
  code: string, resolution: number, tpms_optimizer_mode: TpmsMode,
  signal?: AbortSignal, session_id?: string,
): AsyncGenerator<ExecEvent> {
  const r = await fetch(`${API}/execute/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code, resolution, tpms_optimizer_mode, session_id }),
    signal,
  });
  if (!r.ok) {
    const detail = await r.text();
    throw new Error(`execute failed: ${r.status} ${detail}`);
  }
  if (!r.body) throw new Error('execute: no response body');

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const ev = parseExecSSE(chunk);
      if (ev) yield ev;
    }
  }
}

function parseExecSSE(chunk: string): ExecEvent | null {
  let event = '';
  let data = '';
  for (const line of chunk.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) data += line.slice(5).trim();
  }
  if (!event) return null;
  try {
    const d = data ? JSON.parse(data) : {};
    switch (event) {
      case 'job': return { kind: 'job', job_id: d.job_id };
      case 'progress': return { kind: 'progress', phase: d.phase, attempt: d.attempt, elapsed: d.elapsed, detail: d.detail };
      case 'result': return { kind: 'result', resp: d as ExecuteResponse };
      case 'error': return { kind: 'error', message: d.message };
      case 'cancelled': return { kind: 'cancelled' };
      case 'done': return { kind: 'done' };
      default: return null;
    }
  } catch {
    return null;
  }
}

export interface CachedResults {
  geometry: ExecuteResponse | null;
  sim: SimulateResponse | null;
}

// Latest geometry/sim the backend has computed for this exact code (by hash) —
// used to reuse a copilot-run generation/sim when its proposal is accepted.
export async function getCachedResults(code: string): Promise<CachedResults> {
  return postJson('/results/cached', { code });
}

export async function cancelJob(jobId: string): Promise<void> {
  await fetch(`${API}/jobs/${jobId}/cancel`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
  }).catch(() => { /* best-effort */ });
}

export function decodeMesh(resp: ExecuteResponse): MeshData {
  return {
    vertices: decodeFloat32(resp.vertices_b64),
    triangles: decodeUint32(resp.triangles_b64),
  };
}

function decodeFloat32(b64: string): Float32Array {
  const bytes = atob(b64);
  const buf = new ArrayBuffer(bytes.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bytes.length; i++) view[i] = bytes.charCodeAt(i);
  return new Float32Array(buf);
}

function decodeUint32(b64: string): Uint32Array {
  const bytes = atob(b64);
  const buf = new ArrayBuffer(bytes.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bytes.length; i++) view[i] = bytes.charCodeAt(i);
  return new Uint32Array(buf);
}

// --- chat uploads -------------------------------------------------------

export async function uploadChatFile(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);
  const r = await fetch(`${API}/chat/upload`, { method: 'POST', body: form });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail ?? `HTTP ${r.status}`);
  }
  return r.json();
}

// --- chat (SSE) ---------------------------------------------------------

export async function* streamChat(
  messages: ChatMessage[], state: ChatStateContext,
  signal?: AbortSignal,
  model = 'claude-opus-4-7',
  opts?: { thinking?: boolean; session_id?: string },
): AsyncGenerator<ChatEvent> {
  const r = await fetch(`${API}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, state, model,
                           thinking: opts?.thinking, session_id: opts?.session_id }),
    signal,
  });
  if (!r.ok) {
    const detail = await r.text();
    throw new Error(`chat failed: ${r.status} ${detail}`);
  }
  if (!r.body) throw new Error('chat: no response body');

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const ev = parseSSE(chunk);
      if (ev) yield ev;
    }
  }
}

function parseSSE(chunk: string): ChatEvent | null {
  let event = '';
  let data = '';
  for (const line of chunk.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) data += line.slice(5).trim();
  }
  if (!event) return null;
  try {
    const d = data ? JSON.parse(data) : {};
    switch (event) {
      case 'text': return { kind: 'text', text: d.text };
      case 'thinking': return { kind: 'thinking', text: d.text };
      case 'tool_call_start': return { kind: 'tool_call_start', id: d.id, name: d.name };
      case 'tool_ui': {
        const { tool_id, name, ...payload } = d;
        return { kind: 'tool_ui', tool_id, name, payload };
      }
      case 'tool_result': return { kind: 'tool_result', tool_id: d.tool_id, name: d.name, result: d.result };
      case 'assistant_msg': return { kind: 'assistant_msg', content: d.content };
      case 'done': return { kind: 'done', stop_reason: d.stop_reason };
      case 'error': return { kind: 'error', message: d.message };
      default: return null;
    }
  } catch {
    return null;
  }
}
