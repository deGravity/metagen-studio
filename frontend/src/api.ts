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

export async function executeCode(
  code: string, resolution: number, tpms_optimizer_mode: TpmsMode,
): Promise<ExecuteResponse> {
  return postJson('/execute', { code, resolution, tpms_optimizer_mode });
}

export async function simulate(
  code: string, resolution: number, tpms_optimizer_mode: TpmsMode,
  backend: SimBackend, E = 1.0, nu = 0.45,
): Promise<SimulateResponse> {
  return postJson('/simulate', { code, resolution, tpms_optimizer_mode, backend, E, nu });
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
): AsyncGenerator<ChatEvent> {
  const r = await fetch(`${API}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, state, model }),
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
