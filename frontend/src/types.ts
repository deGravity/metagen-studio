export type TpmsMode = 'current' | 'global' | 'experimental';
export type SimBackend = 'auto' | 'gpu' | 'cpu';

export interface GeometryStats {
  cell_resolution: number;
  volume_fraction: number;
  n_vertices: number;
  n_triangles: number;
  n_active_voxels: number;
  n_total_voxels: number;
}

export interface ExecuteResponse {
  code_hash: string;
  resolution: number;
  tpms_optimizer_mode: TpmsMode;
  stats: GeometryStats;
  vertices_b64: string;
  triangles_b64: string;
  elapsed_geometry_s: number;
  cached: boolean;
}

export interface SimulateResponse {
  code_hash: string;
  resolution: number;
  tpms_optimizer_mode: TpmsMode;
  backend_used: string;
  C_matrix: number[][];
  properties: Record<string, number>;
  elapsed_sim_s: number;
  cached: boolean;
}

export interface InfoResponse {
  gpu_available: boolean;
  valid_gpu_resolutions: number[];
  cache_size: number;
  cache_keys: string[];
  chat_available: boolean;
}

// --- chat ---------------------------------------------------------------

export type AssistantBlock =
  | { type: 'text'; text: string }
  | { type: 'tool_use'; id: string; name: string; input: any };

export type ToolResultBlock = { type: 'tool_result'; tool_use_id: string; content: string };

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string | (AssistantBlock | ToolResultBlock)[];
}

export interface ChatStateContext {
  code: string;
  geometry_code_hash?: string | null;
  geometry_summary?: any;
  sim_code_hash?: string | null;
  sim_summary?: any;
  last_error?: string | null;
}

export interface PendingProposal {
  id: string;
  new_code: string;
  summary: string;
  status: 'pending' | 'applied' | 'discarded';
}

export type ChatEvent =
  | { kind: 'text'; text: string }
  | { kind: 'tool_call_start'; id: string; name: string }
  | { kind: 'tool_ui'; tool_id: string; name: string; payload: any }
  | { kind: 'tool_result'; tool_id: string; name: string; result: any }
  | { kind: 'assistant_msg'; content: AssistantBlock[] }
  | { kind: 'done'; stop_reason: string }
  | { kind: 'error'; message: string };

export interface MeshData {
  vertices: Float32Array;
  triangles: Uint32Array;
}
