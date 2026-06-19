from typing import Optional, Literal
from pydantic import BaseModel, Field


TpmsMode = Literal['current', 'global', 'experimental']
SimBackend = Literal['auto', 'gpu', 'cpu']


class ExecuteRequest(BaseModel):
    code: str
    resolution: int = Field(default=33, ge=8, le=256)
    tpms_optimizer_mode: TpmsMode = 'current'
    session_id: Optional[str] = None   # log to this session if set


class GeometryStats(BaseModel):
    cell_resolution: int
    volume_fraction: float
    n_vertices: int
    n_triangles: int
    n_active_voxels: int
    n_total_voxels: int


class ExecuteResponse(BaseModel):
    code_hash: str
    resolution: int
    tpms_optimizer_mode: TpmsMode
    stats: GeometryStats
    vertices_b64: str  # Float32 [N,3]
    triangles_b64: str  # Uint32 [M,3]
    elapsed_geometry_s: float
    cached: bool


class SimulateRequest(BaseModel):
    code: str
    resolution: int = Field(default=33, ge=8, le=256)
    tpms_optimizer_mode: TpmsMode = 'current'
    backend: SimBackend = 'auto'
    E: float = 1.0
    nu: float = 0.45
    session_id: Optional[str] = None   # log to this session if set


class SimulateResponse(BaseModel):
    code_hash: str
    resolution: int
    tpms_optimizer_mode: TpmsMode
    backend_used: str  # 'gpu' | 'cpu' (resolved from 'auto')
    C_matrix: list[list[float]]  # 6x6
    properties: dict
    elapsed_sim_s: float
    cached: bool


class CodeRequest(BaseModel):
    code: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    traceback: Optional[str] = None


class InfoResponse(BaseModel):
    gpu_available: bool
    valid_gpu_resolutions: list[int]
    cache_size: int
    cache_keys: list[str]
    chat_available: bool


# --- chat ---------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal['user', 'assistant']
    # Free-form: assistant messages on the wire are reconstructed
    # client-side from streamed text + tool blocks; we serialize them
    # back to the API as a list of structured content blocks.
    content: list[dict] | str


class ChatStateContext(BaseModel):
    """Snapshot of editor + last-run artifacts attached to a chat turn."""
    code: str
    geometry_code_hash: Optional[str] = None
    geometry_summary: Optional[dict] = None
    sim_code_hash: Optional[str] = None
    sim_summary: Optional[dict] = None
    last_error: Optional[str] = None


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    state: ChatStateContext
    model: str = 'claude-opus-4-7'
    max_tokens: int = 4096
    session_id: Optional[str] = None       # log to this session if set
    thinking: Optional[bool] = None        # None = use config default


# --- sessions -----------------------------------------------------------

class SessionCreate(BaseModel):
    name: Optional[str] = None
    model: Optional[str] = None


class SessionRename(BaseModel):
    name: str


class CheckoutRequest(BaseModel):
    node_id: str


class SessionEventRequest(BaseModel):
    """Frontend-driven event/node (e.g. proposal accept/reject, edit)."""
    type: str
    payload: dict = {}
    make_node: bool = False
    kind: Optional[str] = None
    label: Optional[str] = None
    snapshot: Optional[dict] = None
