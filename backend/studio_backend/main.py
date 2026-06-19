"""FastAPI entry for the studio backend.

Two run modes share this module:
- Dev: `uvicorn studio_backend.main:app --reload --app-dir backend`
  (driven by run.sh / run.bat; vite serves the frontend on a separate port)
- Packaged: the `metagen-studio` console script (see cli.py) launches a
  single uvicorn that serves both /api routes and the bundled SPA from
  `_frontend_dist/` mounted at /.
"""
from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

# --- import metagen native packages ---------------------------------------
# Try the normal import path first (packaged install, where metagen-dsl /
# metagen-kernel / metagen-simulator are installed in the env). Fall back
# to a dev-checkout sys.path injection: when running from a metagen-dev
# workspace, the sibling submodules live two directories up from the
# studio_backend package.
try:
    import metagen_dsl
    from metagen_dsl import _backend as dsl_backend
except ImportError:
    _WORKSPACE = Path(__file__).resolve().parents[3]
    for sub in ('metagen-dsl', 'metagen-kernel/build',
                'metagen-simulator/build'):
        p = str(_WORKSPACE / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    import metagen_dsl  # noqa: E402
    from metagen_dsl import _backend as dsl_backend  # noqa: E402

from .models import (  # noqa: E402
    ExecuteRequest, ExecuteResponse, GeometryStats,
    SimulateRequest, SimulateResponse, InfoResponse, CodeRequest,
)
from .state import program_cache as _program_cache  # noqa: E402
from .execute import hash_code  # noqa: E402
from .chat import router as chat_router  # noqa: E402
from .kernel_job import stream_geometry, JOBS  # noqa: E402
from . import results_cache  # noqa: E402
from . import sessions as _sessions  # noqa: E402
from .config import cfg  # noqa: E402
from .models import (  # noqa: E402
    SessionCreate, SessionRename, CheckoutRequest, SessionEventRequest,
)


app = FastAPI(title="metaDSL Studio Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(chat_router)


@app.on_event('shutdown')
def _kill_inflight_jobs():
    # Don't let long kernel solves orphan when the server stops or reloads.
    JOBS.cancel_all()

_GPU_VALID_DIMS = [16, 32, 48, 64, 96, 128]

# The DSL's old `tpms_optimizer_mode` (current/global/experimental) was
# replaced by an integer `tpms_multistart_k` (best-of-K in-kernel multistart).
# Map the legacy UI modes onto K so the existing frontend keeps working:
#   current      -> 1  (single solve, production default)
#   global/exper. -> 8 (multistart)
_TPMS_MODE_K = {'current': 1, 'global': 8, 'experimental': 8}


def _multistart_k(mode: str) -> int:
    return _TPMS_MODE_K.get(mode, 1)


def _b64(arr: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode('ascii')


def _pick_mesh(geo) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, triangles) preferring thickened, falling back to voxel surface."""
    tv = np.asarray(geo.thickened_vertices)
    tt = np.asarray(geo.thickened_triangles)
    if tv.size > 0 and tt.size > 0:
        return tv, tt
    return np.asarray(geo.voxel_surface_vertices), np.asarray(geo.voxel_surface_triangles)


@app.get('/api/info', response_model=InfoResponse)
def info() -> InfoResponse:
    import os
    return InfoResponse(
        gpu_available=dsl_backend.gpu_available(),
        valid_gpu_resolutions=[d + 1 for d in _GPU_VALID_DIMS],
        cache_size=len(_program_cache),
        cache_keys=_program_cache.keys(),
        chat_available=bool(os.environ.get('METAGEN_ANTHROPIC_API_KEY')),
    )


@app.post('/api/execute', response_model=ExecuteResponse)
def execute(req: ExecuteRequest) -> ExecuteResponse:
    compiled = _program_cache.get_or_compile(req.code)
    if compiled.error:
        raise HTTPException(status_code=400, detail=compiled.error)
    struct = compiled.structure

    # The Structure has its own LRU; check whether this (resolution, mode)
    # is already cached so we can report `cached=True` in the response.
    pre_cache = struct._cache_get(  # type: ignore[attr-defined]
        ('geometry', req.resolution, req.tpms_optimizer_mode))
    cached = pre_cache is not None

    t0 = time.perf_counter()
    geo = struct.geometry(resolution=req.resolution,
                          tpms_multistart_k=_multistart_k(req.tpms_optimizer_mode))
    elapsed = time.perf_counter() - t0

    verts, tris = _pick_mesh(geo)
    vox = np.asarray(geo.voxel_active_cells)
    n_active = int(vox.sum())

    resp = ExecuteResponse(
        code_hash=compiled.code_hash,
        resolution=req.resolution,
        tpms_optimizer_mode=req.tpms_optimizer_mode,
        stats=GeometryStats(
            cell_resolution=int(geo.cell_resolution),
            volume_fraction=float(geo.volume_fraction),
            n_vertices=int(verts.shape[0]) if verts.ndim == 2 else 0,
            n_triangles=int(tris.shape[0]) if tris.ndim == 2 else 0,
            n_active_voxels=n_active,
            n_total_voxels=int(vox.size),
        ),
        vertices_b64=_b64(verts, np.float32),
        triangles_b64=_b64(tris, np.uint32),
        elapsed_geometry_s=elapsed,
        cached=cached,
    )
    results_cache.put_geometry(compiled.code_hash, {**resp.model_dump(), 'cached': True})
    return resp


@app.post('/api/execute/stream')
async def execute_stream(req: ExecuteRequest):
    """Streaming geometry: runs the kernel in a subprocess so the event loop
    stays responsive, emitting SSE progress events (job → progress* →
    result|error|cancelled → done). The 'result' event carries the same
    payload shape as ExecuteResponse."""
    return StreamingResponse(
        stream_geometry(req.code, req.resolution,
                        _multistart_k(req.tpms_optimizer_mode),
                        req.tpms_optimizer_mode, hash_code(req.code),
                        session_id=req.session_id),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.post('/api/jobs/{job_id}/cancel')
def cancel_job(job_id: str):
    """Kill a running geometry job's process group."""
    return {'ok': JOBS.cancel(job_id)}


@app.get('/api/jobs')
def list_jobs():
    return {'jobs': JOBS.list()}


@app.post('/api/simulate', response_model=SimulateResponse)
def simulate(req: SimulateRequest) -> SimulateResponse:
    compiled = _program_cache.get_or_compile(req.code)
    if compiled.error:
        raise HTTPException(status_code=400, detail=compiled.error)
    struct = compiled.structure

    sim_key = ('simulate', req.resolution, req.backend, req.E, req.nu)
    cached = struct._cache_get(sim_key) is not None  # type: ignore[attr-defined]

    # Honor the requested tpms_optimizer_mode for the underlying geometry.
    # Structure.simulate() doesn't expose that, but it does cache by
    # resolution alone — so call .geometry() first (which is mode-aware)
    # to populate the cache, then .simulate() will pick that up.
    struct.geometry(resolution=req.resolution,
                    tpms_multistart_k=_multistart_k(req.tpms_optimizer_mode))

    t0 = time.perf_counter()
    sim = struct.simulate(resolution=req.resolution, backend=req.backend,
                          E=req.E, nu=req.nu)
    elapsed = time.perf_counter() - t0

    resp = SimulateResponse(
        code_hash=compiled.code_hash,
        resolution=req.resolution,
        tpms_optimizer_mode=req.tpms_optimizer_mode,
        backend_used=str(sim.solver_used),
        C_matrix=np.asarray(sim.C_matrix, dtype=float).tolist(),
        properties={k: float(v) for k, v in sim.properties.items()},
        elapsed_sim_s=elapsed,
        cached=cached,
    )
    results_cache.put_sim(compiled.code_hash, {**resp.model_dump(), 'cached': True})
    if req.session_id:
        _log_sim_node(req.session_id, compiled.code_hash, req, resp, elapsed, origin='button')
    return resp


def _log_sim_node(sid, code_hash, req, resp, elapsed, origin):
    """Append editor_snapshot + sim_run events and a sim node to a session."""
    try:
        e1 = _sessions.append_event(sid, 'editor_snapshot',
                                    {'code': req.code, 'code_hash': code_hash, 'reason': 'sim'})
        e2 = _sessions.append_event(sid, 'sim_run', {
            'code_hash': code_hash, 'resolution': req.resolution,
            'backend': resp.backend_used, 'E': req.E, 'nu': req.nu,
            'origin': origin, 'C_matrix': resp.C_matrix,
            'properties': resp.properties, 'elapsed_s': round(elapsed, 3)})
        ref = _sessions.put_blob(sid, {**resp.model_dump(), 'cached': True})
        _sessions.add_node(sid, 'sim',
                           f"simulated @{req.resolution} · {resp.backend_used}",
                           {'code': req.code, 'code_hash': code_hash,
                            'geometry_ref': None, 'sim_ref': ref,
                            'chat_len': 0},
                           event_ids=[e1['id'], e2['id']])
    except Exception:  # noqa: BLE001 — logging must never break the request
        pass


# --- sessions -------------------------------------------------------------
@app.post('/api/sessions')
def create_session(req: SessionCreate):
    return _sessions.create_session(name=req.name, model=req.model)


@app.get('/api/sessions')
def list_sessions():
    return {'sessions': _sessions.list_sessions()}


@app.get('/api/sessions-usage')
def sessions_usage():
    """Per-session disk usage + total, for the cleanup view."""
    return _sessions.usage()


@app.post('/api/sessions/{sid}/prune')
def prune_session(sid: str):
    """Drop everything off the current branch (keep root→HEAD lineage)."""
    tree = _sessions.prune_to_branch(sid)
    if tree is None:
        raise HTTPException(status_code=404, detail='no such session')
    return tree


@app.post('/api/sessions/{sid}/delete-older')
def delete_older(sid: str):
    """Delete this session and all sessions last updated at/before it."""
    return {'deleted': _sessions.delete_older(sid)}


@app.get('/api/sessions/{sid}')
def get_session(sid: str):
    tree = _sessions.get_tree(sid)
    if tree is None:
        raise HTTPException(status_code=404, detail='no such session')
    return tree


@app.patch('/api/sessions/{sid}')
def rename_session(sid: str, req: SessionRename):
    tree = _sessions.set_name(sid, req.name, 'user')
    if tree is None:
        raise HTTPException(status_code=404, detail='no such session')
    return tree


@app.delete('/api/sessions/{sid}')
def delete_session(sid: str):
    return {'ok': _sessions.delete_session(sid)}


@app.get('/api/sessions/{sid}/events')
def session_events(sid: str, node: str = None, types: str = None):
    tset = set(types.split(',')) if types else None
    return {'events': list(_sessions.read_events(sid, node_id=node, types=tset))}


@app.get('/api/sessions/{sid}/node/{node_id}')
def session_node(sid: str, node_id: str):
    tree = _sessions.get_tree(sid)
    if tree is None or node_id not in tree['nodes']:
        raise HTTPException(status_code=404, detail='no such node')
    node = tree['nodes'][node_id]
    snap = dict(node['snapshot'])
    # resolve referenced result blobs for an instant restore
    snap['geometry'] = _sessions.get_blob(sid, snap['geometry_ref']) if snap.get('geometry_ref') else None
    snap['sim'] = _sessions.get_blob(sid, snap['sim_ref']) if snap.get('sim_ref') else None
    return {'node': node, 'snapshot': snap,
            'events': list(_sessions.read_events(sid, node_id=node_id))}


@app.post('/api/sessions/{sid}/checkout')
def session_checkout(sid: str, req: CheckoutRequest):
    node = _sessions.checkout(sid, req.node_id)
    if node is None:
        raise HTTPException(status_code=404, detail='no such node')
    snap = dict(node['snapshot'])
    snap['geometry'] = _sessions.get_blob(sid, snap['geometry_ref']) if snap.get('geometry_ref') else None
    snap['sim'] = _sessions.get_blob(sid, snap['sim_ref']) if snap.get('sim_ref') else None
    return {'node': node, 'snapshot': snap}


@app.post('/api/sessions/{sid}/event')
def session_event(sid: str, req: SessionEventRequest):
    """Frontend-driven event (+ optional node), e.g. proposal accept/reject."""
    ev = _sessions.append_event(sid, req.type, req.payload)
    node = None
    if req.make_node:
        node = _sessions.add_node(sid, req.kind or req.type, req.label or req.type,
                                  req.snapshot or {'code': None, 'code_hash': None,
                                                   'geometry_ref': None, 'sim_ref': None,
                                                   'chat_len': 0},
                                  event_ids=[ev['id']])
    return {'event': ev, 'node': node}
def results_cached(req: CodeRequest):
    """Latest cached geometry/sim for a given program (by code hash), so an
    accepted proposal can reuse what the copilot already computed."""
    return results_cache.get(hash_code(req.code))


# --- bundled SPA mount (packaged mode) ------------------------------------
# Must come AFTER all /api routes so the catch-all static mount at "/"
# doesn't shadow them. In dev mode this directory doesn't exist and we
# leave the frontend to vite on :5173.
_FRONTEND_DIST = Path(__file__).parent / '_frontend_dist'
if _FRONTEND_DIST.is_dir():
    # html=True turns 404s into index.html so client-side routing works.
    app.mount('/', StaticFiles(directory=_FRONTEND_DIST, html=True), name='frontend')
