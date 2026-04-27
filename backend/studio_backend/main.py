"""FastAPI entry for the studio backend.

Run from the workspace root:
    uvicorn studio_backend.main:app --reload --app-dir studio/backend
"""
from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --- workspace path setup -------------------------------------------------
# studio/backend/studio_backend/main.py → workspace root is parents[3].
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
    SimulateRequest, SimulateResponse, InfoResponse,
)
from .state import program_cache as _program_cache  # noqa: E402
from .chat import router as chat_router  # noqa: E402


app = FastAPI(title="metaDSL Studio Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(chat_router)

_GPU_VALID_DIMS = [16, 32, 48, 64, 96, 128]


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
                          tpms_optimizer_mode=req.tpms_optimizer_mode)
    elapsed = time.perf_counter() - t0

    verts, tris = _pick_mesh(geo)
    vox = np.asarray(geo.voxel_active_cells)
    n_active = int(vox.sum())

    return ExecuteResponse(
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
                    tpms_optimizer_mode=req.tpms_optimizer_mode)

    t0 = time.perf_counter()
    sim = struct.simulate(resolution=req.resolution, backend=req.backend,
                          E=req.E, nu=req.nu)
    elapsed = time.perf_counter() - t0

    return SimulateResponse(
        code_hash=compiled.code_hash,
        resolution=req.resolution,
        tpms_optimizer_mode=req.tpms_optimizer_mode,
        backend_used=str(sim.solver_used),
        C_matrix=np.asarray(sim.C_matrix, dtype=float).tolist(),
        properties={k: float(v) for k, v in sim.properties.items()},
        elapsed_sim_s=elapsed,
        cached=cached,
    )
