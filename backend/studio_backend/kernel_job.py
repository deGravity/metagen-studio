"""Run the geometry kernel in a subprocess, stream its progress, allow cancel.

Why a subprocess: `metagen_kernel.generate()` is a C++ call that holds the
GIL for the whole (potentially minutes-long) TPMS solve. Running it inline in
a sync route blocks the asyncio event loop, so health checks and other
requests stall. Spawning it as a child process keeps the backend responsive,
lets us read the kernel's stdout for live progress, and lets us kill it.

Dual role (mirrors the reprocessing orchestrator's worker/parent split):
  --worker <result_path> : read {code, resolution, multistart_k} JSON from
      stdin, compile + run Structure.geometry(), write mesh + stats to
      <result_path> (npz). The kernel's own stdout (progress) flows through;
      we add a STUDIO_RESULT_READY sentinel on success or STUDIO_ERROR <b64>
      on a compile/run failure.
  (imported)             : stream_geometry() async-generates SSE byte chunks
      from a spawned worker; JOBS.cancel(job_id) kills the process group.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import signal
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)  # .../backend


# --------------------------------------------------------------------------- #
# progress parsing — turn raw kernel stdout lines into structured events
# --------------------------------------------------------------------------- #
_RE_ATTEMPT = re.compile(r'^E\s+(\d+):\s*(.*)$')


def parse_progress(line: str) -> dict | None:
    """Map a kernel stdout line to a progress event, or None to ignore it."""
    s = line.strip()
    if not s:
        return None
    m = _RE_ATTEMPT.match(s)
    if m:
        return {'phase': 'multistart', 'attempt': int(m.group(1)), 'detail': s}
    if 'BOBYQA' in s or 'local optimization' in s:
        return {'phase': 'local_opt', 'detail': s}
    if 'ESCH' in s or 'global optimization' in s or 'DIRECT' in s:
        return {'phase': 'global_opt', 'detail': s}
    if s.startswith('MinF') or 'MinF:' in s:
        return {'phase': 'solve', 'detail': s}
    if s.startswith('#vertices in remeshed') or s.startswith('Tgt len'):
        return {'phase': 'remesh', 'detail': s}
    return None


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode('utf-8')


def _set_pdeathsig():
    """preexec in the worker child: ask the kernel to SIGTERM us if the
    parent (backend) dies, so a --reload / crash / Ctrl-C can't leave the
    kernel solve orphaned and burning CPU. Linux-only; best-effort."""
    try:
        import ctypes
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL('libc.so.6', use_errno=True).prctl(
            PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0)
    except Exception:  # noqa: BLE001
        pass


def _kill_proc(proc) -> bool:
    """SIGTERM a worker by pid. The kernel runs in-process (no grandchildren),
    so terminating the single process kills the solve. Returns False if it was
    already gone."""
    try:
        proc.terminate()
    except ProcessLookupError:
        return False
    return True


# --------------------------------------------------------------------------- #
# parent side: job registry + streaming runner
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    id: str
    proc: object = None          # asyncio subprocess
    status: str = 'running'      # running | done | error | cancelled
    started: float = 0.0


class JobRegistry:
    def __init__(self):
        self._jobs: dict[str, Job] = {}

    def add(self, job: Job):
        self._jobs[job.id] = job

    def remove(self, jid: str):
        self._jobs.pop(jid, None)

    def list(self) -> list[dict]:
        return [{'id': j.id, 'status': j.status,
                 'elapsed': round(time.time() - j.started, 1)}
                for j in self._jobs.values()]

    def cancel(self, jid: str) -> bool:
        job = self._jobs.get(jid)
        if not job or job.proc is None:
            return False
        job.status = 'cancelled'
        return _kill_proc(job.proc)

    def cancel_all(self):
        """Kill every in-flight job. Called on graceful server shutdown; the
        worker's PR_SET_PDEATHSIG is the backstop for non-graceful death."""
        for job in list(self._jobs.values()):
            if job.proc is not None:
                job.status = 'cancelled'
                _kill_proc(job.proc)


JOBS = JobRegistry()


async def stream_geometry(code: str, resolution: int, multistart_k: int,
                          tpms_optimizer_mode: str, code_hash: str):
    """Async-generator of SSE byte chunks for one geometry job.

    Event sequence: job → progress* → (result | error | cancelled) → done.
    """
    job_id = uuid.uuid4().hex[:12]
    fd, result_path = tempfile.mkstemp(suffix='.npz', prefix='studio_geo_')
    os.close(fd)

    cmd = ['stdbuf', '-oL', '-eL', sys.executable, '-u',
           '-m', 'studio_backend.kernel_job', '--worker', result_path]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=_BACKEND_DIR,
        preexec_fn=_set_pdeathsig,   # auto-die if the backend dies
    )
    job = Job(id=job_id, proc=proc, started=time.time())
    JOBS.add(job)
    yield _sse('job', {'job_id': job_id})

    # hand the worker its payload
    proc.stdin.write(json.dumps({
        'code': code, 'resolution': resolution, 'multistart_k': multistart_k,
    }).encode('utf-8'))
    await proc.stdin.drain()
    proc.stdin.close()

    error_b64 = None
    attempts = 0
    try:
        while True:
            try:
                raw = await asyncio.wait_for(proc.stdout.readline(), timeout=1.0)
            except asyncio.TimeoutError:
                # no line in the last second — emit a heartbeat so the UI
                # knows we're alive even during quiet phases (e.g. CGAL).
                yield _sse('progress', {
                    'phase': 'working', 'attempt': attempts,
                    'elapsed': round(time.time() - job.started, 1)})
                continue
            if not raw:
                break
            s = raw.decode('utf-8', 'replace').rstrip('\n')
            if s.startswith('STUDIO_RESULT_READY'):
                continue
            if s.startswith('STUDIO_ERROR '):
                error_b64 = s[len('STUDIO_ERROR '):]
                continue
            ev = parse_progress(s)
            if ev:
                if ev['phase'] == 'multistart':
                    attempts = max(attempts, ev['attempt'])
                ev['elapsed'] = round(time.time() - job.started, 1)
                yield _sse('progress', ev)
        rc = await proc.wait()
    finally:
        # If the generator is torn down before the worker finished (client
        # disconnect / server shutdown), the loop above exits via GeneratorExit
        # and proc is still running — kill it so the solve doesn't orphan.
        if proc.returncode is None:
            _kill_proc(proc)
        JOBS.remove(job_id)

    if job.status == 'cancelled':
        yield _sse('cancelled', {'job_id': job_id})
        _cleanup(result_path)
        return
    if error_b64 is not None:
        msg = base64.b64decode(error_b64).decode('utf-8', 'replace')
        yield _sse('error', {'message': msg})
        _cleanup(result_path)
        return
    if rc != 0:
        yield _sse('error', {'message': f'kernel worker exited with code {rc}'})
        _cleanup(result_path)
        return

    try:
        d = np.load(result_path)
        verts = d['verts']
        tris = d['tris']
        result = {
            'code_hash': code_hash,
            'resolution': resolution,
            'tpms_optimizer_mode': tpms_optimizer_mode,
            'stats': {
                'cell_resolution': int(d['cell_resolution']),
                'volume_fraction': float(d['volume_fraction']),
                'n_vertices': int(verts.shape[0]) if verts.ndim == 2 else 0,
                'n_triangles': int(tris.shape[0]) if tris.ndim == 2 else 0,
                'n_active_voxels': int(d['n_active']),
                'n_total_voxels': int(d['n_total']),
            },
            'vertices_b64': base64.b64encode(
                np.ascontiguousarray(verts, np.float32).tobytes()).decode('ascii'),
            'triangles_b64': base64.b64encode(
                np.ascontiguousarray(tris, np.uint32).tobytes()).decode('ascii'),
            'elapsed_geometry_s': round(time.time() - job.started, 2),
            'cached': False,
        }
        yield _sse('result', result)
    except Exception as e:  # noqa: BLE001
        yield _sse('error', {'message': f'failed to read kernel result: {e}'})
    finally:
        _cleanup(result_path)
    yield _sse('done', {})


def _cleanup(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# non-streaming runners (used by the copilot tools so they don't block the
# event loop — same subprocess, just run to completion and return the result)
# --------------------------------------------------------------------------- #
async def _run_worker(payload: dict, result_path: str):
    """Spawn the worker, feed it `payload`, await completion. Raises
    RuntimeError on a compile/run failure or nonzero exit."""
    cmd = ['stdbuf', '-oL', '-eL', sys.executable, '-u',
           '-m', 'studio_backend.kernel_job', '--worker', result_path]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=_BACKEND_DIR,
        preexec_fn=_set_pdeathsig,
    )
    job = Job(id=uuid.uuid4().hex[:12], proc=proc, started=time.time())
    JOBS.add(job)
    try:
        out, _ = await proc.communicate(json.dumps(payload).encode('utf-8'))
    finally:
        if proc.returncode is None:
            _kill_proc(proc)
        JOBS.remove(job.id)
    text = out.decode('utf-8', 'replace')
    for ln in text.splitlines():
        if ln.startswith('STUDIO_ERROR '):
            raise RuntimeError(base64.b64decode(
                ln[len('STUDIO_ERROR '):]).decode('utf-8', 'replace'))
    if proc.returncode != 0:
        raise RuntimeError(f'kernel worker exited {proc.returncode}: {text[-400:]}')


async def run_geometry_result(code: str, resolution: int, multistart_k: int) -> dict:
    fd, rp = tempfile.mkstemp(suffix='.npz', prefix='studio_geo_')
    os.close(fd)
    try:
        await _run_worker({'mode': 'geometry', 'code': code,
                           'resolution': resolution, 'multistart_k': multistart_k}, rp)
        d = np.load(rp)
        verts = np.ascontiguousarray(d['verts'], np.float32)
        tris = np.ascontiguousarray(d['tris'], np.uint32)
        return {
            'cell_resolution': int(d['cell_resolution']),
            'volume_fraction': float(d['volume_fraction']),
            'n_active_voxels': int(d['n_active']),
            'n_total_voxels': int(d['n_total']),
            'n_vertices': int(verts.shape[0]) if verts.ndim == 2 else 0,
            'n_triangles': int(tris.shape[0]) if tris.ndim == 2 else 0,
            'vertices_b64': base64.b64encode(verts.tobytes()).decode('ascii'),
            'triangles_b64': base64.b64encode(tris.tobytes()).decode('ascii'),
        }
    finally:
        _cleanup(rp)


async def run_sim_result(code: str, resolution: int, multistart_k: int,
                         backend: str, E: float, nu: float) -> dict:
    fd, rp = tempfile.mkstemp(suffix='.json', prefix='studio_sim_')
    os.close(fd)
    try:
        await _run_worker({'mode': 'simulate', 'code': code,
                           'resolution': resolution, 'multistart_k': multistart_k,
                           'backend': backend, 'E': E, 'nu': nu}, rp)
        with open(rp) as f:
            return json.load(f)  # {C_matrix, properties, solver_used}
    finally:
        _cleanup(rp)


# --------------------------------------------------------------------------- #
# worker side: one geometry generation, mesh + stats to npz
# --------------------------------------------------------------------------- #
def _worker_main(result_path: str) -> int:
    # dev-checkout sys.path injection (mirrors main.py): make the sibling
    # metagen packages importable when not pip-installed.
    try:
        import metagen_dsl  # noqa: F401
    except ImportError:
        ws = Path(__file__).resolve().parents[3]
        for sub in ('metagen-dsl', 'metagen-kernel/build',
                    'metagen-simulator/build'):
            p = str(ws / sub)
            if p not in sys.path:
                sys.path.insert(0, p)

    payload = json.loads(sys.stdin.read())
    mode = payload.get('mode', 'geometry')
    code = payload['code']
    resolution = int(payload['resolution'])
    multistart_k = int(payload['multistart_k'])

    def _emit_error(msg: str) -> None:
        sys.stdout.write('STUDIO_ERROR ' + base64.b64encode(
            msg.encode('utf-8')).decode('ascii') + '\n')
        sys.stdout.flush()

    from studio_backend.execute import _compile, hash_code
    compiled = _compile(code, hash_code(code))
    if compiled.error:
        _emit_error(compiled.error)
        return 3

    import traceback
    try:
        if mode == 'simulate':
            backend = payload.get('backend', 'auto')
            E = float(payload.get('E', 1.0))
            nu = float(payload.get('nu', 0.45))
            # prime geometry at the requested multistart_k, then simulate
            compiled.structure.geometry(
                resolution=resolution, tpms_multistart_k=multistart_k)
            sim = compiled.structure.simulate(
                resolution=resolution, backend=backend, E=E, nu=nu)
            with open(result_path, 'w') as f:
                json.dump({
                    'C_matrix': np.asarray(sim.C_matrix, dtype=float).tolist(),
                    'properties': {k: float(v) for k, v in sim.properties.items()},
                    'solver_used': str(sim.solver_used),
                }, f)
        else:
            geo = compiled.structure.geometry(
                resolution=resolution, tpms_multistart_k=multistart_k)
            tv = np.asarray(geo.thickened_vertices)
            tt = np.asarray(geo.thickened_triangles)
            if not (tv.size and tt.size):
                tv = np.asarray(geo.voxel_surface_vertices)
                tt = np.asarray(geo.voxel_surface_triangles)
            vox = np.asarray(geo.voxel_active_cells)
            np.savez(result_path,
                     verts=np.ascontiguousarray(tv, np.float32),
                     tris=np.ascontiguousarray(tt, np.uint32),
                     cell_resolution=int(geo.cell_resolution),
                     volume_fraction=float(geo.volume_fraction),
                     n_active=int(vox.sum()),
                     n_total=int(vox.size))
    except Exception:  # noqa: BLE001
        _emit_error(traceback.format_exc())
        return 4

    sys.stdout.write('STUDIO_RESULT_READY\n')
    sys.stdout.flush()
    return 0


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--worker':
        sys.exit(_worker_main(sys.argv[2]))
    sys.stderr.write('usage: python -m studio_backend.kernel_job --worker <result_path>\n')
    sys.exit(2)
