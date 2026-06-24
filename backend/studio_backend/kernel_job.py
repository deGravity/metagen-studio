"""Streaming geometry runner + job registry (studio transport layer).

The worker subprocess + non-streaming runners now live in
`metagen_domain.kernel_runner` (domain code). This module keeps the studio-only
streaming wrapper: spawn the domain worker, relay its stdout as SSE progress
events, allow cancel, and on success cache + log the result. The kernel is run
as a subprocess because `metagen_kernel.generate()` holds the GIL for the whole
solve; a child keeps the event loop responsive and killable.

Event sequence: job → progress* → (result | error | cancelled) → done.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass

import numpy as np

# worker subprocess machinery lives in the domain package now
from metagen_domain.kernel_runner import (
    _cleanup, _worker_cmd, kill_proc, parse_progress, set_pdeathsig,
)


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode('utf-8')


@dataclass
class Job:
    id: str
    proc: object = None
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
        return kill_proc(job.proc)

    def cancel_all(self):
        for job in list(self._jobs.values()):
            if job.proc is not None:
                job.status = 'cancelled'
                kill_proc(job.proc)


JOBS = JobRegistry()


async def stream_geometry(code: str, resolution: int, multistart_k: int,
                          tpms_optimizer_mode: str, code_hash: str,
                          session_id: str = None):
    """Async-generator of SSE byte chunks for one geometry job."""
    job_id = uuid.uuid4().hex[:12]
    fd, result_path = tempfile.mkstemp(suffix='.npz', prefix='studio_geo_')
    os.close(fd)

    proc = await asyncio.create_subprocess_exec(
        *_worker_cmd(result_path),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        preexec_fn=set_pdeathsig,   # auto-die if the backend dies
    )
    job = Job(id=job_id, proc=proc, started=time.time())
    JOBS.add(job)
    yield _sse('job', {'job_id': job_id})

    proc.stdin.write(json.dumps({
        'mode': 'geometry', 'code': code, 'resolution': resolution,
        'multistart_k': multistart_k,
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
        if proc.returncode is None:
            kill_proc(proc)
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
        from . import results_cache
        results_cache.put_geometry(code_hash, {**result, 'cached': True})
        if session_id:
            try:
                from . import sessions as _sess
                e1 = _sess.append_event(session_id, 'editor_snapshot',
                                        {'code': code, 'code_hash': code_hash, 'reason': 'run'})
                e2 = _sess.append_event(session_id, 'geometry_run', {
                    'code_hash': code_hash, 'resolution': resolution,
                    'tpms_mode': tpms_optimizer_mode, 'multistart_k': multistart_k,
                    'origin': 'button', 'stats': result['stats'],
                    'elapsed_s': result['elapsed_geometry_s']})
                ref = _sess.put_blob(session_id, {**result, 'cached': True})
                vf = result['stats']['volume_fraction']
                _sess.add_node(session_id, 'geometry',
                               f"ran geometry @{resolution} · vf {vf:.3f}",
                               {'code': code, 'code_hash': code_hash,
                                'geometry_ref': ref, 'sim_ref': None, 'chat_len': 0},
                               event_ids=[e1['id'], e2['id']])
            except Exception:  # noqa: BLE001
                pass
        yield _sse('result', result)
    except Exception as e:  # noqa: BLE001
        yield _sse('error', {'message': f'failed to read kernel result: {e}'})
    finally:
        _cleanup(result_path)
    yield _sse('done', {})
