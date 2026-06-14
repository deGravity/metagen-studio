"""Anthropic-backed copilot chat with tool use.

Streams responses to the frontend over SSE. The chat agent runs in a
multi-turn loop on the server: each time the model emits a `tool_use`,
we execute it (geometry/sim runs against the shared program cache, edit
proposals fire-and-forget to the UI), append the result, and continue
the loop until the model returns a non-tool message or hits a guard.

The chat is a **stateless service**: clients pass full message history
on every request. State that the model sees about the workspace (code,
geometry stats, sim results) is in the `state` field and is rebuilt as
a system-prompt preamble each turn.

Set `METAGEN_ANTHROPIC_API_KEY` in the environment to enable.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from typing import Any, AsyncIterator, Optional

import numpy as np
from anthropic import AsyncAnthropic
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from metagen_dsl.docs import render_llm as _render_dsl_docs

from .models import ChatRequest, ChatStateContext
from .state import program_cache


router = APIRouter()


TOOLS: list[dict[str, Any]] = [
    {
        "name": "propose_edit",
        "description": (
            "Propose a complete replacement of the current code. The edit "
            "is shown to the user as a diff; they accept or reject it. "
            "You will not see their decision within this turn — if they "
            "accept, the next user message will reflect the new code. "
            "Always use this tool to make code changes; do not include "
            "code blocks in chat text. Pass the **full** new file."
        ),
        "input_schema": {
            "type": "object",
            "required": ["new_code", "summary"],
            "properties": {
                "new_code": {
                    "type": "string",
                    "description": "The complete new contents of the user's code.py.",
                },
                "summary": {
                    "type": "string",
                    "description": "One-sentence summary of the change.",
                },
            },
        },
    },
    {
        "name": "run_geometry",
        "description": (
            "Generate the voxelized geometry from the current code at the "
            "given resolution and TPMS optimizer mode. Returns volume "
            "fraction, fill fraction, and mesh vertex/triangle counts. "
            "The user's 3D viewer is updated with the resulting mesh."
        ),
        "input_schema": {
            "type": "object",
            "required": ["resolution"],
            "properties": {
                "resolution": {
                    "type": "integer", "minimum": 8, "maximum": 256,
                    "description": "Voxelization resolution. Multigrid-valid "
                                   "values for GPU sim are 17, 33, 49, 65, 97, 129.",
                },
                "tpms_optimizer_mode": {
                    "type": "string",
                    "enum": ["current", "global", "experimental"],
                    "description": "TPMS surface solver. 'current' is fast (BOBYQA only); "
                                   "'global' is deterministic but ~10x slower (adds ESCH "
                                   "global pre-search). Default: current.",
                },
            },
        },
    },
    {
        "name": "run_simulation",
        "description": (
            "Run periodic-homogenization on the current geometry. Returns "
            "the 6x6 stiffness matrix C, plus derived properties (E, K, G, "
            "ν, anisotropy indices). Auto-runs geometry if not yet cached."
        ),
        "input_schema": {
            "type": "object",
            "required": ["resolution"],
            "properties": {
                "resolution": {"type": "integer", "minimum": 8, "maximum": 256},
                "backend": {
                    "type": "string", "enum": ["auto", "gpu", "cpu"],
                    "description": "Solver backend. 'auto' uses GPU when valid, "
                                   "else CPU. Default: auto.",
                },
                "tpms_optimizer_mode": {
                    "type": "string",
                    "enum": ["current", "global", "experimental"],
                },
                "E": {"type": "number", "description": "Young's modulus of solid material. Default 1.0."},
                "nu": {"type": "number", "description": "Poisson's ratio. Default 0.45."},
            },
        },
    },
]


# ------------------------------------------------------------------------
# Tool implementations
# ------------------------------------------------------------------------

async def _tool_propose_edit(args: dict, _state: ChatStateContext) -> tuple[dict, dict]:
    """Returns (tool_result_for_model, ui_event_for_frontend)."""
    new_code = args.get('new_code', '')
    summary = args.get('summary', '')
    return (
        {'ok': True, 'note': 'Edit proposed. The user will accept or reject; '
                             'you will see their decision in the next turn.'},
        {'kind': 'proposal', 'new_code': new_code, 'summary': summary},
    )


def _geometry_summary(geo, code_hash: str, resolution: int, mode: str) -> dict:
    vox = np.asarray(geo.voxel_active_cells)
    return {
        'code_hash': code_hash,
        'resolution': resolution,
        'cell_resolution': int(geo.cell_resolution),
        'tpms_optimizer_mode': mode,
        'volume_fraction': float(geo.volume_fraction),
        'fill_fraction': float(vox.mean()),
        'n_active_voxels': int(vox.sum()),
        'n_total_voxels': int(vox.size),
    }


async def _tool_run_geometry(args: dict, state: ChatStateContext) -> tuple[dict, dict]:
    resolution = int(args.get('resolution', 33))
    mode = args.get('tpms_optimizer_mode', 'current')
    compiled = program_cache.get_or_compile(state.code)
    if compiled.error:
        return ({'ok': False, 'error': compiled.error}, {})
    t0 = time.perf_counter()
    geo = await asyncio.to_thread(
        compiled.structure.geometry,
        resolution=resolution, tpms_optimizer_mode=mode)
    elapsed = time.perf_counter() - t0
    summary = _geometry_summary(geo, compiled.code_hash, resolution, mode)
    summary['elapsed_s'] = elapsed
    return (
        {'ok': True, **summary},
        {'kind': 'geometry_done', **summary},
    )


async def _tool_run_simulation(args: dict, state: ChatStateContext) -> tuple[dict, dict]:
    resolution = int(args.get('resolution', 33))
    backend = args.get('backend', 'auto')
    mode = args.get('tpms_optimizer_mode', 'current')
    E = float(args.get('E', 1.0))
    nu = float(args.get('nu', 0.45))
    compiled = program_cache.get_or_compile(state.code)
    if compiled.error:
        return ({'ok': False, 'error': compiled.error}, {})
    # Prime geometry cache with the right mode first.
    await asyncio.to_thread(
        compiled.structure.geometry,
        resolution=resolution, tpms_optimizer_mode=mode)
    t0 = time.perf_counter()
    sim = await asyncio.to_thread(
        compiled.structure.simulate,
        resolution=resolution, backend=backend, E=E, nu=nu)
    elapsed = time.perf_counter() - t0
    C = np.asarray(sim.C_matrix, dtype=float).tolist()
    properties = {k: float(v) for k, v in sim.properties.items()}
    summary = {
        'code_hash': compiled.code_hash,
        'resolution': resolution,
        'tpms_optimizer_mode': mode,
        'backend_used': str(sim.solver_used),
        'C_matrix': C,
        'properties': properties,
        'elapsed_s': elapsed,
    }
    return ({'ok': True, **summary}, {'kind': 'sim_done', **summary})


_TOOL_DISPATCH = {
    'propose_edit': _tool_propose_edit,
    'run_geometry': _tool_run_geometry,
    'run_simulation': _tool_run_simulation,
}


# ------------------------------------------------------------------------
# Prompt construction
# ------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a copilot embedded in metaDSL Studio — a browser-based CAD tool
where users author Python programs that generate metamaterial unit cells
via the metaDSL (`from metagen import *`). The user has an editor, a 3D
viewer, and a simulation results panel; you help them author and refine
their `make_structure()` function.

Be terse. When a user asks for a change, use the `propose_edit` tool —
do not paste code into chat. When a user asks "what does this do" or
"why is X happening", explain in 2–4 sentences.

Use `run_geometry` and `run_simulation` when the user asks you to test
something or when you need data to answer accurately. Don't run sims
unprompted on every turn — they cost real GPU time.

Resolution guidance: 33 is a fast smoke (sub-second sim on GPU); 65 is
a typical working res; 97 is high-fidelity but takes minutes for dense
structures. Multigrid-valid (i.e. GPU-eligible) resolutions: 17, 33, 49,
65, 97, 129.
"""


# Auto-generated DSL API reference, rendered from docstrings in metagen_dsl.
# Cached at module level; ~6k tokens. Set METAGEN_DSL_DOCS_NO_CACHE=1 to
# re-render on every request (useful when iterating on docstrings).
_DSL_API_DOCS_CACHE: Optional[str] = None


def _get_dsl_api_docs() -> str:
    global _DSL_API_DOCS_CACHE
    if os.environ.get('METAGEN_DSL_DOCS_NO_CACHE') == '1':
        return _render_dsl_docs()
    if _DSL_API_DOCS_CACHE is None:
        _DSL_API_DOCS_CACHE = _render_dsl_docs()
    return _DSL_API_DOCS_CACHE


def _static_system_text() -> str:
    return (
        SYSTEM_PROMPT
        + "\n--- metaDSL API reference ---\n"
        + _get_dsl_api_docs()
    )


def _workspace_state_text(state: ChatStateContext) -> str:
    code_hash = _hash12(state.code)
    parts = ["--- workspace state ---",
             f"current code (hash {code_hash}):",
             "```python",
             state.code,
             "```"]

    if state.geometry_summary:
        gh = state.geometry_code_hash
        stale = (gh and gh != code_hash)
        parts.append(f"\nlast geometry run "
                     f"({'STALE — code edited since' if stale else 'current'}):")
        parts.append(json.dumps(state.geometry_summary, indent=2))

    if state.sim_summary:
        sh = state.sim_code_hash
        stale = (sh and sh != code_hash)
        parts.append(f"\nlast simulation run "
                     f"({'STALE — code edited since' if stale else 'current'}):")
        s = {k: v for k, v in state.sim_summary.items() if k != 'C_matrix'}
        parts.append(json.dumps(s, indent=2))

    if state.last_error:
        parts.append(f"\nlast error:\n{state.last_error}")

    return "\n".join(parts)


def _system_blocks(state: ChatStateContext) -> list[dict]:
    return [
        {
            'type': 'text',
            'text': _static_system_text(),
            'cache_control': {'type': 'ephemeral'},
        },
        {
            'type': 'text',
            'text': _workspace_state_text(state),
        },
    ]


def _hash12(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:12]


# ------------------------------------------------------------------------
# SSE streaming
# ------------------------------------------------------------------------

def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode('utf-8')


async def _agent_loop(req: ChatRequest) -> AsyncIterator[bytes]:
    api_key = os.environ.get('METAGEN_ANTHROPIC_API_KEY')
    if not api_key:
        yield _sse('error', {'message': 'METAGEN_ANTHROPIC_API_KEY not set on backend'})
        return

    client = AsyncAnthropic(api_key=api_key)
    system = _system_blocks(req.state)

    # Model expects content as list[dict] for assistant turns, but plain
    # str works for user messages — normalize.
    api_messages: list[dict] = []
    for m in req.messages:
        if isinstance(m.content, str):
            api_messages.append({'role': m.role, 'content': m.content})
        else:
            api_messages.append({'role': m.role, 'content': m.content})

    MAX_TURNS = 8
    for turn in range(MAX_TURNS):
        try:
            async with client.messages.stream(
                model=req.model, max_tokens=req.max_tokens,
                messages=api_messages, system=system, tools=TOOLS,
                extra_headers={'anthropic-beta': 'files-api-2025-04-14'},
            ) as stream:
                # Track in-flight content blocks so we can reconstruct
                # the final assistant message for the next turn.
                async for event in stream:
                    et = getattr(event, 'type', None)
                    if et == 'text':
                        yield _sse('text', {'text': event.text})
                    elif et == 'content_block_start':
                        block = event.content_block
                        if block.type == 'tool_use':
                            yield _sse('tool_call_start', {
                                'id': block.id, 'name': block.name,
                            })
                    elif et == 'message_stop':
                        pass

                final = await stream.get_final_message()

            # Append the assistant turn to history.
            assistant_blocks: list[dict] = []
            tool_uses: list[tuple[str, str, dict]] = []
            for block in final.content:
                if block.type == 'text':
                    assistant_blocks.append({'type': 'text', 'text': block.text})
                elif block.type == 'tool_use':
                    assistant_blocks.append({
                        'type': 'tool_use', 'id': block.id,
                        'name': block.name, 'input': block.input,
                    })
                    tool_uses.append((block.id, block.name, block.input))
            api_messages.append({'role': 'assistant', 'content': assistant_blocks})
            yield _sse('assistant_msg', {'content': assistant_blocks})

            if final.stop_reason != 'tool_use' or not tool_uses:
                yield _sse('done', {'stop_reason': final.stop_reason})
                return

            # Execute each tool call, emit UI events, and gather results
            # to feed back to the model.
            tool_result_blocks: list[dict] = []
            for tool_id, name, args in tool_uses:
                handler = _TOOL_DISPATCH.get(name)
                if handler is None:
                    result = {'error': f'unknown tool: {name}'}
                    ui = {}
                else:
                    try:
                        result, ui = await handler(args, req.state)
                    except Exception as exc:
                        result = {
                            'ok': False,
                            'error': f'{type(exc).__name__}: {exc}',
                            'traceback': traceback.format_exc(),
                        }
                        ui = {}
                if ui:
                    yield _sse('tool_ui', {'tool_id': tool_id, 'name': name, **ui})
                yield _sse('tool_result', {
                    'tool_id': tool_id, 'name': name, 'result': result,
                })
                tool_result_blocks.append({
                    'type': 'tool_result', 'tool_use_id': tool_id,
                    'content': json.dumps(result),
                })

            api_messages.append({'role': 'user', 'content': tool_result_blocks})

        except Exception as exc:
            yield _sse('error', {
                'message': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc(),
            })
            return

    yield _sse('error', {'message': f'tool-use loop exceeded {MAX_TURNS} turns'})


@router.post('/api/chat')
async def chat_endpoint(req: ChatRequest):
    if not os.environ.get('METAGEN_ANTHROPIC_API_KEY'):
        raise HTTPException(
            status_code=503,
            detail='Set METAGEN_ANTHROPIC_API_KEY in the backend env to enable chat.',
        )
    return StreamingResponse(_agent_loop(req), media_type='text/event-stream')


# Files API cap; documented limit on the upload side is much higher,
# but we keep a sanity ceiling so a bad client can't run the backend
# out of memory. Bump if you need to.
_UPLOAD_MAX_BYTES = 100 * 1024 * 1024  # 100 MB


@router.post('/api/chat/upload')
async def chat_upload(file: UploadFile = File(...)):
    """Proxy a single file upload to Anthropic's Files API.

    Returns the opaque file_id which the frontend then references in
    subsequent chat turns via a `document` block with `source.type=file`.
    Avoids re-shipping large PDFs on every turn.
    """
    api_key = os.environ.get('METAGEN_ANTHROPIC_API_KEY')
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail='Set METAGEN_ANTHROPIC_API_KEY in the backend env to enable uploads.',
        )

    content = await file.read()
    if len(content) > _UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f'file exceeds {_UPLOAD_MAX_BYTES // (1024 * 1024)} MB cap',
        )

    client = AsyncAnthropic(api_key=api_key)
    try:
        result = await client.beta.files.upload(
            file=(file.filename, content, file.content_type or 'application/octet-stream'),
        )
    except Exception as e:  # noqa: BLE001 — surface API errors as 502
        raise HTTPException(status_code=502, detail=f'Anthropic upload failed: {e}')

    return {
        'file_id': result.id,
        'filename': file.filename,
        'size': len(content),
        'media_type': file.content_type,
    }
