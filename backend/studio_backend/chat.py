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
from .config import cfg
from . import sessions as _sess
from .copilot import CopilotEngine, Tool, ToolEnv, ToolOutcome, ToolRegistry
from .copilot.providers.anthropic import AnthropicProvider
from .copilot.types import (
    SystemBlock, Msg, Text, ToolCall, ToolResult, Thinking, Raw, ToolDef,
    TextDelta, ThinkingDelta, ToolCallStarted, AssistantMessage, ToolResultEvent,
    Artifact, Done, ErrorEvent,
)


def _to_msg(m) -> Msg:
    """ChatMessage (frontend content: str | list[dict]) -> normalized Msg.
    Known block types are normalized; anything else (image/file attachments)
    rides through as Raw so it reaches the Anthropic adapter verbatim."""
    c = m.content
    if isinstance(c, str):
        return Msg(m.role, [Text(c)])
    parts = []
    for b in c:
        if not isinstance(b, dict):
            continue
        t = b.get('type')
        if t == 'text':
            parts.append(Text(b.get('text', '')))
        elif t == 'tool_use':
            parts.append(ToolCall(id=b['id'], name=b['name'], input=b.get('input', {})))
        elif t == 'tool_result':
            parts.append(ToolResult(tool_call_id=b['tool_use_id'],
                                    content=b.get('content', ''),
                                    is_error=bool(b.get('is_error'))))
        elif t == 'thinking':
            parts.append(Thinking(text=b.get('thinking', ''), signature=b.get('signature')))
        elif t == 'redacted_thinking':
            parts.append(Thinking(redacted=b.get('data', '')))
        else:
            parts.append(Raw(block=b))
    return Msg(m.role, parts)


def _wrap_tool(handler):
    """Adapt an existing (args, state)->(result, ui) tool to the engine's
    (args, ToolEnv)->ToolOutcome contract; non-empty ui becomes an Artifact."""
    async def h(args, env):
        result, ui = await handler(args, env.get('state'))
        arts = [Artifact(kind=ui.get('kind', 'tool_ui'), data=ui)] if ui else []
        err = isinstance(result, dict) and result.get('ok') is False
        return ToolOutcome(result=result, artifacts=arts, error=err)
    return h


def _build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for t in TOOLS:
        h = _TOOL_DISPATCH.get(t['name'])
        if h is None:
            continue
        reg.register(Tool(
            defn=ToolDef(name=t['name'], description=t['description'],
                         schema=t['input_schema']),
            handler=_wrap_tool(h)))
    return reg


def _turn_label(req: ChatRequest) -> str:
    last_user = next((m for m in reversed(req.messages) if m.role == 'user'), None)
    txt = ''
    if last_user is not None:
        c = last_user.content
        if isinstance(c, str):
            txt = c
        elif isinstance(c, list):
            txt = ' '.join(b.get('text', '') for b in c
                           if isinstance(b, dict) and b.get('type') == 'text')
    txt = (txt or '').strip().replace('\n', ' ')
    return ('copilot: ' + txt[:60]) if txt else 'chat turn'


_AUTONAME_TASKS: set = set()


def _schedule_autoname(sid: str) -> None:
    if not sid:
        return
    try:
        t = asyncio.create_task(_maybe_autoname(sid))
        _AUTONAME_TASKS.add(t)
        t.add_done_callback(_AUTONAME_TASKS.discard)
    except RuntimeError:
        pass  # no running loop (shouldn't happen under uvicorn)


async def _maybe_autoname(sid: str) -> None:
    """Out-of-band: summarize the chat into a short title via a small model.
    Never overwrites a user-set name; runs on a turn cadence."""
    try:
        if not bool(cfg('copilot.autoname.enabled', True)):
            return
        tree = _sess.get_tree(sid)
        if tree is None or tree.get('name_source') == 'user':
            return
        every = int(cfg('copilot.autoname.every_turns', 5))
        n_turns = sum(1 for n in tree['nodes'].values()
                      if n.get('kind') == 'assistant_turn')
        if not (n_turns == 1 or (every > 0 and n_turns % every == 0)):
            return

        lines: list[str] = []
        for ev in _sess.read_events(sid, types={'user_message', 'copilot_response'}):
            p = ev['payload']
            if ev['type'] == 'user_message':
                c = p.get('content')
                txt = (c if isinstance(c, str)
                       else ' '.join(b.get('text', '') for b in c
                                     if isinstance(b, dict) and b.get('type') == 'text')
                       if isinstance(c, list) else '')
                if txt.strip():
                    lines.append('User: ' + txt.strip()[:300])
            else:
                for b in p.get('content_blocks', []):
                    if b.get('type') == 'text' and b.get('text'):
                        lines.append('Assistant: ' + b['text'].strip()[:300])
                        break
        transcript = '\n'.join(lines[-12:])
        if not transcript.strip():
            return

        api_key = os.environ.get('METAGEN_ANTHROPIC_API_KEY')
        if not api_key:
            return
        model = cfg('copilot.autoname.model', 'claude-haiku-4-5-20251001')
        client = AsyncAnthropic(api_key=api_key)
        resp = await client.messages.create(
            model=model, max_tokens=24,
            system=("You title metamaterial-design chat sessions. Reply with ONLY a "
                    "concise title of at most 6 words — no quotes, no trailing "
                    "punctuation."),
            messages=[{'role': 'user', 'content': transcript}])
        title = ''.join(getattr(b, 'text', '') for b in resp.content
                        if getattr(b, 'type', None) == 'text').strip().strip('"').strip()
        if title:
            _sess.set_name(sid, title[:60], 'auto')
    except Exception:  # noqa: BLE001 — naming is best-effort
        pass


def _compact_ui_for_log(sid, ui: dict) -> dict:
    """Spill mesh b64 to a dedup'd blob so events.jsonl stays lean."""
    if not ui or not ('vertices_b64' in ui or 'triangles_b64' in ui):
        return ui
    u = dict(ui)
    if sid:
        try:
            u['mesh_ref'] = _sess.put_blob(sid, {
                'vertices_b64': u.get('vertices_b64'),
                'triangles_b64': u.get('triangles_b64')})
        except Exception:  # noqa: BLE001
            pass
    u.pop('vertices_b64', None)
    u.pop('triangles_b64', None)
    return u


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
            "Generate the voxelized geometry at the given resolution and TPMS "
            "optimizer mode. Returns volume fraction, fill fraction, and mesh "
            "vertex/triangle counts. By default runs the user's current editor "
            "code (and updates their 3D viewer). Pass `code` to instead test a "
            "candidate program privately, in the background, without touching "
            "the editor or viewer — use this to try an idea before proposing it."
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
                "code": {
                    "type": "string",
                    "description": "Optional. A candidate make_structure() program to run "
                                   "INSTEAD of the user's editor code. Use this to test an "
                                   "idea in your own reasoning before proposing it via "
                                   "propose_edit. Omit to run the user's current code.",
                },
            },
        },
    },
    {
        "name": "run_simulation",
        "description": (
            "Run periodic-homogenization on the geometry. Returns the 6x6 "
            "stiffness matrix C, plus derived properties (E, K, G, ν, anisotropy "
            "indices). Auto-runs geometry first. By default uses the user's "
            "current editor code; pass `code` to test a candidate program "
            "privately (in the background) without touching the editor/viewer."
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
                "code": {
                    "type": "string",
                    "description": "Optional. A candidate make_structure() program to "
                                   "simulate INSTEAD of the user's editor code, privately "
                                   "in the background. Omit to use the user's current code.",
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
    k = 1 if mode == 'current' else 8
    code = args.get('code') or state.code
    candidate = bool(args.get('code')) and args['code'] != state.code
    compiled = program_cache.get_or_compile(code)
    if compiled.error:
        return ({'ok': False, 'error': compiled.error, 'ran': 'candidate' if candidate else 'editor'}, {})
    # Run the kernel in a subprocess so the chat SSE stream (on the event
    # loop) keeps flowing — an in-process solve holds the GIL and would drop
    # the chat connection ("network error").
    from .kernel_job import run_geometry_result
    t0 = time.perf_counter()
    try:
        g = await run_geometry_result(code, resolution, k)
    except Exception as e:  # noqa: BLE001
        return ({'ok': False, 'error': str(e),
                 'ran': 'candidate' if candidate else 'editor'}, {})
    elapsed = time.perf_counter() - t0
    summary = {
        'code_hash': compiled.code_hash,
        'resolution': resolution,
        'cell_resolution': g['cell_resolution'],
        'tpms_optimizer_mode': mode,
        'volume_fraction': g['volume_fraction'],
        'fill_fraction': (g['n_active_voxels'] / g['n_total_voxels']
                          if g['n_total_voxels'] else 0.0),
        'n_active_voxels': g['n_active_voxels'],
        'n_total_voxels': g['n_total_voxels'],
        'elapsed_s': elapsed,
        'ran': 'candidate' if candidate else 'editor',
    }
    # Cache the full result (incl mesh) keyed by this code's hash, so that if
    # the user accepts a proposal of this code we can reuse it immediately.
    from . import results_cache
    results_cache.put_geometry(compiled.code_hash, {
        'code_hash': compiled.code_hash, 'resolution': resolution,
        'tpms_optimizer_mode': mode,
        'stats': {'cell_resolution': g['cell_resolution'],
                  'volume_fraction': g['volume_fraction'],
                  'n_vertices': g['n_vertices'], 'n_triangles': g['n_triangles'],
                  'n_active_voxels': g['n_active_voxels'],
                  'n_total_voxels': g['n_total_voxels']},
        'vertices_b64': g['vertices_b64'], 'triangles_b64': g['triangles_b64'],
        'elapsed_geometry_s': round(elapsed, 3), 'cached': True,
    })
    # Candidate (background) runs do NOT touch the user's viewer — return an
    # empty ui event so nothing is rendered. Editor runs update the viewer,
    # carrying the mesh so it refreshes without a second (blocking) refetch.
    if candidate:
        return ({'ok': True, **summary}, {})
    ui = {'kind': 'geometry_done', **summary,
          'n_vertices': g['n_vertices'], 'n_triangles': g['n_triangles'],
          'vertices_b64': g['vertices_b64'], 'triangles_b64': g['triangles_b64']}
    return ({'ok': True, **summary}, ui)


async def _tool_run_simulation(args: dict, state: ChatStateContext) -> tuple[dict, dict]:
    resolution = int(args.get('resolution', 33))
    backend = args.get('backend', 'auto')
    mode = args.get('tpms_optimizer_mode', 'current')
    k = 1 if mode == 'current' else 8
    E = float(args.get('E', 1.0))
    nu = float(args.get('nu', 0.45))
    code = args.get('code') or state.code
    candidate = bool(args.get('code')) and args['code'] != state.code
    compiled = program_cache.get_or_compile(code)
    if compiled.error:
        return ({'ok': False, 'error': compiled.error,
                 'ran': 'candidate' if candidate else 'editor'}, {})
    from .kernel_job import run_sim_result
    t0 = time.perf_counter()
    try:
        s = await run_sim_result(code, resolution, k, backend, E, nu)
    except Exception as e:  # noqa: BLE001
        return ({'ok': False, 'error': str(e),
                 'ran': 'candidate' if candidate else 'editor'}, {})
    elapsed = time.perf_counter() - t0
    summary = {
        'code_hash': compiled.code_hash,
        'resolution': resolution,
        'tpms_optimizer_mode': mode,
        'backend_used': s['solver_used'],
        'C_matrix': s['C_matrix'],
        'properties': s['properties'],
        'elapsed_s': elapsed,
        'ran': 'candidate' if candidate else 'editor',
    }
    from . import results_cache
    results_cache.put_sim(compiled.code_hash, {
        'code_hash': compiled.code_hash, 'resolution': resolution,
        'tpms_optimizer_mode': mode, 'backend_used': s['solver_used'],
        'C_matrix': s['C_matrix'], 'properties': s['properties'],
        'elapsed_sim_s': round(elapsed, 3), 'cached': True,
    })
    # Background candidate sims don't overwrite the user's results panel.
    if candidate:
        return ({'ok': True, **summary}, {})
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

You can test your OWN ideas before proposing them: pass a candidate
`make_structure()` program as the `code` argument to `run_geometry` /
`run_simulation`. That runs your candidate privately, in the background,
WITHOUT changing the user's editor or 3D viewer — so you can draft a
variant, measure it (volume fraction, moduli), iterate, and only then
`propose_edit` the version you're confident in. Omit `code` to run the
user's current editor program (this DOES update their viewer). The tool
result's `ran` field ("candidate" vs "editor") tells you which program was
actually evaluated — check it so you never confuse a background test with
the user's live code.

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


def _hash12(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:12]


# ------------------------------------------------------------------------
# SSE streaming
# ------------------------------------------------------------------------

def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode('utf-8')


async def _agent_loop(req: ChatRequest) -> AsyncIterator[bytes]:
    """Drive the provider-neutral CopilotEngine and re-emit its normalized
    event stream as the studio's SSE protocol. The engine owns the model
    calls + tool loop; this function owns transport (SSE) and session logging.
    The wire protocol and events.jsonl layout are unchanged from the previous
    inline-Anthropic implementation — only the plumbing moved into the engine."""
    api_key = os.environ.get('METAGEN_ANTHROPIC_API_KEY')
    if not api_key:
        yield _sse('error', {'message': 'METAGEN_ANTHROPIC_API_KEY not set on backend'})
        return

    sid = req.session_id

    # --- session logging helpers (no-ops when no session) ---
    turn_event_ids: list[str] = []

    def log(etype, payload):
        if not sid:
            return
        try:
            turn_event_ids.append(_sess.append_event(sid, etype, payload)['id'])
        except Exception:  # noqa: BLE001 — logging must never break the chat
            pass

    def finalize_node():
        if not sid or not turn_event_ids:
            return
        try:
            code = req.state.code
            _sess.add_node(sid, 'assistant_turn', _turn_label(req),
                           {'code': code,
                            'code_hash': _hash12(code) if code else None,
                            'geometry_ref': None, 'sim_ref': None,
                            'chat_len': len(req.messages)},
                           event_ids=list(turn_event_ids))
        except Exception:  # noqa: BLE001
            pass

    # --- thinking config (per-request override > config default) ---
    # Claude 4.x uses ADAPTIVE thinking (the legacy {type:'enabled',
    # budget_tokens} API is rejected by opus-4.7/4.8). The AnthropicProvider
    # translates a non-empty `effort` into {type:adaptive, display:summarized}
    # + output_config.effort; we just decide whether thinking is on and how
    # much room to leave for it. Non-obvious points handled in the adapter:
    #  - opus-4.7/4.8 default display to "omitted"; the adapter forces
    #    "summarized" so the readable CoT streams as 'thinking' deltas.
    #  - thinking is adaptive: 'high' may skip on easy turns; 'xhigh'/'max'
    #    engage it more reliably. effort also governs total token spend, so we
    #    bump max_tokens to leave room for thinking + the answer.
    want_think = (req.thinking if req.thinking is not None
                  else bool(cfg('copilot.thinking.enabled', True)))
    effort = cfg('copilot.thinking.effort', 'high') if want_think else None
    think_max = int(cfg('copilot.thinking.max_tokens', 16000))
    max_tokens = req.max_tokens
    if want_think and max_tokens < think_max:
        max_tokens = think_max

    if sid:
        last_user = next((m for m in reversed(req.messages) if m.role == 'user'), None)
        if last_user is not None:
            log('user_message', {'content': last_user.content})

    system = [SystemBlock(_static_system_text(), cache=True),
              SystemBlock(_workspace_state_text(req.state), cache=False)]
    messages = [_to_msg(m) for m in req.messages]
    registry = _build_registry()
    env = ToolEnv(data={'state': req.state})
    engine = CopilotEngine(AnthropicProvider(api_key=api_key), registry, max_turns=8)

    # per-turn correlation for tool_exec logging: the engine reports tool args
    # on the AssistantMessage and the artifact/result separately, so we stitch
    # them back together by tool_call_id to reproduce the old tool_exec record.
    tool_args_by_id: dict[str, dict] = {}
    ui_by_id: dict[str, dict] = {}

    try:
        async for ev in engine.run(
                model=req.model, system=system, messages=messages, env=env,
                effort=effort, max_tokens=max_tokens, log=log):
            if isinstance(ev, TextDelta):
                yield _sse('text', {'text': ev.text})
            elif isinstance(ev, ThinkingDelta):
                yield _sse('thinking', {'text': ev.text})
            elif isinstance(ev, ToolCallStarted):
                yield _sse('tool_call_start', {'id': ev.id, 'name': ev.name})
            elif isinstance(ev, AssistantMessage):
                # display strips thinking; carry text + tool_use only (the
                # engine retains the full parts, incl. thinking, internally).
                display_blocks: list[dict] = []
                for p in ev.parts:
                    if isinstance(p, Text):
                        display_blocks.append({'type': 'text', 'text': p.text})
                    elif isinstance(p, ToolCall):
                        display_blocks.append({'type': 'tool_use', 'id': p.id,
                                               'name': p.name, 'input': p.input})
                        tool_args_by_id[p.id] = p.input
                yield _sse('assistant_msg', {'content': display_blocks})
            elif isinstance(ev, Artifact):
                ui_by_id[ev.tool_call_id] = ev.data
                yield _sse('tool_ui', {'tool_id': ev.tool_call_id,
                                       'name': ev.tool_name, **ev.data})
            elif isinstance(ev, ToolResultEvent):
                yield _sse('tool_result', {'tool_id': ev.tool_call_id,
                                           'name': ev.name, 'result': ev.result})
                args = tool_args_by_id.get(ev.tool_call_id, {})
                if ev.name == 'propose_edit':
                    log('proposal', {'tool_id': ev.tool_call_id,
                                     'new_code': args.get('new_code', ''),
                                     'summary': args.get('summary', '')})
                log('tool_exec', {
                    'tool_id': ev.tool_call_id, 'name': ev.name, 'args': args,
                    'result': ev.result,
                    'ui': _compact_ui_for_log(sid, ui_by_id.get(ev.tool_call_id, {})),
                    'elapsed_s': ev.elapsed_s})
            elif isinstance(ev, Done):
                yield _sse('done', {'stop_reason': ev.stop_reason})
                finalize_node()
                _schedule_autoname(sid)
                return
            elif isinstance(ev, ErrorEvent):
                payload = {'message': ev.message}
                if ev.detail:
                    payload['traceback'] = ev.detail
                yield _sse('error', payload)
                log('error', {'message': ev.message})
                finalize_node()
                return
    except Exception as exc:  # noqa: BLE001 — defensive: surface anything the
        # engine didn't already wrap as an ErrorEvent.
        yield _sse('error', {
            'message': f'{type(exc).__name__}: {exc}',
            'traceback': traceback.format_exc(),
        })
        log('error', {'message': f'{type(exc).__name__}: {exc}'})
        finalize_node()
        return


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
