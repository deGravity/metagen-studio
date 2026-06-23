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

from .models import ChatRequest, ChatStateContext, ProviderModelsRequest
from .state import program_cache
from .config import cfg
from . import sessions as _sess
from metagen_copilot import CopilotEngine, Tool, ToolEnv, ToolOutcome, ToolRegistry
from metagen_copilot.providers import build_provider
from metagen_copilot.types import (
    SystemBlock, Msg, Text, Document, ToolCall, ToolResult, Thinking, Raw, ToolDef,
    TextDelta, ThinkingDelta, ToolCallStarted, AssistantMessage, ToolResultEvent,
    Artifact, Done, ErrorEvent,
)
from metagen_copilot import pdf as _pdf


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
        elif t == 'document' and isinstance(b.get('source'), dict) \
                and b['source'].get('type') == 'base64':
            # An inline base64 PDF — normalize so the attachment pipeline can
            # route it per the target model's capabilities. A Files-API ref
            # (source.type == 'file') is NOT this; it stays Raw below so the
            # Anthropic-native upload path is untouched.
            src = b['source']
            parts.append(Document(data_b64=src.get('data', ''),
                                  media_type=src.get('media_type', 'application/pdf'),
                                  name=b.get('title', '') or b.get('name', '')))
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


def _infer_provider(model: str) -> str:
    m = (model or '').lower()
    if m.startswith('claude'):
        return 'anthropic'
    if m.startswith(('gpt', 'o1', 'o3', 'o4')):
        return 'openai'
    if m.startswith('gemini'):
        return 'gemini'
    return 'anthropic'


def _resolve_provider(req: ChatRequest):
    """Pick + build the provider for this request. Returns (provider, error):
    on success error is None; on a config/credential problem provider is None
    and error is a user-facing message. Defaults reproduce the previous
    Anthropic-only behavior exactly (METAGEN_ANTHROPIC_API_KEY gate).

    Per-request `api_key` / `base_url` (supplied by the browser's advanced
    settings, stored only in localStorage there) override the backend env/config
    so a user can bring their own credentials without a server-side secret."""
    name = (req.provider or cfg('copilot.provider', None)
            or _infer_provider(req.model)).lower()
    pcfg = cfg(f'copilot.providers.{name}', {}) or {}
    key_env = pcfg.get('key_env') or (
        'METAGEN_ANTHROPIC_API_KEY' if name in ('anthropic', 'claude') else None)
    api_key = getattr(req, 'api_key', None) or (os.environ.get(key_env) if key_env else None)
    base_url = _normalize_base_url(getattr(req, 'base_url', None) or pcfg.get('base_url'))
    mode = pcfg.get('mode')
    profile = (pcfg.get('models') or {}).get(req.model, {}) or {}

    is_local = name in ('vllm', 'openai-compat', 'openai_compat', 'chat_completions')
    if is_local:
        if not base_url:
            return None, (f"provider '{name}' has no base_url — set one in the "
                          f"advanced settings, or copilot.providers.{name}.base_url")
    elif not api_key:
        return None, (f'no API key for {name}: set {key_env} on the backend or '
                      f'add a key in advanced settings' if key_env
                      else f"no API key configured for provider '{name}'")

    try:
        provider = build_provider(name, api_key=api_key, base_url=base_url,
                                  mode=mode, profile=profile)
    except ValueError as exc:
        return None, str(exc)
    return provider, None


# canonical provider order for the UI + which need a base_url vs a key
_UI_PROVIDERS = [
    ('anthropic', 'Anthropic', False),
    ('openai', 'OpenAI', False),
    ('gemini', 'Gemini', False),
    ('vllm', 'vLLM (local/open)', True),
]


def provider_status() -> list[dict]:
    """Per-provider availability + curated models for the UI selector. Reports
    what the *backend* has (env key / configured base_url); the frontend merges
    in any browser-local credentials on top of this."""
    out = []
    for name, label, needs_base_url in _UI_PROVIDERS:
        pcfg = cfg(f'copilot.providers.{name}', {}) or {}
        key_env = pcfg.get('key_env')
        has_key = bool(os.environ.get(key_env)) if key_env else False
        base_url = pcfg.get('base_url')
        if needs_base_url:
            available = bool(base_url)
            need = 'base_url'
        else:
            available = has_key
            need = key_env or 'api_key'
        # For discovery providers (vLLM) the `models` map holds capability
        # profiles keyed by short names (qwen3.6) — those are NOT valid model
        # ids to send, so don't surface them; the UI fills the list from
        # {base_url}/v1/models. Other providers use their curated model_options.
        if needs_base_url:
            models: list = []
        else:
            models = pcfg.get('model_options') or list((pcfg.get('models') or {}).keys())
        out.append({'name': name, 'label': label, 'available': available,
                    'needs': need, 'needs_base_url': needs_base_url,
                    'key_env': key_env, 'models': models,
                    'default_model': models[0] if models else None,
                    'discover': needs_base_url, 'base_url': base_url})
    return out


def _normalize_base_url(url: Optional[str]) -> Optional[str]:
    """Tolerate a base_url with no scheme (the common paste mistake): default
    to http://. Leaves an explicit http(s):// untouched."""
    if not url:
        return url
    u = url.strip()
    if u and not u.startswith(('http://', 'https://')):
        u = 'http://' + u
    return u


async def discover_models(base_url: str, api_key: Optional[str] = None) -> dict:
    """List the models an OpenAI-compatible server (vLLM) is currently serving,
    via {base_url}/v1/models. Returns {models:[ids]} or {models:[], error:...}."""
    base_url = _normalize_base_url(base_url)
    if not base_url:
        return {'models': [], 'error': 'no base_url'}
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key or 'EMPTY', base_url=base_url)
        resp = await client.models.list()
        return {'models': sorted(m.id for m in resp.data)}
    except Exception as exc:  # noqa: BLE001 — include the URL so a bad one is obvious
        return {'models': [], 'error': f'{type(exc).__name__}: {exc} (url={base_url})'}


# Process-local ingest cache: repeated turns in a chat (and benchmark reruns
# in one process) reuse the extraction. Persisting across processes via the
# session blob store is a future optimization (§4.4).
_PDF_CACHE = _pdf.InMemoryIngestCache()


def _vision_transcriber():
    """Build a sync (images, prompt)->markdown transcriber for the vision_ocr
    backend from config, or None if not configured. Sync on purpose so it runs
    inside asyncio.to_thread without touching the event loop."""
    vc = cfg('copilot.pdf.vision_ocr', {}) or {}
    provider = (vc.get('provider') or 'anthropic').lower()
    model = vc.get('model')
    if not model:
        return None
    if provider in ('anthropic', 'claude'):
        key = os.environ.get(cfg('copilot.providers.anthropic.key_env',
                                 'METAGEN_ANTHROPIC_API_KEY'))
        if not key:
            return None

        def transcribe(images, prompt):
            from anthropic import Anthropic
            content = [{'type': 'image', 'source': {'type': 'base64',
                        'media_type': im.media_type, 'data': im.data_b64}}
                       for im in images]
            content.append({'type': 'text', 'text': prompt})
            r = Anthropic(api_key=key).messages.create(
                model=model, max_tokens=4096,
                messages=[{'role': 'user', 'content': content}])
            return ''.join(getattr(b, 'text', '') for b in r.content
                           if getattr(b, 'type', None) == 'text')
        return transcribe
    if provider in ('openai', 'gpt'):
        key = os.environ.get(cfg('copilot.providers.openai.key_env',
                                 'METAGEN_OPENAI_KEY'))
        if not key:
            return None

        def transcribe(images, prompt):
            from openai import OpenAI
            content = [{'type': 'text', 'text': prompt}]
            for im in images:
                content.append({'type': 'image_url', 'image_url': {
                    'url': f'data:{im.media_type};base64,{im.data_b64}'}})
            r = OpenAI(api_key=key).chat.completions.create(
                model=model, max_tokens=4096,
                messages=[{'role': 'user', 'content': content}])
            return r.choices[0].message.content or ''
        return transcribe
    return None


def _build_pdf_backend():
    name = cfg('copilot.pdf.backend', 'pymupdf4llm')
    endpoint = cfg(f'copilot.pdf.{name}.endpoint', None)
    transcribe = _vision_transcriber() if name in ('vision_ocr', 'vision') else None
    return _pdf.build_backend(name, endpoint=endpoint, transcribe=transcribe)


def _prepare_attachments(messages: list, provider, model: str) -> list:
    """Route inline PDF Documents to what `model` can consume. No-op (fast
    path) when there are no inline PDFs, or when the model takes PDFs natively
    and config doesn't force a backend — so the Anthropic default is unchanged.
    Runs the (CPU/network) backend off the event loop via the caller's thread."""
    has_pdf = any(isinstance(p, Document) and 'pdf' in p.media_type
                  for m in messages for p in m.parts)
    if not has_pdf:
        return messages
    try:
        caps = provider.capabilities(model)
    except Exception:  # noqa: BLE001
        caps = None
    force = bool(cfg('copilot.pdf.force_backend', False))
    if caps is not None and caps.native_pdf and not force:
        return messages   # native PDF: leave Documents in place

    mode = cfg('copilot.pdf.mode', 'both')
    dpi = int(cfg('copilot.pdf.image_dpi', 150))
    max_pages = cfg('copilot.pdf.max_pages', None)
    backend = _build_pdf_backend()
    opts = _pdf.IngestOpts(image_dpi=dpi,
                           max_pages=int(max_pages) if max_pages else None)
    out = []
    for m in messages:
        r = _pdf.prepare_parts(list(m.parts), caps or _pdf_default_caps(),
                               backend=backend, mode=mode, force_backend=force,
                               cache=_PDF_CACHE, opts=opts)
        out.append(Msg(m.role, r.parts))
    return out


def _pdf_default_caps():
    from metagen_copilot.types import Capabilities
    return Capabilities(native_pdf=False, native_images=False)


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
    provider, perr = _resolve_provider(req)
    if perr:
        yield _sse('error', {'message': perr})
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
    # Route inline PDF attachments to what this model can consume (no-op for
    # native-PDF models / no attachments). Runs the extractor off the event
    # loop so a slow ingest doesn't stall the SSE stream.
    try:
        messages = await asyncio.to_thread(
            _prepare_attachments, messages, provider, req.model)
    except Exception as exc:  # noqa: BLE001 — never let ingest break the chat
        log('error', {'message': f'attachment preprocessing failed: {exc}'})
    registry = _build_registry()
    env = ToolEnv(data={'state': req.state})
    engine = CopilotEngine(provider, registry, max_turns=8)

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


@router.get('/api/providers')
def providers_endpoint():
    """Provider availability + curated models for the UI selector. Includes the
    resolved default provider so the UI can preselect it."""
    return {'providers': provider_status(),
            'default_provider': (cfg('copilot.provider', None) or 'anthropic')}


@router.post('/api/provider-models')
async def provider_models_endpoint(req: ProviderModelsRequest):
    """Live-discover the models an OpenAI-compatible server is serving. The
    base_url/api_key may be client-supplied (advanced settings); if base_url is
    omitted we fall back to the provider's configured base_url."""
    base_url = req.base_url or cfg(f'copilot.providers.{req.provider}.base_url', None)
    return await discover_models(base_url, req.api_key)


@router.post('/api/chat')
async def chat_endpoint(req: ChatRequest):
    _, perr = _resolve_provider(req)
    if perr:
        raise HTTPException(status_code=503, detail=perr)
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
