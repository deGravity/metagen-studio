# metaDSL Studio: Copilot & UI Architecture

This document describes how the studio is currently put together — backend,
frontend, and the AI copilot that's wired into them — so that the assistant
model, the harness around it, and the UI can be cleanly decoupled.

It's intended as a snapshot of present state, not an aspirational design.
Where coupling is non-obvious or load-bearing, it's called out explicitly
in the **Coupling points** section near the end.

---

## 1. High-level shape

```
                        ┌─────────────────────────────────────┐
                        │              Browser                │
                        │ ┌─────────────┐  ┌───────────────┐  │
                        │ │ Editor      │  │ Viewer3D      │  │
                        │ │ (Monaco)    │  │ (three.js,    │  │
                        │ │             │  │  R3F)         │  │
                        │ └─────────────┘  └───────────────┘  │
                        │ ┌─────────────────────────────────┐ │
                        │ │ Right pane: Settings │ Results  │ │
                        │ │                      │ Copilot  │ │
                        │ └─────────────────────────────────┘ │
                        └───────────────┬─────────────────────┘
                                        │ /api/info, /api/execute,
                                        │ /api/simulate, /api/chat (SSE),
                                        │ /api/chat/upload
                        ┌───────────────▼─────────────────────┐
                        │     FastAPI backend (uvicorn)       │
                        │ ┌──────────────────────────────────┐│
                        │ │  main.py                         ││
                        │ │   • /api/info                    ││
                        │ │   • /api/execute  ─┐             ││
                        │ │   • /api/simulate ─┼─► program_  ││
                        │ │                    │   cache     ││
                        │ │  chat.py           │             ││
                        │ │   • /api/chat (SSE)│   (LRU on   ││
                        │ │   • /api/chat/upload  code_hash) ││
                        │ │     ┌─────────┐    │             ││
                        │ │     │ agent   │ ───┘             ││
                        │ │     │ loop    │                  ││
                        │ │     └────┬────┘                  ││
                        │ │          │ messages.stream()     ││
                        │ └──────────┼───────────────────────┘│
                        └────────────┼─────────────────────────┘
                                     ▼
                              Anthropic API
                          (Claude Opus 4.7, Files API)
```

Three runtime processes during dev:

- **uvicorn** serving `/api/*` from `backend/studio_backend/`.
- **vite** serving the frontend on `:5173` and proxying `/api` to uvicorn.
- The user's browser running the SPA.

`run.sh` is the orchestrator; it launches both and runs a watchdog that
polls `/api/info` and tears the run down if the backend goes unresponsive
(catches the `--reload` zombie-worker case). See `run.sh` for the
`STUDIO_DEV_WATCHDOG` env var.

In **packaged** mode the same `main.py` mounts the prebuilt SPA at `/`
(see `_FRONTEND_DIST`); no vite, single port. The chat path is identical.

---

## 2. Backend

### 2.1 File map

```
backend/studio_backend/
  main.py        FastAPI app, /api/info, /api/execute, /api/simulate,
                 SPA mount.
  chat.py        Copilot router. /api/chat (SSE), /api/chat/upload,
                 tool definitions, agent loop, system prompt.
  execute.py     ProgramCache + _compile(): exec user code in an
                 isolated namespace, return a Structure.
  state.py       Shared `program_cache` singleton.
  models.py      Pydantic request/response schemas.
  cli.py         `metagen-studio` console entry (packaged mode).
```

### 2.2 The `program_cache` and `Structure`

User code is a Python string defining `make_structure() -> Structure`.
`ProgramCache` (`state.py`, implemented in `execute.py`) keeps an
LRU of up to 32 compiled `Structure` objects keyed on a 12-char SHA-256
of the code. Compilation:

1. `sys.modules.setdefault('metagen', metagen_dsl)` so legacy
   `from metagen import *` still resolves.
2. Build a fresh `ModuleType`, seed its namespace with the public
   `metagen_dsl` symbols.
3. `exec(compile(code, '<user:HASH>', 'exec'), mod.__dict__)`.
4. Pull `make_structure` out, call it, store the resulting `Structure`.

The cache is shared between the HTTP routes and the chat tool dispatch
through the singleton in `state.py`, so a `/api/execute` from the UI and
a `run_geometry` from the copilot hit the same `Structure` instance and
share the per-`Structure` geometry/sim LRU.

> ⚠️ **Trust model.** The DSL is `exec`'d verbatim. Single-user dev tool;
> not safe for untrusted input. Documented in `execute.py`'s module
> docstring.

### 2.3 HTTP routes (non-chat)

| Method | Path           | Purpose                                            |
| ------ | -------------- | -------------------------------------------------- |
| GET    | `/api/info`    | `gpu_available`, valid GPU resolutions, cache size, `chat_available` (true iff `METAGEN_ANTHROPIC_API_KEY` is set). |
| POST   | `/api/execute` | Run geometry, return base64 vertices + triangles + stats. |
| POST   | `/api/simulate`| Run homogenization, return `C_matrix`, derived properties. |

Both `/execute` and `/simulate` go through `program_cache.get_or_compile()`
then delegate to the `Structure`'s own per-`(resolution, mode)` LRU. They
do not touch the chat path.

### 2.4 The copilot harness (`chat.py`)

`chat.py` is the entire copilot. Four pieces:

1. **System prompt construction** — `SYSTEM_PROMPT` literal + auto-rendered
   DSL API reference + per-request workspace state.
2. **Tool definitions** — JSON schemas (`TOOLS`) + Python handlers
   (`_TOOL_DISPATCH`).
3. **Agent loop** (`_agent_loop`) — drives `client.messages.stream`, runs
   tools when the model emits `tool_use`, streams events to the frontend
   over SSE.
4. **Two HTTP routes** — `/api/chat` (the SSE stream) and
   `/api/chat/upload` (proxy to Anthropic Files API for PDFs).

#### 2.4.1 System prompt

Composed each request from two text blocks, sent as a `list[dict]` to
the Messages API so the static prefix can be cached:

```python
[
  { 'type': 'text',
    'text': SYSTEM_PROMPT + "\n--- metaDSL API reference ---\n" + dsl_docs,
    'cache_control': {'type': 'ephemeral'} },          # ~6k tokens, cacheable
  { 'type': 'text',
    'text': _workspace_state_text(state) },            # per-request, uncached
]
```

- `SYSTEM_PROMPT` (in `chat.py:212`): copilot persona, tool-use guidance,
  resolution recommendations.
- DSL API reference: `from metagen_dsl.docs import render_llm`, cached at
  module level in `_DSL_API_DOCS_CACHE`. `METAGEN_DSL_DOCS_NO_CACHE=1`
  forces re-render per request (for iterating on DSL docstrings).
- Workspace state: current code (with 12-char hash), last geometry run
  summary (marked STALE if the code hash changed since), last sim run
  summary (`C_matrix` stripped to save tokens), and `last_error`.

#### 2.4.2 Tools

Three tools, all defined in `chat.py:39-118`. Each has a JSON-schema
definition for the model and a Python handler that returns a pair
`(tool_result_for_model, ui_event_for_frontend)`.

| Tool             | Args (required ✱)                                          | Result to model                                    | UI side-effect                          |
| ---------------- | ---------------------------------------------------------- | -------------------------------------------------- | --------------------------------------- |
| `propose_edit`   | `new_code`✱, `summary`✱                                    | `{ok: true, note: "Edit proposed…"}`               | `tool_ui: {kind:'proposal', …}`         |
| `run_geometry`   | `resolution`✱, `tpms_optimizer_mode?`                      | volume_fraction, fill_fraction, voxel counts, …    | `tool_ui: {kind:'geometry_done', …}`    |
| `run_simulation` | `resolution`✱, `backend?`, `tpms_optimizer_mode?`, `E?`, `nu?` | `C_matrix`, `properties`, `backend_used`, …    | `tool_ui: {kind:'sim_done', …}`         |

The `propose_edit` handler does **no actual edit** — it just emits a
`proposal` UI event and tells the model the user will decide later. The
model never observes acceptance directly; the next user turn carries
the new code via the `state` field, so the model infers it from the
hash changing.

`run_geometry` and `run_simulation` go through `program_cache` and call
`Structure.geometry()` / `Structure.simulate()` via `asyncio.to_thread`
to avoid blocking the event loop during long native calls.

Tool dispatch is name-based via `_TOOL_DISPATCH` (`chat.py:201`). Adding
a new tool means: add a `TOOLS` entry, write a handler, add to dispatch.

#### 2.4.3 Agent loop

The whole loop is `_agent_loop()` (`chat.py:313`). Pseudocode:

```
api_messages = req.messages                     # full client history
for turn in range(MAX_TURNS=8):
    stream = client.messages.stream(
        model=req.model, system=<2 blocks>, tools=TOOLS,
        messages=api_messages,
        extra_headers={'anthropic-beta': 'files-api-2025-04-14'},
    )

    async for event in stream:
        if event.type == 'text':              yield SSE 'text'
        if content_block_start tool_use:      yield SSE 'tool_call_start'

    final = await stream.get_final_message()

    append assistant turn to api_messages
    yield SSE 'assistant_msg' with full block list

    if stop_reason != 'tool_use':
        yield SSE 'done'; return

    tool_results = []
    for each tool_use in final.content:
        result, ui_event = await dispatch(name, args)
        if ui_event: yield SSE 'tool_ui'
        yield SSE 'tool_result'
        tool_results.append(tool_result block)

    append {'role':'user', 'content': tool_results} to api_messages
    # loop iterates: model sees the tool results and responds

yield SSE 'error' ('loop exceeded MAX_TURNS')
```

Notes:

- **MAX_TURNS = 8** (`chat.py:331`) — hard cap on tool-use round-trips
  within a single request. The user must send another message to extend.
- The model's beta `files-api-2025-04-14` header is unconditionally
  attached so `document` blocks with `source.type=file` resolve.
- Exceptions in tool handlers are caught and surfaced to the model as
  `{ok: false, error, traceback}` so the model can react instead of the
  whole stream blowing up.
- Stream-level exceptions yield SSE `error` and return.

#### 2.4.4 SSE event protocol

Wire format: standard `text/event-stream`, one event per `event:`/`data:`
pair, double-newline separated.

| `event:`           | `data:` shape                                                                       | When                                              |
| ------------------ | ----------------------------------------------------------------------------------- | ------------------------------------------------- |
| `text`             | `{text}`                                                                            | Per text-token chunk during streaming.            |
| `tool_call_start`  | `{id, name}`                                                                        | Model started emitting a `tool_use` block.        |
| `tool_ui`          | `{tool_id, name, kind, ...payload}`                                                 | After tool ran, *only if* the handler returned a non-empty UI event (proposal / geometry_done / sim_done). |
| `tool_result`      | `{tool_id, name, result}`                                                           | After each tool ran (always emitted).             |
| `assistant_msg`    | `{content: AssistantBlock[]}`                                                       | End of a model turn; full assembled block list.   |
| `done`             | `{stop_reason}`                                                                     | Model returned without a tool_use; stream ends.   |
| `error`            | `{message, traceback?}`                                                             | Anything raising in the loop; stream ends.        |

The frontend reconstructs the conversation from `text` (token-by-token,
appended to the current assistant turn's tail text block) plus
`assistant_msg` (canonical block list at turn end). `tool_ui` carries
the *UI* side effect (which the chat panel forwards to App), while
`tool_result` carries the data for in-chat display.

#### 2.4.5 Files API integration

`POST /api/chat/upload` (`chat.py:431`) is a thin proxy:

1. Receives a single multipart `file` via FastAPI's `UploadFile`.
2. Hard ceiling: `_UPLOAD_MAX_BYTES = 100 MB` (returns 413 if exceeded).
3. Calls `client.beta.files.upload(file=(name, content, content_type))`.
4. Returns `{file_id, filename, size, media_type}`.

The file_id is opaque and goes into a `document` content block on the
next chat turn. Beta header `files-api-2025-04-14` is on the messages
stream as well.

#### 2.4.6 Stateless service contract

The service is stateless across requests *except* for the
`program_cache` (which keys on code_hash, so the same code from
different sessions hits the same entry by design). Specifically:

- Conversation history lives on the client. Every request resends the
  full transcript.
- The system prompt is rebuilt on every request from `req.state`.
- No session cookies, no per-user state.

This makes the harness easy to scale horizontally but means re-sending
attachments is expensive on long conversations *unless* they're in the
Files API (where they're referenced by file_id, not bytes).

### 2.5 Models (`models.py`)

The chat-relevant pydantic models:

```python
class ChatMessage:
    role: 'user' | 'assistant'
    content: list[dict] | str            # passed through to Anthropic verbatim

class ChatStateContext:
    code: str
    geometry_code_hash, geometry_summary
    sim_code_hash, sim_summary
    last_error

class ChatRequest:
    messages: list[ChatMessage]
    state: ChatStateContext
    model: str = 'claude-opus-4-7'
    max_tokens: int = 4096
```

`ChatMessage.content` being `list[dict] | str` is what lets the frontend
ship arbitrary content blocks (image, document, text) without the backend
needing to understand them — they go straight to the Messages API.

---

## 3. Frontend

### 3.1 File map

```
frontend/src/
  main.tsx               React root.
  App.tsx                Top-level composition; owns all shared state.
  api.ts                 Wire layer: postJson, getInfo, executeCode,
                         simulate, uploadChatFile, streamChat (SSE).
  types.ts               TypeScript mirrors of backend models + chat types.
  styles.css             Whole app's CSS.
  components/
    Editor.tsx           Monaco wrapper.
    Viewer3D.tsx         three.js / R3F mesh viewer + tile slider.
    Settings.tsx         Resolution, TPMS mode, backend, run buttons.
    Results.tsx          Geometry + sim numerical readout.
    Chat.tsx             Copilot panel. Conversation, attachments, proposals.
```

### 3.2 `App.tsx` — root composition

`App` owns the **single source of truth** for everything cross-panel:

| State            | Purpose                                                 |
| ---------------- | ------------------------------------------------------- |
| `code`           | Editor contents.                                        |
| `currentHash`    | 12-char SHA-256 of `code`, recomputed on change.        |
| `resolution`, `tpmsMode`, `simBackend` | Settings values.                  |
| `geometry`       | Latest `ExecuteResponse` (with `code_hash` for staleness checks). |
| `sim`            | Latest `SimulateResponse`.                              |
| `mesh`           | Decoded `MeshData` for the viewer.                      |
| `info`           | `/api/info` snapshot (gpu_available, chat_available, …).|
| `tab`            | Which right-pane tab is visible (`'results' | 'chat'`). |
| `busy`, `error`  | Cross-panel loading / error state.                      |

It also derives the `chatState: ChatStateContext` object passed to
`ChatPanel`, summarizing the editor + last-run artifacts for the model.

Layout: a 3-column grid (`editor-pane` | `viewer-pane` | `right-pane`).
The right pane is split: top `SettingsPanel`, bottom a tabbed view with
`ResultsPanel` and `ChatPanel`. Both panels stay mounted; the inactive
one is hidden via the HTML `hidden` attribute so chat scrollback and
streaming state survive tab switches.

### 3.3 `ChatPanel` (chat UI + harness consumer)

Local state (lives in `ChatPanel`, not `App`):

| State                  | Shape                                            |
| ---------------------- | ------------------------------------------------ |
| `turns`                | `ChatTurn[]` — chronological transcript.         |
| `input`                | Current textarea contents.                       |
| `pendingAttachments`   | `Attachment[]` — uploads in flight or ready.     |
| `busy`                 | A request is in-flight.                          |
| `error`                | Last error string.                               |
| Refs: `proposalsRef`, `abortRef`, `scrollRef`, `fileInputRef` |   |

A `ChatTurn` is a UI-level concept, not a wire-level one:

```ts
interface ChatTurn {
  id: string;
  role: 'user' | 'assistant';
  text?: string;                    // user turn raw text
  attachments?: Attachment[];       // user turn uploads
  blocks?: AssistantBlock[];        // assistant turn content blocks
  proposals?: PendingProposal[];    // proposals from this turn's propose_edit calls
  toolResults?: { tool_id, name, result }[];
  streaming?: boolean;
}
```

Key methods:

- **`buildApiMessages()`** — reconstructs the wire `ChatMessage[]` from
  `turns`. For each user turn, calls `buildUserContent(text, attachments)`
  which returns either a plain string (no attachments) or a list of
  content blocks (`[image|document, …, text]`). Assistant turns get
  their raw `blocks` shipped back.
- **`addFiles()`** — validates files (5 MB cap for images, 100 MB for
  PDFs), inlines images as base64 immediately, kicks off Files API
  upload for PDFs. PDFs sit in `pendingAttachments` with `uploading:
  true` until the POST returns a `fileId`.
- **`send()`** — guards against in-flight uploads, appends user +
  placeholder-assistant turns, then drives `streamChat()` and dispatches
  each SSE event into local state and (for some events) into props.

SSE event handling in `send()`:

| Incoming event   | Effect                                                                 |
| ---------------- | ---------------------------------------------------------------------- |
| `text`           | Append to current assistant turn's tail text block (streaming).        |
| `tool_ui` (`proposal`) | Add `PendingProposal` to current assistant turn.                 |
| `tool_ui` (`geometry_done`) | Call `props.onGeometryDone(payload)` → App refreshes mesh. |
| `tool_ui` (`sim_done`) | Call `props.onSimDone(payload)` → App sets `sim` state.        |
| `tool_result`    | Append to current assistant turn's `toolResults` (for display).        |
| `assistant_msg`  | Replace current assistant turn's blocks with canonical list; reset live-text buffer. |
| `done`           | Clear `streaming` flag.                                                |
| `error`          | Set local `error`; break out of loop.                                  |

### 3.4 The App ↔ ChatPanel contract

`ChatPanel` props (the entire surface between the chat and the rest of
the UI):

```ts
interface Props {
  state: ChatStateContext;                       // read-only, derived in App
  available: boolean;                            // gates the disabled banner
  onApplyProposal: (newCode, proposalId, summary) => void;
  onGeometryDone: (summary) => void;
  onSimDone: (summary) => void;
}
```

So:

- **In:** workspace snapshot + chat-available flag.
- **Out:** three callbacks. Apply-proposal mutates `code`; geometry-done
  reruns `/api/execute` on the App side to fetch the actual mesh
  (because the chat tool returns *summary* numbers, not vertex data);
  sim-done synthesizes a `SimulateResponse` straight from the summary.

This contract is small but worth noting for decoupling: the chat panel
doesn't directly touch `code`, `geometry`, `sim`, `mesh`, or `info`
state. It only receives a snapshot and emits callbacks.

### 3.5 `api.ts` — wire layer

Thin: `postJson` helper, one function per route. The chat-specific bits:

- **`uploadChatFile(file)`** — multipart POST to `/api/chat/upload`.
- **`streamChat(messages, state, signal, model='claude-opus-4-7')`** —
  async generator. Posts JSON to `/api/chat`, reads the SSE stream by
  hand (splits on `\n\n`, parses `event:`/`data:` lines), yields
  `ChatEvent` objects.

The model parameter defaults to `'claude-opus-4-7'`. This is the only
place the model name appears in the frontend.

### 3.6 Other components (briefly)

- **`Editor`** — Monaco wrapper; emits `onChange(code)`. Doesn't know about chat.
- **`Viewer3D`** — three.js via R3F; renders `mesh` as instanced copies
  with a 1–6 tile slider. Doesn't know about chat.
- **`Settings`** — UI for resolution/mode/backend + the manual
  Geometry/Sim run buttons. Doesn't know about chat.
- **`Results`** — readout of `geometry.stats`, `sim.C_matrix`, derived
  properties; flags staleness if a run's `code_hash` differs from the
  editor's. Doesn't know about chat.

All four are pure functions of their props plus their own ephemeral UI
state; they integrate via `App`, not via the chat.

---

## 4. End-to-end flows

### 4.1 User runs geometry from the Settings panel (no chat involved)

```
User clicks "Run geometry"
  → App.runGeometry()
     → POST /api/execute  {code, resolution, mode}
        → program_cache.get_or_compile(code)   (LRU on code_hash)
        → Structure.geometry(resolution, mode) (per-Structure LRU)
        → pick thickened mesh, b64-encode arrays
     ← ExecuteResponse
     → setGeometry(r); setMesh(decodeMesh(r))
  → Viewer3D rerenders with new mesh
  → Results panel rerenders with new stats
```

### 4.2 User asks the copilot to refactor; copilot proposes an edit

```
User types "make the beams thicker" → ChatPanel.send()
  ↓ POST /api/chat (SSE)  with full history + state snapshot
  → _agent_loop:
      messages.stream(...) tool_use=propose_edit
      handler returns ({ok:true,note:…}, {kind:'proposal', new_code, summary})
      SSE 'tool_ui' {kind:'proposal', ...}
      SSE 'tool_result' {ok:true, ...}
      SSE 'assistant_msg' {content}
      stop_reason='end_turn' → SSE 'done'
  ← stream ends
  ChatPanel:
      turn.proposals += PendingProposal(status='pending')
User clicks "apply"
  → ChatPanel.applyProposal → props.onApplyProposal(newCode)
      → App.setCode(newCode)        # editor updates; currentHash changes;
                                    # geomStale/simStale flip true.
```

The model never directly sees acceptance/rejection. On the next chat
turn it sees:

- the new code (in the state preamble),
- a different `code_hash`,
- last geometry/sim runs marked **STALE — code edited since**.

### 4.3 User asks the copilot to "run a sim and tell me E"

```
ChatPanel.send()
  → POST /api/chat
  → agent loop:
      tool_use=run_simulation {resolution:33, backend:'auto'}
      handler:
        - program_cache.get_or_compile(state.code)
        - Structure.geometry(33, mode)         (warm cache)
        - Structure.simulate(33, 'auto', E, nu)
      returns ({ok:true, C_matrix, properties, …},
              {kind:'sim_done', C_matrix, properties, …})
      SSE 'tool_ui' {kind:'sim_done', ...} ─────────────► ChatPanel
                                                          ↓ props.onSimDone
                                                          App.setSim(...)
                                                          Results panel updates
      SSE 'tool_result' {properties: {E_VRH, ...}}
  → next loop turn: model sees the tool result, writes a reply
  SSE 'text' (streaming…)
  SSE 'assistant_msg' (final blocks)
  SSE 'done'
```

Same as `geometry_done`, the chat panel forwards the payload to `App`
via its callback, and `App` updates its own state so the Results panel
reflects what the copilot just computed.

### 4.4 User attaches a PDF and asks a question

```
User picks 34 MB PDF
  → ChatPanel.addFiles:
      push pending attachment {uploading:true}
      fetch POST /api/chat/upload (multipart)
        → client.beta.files.upload(...)
        ← {file_id: 'file_...'}
      update attachment {fileId, uploading:false}
  Send button enables.
User clicks "send"
  → buildUserContent(text, [pdf-att]) =
      [ {type:'document', source:{type:'file', file_id:'file_...'}},
        {type:'text', text:'<question>'} ]
  → POST /api/chat with this user content
  → messages.stream(..., extra_headers={'anthropic-beta': 'files-api-2025-04-14'})
  → model reads PDF + answers; SSE 'text' chunks back
```

On subsequent turns, the same `file_id` is resent (no re-upload).

---

## 5. Coupling points (and what would need to move)

The user's goal is to decouple **assistant model**, **copilot harness**,
and **UI**. Here's where they're currently fused, ordered from
most-coupled to least:

### 5.1 `chat.py` is the harness *and* the assistant binding

Inside one ~470-line module live:

- The Anthropic SDK client construction (`AsyncAnthropic`).
- The Anthropic-specific message shape (content blocks, `messages.stream`,
  `tool_use` semantics, `cache_control`, beta headers).
- The tool definitions (in Anthropic's JSON-schema flavor).
- The tool dispatch table.
- The agent loop control flow (MAX_TURNS, tool-use → continue, else end).
- The wire protocol *to the frontend* (SSE event names + payload shapes).
- The system prompt content (copilot persona + DSL docs).
- File upload proxy to Anthropic Files API.

**To swap models** (say, an OpenAI- or local-LLM-backed copilot), almost
every part of this file would need to move behind an interface. The
practical seam is between `_agent_loop` (control flow + tool dispatch)
and `client.messages.stream` (Anthropic-specific). A model-agnostic
`AssistantClient` protocol with a `stream(messages, system, tools)
-> AsyncIterator[Event]` would isolate the binding.

### 5.2 The harness owns the tool implementations

The tools call into `program_cache` and `Structure` directly. They're
also the only place the model can interact with workspace state.

A cleaner split: tools become a registry the harness consumes:

```python
class Tool(Protocol):
    name: str
    schema: dict
    async def run(self, args, ctx) -> tuple[ModelResult, UIEvent | None]: ...
```

That puts geometry/sim/edit operations behind a single interface the
harness exercises without knowing about Structure or program_cache —
and would also let non-chat surfaces (e.g. a CLI) reuse the same tools.

### 5.3 SSE event names are an ad-hoc protocol

`tool_ui` carries arbitrary kind-tagged payloads (`'proposal'`,
`'geometry_done'`, `'sim_done'`) that the frontend pattern-matches on.
There's no schema; the only enforcement is "both sides agree."

If `ChatPanel` is to become reusable across studios/models, this
protocol should be versioned (or at least typed in one place).
Candidates: a generated TypeScript from pydantic models, or a small
hand-written protocol module imported by both.

### 5.4 `ChatPanel` knows about specific tool names

In rendering: `tr.name === 'run_geometry'` formats vf + elapsed_s;
`tr.name === 'run_simulation'` formats E_VRH; `tr.name === 'propose_edit'`
suppresses the default tool-result row in favor of the proposal card.

If tools are pluggable, the panel needs a render registry — small
per-tool render fragments keyed by tool name — instead of hard-coded
branches.

### 5.5 Workspace state is mutually agreed-upon shape

`ChatStateContext` carries:

```ts
{
  code, geometry_code_hash, geometry_summary,
  sim_code_hash, sim_summary, last_error
}
```

…and the backend embeds this into the system prompt verbatim. The
shape is fine for the current domain (DSL editor + geometry/sim
artifacts) but baked into both ends. A studio that's also producing
something else (e.g. layout diagrams, optimization runs) would extend
this; today there's no extension point.

### 5.6 The system prompt is a constant string in `chat.py`

`SYSTEM_PROMPT` (`chat.py:212`) is hand-edited Python source. The DSL
API reference is auto-rendered but the persona/guidance is static. Pulling
the prompt out into a versioned file (e.g. `prompts/copilot.md`) would
make it editable without code review and enable per-model variants.

### 5.7 Model identifier lives in two places

- Frontend default: `streamChat(..., model='claude-opus-4-7')` in `api.ts`.
- Backend default: `ChatRequest.model = 'claude-opus-4-7'` in `models.py`.

Either is overrideable but both encode an Anthropic-specific id. If the
backend grew a model registry (`{'opus-4.7': AnthropicClient, ...}`),
the frontend would pick a logical name instead.

### 5.8 Things that are *not* coupling problems

For completeness, these are clean today and likely don't need work:

- `App ↔ ChatPanel` contract is small and well-defined.
- `program_cache` is genuinely shared infrastructure, not a chat concern.
- `Editor`, `Viewer3D`, `Settings`, `Results` have no knowledge of chat.
- The Files API path (`/api/chat/upload`) is self-contained.

---

## 6. Quick reference: key files & line ranges

| Concern                                       | File / range                                          |
| --------------------------------------------- | ----------------------------------------------------- |
| Tool schemas + descriptions                   | `backend/studio_backend/chat.py:39-118`               |
| Tool handlers                                 | `backend/studio_backend/chat.py:125-205`              |
| System prompt (literal)                       | `backend/studio_backend/chat.py:212-231`              |
| System block assembly (+ cache_control)       | `backend/studio_backend/chat.py:249-297`              |
| Agent loop                                    | `backend/studio_backend/chat.py:313-412`              |
| SSE event encoder                             | `backend/studio_backend/chat.py:309-310`              |
| `/api/chat`                                   | `backend/studio_backend/chat.py:415-422`              |
| `/api/chat/upload` (Files API proxy)          | `backend/studio_backend/chat.py:431-466`              |
| Program cache                                 | `backend/studio_backend/execute.py`, `state.py`       |
| `/api/info`, `/api/execute`, `/api/simulate`  | `backend/studio_backend/main.py:76-160`               |
| Pydantic models                               | `backend/studio_backend/models.py`                    |
| App root + shared state                       | `frontend/src/App.tsx`                                |
| Chat panel                                    | `frontend/src/components/Chat.tsx`                    |
| Wire layer (`streamChat`, `uploadChatFile`)   | `frontend/src/api.ts`                                 |
| Frontend chat types                           | `frontend/src/types.ts:43-83`                         |
