# Studio Sessions — design & implementation plan

Status: **proposal** (no code yet). Branch: `feat/session-history`.

## 1. Goals (from the request)

1. **Save/restore sessions**, auto-saved continuously by default.
2. **Log everything**: full input/output of every copilot call (incl.
   thinking / CoT), every tool call + its response, and the history of editor
   state at each meaningful action (run geometry, simulate, send-to-copilot,
   accept-proposal).
3. Keep the **current chat view** as-is; add a **second, hidden-by-default
   "log explorer"** view (opened separately) for the full logs.
4. From the explorer, **rewind** to an earlier state and continue → the
   session history becomes a **git-like tree with optional branching**.
5. **Auto-name** sessions via a small out-of-band LLM call that summarizes the
   chat; **never overwrite a user-chosen name**.

## 2. Mental model: a session is an event log + a state DAG

Two layers, both server-side (single-user dev tool):

- **Event log** (append-only, the source of truth): every observable thing
  that happens, in order, with full fidelity. Nothing is summarized or
  dropped here. This satisfies "log everything."
- **State DAG** (derived, the navigable structure): a git-like tree of
  *checkpoints*. Each checkpoint is a restorable studio state; edges are the
  actions that produced them. Rewinding moves a `HEAD` pointer to an older
  checkpoint; acting from there creates a new child → a branch.

The conversation history sent to the model for any checkpoint = the
root→checkpoint path through the DAG. So "rewind + continue" naturally
reconstructs the right prompt and forks cleanly, exactly like git.

## 3. Data structures

### 3.1 Event log — `events.jsonl` (one JSON object per line)

Common envelope:
```jsonc
{
  "id": "evt_<ulid>",          // monotonic, sortable
  "ts": "2026-06-18T19:00:00.123Z",
  "node_id": "node_<ulid>",    // checkpoint this event belongs to / created
  "parent_node": "node_<ulid>|null",
  "type": "<see below>",
  "payload": { ... }
}
```

Event `type`s and payloads:

- `editor_snapshot` — `{ code, code_hash, reason: "run|sim|chat|accept|edit", from_node }`
  Recorded whenever code is about to be acted on (run/sim/sent-to-copilot) or
  changed by an accepted proposal.
- `geometry_run` — `{ code_hash, resolution, tpms_mode, multistart_k, origin:
  "button|copilot|candidate", stats, elapsed_s, result_blob }`
- `sim_run` — `{ code_hash, resolution, backend, E, nu, origin, C_matrix,
  properties, elapsed_s }`
- `copilot_request` — the **full** request for one model call:
  `{ call_index, model, max_tokens, system_blocks, messages, tools_digest,
     thinking_config }`
- `copilot_response` — the **full** response for that call:
  `{ call_index, stop_reason, usage, content_blocks: [
       {type:"thinking", thinking, signature?},
       {type:"text", text},
       {type:"tool_use", id, name, input} ], raw_text }`
  (A single user turn may produce several request/response pairs — one per
  tool round-trip in the agent loop. `call_index` orders them.)
- `tool_exec` — `{ tool_id, name, args, result, ui_event, elapsed_s, error? }`
- `proposal` — `{ proposal_id, tool_id, new_code, summary }`
- `proposal_decision` — `{ proposal_id, status: "accepted|rejected" }`
- `user_message` — `{ text, attachments }` (what the user typed/attached)
- `session_meta` — `{ name, name_source: "auto|user", model, created }`

Everything is captured: CoT lives in `copilot_response.content_blocks` (type
`thinking`); raw I/O in `copilot_request`/`copilot_response`; tool I/O in
`tool_exec`; editor history in `editor_snapshot`.

### 3.2 State DAG — `tree.json`

```jsonc
{
  "session_id": "...",
  "name": "Auxetic honeycomb tuning",
  "name_source": "auto",            // "auto" | "user"
  "created": "...", "updated": "...",
  "head": "node_<ulid>",            // current checkpoint
  "model": "claude-opus-4-7",
  "nodes": {
    "node_<ulid>": {
      "id": "...", "parent": "node_<ulid>|null",
      "children": ["node_..."],     // >1 child = a branch point
      "ts": "...",
      "kind": "root|user_turn|assistant_turn|geometry|sim|edit",
      "label": "ran geometry @65 · vf 0.21",   // short human label
      "event_ids": ["evt_...", ...],            // events that compose this node
      "snapshot": {
        "code": "...", "code_hash": "...",
        "geometry_ref": "blob_<hash>|null",      // → results blob
        "sim_ref": "blob_<hash>|null",
        "chat_len": 7                            // # of messages on root→here path
      }
    }
  }
}
```

A node is created for each state-changing action. A **chat user-turn** that
triggers a multi-tool agent loop is a single node whose `event_ids` reference
all the request/response/tool_exec/proposal events of that turn (the explorer
expands them). Editor runs/sims/edits are their own nodes.

### 3.3 Content-addressed blob store — `blobs/<sha256>.json(.gz)`

Geometry/sim results (incl. mesh b64) are **deduplicated by `code_hash`+params**
so restore is instant (no re-run) and storage stays bounded — many nodes share
the same code. Meshes are the only large items; gzip them. A simple GC can
drop blobs unreferenced by any node if space is tight. (Reuses the existing
`results_cache` shape; persistence is just spilling that cache to disk per
session.)

## 4. Storage layout

```
<STUDIO_SESSION_DIR>/                 # default ~/.metagen-studio/sessions, overridable by env
  index.json                         # [{id, name, name_source, updated, n_nodes}] for the picker
  <session_id>/
    events.jsonl                     # append-only full log
    tree.json                        # state DAG + HEAD + name
    blobs/<sha256>.json.gz           # dedup'd geometry/sim results
```

Append-only `events.jsonl` is crash-safe (fsync on append) and the tree is
rebuildable from it if `tree.json` is ever lost.

## 5. Backend API (additions)

Session lifecycle:
- `POST   /api/sessions` → create `{id, name}` (auto-name "Untitled" initially).
- `GET    /api/sessions` → index list.
- `GET    /api/sessions/{id}` → `tree.json` (+ HEAD).
- `PATCH  /api/sessions/{id}` → `{name}` sets user name (`name_source=user`).
- `DELETE /api/sessions/{id}`.
- `GET    /api/sessions/{id}/events?node=&type=` → filtered event log (for the explorer).
- `GET    /api/sessions/{id}/node/{node_id}` → full restorable snapshot (code + result blobs).
- `POST   /api/sessions/{id}/checkout {node_id}` → set HEAD (rewind). Returns the snapshot to load.
- (Branching is implicit: act after a checkout → new child of HEAD.)

Threading: existing endpoints gain an optional `session_id` (+ `node` parent)
in their request bodies — `/api/execute`, `/api/execute/stream`,
`/api/simulate`, `/api/chat`. When present, the handler appends the relevant
events and creates/extends nodes. When absent, behavior is unchanged
(sessions are additive, never required).

## 6. Logging integration points

- **`/api/chat`** (`_agent_loop`): wrap each `client.messages.stream(...)` call
  to emit `copilot_request` before and `copilot_response` after (capturing the
  reconstructed `content_blocks` incl. thinking). Each tool dispatch emits
  `tool_exec`; proposals emit `proposal`. The whole user turn becomes one
  `assistant_turn` node. **To have CoT to log, enable extended thinking**
  (`thinking={type:"enabled", budget_tokens:N}`) — see §10 decisions.
- **`/api/execute[/stream]`, `/api/simulate`**: emit `editor_snapshot` (reason
  run/sim) + `geometry_run`/`sim_run`, persist result blob, create a node.
- **accept proposal** (frontend → a new `POST /api/sessions/{id}/event` or
  folded into the next `/api/chat`): emit `proposal_decision` + `editor_snapshot`
  (reason accept) and create an `edit` node.

All appends are server-side so logging can't be lost if the browser closes.

## 7. Frontend

### 7.1 Session lifecycle
- On load: create or resume a session (last-open id in localStorage); pass
  `session_id` on every action. A small **session bar** (header): current name
  (click to rename), a session picker (recent list), "New session".
- Auto-save is automatic — every action already round-trips to the backend,
  which logs it. No explicit save button.

### 7.2 Two views
- **Chat view** — unchanged (the dock under the editor).
- **Log Explorer** — hidden by default; opened via a header button **and** as a
  standalone route (`/explorer?session=<id>`) so it can live in a separate
  browser window/tab. Layout:
  - **Left: the tree** — git-graph rendering of `tree.json` (nodes = checkpoints,
    branches drawn as in a commit graph), HEAD highlighted. Node labels show the
    action + key stat.
  - **Right: node detail** — the full expanded log for the selected node: user
    message, each copilot call's *thinking*, text, tool calls + args + results,
    proposals, and the geometry/sim numbers. Raw request/response viewable
    (collapsible "raw JSON").
  - **Actions on a node**: "Rewind here" (checkout → loads that snapshot into the
    editor/viewer/results and truncates the live chat to that path; continuing
    branches), "Open in chat", "Copy code".

Rewind UX mirrors git: rewinding doesn't delete the old branch; it moves HEAD,
and the explorer keeps showing all branches.

## 8. Auto-naming

- A debounced background job (server-side) calls a **small model**
  (`claude-haiku`) with the running chat summary → a ≤6-word title.
- Triggers: after the first assistant turn, then every ~5 turns or after ~60s
  idle, whichever first; skipped entirely if `name_source == "user"`.
- Runs **out-of-band** (its own one-shot `messages.create`, not in the agent
  loop, not logged into the conversation) so it never pollutes context or cost
  accounting of the main chat. Result updates `tree.json.name` +
  `index.json`; pushed to the UI via a lightweight poll or the next SSE.
- A `PATCH /api/sessions/{id} {name}` from the rename box sets
  `name_source=user` and disables further auto-naming.

## 9. Rewind / branch semantics (git analogy)

| git            | sessions                                             |
|----------------|------------------------------------------------------|
| commit         | node (checkpoint = code + results + chat cursor)     |
| HEAD           | `tree.head`                                          |
| checkout <c>   | `POST checkout {node_id}` → load snapshot            |
| branch (auto)  | acting after a checkout that isn't a leaf            |
| log --graph    | the explorer's tree pane                             |
| diff           | code diff between a node and its parent (nice-to-have)|

No rebasing/merging in v1 — only branch + navigate. The model prompt for a
continuation is always the root→HEAD path.

## 10. Decisions (resolved 2026-06-18)

1. **Extended thinking** — ENABLE as a **UI toggle** (default on, modest
   budget). Thinking blocks are logged in `copilot_response.content_blocks`.
2. **Mesh persistence** — STORE meshes in blobs (content-addressed + gz).
   Revisit if disk gets out of hand.
3. **Config + storage location** — driven by a layered config:
   - packaged default `metagen-studio/config.yaml` (checked in),
   - overridden by `~/.config/metagen.yaml`, then `~/.metagenconfig.yaml`,
   - then env vars (`METAGEN_STUDIO_SESSION_DIR`, etc.).
   Default `sessions.dir = ~/.metagen-studio/sessions`. **On this machine**:
   `~/.config/metagen.yaml` sets it to `/ssd/benjones/metagen-dev/studio-sessions`.
   Session dirs are **gitignored** (metagen-dev repo + metagen-studio).
4. **Node granularity** — every action is a node. May add compaction later.
5. **Retention** — keep everything forever. Provide a **cleanup view/modal**
   showing per-session size + total, with: delete a session; delete everything
   off the current branch within a session; delete this session + all older
   (by last-use timestamp). No automatic GC.

## 11. Phased implementation plan

- **P1 — Logging spine (no UI rewind yet).** Session store + `events.jsonl` +
  thread `session_id` through all endpoints + emit all event types. Sessions
  auto-create/resume. Verifiable by inspecting the log files. *Lowest risk,
  unlocks everything.*
- **P2 — Tree + restore.** Build/maintain `tree.json`, blob store, `checkout`.
  Session bar (name, picker, new). Restoring a node loads editor/viewer/results.
- **P3 — Log Explorer view.** Standalone route + tree graph + node detail
  (thinking/tool/raw), "rewind here" wired to checkout → branch.
- **P4 — Auto-naming.** Haiku summarizer job + rename box + `name_source` guard.
- **P5 — Polish.** Node diffs, GC, export/import a session, keyboard nav.

Each phase is independently shippable and leaves the studio fully working.
```
