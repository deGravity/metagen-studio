# Copilot agent loop & "thinking" — before vs after

This documents the studio copilot's server-side agent loop (`backend/studio_backend/chat.py::_agent_loop`)
before the sessions work and now, and clarifies what the **thinking** toggle does.

---

## 1. Before (pre-sessions)  — chat.py @ `645be02`

```
browser                        backend  /api/chat (SSE)                     Anthropic API
───────                        ──────────────────────────────              ─────────────
POST {messages, state, model}
        │
        ▼
   build system = [ static prompt+DSL docs (cached) , workspace-state ]
   api_messages = client-supplied history
        │
        ▼
   ┌─ for turn in 0..8 ────────────────────────────────────────────────┐
   │  client.messages.stream(model, max_tokens, messages,              │
   │                          system, tools, files-beta)   ───────────────►  (NO thinking arg)
   │     stream events:                                                │      model replies
   │        text         ──► SSE 'text'         ─────────────► browser │
   │        tool_use start──► SSE 'tool_call_start' ────────► browser  │
   │  final = get_final_message()                                      │
   │  assistant_blocks = [text, tool_use]   ← thinking blocks DROPPED   │
   │  SSE 'assistant_msg' ──────────────────────────────────► browser  │
   │  if stop_reason != tool_use:  SSE 'done'; return                  │
   │  else: for each tool_use:                                         │
   │           result = handler(args, state)   ◄── RUNS IN-PROCESS     │
   │                                               (asyncio.to_thread, │
   │                                                holds the GIL)     │
   │           SSE 'tool_ui' + 'tool_result' ───────────────► browser  │
   │        append tool_result blocks; continue loop                   │
   └───────────────────────────────────────────────────────────────────┘
```

Notes:
- **No thinking argument** was ever sent. The model used its own default.
- Tool handlers (`run_geometry` / `run_simulation`) ran **in-process** → the C++
  kernel held the GIL → the event loop stalled (this is the "network error"
  we later fixed by moving them to subprocesses).
- The reconstructed assistant message kept only `text` + `tool_use`.

---

## 2. After (current) — with sessions, subprocess tools, thinking toggle

```
browser                        backend /api/chat (SSE)                      Anthropic API
───────                        ──────────────────────────────              ─────────────
POST {messages, state, model, thinking?, session_id?}
        │
        ▼
   thinking → adaptive(+effort) | disabled        (per-request > config)
   if session_id: append_event(user_message)
        │
        ▼
   ┌─ for call_index in 0..8 ──────────────────────────────────────────┐
   │  log copilot_request  (full system+messages+thinking)             │
   │  client.messages.stream(model, max_tokens, messages, system,      │
   │       tools, thinking={adaptive|disabled},                ──────────►  adaptive thinking
   │       output_config={effort} , files-beta)                        │   (internal for 4.x)
   │     stream events:                                                │
   │        text     ──► SSE 'text'                ───────► browser    │
   │        thinking ──► SSE 'thinking'            ───────► browser    │  (only if model
   │        tool_use ──► SSE 'tool_call_start'     ───────► browser    │   emits it)
   │  final = get_final_message()                                      │
   │  api_blocks   = [thinking?, text, tool_use]  ← kept for API valid │
   │  display_blocks = [text, tool_use]                                │
   │  log copilot_response (content_blocks incl thinking + usage)      │
   │  SSE 'assistant_msg' (display_blocks) ──────────────► browser     │
   │  if stop_reason != tool_use:                                      │
   │       SSE 'done'; finalize_node(assistant_turn);                  │
   │       schedule out-of-band auto-name;  return                     │
   │  else: for each tool_use:                                         │
   │          result = handler(args|candidate-code, state)            │
   │                       ◄── RUNS IN A SUBPROCESS (kernel_job),      │
   │                           event loop stays free                   │
   │          log tool_exec (+ proposal); SSE 'tool_ui'+'tool_result'  │
   │        append tool_result blocks; continue loop                   │
   └───────────────────────────────────────────────────────────────────┘

   (button-initiated /api/execute & /api/simulate also append events +
    create geometry/sim nodes in the session DAG — see SESSIONS_DESIGN.md)
```

What changed vs. before:

| aspect            | before                       | after                                            |
|-------------------|------------------------------|--------------------------------------------------|
| thinking arg      | none                         | `thinking={adaptive|disabled}` + `output_config.effort` |
| tool execution    | in-process (GIL-blocking)    | subprocess (event loop free); candidate-code testing |
| assistant rebuild | text + tool_use              | thinking + text + tool_use (carried for API validity) |
| session logging   | none                         | user_message / copilot_request / copilot_response (incl thinking + usage) / tool_exec / proposal → one `assistant_turn` node |
| SSE events        | text, tool_call_start, tool_ui, tool_result, assistant_msg, done, error | + `thinking` |
| after a turn      | —                            | finalize node + out-of-band auto-name            |

---

## 3. What "thinking" actually is here

There are **two different thinking APIs**, and the model decides which applies:

- **Legacy (Claude 3.7-era):** `thinking={type:'enabled', budget_tokens:N}` →
  the model returns its chain-of-thought as `thinking` content blocks.
- **Adaptive (Claude 4.x, incl. `claude-opus-4-7`):** `thinking={type:'adaptive'}`
  + `output_config={effort: low|medium|high}` → the model **adaptively** decides
  how much to reason; effort tunes depth.

### 3a. The two APIs and what opus-4-7 requires

- **Manual / extended (Claude 3.7–Opus 4.5):** `thinking={type:'enabled',
  budget_tokens:N}` (budget < max_tokens). Returns `thinking` content blocks.
- **Adaptive (Claude 4.6+ incl. opus-4-7/4.8 — the ONLY mode they accept):**
  `thinking={type:'adaptive'}` and **`output_config={effort: low|medium|high|xhigh|max}`**.
  On opus-4-7/4.8, manual `enabled+budget` is **rejected with a 400** (the error
  you hit). Thinking is **off** unless you explicitly pass `thinking=adaptive`.

Two non-obvious wrinkles that produce the "no thinking" surprise:

1. **`display` defaults to `"omitted"` on opus-4.7/4.8.** The model still thinks,
   but the `thinking` field comes back **empty** (only an encrypted `signature`).
   You must set **`display:"summarized"`** to receive the readable summary.
2. **Adaptive *skips* thinking on easy turns at `effort:high`.** It only "almost
   always" thinks; `xhigh`/`max` engage it reliably and more deeply (`xhigh` is
   the documented opus-4-7 sweet spot for agentic/coding work). `effort` also
   bounds *total* token spend (text + tool calls + thinking), so thinking needs
   `max_tokens` headroom.

**Empirical (probed against `claude-opus-4-7`):**

| request                                              | thinking block returned?     |
|------------------------------------------------------|------------------------------|
| `enabled` + `budget_tokens` (legacy)                 | **400 — not supported**      |
| `adaptive` + `effort:high`, **no `display`**         | none (omitted + often skips) |
| `adaptive` + `display:summarized` + `effort:high`    | usually none (skipped)       |
| `adaptive` + `display:summarized` + `effort:xhigh`   | **yes — 815 chars, 889 thinking_tokens** |
| `adaptive` + `display:summarized` + `effort:max`     | **yes — 2360 chars, 4312 thinking_tokens** |
| streaming, `adaptive+summarized+xhigh`               | **12 `thinking` deltas → our `thinking` SSE fires** |

So **opus-4-7 *does* expose a (summarized) chain-of-thought** — earlier I
concluded otherwise because I omitted `display:"summarized"` and tested at
`high` (where adaptive skipped). Corrected understanding:

1. The "thinking" seen **before** our changes was the model's normal step-by-step
   **text** (pre-sessions never sent a thinking arg, so no thinking blocks).
2. P1 used the **legacy `enabled+budget`** API → **400** on opus-4-7, and since
   thinking defaulted on it broke every turn (the `BadRequestError`).
3. **Fix (current):** toggle on → `thinking={type:'adaptive', display:'summarized'}`
   + `output_config={effort}` (config `copilot.thinking.effort`, default `high`)
   + `max_tokens` raised to `copilot.thinking.max_tokens` (16k) for headroom;
   toggle off → `thinking={type:'disabled'}`. When the model thinks, the summary
   streams as `thinking` deltas → the `thinking` SSE → the live "💭 thinking"
   panel and the session log populate. At `high` thinking is adaptive (shows on
   the turns the model deems worth it); set effort to `xhigh`/`max` to force it.

**Tool-use note:** with thinking active, `thinking` blocks (incl. their
`signature`) must be passed back unchanged on tool round-trips — the loop
reconstructs `api_blocks` with thinking + signature to satisfy this; adaptive
auto-enables interleaved thinking (thinking between tool calls).
```
