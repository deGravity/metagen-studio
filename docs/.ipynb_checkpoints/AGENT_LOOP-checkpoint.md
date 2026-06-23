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

**Key empirical finding (probed against `claude-opus-4-7`):**

| request                                   | result                          |
|-------------------------------------------|---------------------------------|
| baseline (no thinking arg)                | 200 OK, **0 thinking blocks**   |
| `enabled` + `budget_tokens` (legacy)      | **400** "not supported … use adaptive + output_config.effort" |
| `adaptive` + `effort:high` (and `disabled`)| 200 OK, **0 thinking blocks**   |
| + `interleaved-thinking` beta             | 200 OK, **0 thinking blocks**   |

So for **opus-4-7 the thinking trace is internal and never returned as content**
— not via create, not via streaming, not with the interleaved-thinking beta.
Consequences:

1. The "thinking" you saw **before** our changes was **not** a thinking trace —
   the model sent no thinking blocks. It was the model's normal, structured
   step-by-step **text** output (which reads like reasoning). Nothing in the
   pre-sessions harness requested or surfaced thinking.
2. Our P1 code used the **legacy** `enabled+budget` API, which **400s on
   opus-4-7** — and since thinking defaulted *on*, that broke every chat turn.
   (This is the `BadRequestError` you hit.)
3. **Fix:** the toggle now maps to the adaptive API —
   on → `thinking={type:'adaptive'}` + `output_config={effort}` (default
   `high`); off → `thinking={type:'disabled'}`. This **tunes internal reasoning
   depth / latency / quality**, but for opus-4-7 it does **not** produce a
   visible chain-of-thought, so the live "💭 thinking" panel and the logged
   `thinking` content blocks stay empty for this model.

The harness still *handles* returned thinking blocks (streams a `thinking` SSE
event; preserves them through tool round-trips; logs them) — so if you point the
studio at a model that *does* return a thinking trace, the CoT panel and the
session log will populate automatically with no further changes.
```
