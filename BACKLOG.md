# studio backlog

Items deferred from in-flight work. Each entry: short title, observed
behavior, suspected cause / starting point.

## bugs

### staleness tag is always shown, even right after a fresh run
Observed Phase 1. Geometry and Sim sections in the right pane render the
amber `(stale)` tag immediately after a successful run, with no edits.

Background-check that already passed: a one-shot test confirmed
`hashlib.sha256(b'from metagen import *\n').hexdigest()[:12]` and the
equivalent JS `crypto.subtle.digest` → first 6 bytes → hex both produce
`aa489e0337a6`, so the hashing logic itself agrees byte-for-byte.

Suspected causes (try in this order):
1. **Monaco end-of-line normalization.** The default `Editor` may emit
   `\r\n` line endings via `onChange` even when `value` was passed in as
   `\n`-terminated. The first onChange fires after mount and silently
   replaces `code` with a CRLF version; backend hashes whatever bytes the
   request body contains, frontend hashes its current state; they may
   diverge. Fix candidate: set `EOL: '\n'` in Monaco options, or normalize
   `\r\n` → `\n` in the `onChange` callback before `setCode`.
2. **useEffect race.** `useEffect(() => hashCode(code).then(setCurrentHash), [code])`
   runs an async hash. If `runGeometry` fires before that promise resolves
   (e.g. immediately after page load), `currentHash` is still `''` when
   the response lands, so `geometry.code_hash !== currentHash` is true.
   Fix candidate: hash synchronously in the same effect that triggers
   the API call, or compute the hash inline at staleness-check time.
3. **Trailing newline mismatch.** The DEFAULT_CODE template literal in
   `App.tsx` ends with a `\n`, but Monaco may strip it on read-back.
   Compare `code.length` and the last char on the wire vs in state.

To debug: log `code` length and first/last 50 chars on both sides next
to each request, plus `currentHash` at click time.

## features

### geometry-run progress feedback
No live indicator while `kernel.generate` runs. At res 97 dense crystals,
`generate()` can take 100+ seconds; the user just sees the disabled "Run
geometry" button with no signal of liveness.

Approach options:
- **(easy)** Indeterminate spinner + elapsed-time counter on the button
  while the request is in flight.
- **(medium)** Backend instrumentation: have `metagen-kernel` expose
  progress via callbacks. The kernel's internals report `Tgt len`, `E N`
  iteration counts etc. to stdout — pipe that into a per-request log
  stream the frontend can consume via SSE.
- **(harder)** Phase-based progress: emit "preparing graph", "evaluating
  level 4", "voxelizing", "extracting mesh" as discrete events. Requires
  C++ bindings for progress hooks.

Start with the easy version; promote to SSE log streaming once it's
worth the kernel-side work.

### cancel + restart on parameter change during long-running generation
When a generation is in flight and the user changes resolution (or TPMS
mode), the in-flight request continues to its full duration even though
the user no longer wants it. They have to wait + then re-click.

Approach:
1. Frontend tracks `currentRequestId` (UUID per click). New request →
   cancels previous via `AbortController`. Frontend logic: on parameter
   change while busy, abort + re-fire automatically (or surface a "cancel
   + restart" prompt).
2. Backend needs to honor cancellation. FastAPI's `Request.is_disconnected()`
   helps, but the heavy work happens inside `kernel.generate()` which is a
   blocking C++ call holding the GIL — Python can't preempt it. Real
   cancellation needs C++ cooperation: a `std::atomic<bool>` cancel flag
   that `StructureRepresentation::evaluateGraphBasedOnLevel` periodically
   checks. Until that's wired through, the best the backend can do is
   ignore the (now-stale) result when it finishes.
3. Concurrency: while a generation runs, the kernel is single-threaded
   in this process. Auto-restart should queue the new generation; if the
   in-flight one is uncancellable yet, finish it (silently discard) then
   start the new one. Show a "restarting…" badge.

This unlocks meaningful interactivity at high resolutions. Start with
the frontend-side abort + queueing; the C++ cancel hook is a bigger
ticket against `metagen-kernel`.
