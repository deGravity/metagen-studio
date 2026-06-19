# Provider-agnostic copilot — research & plan

Status: **proposal**. Goal: decouple the copilot from (a) Anthropic and (b) the
studio UI, so we can swap the backing LLM (Claude / Gemini / GPT / open models
via our vLLM server), run it **headless for benchmarking**, and later embed it
in other tools (e.g. a CAD plugin).

---

## 1. Goals / non-goals

**Goals**
- One **copilot engine** (agent loop + tools + transcript) that is LLM- and
  UI-agnostic.
- **Provider adapters** for Anthropic, Google Gemini, OpenAI, and OpenAI-
  compatible vLLM (Qwen 3.x, gpt-oss-120b).
- A **capability model** so the engine adapts to what each model supports
  (tool calling, PDF, images, reasoning), with our own **preprocessing** where a
  capability is missing (notably PDF for text-only open models).
- **Headless mode** for benchmarking many models over task suites.
- A clean **embedding boundary** so a CAD plugin (or anything) can host the
  copilot with its own tool implementations.

**Non-goals (now)**: changing the DSL/kernel; a universal "best" model; fine-
tuning. Streaming-to-UI stays, but becomes one consumer among several.

---

## 2. What's coupled today (to undo)

`backend/studio_backend/chat.py` is Anthropic- and UI-specific throughout:
- `AsyncAnthropic`, Anthropic message/content-block shape, `tool_use`/
  `tool_result` blocks, `cache_control` system blocks, adaptive-thinking params.
- Tool **schemas** are written in Anthropic's `input_schema` form; tool
  **handlers** call the kernel *and* emit UI/SSE events (`tool_ui`, geometry/sim
  summaries) — i.e. transport + business logic + provider format are entangled.
- PDF ingest = Anthropic Files API (`/api/chat/upload`, files-api beta).
- The loop yields studio SSE events directly; there is no transcript object the
  caller owns — the browser is the source of truth for history.

The **session store** (events.jsonl + state DAG) we already built is the right
substrate for transcripts in *all* modes, including benchmarking.

---

## 3. Capability matrix (researched)

| capability | Claude (opus-4.x) | OpenAI (GPT-5.x, Responses API) | Gemini (2.5/3) | vLLM · Qwen3 | vLLM · gpt-oss-120b |
|---|---|---|---|---|---|
| API shape | Anthropic Messages | Responses API (preferred) | `generateContent` | OpenAI Chat Completions (compat) | OpenAI Chat Completions (compat) |
| tool calling | native (`tool_use`), parallel | native functions, parallel | native `functionDeclarations`, parallel | yes — `--enable-auto-tool-choice --tool-call-parser qwen3_coder`/hermes | yes — harmony/`openai` tool parser |
| tool_choice | `auto`/`none` (not `any`/named while thinking) | `auto`/`required`/`none`/named | `auto`/`any`/`none` | `auto`/`required`/`none` | `auto`/`required`/`none` |
| native PDF | ✅ document block + Files API | ✅ `input_file` (extracts text **and** page images for vision models) | ✅ inline_data / File API, ≤1000 pg / 50 MB, native (no manual OCR) | ❌ text-only | ❌ text-only |
| images | ✅ | ✅ (vision) | ✅ | ✅ on VL variants (Qwen3-VL); base/text Qwen3 = text-only | ❌ (text-only) |
| reasoning / "thinking" | adaptive + `output_config.effort`; summarized CoT + `signature` | `reasoning.effort` low/med/high/xhigh + reasoning summaries | `thinkingConfig.thinkingBudget` + thought summaries | `enable_thinking` + `reasoning_content` (`--reasoning-parser qwen3`) | reasoning effort via harmony "analysis" channel |
| streaming | SSE: text/thinking/tool deltas | SSE: Responses events | `streamGenerateContent` | OpenAI-compat stream; `reasoning_content` + tool deltas (parser-dependent) | same |

**Consequences for our harness**
- **Tool calling** is universal but in 3 wire formats (Anthropic blocks, OpenAI
  functions, Gemini declarations) + vLLM's OpenAI-compatible variant whose
  *streaming* tool-delta shape depends on the server's `--tool-call-parser`.
  → normalize to one internal tool/call/result model.
- **PDF** is native only on Claude/Gemini/OpenAI-vision. For text-only open
  models (gpt-oss; base Qwen3) we must **preprocess**; even for vision-capable
  open models (Qwen3-VL) we still need to *render* the PDF to images ourselves.
  So PDF ingest is a **pluggable preprocessing pipeline** (§4.4), not a single
  extractor — text-only and vision passes, technique selectable in config.
- **Reasoning** exists everywhere but is exposed differently (Anthropic effort+
  summarized blocks; OpenAI reasoning.effort+summaries; Gemini thinkingBudget;
  vLLM `reasoning_content` / harmony). → one normalized "effort" knob + a
  normalized `thinking` event; not all providers return visible CoT.

Sources: vLLM tool calling & reasoning docs; Gemini document-processing; OpenAI
Responses file-input / reasoning docs (see chat transcript for links).

---

## 4. Architecture

```
            ┌──────────────────────────────────────────────────────────┐
 consumers  │  studio FastAPI/SSE   headless benchmark runner   CAD plugin│
            └───────────────▲───────────────▲───────────────▲────────────┘
                            │  normalized events (async stream) + ToolEnv
            ┌───────────────┴───────────────────────────────────────────┐
 engine     │  CopilotEngine: agent loop, transcript, tool dispatch,     │
            │  attachment preprocessing, reasoning/effort normalization  │
            └───────────────▲───────────────────────────────────────────┘
                            │  Provider interface (normalized in/out)
            ┌───────────────┴───────────────────────────────────────────┐
 adapters   │  Anthropic   OpenAI(Responses)   Gemini   OpenAICompat(vLLM)│
            └────────────────────────────────────────────────────────────┘
```

### 4.1 Normalized data model (`copilot/types.py`)
- `Msg{role, parts}` where `Part` ∈ `Text | Image | Document(pdf) | ToolCall |
  ToolResult | Thinking{text?, signature?}`.
- `ToolDef{name, description, json_schema}` (provider-neutral JSON Schema).
- `Capabilities{tools, tool_streaming, parallel_tools, native_pdf,
  native_images, reasoning: none|effort|budget|toggle, returns_cot}`.
- `Event` (engine output stream): `text_delta | thinking_delta | tool_call |
  tool_result | message | usage | done | error` **+ sidecar** `artifact`
  (typed: `geometry`, `sim`, `proposal`, …) so non-UI consumers can ignore them.

### 4.2 Provider interface (`copilot/providers/base.py`)
```python
class Provider(Protocol):
    name: str
    def capabilities(self, model: str) -> Capabilities: ...
    async def stream(self, *, model, system, messages: list[Msg],
                     tools: list[ToolDef], reasoning_effort: str|None,
                     max_tokens: int) -> AsyncIterator[Event]: ...
```
Each adapter (a) maps `messages`/`tools` to its wire format, (b) sets the right
reasoning param for the model, (c) parses its stream into normalized `Event`s
(incl. tool-call assembly and `thinking`/`reasoning_content`), (d) reconstructs
the assistant turn (preserving thinking signatures where required). vLLM is just
the OpenAI adapter pointed at a `base_url` with a per-model profile (which tool
parser / reasoning field the server uses).

### 4.3 Tool registry — decoupled from UI (`copilot/tools.py`)
```python
@dataclass
class Tool:
    defn: ToolDef
    handler: Callable[[dict, ToolEnv], Awaitable[ToolOutcome]]
# ToolOutcome = {result_for_model, artifacts:[Event], error?}
```
- `ToolEnv` carries the *environment* (compiled program, kernel runner, session
  id) — **not** the transport. The studio binds `run_geometry`/`run_simulation`/
  `propose_edit` to kernel_job + emits `artifact` events; a CAD plugin binds the
  same names to its own geometry engine; the benchmark runner binds them to a
  headless kernel with no UI emission. **Same engine, different `ToolEnv`.**

### 4.4 PDF / attachment preprocessing pipeline (`copilot/pdf/`)
A **pluggable** pipeline (not one extractor), so we can A/B techniques per the
research and pick winners in config. A backend ingests a PDF → an
`IngestedDoc{markdown|text, page_images?, per_page_confidence?, meta}`.

**Backends (selectable, `copilot.pdf.backend`):**
- `pymupdf4llm` — fast, no-ML, native-text → markdown. Default for digital PDFs;
  great latency, weak on scans/complex tables.
- `marker` (datalab) — deep-learning pipeline → markdown+JSON, LaTeX equations,
  tables, extracted images. Best for papers/equations; **GPU-recommended**
  (5–10× slower on CPU) — a natural fit to run **on the vLLM host/GPU box**, not
  here.
- `pdfmux` — RAG-oriented hybrid (PyMuPDF-speed pages, escalates table pages to
  Docling) with **per-page confidence scores**; strong headings/tables.
- `vision_ocr` — render pages → images and run a **vision model** (the active
  provider if vision-capable, or a configured dedicated VLM) to emit
  markdown/text. Highest fidelity on scans/diagrams; highest cost.
- (`docling`, `mineru` are easy future drop-ins behind the same interface.)

**Routing by target-model capability × config `mode` (`text_only|images|both`):**
- `native_pdf` provider (Claude/Gemini/OpenAI-vision) → pass the `Document`
  through **unless** config forces a backend (so we can benchmark "native PDF"
  vs "our markdown" head-to-head).
- `native_images` provider (Qwen3-VL) → page images (+ optional extracted text).
- text-only provider (gpt-oss, base Qwen3) → markdown/text from the chosen
  backend; **never** images.

The pipeline caches `IngestedDoc` by `(pdf_hash, backend, opts)` (reuse the
session blob store), so repeated turns and benchmark reruns don't re-extract.
Backends declare their own deps/placement (e.g. `marker` may live as a small
service on the GPU host) so this machine isn't forced to host heavy models.

### 4.5 Reasoning normalization
- Engine exposes one `effort ∈ {off,low,medium,high,xhigh,max}`. Adapters map:
  Anthropic→`thinking{adaptive,display:summarized}`+`effort`; OpenAI→
  `reasoning.effort`; Gemini→`thinkingBudget`; vLLM→`enable_thinking`/harmony
  effort. `returns_cot` tells consumers whether to expect a `thinking` stream.

### 4.6 The agent loop (provider-neutral)
Owns: build system+messages from a `Transcript`, call `provider.stream`, relay
`Event`s, on `tool_call` look up the `Tool`, run its handler with `ToolEnv`,
append `tool_result`, loop to `MAX_TURNS`; persist every step to the session
event log. No Anthropic or HTTP specifics here.

---

## 5. Headless / benchmarking mode

- A `BenchmarkRunner` drives `CopilotEngine` over a task suite (prompt + optional
  attachments + a scoring fn) with a chosen provider/model/effort, a **no-UI
  ToolEnv** (kernel runs, results captured, no SSE), recording each run as a
  **session** (reuse events.jsonl + DAG). Deterministic where the provider
  allows (temperature/seed); otherwise N repeats.
- Scoring hooks per task: did `propose_edit` compile? does the geometry sim hit
  target moduli / vf within tolerance? tool-call efficiency / token cost. Output:
  a per-(model, task) table + the full transcripts for inspection in the existing
  Log Explorer.
- This is the payoff of decoupling: **the same engine + tools** evaluated across
  Claude/Gemini/GPT/Qwen3/gpt-oss with one config switch.

---

## 6. Configuration (`config.yaml`)
```yaml
copilot:
  provider: anthropic            # anthropic | openai | gemini | vllm
  model: claude-opus-4-7
  effort: high
  providers:
    anthropic: { key_env: METAGEN_ANTHROPIC_API_KEY }
    openai:    { key_env: METAGEN_OPENAI_KEY }
    gemini:    { key_env: METAGEN_GOOGLE_KEY }
    vllm:                         # OpenAI-compatible
      base_url: http://<vllm-host>:8000/v1
      key_env: METAGEN_VLLM_KEY   # often a dummy
      models:                     # per-model profile (parser quirks, caps)
        qwen3:       { tool_parser: qwen3_coder, reasoning: toggle, native_pdf: false, native_images: false }
        qwen3-vl:    { tool_parser: qwen3_coder, reasoning: toggle, native_pdf: false, native_images: true  }
        gpt-oss-120b:{ tool_parser: harmony,     reasoning: effort,  native_pdf: false, native_images: false }
  pdf:
    backend: pymupdf4llm          # pymupdf4llm | marker | pdfmux | vision_ocr
    mode: both                    # text_only | images | both (gated by model caps)
    vision_ocr: { provider: gemini, model: gemini-2.5-flash }   # used by vision_ocr backend
    marker: { endpoint: http://<gpu-host>:8xxx }                # optional remote GPU service
```
Per-request overrides (provider/model/effort/pdf-backend) flow through
`ChatRequest`, so the UI dropdown and the benchmark runner use the same path.

**Reproduce-current-behavior (compat):** the shipped defaults reproduce today's
studio exactly — `provider: anthropic`, `model: claude-opus-4-7`, adaptive
thinking (`display: summarized`, `effort: high`), the existing system prompt +
DSL-docs injection, and Anthropic-native PDF. Determinism knobs (temperature,
seed) are **opt-in settings** used by the benchmark runner; the interactive
default leaves them unset so the live copilot behaves identically to now.

---

## 7. Phased implementation

- **P1 — extract the engine.** Pull the agent loop out of `chat.py` into a
  provider-neutral `CopilotEngine` + normalized types + tool registry, with the
  **Anthropic adapter** as the first provider. Studio behaves identically; this
  is a pure refactor (sessions logging preserved). *De-risks everything.*
- **P2 — OpenAI-style adapter, two modes.** `mode: responses` for **OpenAI**
  (GPT-5.x, best tool/reasoning story) and `mode: chat_completions` for **vLLM**
  (Qwen3, gpt-oss) via `base_url` + per-model profiles; tool-call + `reasoning_
  content` parsing; `tool_choice`. Unlocks the open models on your server first
  (no PDF needed for code tasks).
- **P3 — PDF preprocessing pipeline** (§4.4): backend interface +
  `pymupdf4llm`/`pdfmux`/`marker`/`vision_ocr`, capability-gated routing, cached
  ingest. Config-selectable so we can compare techniques.
- **P4 — Gemini adapter** (native PDF + thinkingBudget + thought summaries).
- **P5 — headless runner + adapt existing evals.** Wire the BenchmarkRunner over
  `CopilotEngine` with a no-UI `ToolEnv`; **adapt the existing non-agentic
  codegen eval suites first** (lower lift) before authoring a new agentic suite.
  Capture CoT + token cost per run into the session log; cross-model report.
- **P6 — embedding boundary doc + example** (CAD-plugin-style host that supplies
  its own `ToolEnv`).

Each phase ships independently; P1 leaves the studio unchanged.

---

## 8. Decisions (resolved 2026-06-19)
1. **OpenAI surface** — one OpenAI-style adapter with **two modes**: `responses`
   for OpenAI (GPT-5.x), `chat_completions` for vLLM (Qwen3, gpt-oss).
2. **PDF** — build **both** a text-only pipeline **and** a vision-model OCR pass,
   as selectable backends (§4.4), so we can compare techniques; choice lives in
   config. (Qwen3-VL can take page images directly; gpt-oss stays text-only.)
3. **CoT capture** — yes: record provider CoT where available (Anthropic/OpenAI/
   Gemini summaries, vLLM `reasoning_content`) into the session log.
4. **Determinism** — make temp/seed/effort **settings** (used by the benchmark
   runner); the interactive defaults must **reproduce current studio behavior
   exactly** (see compat note in §6). Exact N / temperature policy: TBD.
5. **Eval suite** — **adapt the existing non-agentic codegen eval suites first**;
   a new agentic eval suite is not yet critical (deferred within P5).

### Still to nail down later
- Determinism policy for vendor APIs (temperature/seed support varies) + repeat
  count N for non-deterministic providers.
- Which PDF backend(s) become the default after the A/B comparison.
