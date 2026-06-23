# metagen-copilot

A provider-neutral, UI-decoupled agent engine for metamaterial-design copilots.
Extracted from metagen-studio (see `docs/COPILOT_PROVIDERS.md`); the studio, the
headless benchmark runner, and CAD-host integrations all depend on it.

## What's in here

- `engine` — `CopilotEngine`: the provider-neutral agent loop (stream → tool
  calls → results → loop), emitting a normalized `Event` stream. No HTTP/SSE.
- `providers/` — adapters mapping the normalized request to a vendor API and
  back: `anthropic`, `openai` (Responses **and** OpenAI-compatible
  chat-completions for vLLM), `gemini`. Each SDK is an optional extra, imported
  lazily — `import metagen_copilot` works with none installed.
- `tools` — `ToolRegistry` / `ToolEnv` / `Tool` / `ToolOutcome`. The registry is
  host-agnostic; the *host* binds tool handlers to its environment (the studio
  binds the metagen kernel; a CAD host adds its own integration tools).
- `pdf/` — pluggable attachment-ingest pipeline (PyMuPDF default; vision-OCR;
  remote marker/docling). Optional `pdf` extra.
- `benchmark` — `BenchmarkRunner` to drive the engine over a task suite headless.
- `types` — the normalized data model (`Msg`/`Part`/`Event`/`Capabilities`/…).

## Install

```bash
pip install metagen-copilot[anthropic,openai,gemini,pdf]   # or any subset
```

Core (`pip install metagen-copilot`) is dependency-free; add the extras for the
providers / PDF backends you actually use.

## Boundary

The engine imports nothing host-specific: it depends only on injected interfaces
(`ToolEnv`, tool handlers, an optional log callback) and yields `Event`s. This is
what lets the same engine power the studio, a benchmark sweep, and a CAD plugin.
See `metagen-studio/docs/COPILOT_PROVIDERS.md` §9.
