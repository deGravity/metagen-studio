# metagen-studio

Browser-based interactive CAD studio + AI copilot for [metaDSL][dsl]
metamaterial programs. Edit code in a Monaco editor, regenerate
geometry on demand, run simulations, and chat with a copilot that can
read your code, propose edits, and trigger runs.

> **Status:** internal pre-publication. The associated paper is not yet
> released; please don't redistribute.

## Install (production)

The studio and its native dependencies (geometry kernel, FEM simulator)
are published as conda packages on a private S3 channel. One-time:

```sh
conda config --add channels https://metagen-dist.s3.amazonaws.com/conda/
conda create -n metagen-studio metagen-studio
conda activate metagen-studio
```

Then launch:

```sh
metagen-studio
# opens a browser tab on http://127.0.0.1:8000
```

Useful flags:

```
metagen-studio --port 8000          # bind port
metagen-studio --host 0.0.0.0       # expose on the network (default 127.0.0.1)
metagen-studio --no-browser         # don't auto-open a tab
```

The packaged install runs as a single uvicorn process serving both
`/api/*` (FastAPI) and `/` (the bundled Vite/React SPA).

## Development setup

Clone alongside [metagen-dev][dev] (which carries the kernel/simulator
build paths) or as a standalone checkout, then:

```sh
# Linux/macOS
./run.sh

# Windows
run.bat
```

This launches **two** processes — `vite` on `:5173` (frontend, hot
reload) and `uvicorn --reload` on `:8000` (backend, hot reload) —
proxied so the browser sees a unified site at `:5173`. Edits to either
side restart the appropriate half.

One-time setup before the first `run.sh`:

```sh
# backend Python deps (use the dev env from metagen-dev/environment-dev.yml)
cd backend && pip install -e . && cd ..

# frontend deps
cd frontend && npm install && cd ..
```

The dev mode imports the metagen native extensions from a sibling
metagen-dev checkout via a `sys.path` fallback in `studio_backend/main.py`
(triggered when the conda-installed copies aren't found). This lets you
hack on the kernel and simulator alongside the studio without
reinstalling.

## Configuration

Environment variables read at startup:

| Variable | Default | Purpose |
|---|---|---|
| `METAGEN_ANTHROPIC_API_KEY` | unset | Enables the Copilot tab in the UI. Without it, the chat panel shows a "set the env var" message. Prefixed to avoid colliding with Claude Code's own `ANTHROPIC_API_KEY`. |
| `STUDIO_PY` (dev only) | metagen-dev's `metamaterials-dev` Python | Path to the Python interpreter used by `run.sh` / `run.bat`. |
| `STUDIO_BACKEND_PORT` (dev only) | `8000` | Backend port. |
| `STUDIO_FRONTEND_PORT` (dev only) | `5173` | Vite port. |

## What's inside

```
metagen-studio/
  backend/                    Python package (FastAPI app, copilot, exec cache)
    pyproject.toml
    studio_backend/
      main.py                 routes + conditional SPA mount
      cli.py                  metagen-studio console entry point
      chat.py                 Anthropic streaming + tool dispatch
      execute.py              compiles user DSL code with caching
      models.py               request/response schemas
      state.py                shared program cache singleton
  frontend/                   Vite + React + TypeScript SPA
    package.json
    src/
      App.tsx                 layout
      components/             Editor, Viewer3D, Settings, Results, Chat
      api.ts, types.ts, ...
  run.sh, run.bat             dev launchers
  BACKLOG.md                  known issues + deferred features
```

## API surface (backend)

- `GET /api/info` — runtime probe (GPU available, valid GPU
  resolutions, chat enabled?, cache stats)
- `POST /api/execute` — compile DSL → run kernel → return mesh + stats
- `POST /api/simulate` — run homogenization → return 6×6 C matrix +
  derived properties
- `POST /api/chat` — SSE-streamed copilot agent loop with tool use
  (`propose_edit`, `run_geometry`, `run_simulation`)

## Related repos

- [metagen-dev][dev] — top-level workspace (kernel, simulator, dsl as
  submodules; build/test infrastructure)
- [metagen-dsl][dsl] — pure-Python DSL (geometry authoring)
- metagen-kernel — C++ pybind11 geometry generator
- metagen-simulator — C++/CUDA pybind11 FEM homogenization

[dsl]: https://github.com/deGravity/metagen-dsl
[dev]: https://github.com/deGravity/metagen-dev
