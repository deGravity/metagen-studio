#!/usr/bin/env bash
# Start the studio backend and frontend dev servers.
# Usage:  ./studio/run.sh
# Env:    STUDIO_BACKEND_PORT (default 8000), STUDIO_FRONTEND_PORT (default 5173)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${STUDIO_PY:-/home/ben/miniconda3/envs/metamaterials-dev/bin/python}"
BACKEND_PORT="${STUDIO_BACKEND_PORT:-8000}"
FRONTEND_PORT="${STUDIO_FRONTEND_PORT:-5173}"

if [[ ! -x "$PY" ]]; then
  echo "Python not found at $PY; set STUDIO_PY env var" >&2
  exit 1
fi

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[studio] backend  → http://localhost:$BACKEND_PORT"
"$PY" -m uvicorn studio_backend.main:app \
  --app-dir studio/backend \
  --host 0.0.0.0 --port "$BACKEND_PORT" \
  --reload &

echo "[studio] frontend → http://localhost:$FRONTEND_PORT"
( cd studio/frontend && npm run dev -- --port "$FRONTEND_PORT" --host ) &

wait
