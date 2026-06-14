#!/usr/bin/env bash
# Start the studio backend and frontend dev servers (development mode).
# For a packaged install instead, use the `metagen-studio` CLI entry
# point that ships with the conda package.
#
# Usage:  ./run.sh
# Env:    STUDIO_BACKEND_PORT      (default 8000)
#         STUDIO_FRONTEND_PORT     (default 5173)
#         STUDIO_DEV_WATCHDOG      (default 1) — poll /api/info and abort
#                                    loudly if the backend stops responding.
#                                    `uvicorn --reload` occasionally leaves
#                                    a zombie worker behind a still-listening
#                                    socket; this catches that.
set -euo pipefail
cd "$(dirname "$0")"

PY="${STUDIO_PY:-/home/ben/miniconda3/envs/metamaterials-dev/bin/python}"
BACKEND_PORT="${STUDIO_BACKEND_PORT:-8000}"
FRONTEND_PORT="${STUDIO_FRONTEND_PORT:-5173}"
DEV_WATCHDOG="${STUDIO_DEV_WATCHDOG:-1}"

if [[ ! -x "$PY" ]]; then
  echo "Python not found at $PY; set STUDIO_PY env var" >&2
  exit 1
fi

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[studio] backend  → http://localhost:$BACKEND_PORT"
"$PY" -m uvicorn studio_backend.main:app \
  --app-dir backend \
  --host 0.0.0.0 --port "$BACKEND_PORT" \
  --reload &

echo "[studio] frontend → http://localhost:$FRONTEND_PORT"
( cd frontend && npm run dev -- --port "$FRONTEND_PORT" --host ) &

watchdog() {
  # Health-poll the backend. If /api/info doesn't respond for
  # WATCHDOG_FAIL_THRESHOLD consecutive polls, log loudly and TERM
  # the parent script so the trap fires and tears everything down.
  local startup_grace=15
  local interval=5
  local timeout=3
  local fails=0
  local max_fails=3
  sleep "$startup_grace"
  while sleep "$interval"; do
    if curl -sf -m "$timeout" -o /dev/null "http://localhost:$BACKEND_PORT/api/info"; then
      fails=0
    else
      fails=$((fails + 1))
      echo "[studio][watchdog] backend not responding (fail $fails/$max_fails)" >&2
      if (( fails >= max_fails )); then
        echo "[studio][watchdog] backend unresponsive — likely a zombie worker after a --reload crash. Shutting down so you can restart cleanly." >&2
        kill -TERM $$
        return
      fi
    fi
  done
}

if [[ "$DEV_WATCHDOG" == "1" ]]; then
  echo "[studio] watchdog enabled (set STUDIO_DEV_WATCHDOG=0 to disable)"
  watchdog &
fi

wait
