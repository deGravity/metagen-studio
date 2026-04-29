"""Command-line entry point for the packaged metaGen Studio.

`conda install metagen-studio` exposes a `metagen-studio` shell command
that launches a single uvicorn process serving both the FastAPI routes
and the bundled SPA. In dev mode, prefer `run.sh` / `run.bat` which run
vite + uvicorn separately with hot reload on both halves.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog='metagen-studio',
        description='Launch the metaGen Studio (single-process packaged mode).',
    )
    p.add_argument('--host', default='127.0.0.1',
                   help='Bind address (default 127.0.0.1; use 0.0.0.0 to expose).')
    p.add_argument('--port', type=int, default=8000, help='Port (default 8000).')
    p.add_argument('--no-browser', action='store_true',
                   help="Don't open a browser tab on startup.")
    p.add_argument('--log-level', default='info',
                   choices=['critical', 'error', 'warning', 'info', 'debug'])
    args = p.parse_args(argv)

    url = f'http://{args.host}:{args.port}'
    has_display = 'DISPLAY' in os.environ or sys.platform.startswith('win') or sys.platform == 'darwin'
    if not args.no_browser and has_display:
        # Open the browser shortly after uvicorn comes up. Daemon thread
        # so it doesn't block server shutdown if the browser hangs.
        def _open() -> None:
            time.sleep(1.5)
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    print(f'[metagen-studio] {url}')
    print('  Press Ctrl-C to stop.')

    import uvicorn
    uvicorn.run(
        'studio_backend.main:app',
        host=args.host, port=args.port,
        log_level=args.log_level,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
