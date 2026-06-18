"""In-memory cache of geometry/sim results keyed by code hash.

Lets an accepted copilot proposal reuse a generation/simulation the copilot
already ran on that exact code — so accepting immediately updates the viewer
and results instead of forcing a re-run. Populated by every result-producing
path (the Run buttons and the copilot tools), looked up on accept via
POST /api/results/cached.

One entry per code_hash (latest result wins); small LRU bound. Process-local
and ephemeral — purely a convenience cache.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional

_MAX = 64
_geom: "OrderedDict[str, dict]" = OrderedDict()
_sim: "OrderedDict[str, dict]" = OrderedDict()
_lock = threading.Lock()


def _put(store: "OrderedDict[str, dict]", code_hash: str, payload: dict) -> None:
    with _lock:
        store[code_hash] = payload
        store.move_to_end(code_hash)
        while len(store) > _MAX:
            store.popitem(last=False)


def put_geometry(code_hash: str, payload: dict) -> None:
    _put(_geom, code_hash, payload)


def put_sim(code_hash: str, payload: dict) -> None:
    _put(_sim, code_hash, payload)


def get(code_hash: str) -> dict:
    """Latest cached geometry/sim for this code (either may be None)."""
    with _lock:
        return {'geometry': _geom.get(code_hash), 'sim': _sim.get(code_hash)}
