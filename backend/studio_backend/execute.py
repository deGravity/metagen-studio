"""DSL code execution + per-program Structure caching.

A user-supplied `code.py` defines `make_structure(...) -> Structure`. We
exec the source in a fresh module namespace seeded with `metagen_dsl`
star-imports (matching the legacy `from metagen import *` convention),
then call `make_structure()` and hand the resulting `Structure` to the
caller. Each `Structure` carries its own LRU on `(resolution, mode)` etc.,
so re-running with the same parameters costs nothing.

Trust model: this is single-user dev tooling. The DSL code is exec'd
verbatim in the backend process; do not expose this service to untrusted
input.
"""
from __future__ import annotations

import hashlib
import sys
import threading
import traceback
import types
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional


def hash_code(code: str) -> str:
    return hashlib.sha256(code.encode('utf-8')).hexdigest()[:12]


@dataclass
class CompiledProgram:
    code_hash: str
    structure: object  # metagen_dsl.Structure
    error: Optional[str] = None


class ProgramCache:
    """LRU keyed on code_hash → compiled Structure."""

    def __init__(self, max_entries: int = 32):
        self._d: OrderedDict[str, CompiledProgram] = OrderedDict()
        self._max = max_entries
        self._lock = threading.Lock()

    def get_or_compile(self, code: str) -> CompiledProgram:
        h = hash_code(code)
        with self._lock:
            if h in self._d:
                self._d.move_to_end(h)
                return self._d[h]
        compiled = _compile(code, h)
        with self._lock:
            self._d[h] = compiled
            self._d.move_to_end(h)
            while len(self._d) > self._max:
                self._d.popitem(last=False)
        return compiled

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._d.keys())

    def __len__(self) -> int:
        return len(self._d)


def _compile(code: str, code_hash: str) -> CompiledProgram:
    """Exec code in an isolated namespace and grab make_structure()."""
    import metagen_dsl

    # Match the legacy `from metagen import *` shim: alias metagen_dsl as
    # `metagen` in sys.modules so user code that says `from metagen import *`
    # still works.
    sys.modules.setdefault('metagen', metagen_dsl)

    mod = types.ModuleType(f'studio_user_{code_hash}')
    # Seed the namespace with the DSL public API so user code can `from
    # metagen import *` or just reference symbols directly.
    for name in dir(metagen_dsl):
        if not name.startswith('_'):
            setattr(mod, name, getattr(metagen_dsl, name))
    mod.__dict__['__name__'] = mod.__name__

    try:
        exec(compile(code, f'<user:{code_hash}>', 'exec'), mod.__dict__)
    except Exception:
        return CompiledProgram(
            code_hash=code_hash, structure=None,
            error=traceback.format_exc())

    make = mod.__dict__.get('make_structure')
    if not callable(make):
        return CompiledProgram(
            code_hash=code_hash, structure=None,
            error="user code defines no make_structure() function")

    try:
        struct = make()
    except Exception:
        return CompiledProgram(
            code_hash=code_hash, structure=None,
            error=traceback.format_exc())

    return CompiledProgram(code_hash=code_hash, structure=struct)
