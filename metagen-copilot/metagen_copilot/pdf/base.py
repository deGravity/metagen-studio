"""PDF backend interface + ingest cache protocol.

A PdfBackend ingests raw PDF bytes into an IngestedDoc. Backends declare their
own availability (a heavy/remote backend reports unavailable instead of
importing torch on this box). The cache is an injected protocol so the copilot
package stays free of studio coupling — the studio binds it to the session blob
store; the benchmark runner / a CAD host bind their own. See §4.4 / §9.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from .types import IngestedDoc, IngestOpts


class BackendUnavailable(RuntimeError):
    """Raised when a backend's dependencies/service aren't reachable here.
    Carries an actionable message (what to install / which host to point at)."""


@runtime_checkable
class PdfBackend(Protocol):
    name: str

    def available(self) -> bool:
        """True if this backend can run in the current environment."""
        ...

    def ingest(self, data: bytes, opts: IngestOpts) -> IngestedDoc:
        """Ingest PDF bytes. Raises BackendUnavailable if not runnable."""
        ...


@runtime_checkable
class IngestCache(Protocol):
    """Content-addressed cache for ingest results, keyed by a string the
    pipeline derives from (pdf_hash, backend, opts)."""

    def get(self, key: str) -> Optional[IngestedDoc]:
        ...

    def put(self, key: str, doc: IngestedDoc) -> None:
        ...


class InMemoryIngestCache:
    """Default process-local cache. The studio swaps in a session-blob-backed
    implementation so reruns/benchmarks reuse extractions across processes."""

    def __init__(self) -> None:
        self._d: dict[str, IngestedDoc] = {}

    def get(self, key: str) -> Optional[IngestedDoc]:
        return self._d.get(key)

    def put(self, key: str, doc: IngestedDoc) -> None:
        self._d[key] = doc
