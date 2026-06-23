"""Pluggable PDF/attachment preprocessing for the copilot.

Self-contained: no studio/FastAPI imports. A backend ingests a PDF →
IngestedDoc; the pipeline caches it and routes it into provider-neutral Parts
gated by the target model's Capabilities. See docs/COPILOT_PROVIDERS.md §4.4.
"""
from .backends import build_backend, rasterize
from .base import BackendUnavailable, IngestCache, InMemoryIngestCache, PdfBackend
from .pipeline import RouteResult, ingest, prepare_parts
from .types import IngestedDoc, IngestOpts, PageImage

__all__ = [
    "build_backend", "rasterize", "ingest", "prepare_parts", "RouteResult",
    "PdfBackend", "IngestCache", "InMemoryIngestCache", "BackendUnavailable",
    "IngestedDoc", "IngestOpts", "PageImage",
]
