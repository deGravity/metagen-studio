"""Attachment pipeline: ingest (cached) + capability×mode routing.

`ingest()` runs a backend with a content-addressed cache. `prepare_parts()`
rewrites a message's Parts for a target model: Document(pdf) parts become either
a native passthrough, page-image Parts, or extracted-text Parts, gated by the
model's Capabilities and the config `mode`. Everything else passes through
untouched — crucially, an Anthropic Files-API reference (carried as a Raw block)
is left alone so the existing studio upload path is unaffected. See §4.4.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Optional

from ..types import Capabilities, Document, Image, Part, Text
from .base import IngestCache, InMemoryIngestCache, PdfBackend
from .types import IngestedDoc, IngestOpts

Mode = Literal["text_only", "images", "both"]


def _doc_key(data: bytes, backend: str, opts: IngestOpts) -> str:
    h = hashlib.sha256(data).hexdigest()[:16]
    return f"{h}:{backend}:{opts.key()}"


def ingest(data: bytes, backend: PdfBackend, opts: IngestOpts,
           cache: Optional[IngestCache] = None) -> IngestedDoc:
    """Ingest with caching by (pdf_hash, backend, opts)."""
    cache = cache if cache is not None else InMemoryIngestCache()
    key = _doc_key(data, backend.name, opts)
    hit = cache.get(key)
    if hit is not None:
        return hit
    doc = backend.ingest(data, opts)
    cache.put(key, doc)
    return doc


@dataclass
class RouteResult:
    parts: list[Part]
    notes: list[str]   # human-readable record of what each document became


def _b64_to_bytes(data_b64: str) -> bytes:
    import base64
    return base64.b64decode(data_b64)


def prepare_parts(parts: list[Part], caps: Capabilities, *,
                  backend: PdfBackend, mode: Mode = "both",
                  force_backend: bool = False,
                  cache: Optional[IngestCache] = None,
                  opts: Optional[IngestOpts] = None) -> RouteResult:
    """Rewrite Document(pdf) parts for a target model. Routing:

    - native_pdf model and NOT force_backend → pass the Document through
      (lets us A/B native-PDF vs our extraction by flipping force_backend).
    - native_images model, mode in {images, both} → page images (+ text if the
      backend produced it and mode == both).
    - otherwise (text-only model, or images-incapable) → extracted text only;
      never emit images to a model that can't take them.

    Non-Document parts pass through unchanged (incl. Raw Files-API refs)."""
    out: list[Part] = []
    notes: list[str] = []
    for p in parts:
        if not isinstance(p, Document) or "pdf" not in p.media_type:
            out.append(p)
            continue

        if caps.native_pdf and not force_backend:
            out.append(p)
            notes.append(f"{p.name or 'pdf'}: passed through (native_pdf)")
            continue

        want_images = caps.native_images and mode in ("images", "both")
        want_text = (not caps.native_images) or mode in ("text_only", "both")
        o = opts or IngestOpts()
        o = IngestOpts(want_text=want_text, want_images=want_images,
                       image_dpi=o.image_dpi, max_pages=o.max_pages)
        doc = ingest(_b64_to_bytes(p.data_b64), backend, o, cache)

        emitted = []
        label = p.name or "pdf"
        if want_text and doc.text:
            out.append(Text(f"[Attached document '{label}', extracted via "
                            f"{doc.backend}]\n\n{doc.text}"))
            emitted.append("text")
        if want_images and doc.page_images:
            for pi in doc.page_images:
                out.append(Image(data_b64=pi.data_b64, media_type=pi.media_type))
            emitted.append(f"{len(doc.page_images)} page-images")
        if not emitted:
            out.append(Text(f"[Attached document '{label}' could not be "
                            f"rendered for this model.]"))
            emitted.append("placeholder")
        notes.append(f"{label}: {doc.backend} → {', '.join(emitted)}")
    return RouteResult(parts=out, notes=notes)
