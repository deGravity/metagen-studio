"""Built-in PDF backends + a name→backend builder.

Config-shape-agnostic: the host passes already-resolved per-backend settings
(endpoints, a vision transcriber) so the package stays free of studio config.
"""
from __future__ import annotations

from typing import Optional

from ..base import PdfBackend
from .pymupdf_backend import PyMuPDFBackend, rasterize
from .remote import REMOTE_KINDS, RemoteBackend
from .vision_ocr import Transcriber, VisionOCRBackend

# aliases → canonical backend name
_ALIASES = {
    "pymupdf": "pymupdf4llm", "pymupdf4llm": "pymupdf4llm",
    "vision_ocr": "vision_ocr", "vision": "vision_ocr",
    **{k: k for k in REMOTE_KINDS},
}


def build_backend(name: str, *, endpoint: Optional[str] = None,
                  transcribe: Optional[Transcriber] = None,
                  **opts) -> PdfBackend:
    canon = _ALIASES.get((name or "pymupdf4llm").lower())
    if canon == "pymupdf4llm":
        return PyMuPDFBackend()
    if canon == "vision_ocr":
        return VisionOCRBackend(transcribe=transcribe,
                                **{k: v for k, v in opts.items()
                                   if k in ("prompt", "batch_pages")})
    if canon in REMOTE_KINDS:
        return RemoteBackend(canon, endpoint=endpoint)
    raise ValueError(f"unknown pdf backend: {name!r}")


__all__ = ["build_backend", "rasterize", "PyMuPDFBackend",
           "VisionOCRBackend", "RemoteBackend", "REMOTE_KINDS"]
