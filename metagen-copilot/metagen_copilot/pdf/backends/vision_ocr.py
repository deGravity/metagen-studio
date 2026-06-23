"""vision_ocr — rasterize pages and have a vision model transcribe them.

Highest fidelity on scans/diagrams/complex layout, highest cost. The actual
model call is injected as a `transcribe` callable so this package stays free of
any provider/event-loop coupling: the studio supplies a transcriber that wraps
its configured VLM (the active provider if vision-capable, else a dedicated
one); the benchmark runner supplies its own. See §4.4.
"""
from __future__ import annotations

from typing import Callable, Optional

from ..base import BackendUnavailable
from ..types import IngestedDoc, IngestOpts, PageImage
from .pymupdf_backend import rasterize

# (page_images, prompt) -> markdown/text for those pages.
Transcriber = Callable[[list[PageImage], str], str]

_DEFAULT_PROMPT = (
    "Transcribe these document pages to clean GitHub-flavored Markdown. "
    "Preserve headings, lists, tables, and equations (LaTeX). Output only the "
    "transcription, no commentary.")


class VisionOCRBackend:
    name = "vision_ocr"

    def __init__(self, transcribe: Optional[Transcriber] = None, *,
                 prompt: str = _DEFAULT_PROMPT, batch_pages: int = 4):
        self._transcribe = transcribe
        self._prompt = prompt
        self._batch = max(1, batch_pages)

    def available(self) -> bool:
        # Needs both a rasterizer (PyMuPDF) and an injected model call.
        if self._transcribe is None:
            return False
        try:
            import fitz  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False

    def ingest(self, data: bytes, opts: IngestOpts) -> IngestedDoc:
        if self._transcribe is None:
            raise BackendUnavailable(
                "vision_ocr needs a transcribe callable (configure a vision "
                "model for copilot.pdf.vision_ocr).")
        images = rasterize(data, opts.image_dpi, opts.max_pages)
        if not images:
            return IngestedDoc(text="", n_pages=0, backend=self.name)
        chunks: list[str] = []
        for start in range(0, len(images), self._batch):
            batch = images[start:start + self._batch]
            chunks.append(self._transcribe(batch, self._prompt))
        return IngestedDoc(
            text="\n\n".join(c.strip() for c in chunks if c and c.strip()),
            page_images=images if opts.want_images else [],
            n_pages=len(images), backend=self.name,
            meta={"pages_transcribed": len(images), "batch_pages": self._batch})
