"""PyMuPDF-based ingest — the lightweight, no-ML default.

Native-text extraction → markdown-ish text, plus optional page rasterization
(shared with the vision_ocr backend). If `pymupdf4llm` is also installed we use
its richer markdown converter; otherwise we fall back to core PyMuPDF block
text. Great latency on digital PDFs; weak on scans (use vision_ocr for those).
"""
from __future__ import annotations

import base64

from ..base import BackendUnavailable
from ..types import IngestedDoc, IngestOpts, PageImage


def _import_fitz():
    try:
        import fitz  # PyMuPDF
        return fitz
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(
            "PyMuPDF not installed — `pip install pymupdf` (core, no ML deps).") from exc


def rasterize(data: bytes, dpi: int = 150, max_pages: int | None = None) -> list[PageImage]:
    """Render PDF pages to PNG images. Shared by vision_ocr. Raises
    BackendUnavailable if PyMuPDF is missing."""
    fitz = _import_fitz()
    out: list[PageImage] = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    with fitz.open(stream=data, filetype="pdf") as doc:
        n = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
        for i in range(n):
            pix = doc.load_page(i).get_pixmap(matrix=mat, alpha=False)
            out.append(PageImage(page=i, data_b64=base64.b64encode(pix.tobytes("png")).decode(),
                                 media_type="image/png", width=pix.width, height=pix.height))
    return out


def _markdown_via_pymupdf4llm(data: bytes, max_pages: int | None) -> str | None:
    try:
        import pymupdf4llm
        import fitz
    except Exception:  # noqa: BLE001 — optional upgrade
        return None
    with fitz.open(stream=data, filetype="pdf") as doc:
        pages = None
        if max_pages is not None:
            pages = list(range(min(doc.page_count, max_pages)))
        return pymupdf4llm.to_markdown(doc, pages=pages)


class PyMuPDFBackend:
    name = "pymupdf4llm"   # canonical name in config; degrades to core text

    def available(self) -> bool:
        try:
            _import_fitz()
            return True
        except BackendUnavailable:
            return False

    def ingest(self, data: bytes, opts: IngestOpts) -> IngestedDoc:
        fitz = _import_fitz()
        text = ""
        used = "pymupdf"
        if opts.want_text:
            md = _markdown_via_pymupdf4llm(data, opts.max_pages)
            if md is not None:
                text, used = md, "pymupdf4llm"
            else:
                parts: list[str] = []
                with fitz.open(stream=data, filetype="pdf") as doc:
                    n = (doc.page_count if opts.max_pages is None
                         else min(doc.page_count, opts.max_pages))
                    for i in range(n):
                        t = doc.load_page(i).get_text("text").strip()
                        if t:
                            parts.append(f"<!-- page {i + 1} -->\n{t}")
                text = "\n\n".join(parts)

        images = rasterize(data, opts.image_dpi, opts.max_pages) if opts.want_images else []
        with fitz.open(stream=data, filetype="pdf") as doc:
            n_pages = doc.page_count
        return IngestedDoc(text=text, page_images=images, n_pages=n_pages,
                           backend=self.name, meta={"extractor": used})
