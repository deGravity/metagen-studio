"""Heavy/remote backends — marker, pdfmux, docling, mineru.

These are deep-learning pipelines that are GPU-recommended and shouldn't import
torch on this box. They're exposed as a single RemoteBackend that POSTs the PDF
to a small HTTP service (e.g. running on the GPU host) and expects an
IngestedDoc-shaped JSON back. With no `endpoint` configured the backend simply
reports unavailable with an actionable message, so the pipeline can fall back
to a local backend. See §4.4 (marker "natural fit to run on the GPU box").

Wire format (service → us): {"text": str, "per_page_confidence": [float]?,
"n_pages": int?, "meta": {...}?}.
"""
from __future__ import annotations

from typing import Optional

from ..base import BackendUnavailable
from ..types import IngestedDoc, IngestOpts

# backends that, today, only make sense behind a remote GPU service
REMOTE_KINDS = ("marker", "pdfmux", "docling", "mineru")


class RemoteBackend:
    def __init__(self, kind: str, *, endpoint: Optional[str] = None,
                 timeout_s: float = 120.0):
        self.name = kind
        self._endpoint = endpoint
        self._timeout = timeout_s

    def available(self) -> bool:
        return bool(self._endpoint)

    def ingest(self, data: bytes, opts: IngestOpts) -> IngestedDoc:
        if not self._endpoint:
            raise BackendUnavailable(
                f"'{self.name}' is a GPU/remote backend; set "
                f"copilot.pdf.{self.name}.endpoint to a running service "
                f"(it's not run on this host). Falling back requires choosing "
                f"a local backend (pymupdf4llm / vision_ocr).")
        try:
            import httpx
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable("httpx required to reach a remote PDF "
                                     "service (`pip install httpx`).") from exc
        files = {"file": ("doc.pdf", data, "application/pdf")}
        params = {"want_text": opts.want_text, "want_images": opts.want_images,
                  "dpi": opts.image_dpi}
        if opts.max_pages is not None:
            params["max_pages"] = opts.max_pages
        with httpx.Client(timeout=self._timeout) as client:
            r = client.post(self._endpoint, files=files, params=params)
            r.raise_for_status()
            j = r.json()
        return IngestedDoc(
            text=j.get("text", ""),
            per_page_confidence=j.get("per_page_confidence"),
            n_pages=int(j.get("n_pages", 0)),
            backend=self.name, meta=j.get("meta", {}))
