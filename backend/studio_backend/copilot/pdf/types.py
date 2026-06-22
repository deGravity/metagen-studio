"""Provider-neutral attachment-ingest data model.

A PDF (or other document) is ingested by a backend into an IngestedDoc: a
text/markdown rendering plus optional page images and per-page confidence. The
routing layer (pipeline.py) then turns that into the normalized message Parts
the target model can actually consume, gated by the model's Capabilities.
Nothing here imports a vendor SDK or studio internals. See
docs/COPILOT_PROVIDERS.md §4.4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PageImage:
    """One rasterized PDF page."""
    page: int                       # 0-based page index
    data_b64: str                   # base64 PNG
    media_type: str = "image/png"
    width: int = 0
    height: int = 0


@dataclass
class IngestOpts:
    """Knobs a backend may honor. Part of the cache key, so changing any of
    these invalidates a cached ingest."""
    want_text: bool = True
    want_images: bool = False
    image_dpi: int = 150            # rasterization DPI for page images
    max_pages: Optional[int] = None  # cap pages processed (None = all)

    def key(self) -> tuple:
        return (self.want_text, self.want_images, self.image_dpi, self.max_pages)


@dataclass
class IngestedDoc:
    """Backend output. `text` is markdown when the backend can produce it,
    else plain text. `page_images` populated only when want_images. `meta`
    carries backend-specific extras (page count, table count, model used…)."""
    text: str = ""
    page_images: list[PageImage] = field(default_factory=list)
    per_page_confidence: Optional[list[float]] = None
    n_pages: int = 0
    backend: str = ""
    meta: dict = field(default_factory=dict)
