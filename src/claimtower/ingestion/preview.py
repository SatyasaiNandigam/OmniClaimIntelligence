"""A page of a stored document as a picture, with the place a finding came from marked on it (T7.3): what the system READ, shown where it read it.

    png, how = render_highlighted(data, page=2, region=BBox(...), text="Assessed repair cost: INR 98,000")
    how    "region"  the normalised box the extractor recorded was drawn
           "text"    no box was recorded (a text layer without geometry): the quoted line was searched for on the page and drawn
           "none"    nothing could be marked; the page is shown as it is

Only a PDF or an image has a page picture; any other kind (text, e-mail, Word) raises `UnsupportedDocument` and the caller shows the stored text instead. The original file is never
changed: the mark is drawn on a copy of the rendered page.
"""
from __future__ import annotations

import io
from typing import Literal

import pymupdf
from PIL import Image, ImageDraw

from claimtower.core.schemas.common import BBox
from claimtower.ingestion.loaders import UnsupportedDocument, detect_kind, render_page_image

DPI = 110
MARK = (255, 190, 0)


def _search_text(text: str) -> list[str]:
    words = " ".join(text.split())
    return [words[:80], words[:40]] if words else []


def render_highlighted(data: bytes, page: int, region: BBox | None = None, text: str | None = None, dpi: int = DPI) -> tuple[bytes, Literal["region", "text", "none"]]:
    png, _ = render_page_image(data, page, dpi)
    img = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = img.size
    boxes: list[tuple[float, float, float, float]] = []
    how: Literal["region", "text", "none"] = "none"
    if region is not None:
        boxes, how = [(region.x0 * w, region.y0 * h, region.x1 * w, region.y1 * h)], "region"
    elif text and detect_kind(data)[0] == "pdf":
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            pg = pdf[page - 1]
            for needle in _search_text(text):
                hits = pg.search_for(needle)
                if hits:
                    s = dpi / 72
                    boxes, how = [(r.x0 * s, r.y0 * s, r.x1 * s, r.y1 * s) for r in hits[:3]], "text"
                    break
    if boxes:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for b in boxes:
            d.rectangle(b, fill=(*MARK, 90), outline=(*MARK, 255), width=3)
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue(), how


__all__ = ["UnsupportedDocument", "render_highlighted"]
