"""Logical lines (T2.3): the unit the extractors read, with page and region.

A PDF wraps long values over several physical lines. Guessing from punctuation would truncate or glue values, so wrapped lines are
merged by GEOMETRY: a line directly under the previous one (gap smaller than `wrap_gap_ratio` x line height) that starts at about the
same x belongs to the same paragraph; a larger gap starts a new line. Each logical line carries the union bounding box of its parts,
normalised 0..1, which becomes the field's `region`.

Pages without geometry (OCR text, DOCX, plain text, e-mail, or a PDF whose original is unavailable) use one logical line per text line
and have no region. A merged line's text is its parts joined with single spaces, so it is a substring of the whitespace-normalised page
text (the provenance basis, `loaders.normalise`).
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import pymupdf

from claimtower.core.config import ExtractorConfig
from claimtower.core.schemas.common import BBox


class PageLike(Protocol):
    number: int
    text: str
    source: str
    needs_ocr: bool


@dataclass(frozen=True)
class Line:
    page: int
    text: str
    bbox: BBox | None
    source: str = "text_layer"          # "text_layer" | "ocr"


_LABELISH = re.compile(r"^[A-Za-z][A-Za-z0-9 .'/#&()-]{0,40}:(\s|$)")     # starts like 'Label: value'


def _same_row(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """Two boxes on the same baseline (a label run and its right-aligned amount): vertical overlap of at least 60% of the smaller one."""
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    return overlap >= 0.6 * min(a[3] - a[1], b[3] - b[1])


def _pdf_lines(pdf: pymupdf.Document, page_no: int, cfg: ExtractorConfig) -> list[Line]:
    page = pdf[page_no - 1]
    w, h = page.rect.width, page.rect.height
    physical: list[tuple[str, tuple[float, float, float, float]]] = []
    for block in page.get_text("dict")["blocks"]:
        for ln in block.get("lines", []):
            dx, dy = ln.get("dir", (1.0, 0.0))
            if abs(dx - 1.0) > 0.01 or abs(dy) > 0.01:          # rotated text (stamps) is not body text
                continue
            text = "".join(span["text"] for span in ln["spans"]).strip()
            if text:
                physical.append((text, tuple(ln["bbox"])))

    # 1. pieces on the same baseline form ONE row, read left to right ('Kitchen floor:' + 'INR 110,000')
    rows: list[list[tuple[str, tuple[float, float, float, float]]]] = []
    for text, bb in physical:
        for r in reversed(rows[-3:]):
            row_box = (0.0, min(p[1][1] for p in r), 0.0, max(p[1][3] for p in r))       # only the vertical extent matters
            if _same_row(row_box, bb):
                r.append((text, bb))
                break
        else:
            rows.append([(text, bb)])
    flat: list[tuple[str, tuple[float, float, float, float], int]] = []
    for r in rows:
        r.sort(key=lambda part: part[1][0])
        box = (min(p[1][0] for p in r), min(p[1][1] for p in r), max(p[1][2] for p in r), max(p[1][3] for p in r))
        flat.append((" ".join(p[0] for p in r), box, len(r)))

    # 2. rows directly under each other, left-aligned, that do not start a new 'Label: value' belong to one paragraph
    merged: list[tuple[str, tuple[float, float, float, float]]] = []
    last_row: tuple[str, tuple[float, float, float, float], int] | None = None
    for text, bb, parts in flat:
        if merged and last_row is not None:
            ptext, pbox = merged[-1]
            lbb = last_row[1]
            height = lbb[3] - lbb[1]                              # ONE line's height, not the merged paragraph's
            gap = bb[1] - lbb[3]
            tight = abs(bb[0] - lbb[0]) <= cfg.wrap_x_tolerance and -0.5 * height <= gap <= cfg.wrap_gap_ratio * height and bb[1] >= lbb[1]
            starts_new_field = bool(_LABELISH.match(text)) and bool(_LABELISH.match(merged[-1][0]))
            if tight and parts == 1 and last_row[2] == 1 and not starts_new_field:
                merged[-1] = (f"{ptext} {text}", (min(pbox[0], bb[0]), min(pbox[1], bb[1]), max(pbox[2], bb[2]), max(pbox[3], bb[3])))
                last_row = (text, bb, parts)
                continue
        merged.append((text, bb))
        last_row = (text, bb, parts)
    return [Line(page_no, t, BBox(x0=b[0] / w, y0=b[1] / h, x1=b[2] / w, y1=b[3] / h)) for t, b in merged]


def logical_lines(pages: Sequence[PageLike], *, kind: str | None, original: Callable[[], bytes] | None,
                  cfg: ExtractorConfig) -> list[Line]:
    pdf = None
    if kind == "pdf" and original is not None:
        try:
            pdf = pymupdf.open(stream=original(), filetype="pdf")
        except Exception:                                      # unreadable original: fall back to plain text lines
            pdf = None
    out: list[Line] = []
    for p in pages:
        if pdf is not None and p.source == "text_layer" and p.number <= pdf.page_count:
            out.extend(_pdf_lines(pdf, p.number, cfg))
        else:
            src = "ocr" if p.source == "ocr" else "text_layer"
            out.extend(Line(p.number, ln.strip(), None, src) for ln in p.text.splitlines() if ln.strip())
    return out
