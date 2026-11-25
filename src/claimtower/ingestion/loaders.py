"""Loaders: turn a file's bytes into addressable pages (T2.1). PDF, images, DOCX, e-mail (.eml) and plain text.

    doc = load(data, "report.pdf")
    for page in doc.pages: page.number, page.text, page.source, page.needs_ocr

Rules
  * The format is detected from the CONTENT (magic bytes), never trusted from the extension.
  * A loader only reads. Originals are evidence and are never altered (the store keeps them byte-identical).
  * `Page.text` is the RAW text. Provenance matching (T2.4) uses `normalise(text)` - whitespace collapsed - because a long line can
    wrap in a PDF; this is the same basis as the policy clause spans in data/synthetic/policies/clause_index.json.
  * A page without a usable text layer is FLAGGED (`needs_ocr=True`, `source="none"`), never dropped and never guessed.
    Photographs also arrive as image pages with no text; telling a photo from a scanned form is the classifier's job (T2.2).
  * Pages: PDF/TIFF = real pages; DOCX = sections separated by explicit page breaks (Word's own pagination is not reproducible
    without a layout engine); e-mail and plain text = one page (plain text may use form feeds).
  * Failures raise a DocumentLoadError subclass. The store still keeps the original and flags the file for a human.
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from email import message_from_bytes, policy
from html.parser import HTMLParser
from typing import Literal

import pymupdf
from docx import Document as DocxDocument
from PIL import Image, UnidentifiedImageError

Kind = Literal["pdf", "image", "docx", "email", "text"]
TextSource = Literal["text_layer", "ocr", "none"]
MIN_TEXT_CHARS = 20                 # fewer non-space characters than this on a page with images = no usable text layer
RENDER_DPI = 150
MAX_FRAMES = 200                    # sanity limit for multi-frame images


class DocumentLoadError(Exception):
    """The file could not be turned into pages."""


class UnsupportedDocument(DocumentLoadError):
    """A format this prototype does not read (archives, binaries, ...)."""


class CorruptDocument(DocumentLoadError):
    """Empty, truncated or malformed file."""


class ProtectedDocument(DocumentLoadError):
    """Password-protected: needs a human."""


@dataclass(frozen=True)
class Page:
    number: int                     # 1-based
    text: str
    source: TextSource
    needs_ocr: bool
    width: float = 0.0              # PDF points / image pixels
    height: float = 0.0


@dataclass(frozen=True)
class Attachment:
    filename: str
    media_type: str
    data: bytes


@dataclass
class LoadedDocument:
    kind: Kind
    media_type: str
    pages: list[Page]
    attachments: list[Attachment] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)      # e.g. e-mail headers

    @property
    def page_count(self) -> int:
        return len(self.pages)


def normalise(text: str) -> str:
    """Whitespace-collapsed text: the basis for verbatim provenance checks."""
    return " ".join(text.split())


# ---------------------------------------------------------------------------------------------------- detection
_IMAGE_MAGIC = [
    (b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"), (b"II*\x00", "image/tiff"), (b"MM\x00*", "image/tiff"),
    (b"GIF87a", "image/gif"), (b"GIF89a", "image/gif"), (b"BM", "image/bmp"),
]
_EXT_MEDIA = {
    "pdf": "application/pdf", "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "tif": "image/tiff",
    "tiff": "image/tiff", "gif": "image/gif", "bmp": "image/bmp", "webp": "image/webp", "txt": "text/plain",
    "eml": "message/rfc822", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
_HEADER_LINE = re.compile(rb"^[A-Za-z][A-Za-z0-9-]*:\s", re.MULTILINE)


def claimed_media_type(filename: str) -> str | None:
    """What the extension SAYS the file is (informational only)."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _EXT_MEDIA.get(ext)


def _looks_like_text(sample: bytes) -> bool:
    """Text in ANY common encoding (UTF-8, Windows-1252, ...): no NUL bytes and almost no control characters."""
    if b"\x00" in sample:
        return False
    control = sum(1 for b in sample if b < 32 and b not in (9, 10, 12, 13))
    return control / max(1, len(sample)) < 0.02


def detect_kind(data: bytes, filename: str = "") -> tuple[Kind, str]:
    """(kind, media type) from the bytes. Raises CorruptDocument for empty input, UnsupportedDocument for unknown formats."""
    if not data:
        raise CorruptDocument("empty file")
    if b"%PDF-" in data[:1024]:
        return "pdf", "application/pdf"
    for magic, media in _IMAGE_MAGIC:
        if data.startswith(magic):
            return "image", media
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image", "image/webp"
    if data.startswith(b"PK\x03\x04"):
        try:
            names = zipfile.ZipFile(io.BytesIO(data)).namelist()
        except zipfile.BadZipFile:
            raise CorruptDocument("damaged zip container") from None
        if "word/document.xml" in names:
            return "docx", _EXT_MEDIA["docx"]
        raise UnsupportedDocument("zip archive (not a Word document)")
    if not _looks_like_text(data[:200_000]):
        raise UnsupportedDocument("binary file of unknown format")
    top = data[:4000]
    looks_like_mail = len(_HEADER_LINE.findall(top)) >= 2 and re.search(rb"^(From|To|Subject):", top, re.MULTILINE | re.IGNORECASE)
    if filename.lower().endswith(".eml") or looks_like_mail:
        return "email", "message/rfc822"
    return "text", "text/plain"


# ---------------------------------------------------------------------------------------------------- per-format loaders
def _load_pdf(data: bytes) -> LoadedDocument:
    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise CorruptDocument(f"cannot open PDF: {e}") from e
    if pdf.needs_pass:
        raise ProtectedDocument("password-protected PDF")
    if pdf.page_count == 0:
        raise CorruptDocument("PDF has no pages")
    pages = []
    for i, p in enumerate(pdf, start=1):
        text = p.get_text("text")
        usable = len(re.sub(r"\s", "", text)) >= MIN_TEXT_CHARS
        needs_ocr = not usable and bool(p.get_images())
        source: TextSource = "text_layer" if usable or (text.strip() and not needs_ocr) else "none"
        pages.append(Page(i, text if source == "text_layer" else "", source, needs_ocr, p.rect.width, p.rect.height))
    return LoadedDocument("pdf", "application/pdf", pages)


def _load_image(data: bytes, media_type: str) -> LoadedDocument:
    try:
        Image.open(io.BytesIO(data)).verify()
        img = Image.open(io.BytesIO(data))
        frames = min(getattr(img, "n_frames", 1), MAX_FRAMES)
        pages = []
        for n in range(frames):
            img.seek(n)
            pages.append(Page(n + 1, "", "none", True, float(img.width), float(img.height)))
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as e:
        raise CorruptDocument(f"cannot read image: {e}") from e
    return LoadedDocument("image", media_type, pages)


def _load_docx(data: bytes) -> LoadedDocument:
    try:
        doc = DocxDocument(io.BytesIO(data))
    except Exception as e:
        raise CorruptDocument(f"cannot open Word document: {e}") from e
    pages: list[list[str]] = [[]]
    for item in doc.iter_inner_content():
        if hasattr(item, "rows"):                                         # table: one line per row, cells tab-separated
            for row in item.rows:
                cells = []
                for c in row.cells:
                    if c.text not in cells[-1:]:                          # merged cells repeat their text
                        cells.append(c.text)
                pages[-1].append("\t".join(cells))
            continue
        el = item._p
        if el.xpath("./w:pPr/w:pageBreakBefore") and pages[-1]:
            pages.append([])
        pages[-1].append(item.text)
        if el.xpath('.//w:br[@w:type="page"]'):
            pages.append([])
    while len(pages) > 1 and not any(line.strip() for line in pages[-1]):
        pages.pop()                                                       # a trailing break must not create an empty page
    return LoadedDocument("docx", _EXT_MEDIA["docx"],
                          [Page(i, "\n".join(lines).strip("\n"), "text_layer", False) for i, lines in enumerate(pages, start=1)])


class _HtmlText(HTMLParser):
    _BLOCK = {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "table"}

    def __init__(self) -> None:
        super().__init__()
        self.out: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in self._BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in self._BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.out.append(data)


def html_to_text(html: str) -> str:
    parser = _HtmlText()
    parser.feed(html)
    text = "".join(parser.out)
    return re.sub(r"\n\s*\n+", "\n", re.sub(r"[ \t]+", " ", text)).strip()


def _load_email(data: bytes) -> LoadedDocument:
    try:
        msg = message_from_bytes(data, policy=policy.default)
    except Exception as e:
        raise CorruptDocument(f"cannot parse e-mail: {e}") from e
    headers = {h: str(msg[h]) for h in ("From", "To", "Cc", "Date", "Subject", "Message-ID") if msg[h]}
    body = ""
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is not None:
        try:
            body = part.get_content()
        except (LookupError, UnicodeError):                                # unknown charset: decode leniently
            body = (part.get_payload(decode=True) or b"").decode("utf-8", errors="replace")
        if part.get_content_type() == "text/html":
            body = html_to_text(body)
    attachments = []
    for i, att in enumerate(msg.iter_attachments(), start=1):
        payload = att.get_payload(decode=True)
        if payload is not None:
            attachments.append(Attachment(att.get_filename() or f"attachment-{i}", att.get_content_type(), payload))
    head = "\n".join(f"{k}: {v}" for k, v in headers.items() if k != "Message-ID")
    return LoadedDocument("email", "message/rfc822", [Page(1, f"{head}\n\n{body}".strip(), "text_layer", False)], attachments,
                          {k.lower(): v for k, v in headers.items()})


def _load_text(data: bytes) -> LoadedDocument:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1252", errors="replace")                     # legacy single-byte encodings
    chunks = text.split("\f") if "\f" in text else [text]
    return LoadedDocument("text", "text/plain", [Page(i, c.strip("\n"), "text_layer", False) for i, c in enumerate(chunks, start=1)])


def load(data: bytes, filename: str = "") -> LoadedDocument:
    kind, media = detect_kind(data, filename)
    if kind == "pdf":
        return _load_pdf(data)
    if kind == "image":
        return _load_image(data, media)
    if kind == "docx":
        return _load_docx(data)
    if kind == "email":
        return _load_email(data)
    return _load_text(data)


# ---------------------------------------------------------------------------------------------------- rendering (for OCR / vision)
def render_page_image(data: bytes, page_number: int, dpi: int = RENDER_DPI) -> tuple[bytes, str]:
    """PNG bytes of one page, for OCR or a vision model. Only PDFs and images have a visual page."""
    kind, _ = detect_kind(data)
    if kind == "pdf":
        pdf = pymupdf.open(stream=data, filetype="pdf")
        if not 1 <= page_number <= pdf.page_count:
            raise ValueError(f"page {page_number} out of range 1..{pdf.page_count}")
        return pdf[page_number - 1].get_pixmap(dpi=dpi).tobytes("png"), "image/png"
    if kind == "image":
        img = Image.open(io.BytesIO(data))
        if not 1 <= page_number <= getattr(img, "n_frames", 1):
            raise ValueError(f"page {page_number} out of range")
        img.seek(page_number - 1)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "PNG")
        return buf.getvalue(), "image/png"
    raise UnsupportedDocument(f"{kind} documents have no page image to render")
