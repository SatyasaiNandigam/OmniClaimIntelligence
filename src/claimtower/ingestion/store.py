"""Document store (T2.1): originals are immutable evidence; everything derived from them lives beside them.

Layout under the store root (default `var/store/`, git-ignored runtime data):
    blobs/<sha[:2]>/<sha256>          the ORIGINAL bytes, content-addressed, written once, read-only, verified on every read
    meta/<claim_id>/<document_id>.json  metadata + extracted page text for one upload (derived, regenerable from the blob)

Guarantees
  * The original is never altered: there is no API that rewrites a blob. `original()` re-hashes on every read and raises
    IntegrityError if the bytes no longer match - tampering or bit-rot is detected, not silently served.
  * Nothing is ever lost: a corrupt / protected / unsupported file is still stored and flagged (`load_error`) for a human.
  * Every upload is its own document (who uploaded, when, via which channel) even when the bytes are identical; identical bytes share
    one blob, and `same_content_as` records the earlier uploads (input for the document-duplicate stretch task).
  * E-mail attachments are ingested as child documents (`parent_document_id`).
  * OCR only ever adds text to the derived metadata; pages stay flagged by `source` ("text_layer" | "ocr" | "none").
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from claimtower.core.ids import new_id
from claimtower.core.schemas import DocumentRecord
from claimtower.core.schemas.common import Contract
from claimtower.ingestion.extractors.models import ExtractionResult
from claimtower.ingestion.loaders import (
    DocumentLoadError, LoadedDocument, claimed_media_type, detect_kind, load, render_page_image,
)
from claimtower.ingestion.ocr import OcrEngine

log = logging.getLogger("claimtower.ingestion")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MAX_EMAIL_DEPTH = 3


class Classifier(Protocol):
    """What the store needs from a classifier (ingestion.classifier.HybridClassifier / RuleClassifier satisfy it)."""

    def classify(self, *, kind: str | None, pages: Sequence[StoredPage], original: Callable[[], bytes] | None = None,
                 load_error: str | None = None): ...


class Extractor(Protocol):
    """What the store needs from an extractor (ingestion.extractors.HybridExtractor / RuleExtractor satisfy it)."""

    def extract(self, *, doc_type: str, kind: str | None, pages: Sequence[StoredPage],
                original: Callable[[], bytes] | None = None) -> ExtractionResult: ...


class IntegrityError(Exception):
    """Stored bytes no longer match their SHA-256."""


class StoreError(Exception):
    """Invalid request to the store (bad id, unknown document, duplicate id)."""


class StoredPage(Contract):
    number: int
    text: str
    source: Literal["text_layer", "ocr", "none"]
    needs_ocr: bool
    width: float = 0.0
    height: float = 0.0


class StoredDocument(Contract):
    document_id: str
    claim_id: str
    filename: str
    sha256: str
    size_bytes: int
    media_type: str | None = None                  # detected from the bytes; None when the format could not be determined
    claimed_media_type: str | None = None          # what the extension says
    media_type_mismatch: bool = False
    kind: str | None = None
    blob_path: str                                 # relative to the store root
    uploaded_by: str | None = None
    source_channel: str | None = None
    uploaded_at: datetime
    page_count: int
    pages: list[StoredPage]
    load_error: str | None = None
    parent_document_id: str | None = None
    same_content_as: list[str] = []
    email_headers: dict[str, str] = {}
    ocr_engine: str | None = None
    ocr_errors: list[str] = []
    # classification (T2.2): derived data, never touches the original
    detected_type: str = "unknown"                 # a pack type, or "unknown" = routed to a human
    type_confidence: float = 0.0
    type_evidence_snippet: str | None = None
    type_page: int | None = None
    type_method: str | None = None                 # rules | container | visual | llm | human
    type_code: str | None = None                   # ok | needs_ocr | no_text | unreadable | low_score | ambiguous | ...
    type_reason: str | None = None                 # human-readable justification or abstention reason
    type_scores: dict[str, float] = {}
    type_by: str | None = None                     # who assigned it when method == human
    # extraction (T2.3): derived data, never touches the original
    extraction: ExtractionResult | None = None

    def to_record(self) -> DocumentRecord:
        """The frozen contract view. `detected_type` is 'unknown' until classified, and stays 'unknown' when a human must decide."""
        return DocumentRecord(
            document_id=self.document_id, claim_id=self.claim_id, filename=self.filename, sha256=self.sha256, page_count=self.page_count,
            detected_type=self.detected_type, type_confidence=self.type_confidence, type_evidence_snippet=self.type_evidence_snippet,
            source_channel=self.source_channel, uploaded_by=self.uploaded_by, uploaded_at=self.uploaded_at,
            needs_ocr=any(p.needs_ocr for p in self.pages), blob_path=self.blob_path)


def _check_id(value: str, what: str) -> str:
    if not _SAFE_ID.match(value):
        raise StoreError(f"invalid {what} '{value}': use letters, digits, '.', '_' or '-' (no path separators)")
    return value


class DocumentStore:
    def __init__(self, root: Path | str, clock: Callable[[], datetime] | None = None) -> None:
        self.root = Path(root)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        (self.root / "blobs").mkdir(parents=True, exist_ok=True)
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        self._docs: dict[str, StoredDocument] = {}
        self._by_sha: dict[str, list[str]] = {}
        for f in sorted((self.root / "meta").glob("*/*.json")):
            self._remember(StoredDocument.model_validate_json(f.read_text(encoding="utf-8")))

    @classmethod
    def default(cls, repo_root: Path | str) -> "DocumentStore":
        return cls(Path(repo_root) / "var" / "store")

    # ------------------------------------------------------------------------------------------ ingest
    def ingest(self, claim_id: str, filename: str, data: bytes, *, uploaded_by: str | None = None, source_channel: str | None = None,
               uploaded_at: datetime | None = None, document_id: str | None = None, parent_document_id: str | None = None,
               ocr: OcrEngine | None = None, classifier: Classifier | None = None, extractor: Extractor | None = None,
               _depth: int = 0) -> DocumentRecord:
        """Store the original, extract pages, return the DocumentRecord. Never raises for a bad FILE (it is flagged); raises StoreError
        for a bad REQUEST (invalid or duplicate id)."""
        _check_id(claim_id, "claim id")
        doc_id = _check_id(document_id, "document id") if document_id is not None else new_id("DOC")
        if doc_id in self._docs:
            raise StoreError(f"document id '{doc_id}' already exists")
        sha = hashlib.sha256(data).hexdigest()
        blob = self._write_blob(sha, data)

        loaded: LoadedDocument | None = None
        error: str | None = None
        try:
            loaded = load(data, filename)
        except DocumentLoadError as e:
            error = f"{type(e).__name__}: {e}"
            log.warning("document %s (%s) stored but not readable: %s", doc_id, filename, error)

        detected: str | None = None
        try:
            detected = detect_kind(data, filename)[1]
        except DocumentLoadError:
            pass
        claimed = claimed_media_type(filename)
        pages = ([StoredPage(number=p.number, text=p.text, source=p.source, needs_ocr=p.needs_ocr, width=p.width, height=p.height)
                  for p in loaded.pages] if loaded else [StoredPage(number=1, text="", source="none", needs_ocr=False)])
        stored = StoredDocument(
            document_id=doc_id, claim_id=claim_id, filename=os.path.basename(filename.replace("\\", "/")) or "unnamed", sha256=sha,
            size_bytes=len(data), media_type=detected, claimed_media_type=claimed,
            media_type_mismatch=bool(detected and claimed and detected != claimed), kind=loaded.kind if loaded else None,
            blob_path=blob, uploaded_by=uploaded_by, source_channel=source_channel, uploaded_at=uploaded_at or self._clock(),
            page_count=len(pages), pages=pages, load_error=error, parent_document_id=parent_document_id,
            same_content_as=list(self._by_sha.get(sha, [])), email_headers=loaded.metadata if loaded and loaded.kind == "email" else {})
        self._save(stored)

        if loaded and loaded.attachments and _depth < MAX_EMAIL_DEPTH:
            for att in loaded.attachments:
                self.ingest(claim_id, att.filename, att.data, uploaded_by=uploaded_by, source_channel=source_channel,
                            uploaded_at=stored.uploaded_at, parent_document_id=doc_id, ocr=ocr, classifier=classifier, extractor=extractor,
                            _depth=_depth + 1)
        if ocr and any(p.needs_ocr for p in stored.pages):
            self.run_ocr(doc_id, ocr)
        if classifier:
            self.classify(doc_id, classifier)
        if extractor:
            self.extract(doc_id, extractor)
        return self.record(doc_id)

    # ------------------------------------------------------------------------------------------ classification (T2.2)
    def classify(self, document_id: str, classifier: Classifier) -> DocumentRecord:
        """Classify and persist the result with its justification. A human assignment is never overwritten by a re-run."""
        doc = self.stored(document_id)
        if doc.type_method == "human":
            return doc.to_record()
        result = classifier.classify(kind=doc.kind, pages=doc.pages, original=lambda: self.original(document_id), load_error=doc.load_error)
        self._save(doc.model_copy(update={
            "detected_type": result.type, "type_confidence": result.confidence, "type_evidence_snippet": result.evidence_snippet,
            "type_page": result.page, "type_method": result.method, "type_code": result.code, "type_reason": result.reason,
            "type_scores": result.scores, "type_by": None}))
        return self.record(document_id)

    def set_type(self, document_id: str, doc_type: str, *, by: str, allowed: Iterable[str] | None = None) -> DocumentRecord:
        """A human decides the type (the only way to record 'other'). Survives re-classification."""
        doc = self.stored(document_id)
        if not by.strip():
            raise StoreError("a human type assignment needs 'by' (who decided)")
        if doc_type == "unknown" or (allowed is not None and doc_type not in set(allowed)):
            raise StoreError(f"'{doc_type}' is not an allowed document type")
        self._save(doc.model_copy(update={
            "detected_type": doc_type, "type_confidence": 1.0, "type_evidence_snippet": None, "type_page": None, "type_method": "human",
            "type_code": "ok", "type_reason": f"assigned by {by}", "type_by": by}))
        return self.record(document_id)

    # ------------------------------------------------------------------------------------------ extraction (T2.3)
    def extract(self, document_id: str, extractor: Extractor) -> ExtractionResult:
        """Extract fields and tables for the document's CURRENT type and persist them with their provenance. Re-run after OCR or after
        a human changes the type. A document whose type is unknown is skipped (with the reason), never guessed at."""
        doc = self.stored(document_id)
        result = extractor.extract(doc_type=doc.detected_type, kind=doc.kind, pages=doc.pages, original=lambda: self.original(document_id))
        self._save(doc.model_copy(update={"extraction": result}))
        return result

    def needs_human(self, claim_id: str | None = None) -> list[StoredDocument]:
        """Documents the classifier could not decide (type 'unknown'), with the reason in `type_reason` / `type_code`."""
        return [d for d in self._docs.values() if d.detected_type == "unknown" and (claim_id is None or d.claim_id == claim_id)]

    # ------------------------------------------------------------------------------------------ OCR
    def run_ocr(self, document_id: str, engine: OcrEngine) -> DocumentRecord:
        """Fill in text for pages that have none. The original is untouched; only derived metadata changes. A page whose OCR fails
        stays flagged (`needs_ocr`) and the error is recorded."""
        doc = self.stored(document_id)
        data = self.original(document_id)
        pages, errors = [], list(doc.ocr_errors)
        for p in doc.pages:
            if not p.needs_ocr:
                pages.append(p)
                continue
            try:
                image, mime = render_page_image(data, p.number)
                text = engine.recognise(image, mime)
            except Exception as e:                                     # a failing page must not lose the others
                errors.append(f"page {p.number}: {type(e).__name__}: {e}")
                pages.append(p)
                continue
            pages.append(p.model_copy(update={"text": text, "source": "ocr", "needs_ocr": False}))
        updated = doc.model_copy(update={"pages": pages, "ocr_engine": engine.name, "ocr_errors": errors})
        self._save(updated)
        return updated.to_record()

    # ------------------------------------------------------------------------------------------ read
    def stored(self, document_id: str) -> StoredDocument:
        try:
            return self._docs[document_id]
        except KeyError:
            raise StoreError(f"unknown document '{document_id}'") from None

    def record(self, document_id: str) -> DocumentRecord:
        return self.stored(document_id).to_record()

    def records(self, claim_id: str) -> list[DocumentRecord]:
        return [d.to_record() for d in self._docs.values() if d.claim_id == claim_id]

    def pages(self, document_id: str) -> list[StoredPage]:
        return list(self.stored(document_id).pages)

    def children(self, document_id: str) -> list[DocumentRecord]:
        return [d.to_record() for d in self._docs.values() if d.parent_document_id == document_id]

    def find_by_sha256(self, sha256: str) -> list[DocumentRecord]:
        return [self._docs[i].to_record() for i in self._by_sha.get(sha256, [])]

    def original(self, document_id: str) -> bytes:
        """The original bytes, verified against the recorded SHA-256 on every read."""
        doc = self.stored(document_id)
        data = (self.root / doc.blob_path).read_bytes()
        if hashlib.sha256(data).hexdigest() != doc.sha256:
            raise IntegrityError(f"{document_id}: stored bytes no longer match sha256 {doc.sha256[:12]}...")
        return data

    def verify_all(self) -> list[str]:
        """Integrity sweep: ids of documents whose blob is missing or altered."""
        bad = []
        for doc_id in self._docs:
            try:
                self.original(doc_id)
            except (IntegrityError, FileNotFoundError):
                bad.append(doc_id)
        return bad

    # ------------------------------------------------------------------------------------------ internals
    def _write_blob(self, sha: str, data: bytes) -> str:
        rel = f"blobs/{sha[:2]}/{sha}"
        path = self.root / rel
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise IntegrityError(f"existing blob {sha[:12]}... is corrupt; refusing to overwrite evidence")
            return rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)                                          # atomic: a reader never sees a partial blob
        try:
            os.chmod(path, 0o444)                                      # read-only: originals are never rewritten
        except OSError:
            pass
        return rel

    def _save(self, doc: StoredDocument) -> None:
        path = self.root / "meta" / doc.claim_id / f"{doc.document_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)
        self._remember(doc)

    def _remember(self, doc: StoredDocument) -> None:
        self._docs[doc.document_id] = doc
        ids = self._by_sha.setdefault(doc.sha256, [])
        if doc.document_id not in ids:
            ids.append(doc.document_id)
