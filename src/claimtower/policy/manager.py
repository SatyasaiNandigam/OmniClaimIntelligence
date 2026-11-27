"""The policy library as something that can change while the system runs: add a wording or an endorsement, or re-read the folder, without restarting.

    mgr = PolicyDocumentManager(decomposer, identity, builtin_dir, upload_dir)
    mgr.reload()                                          # read both folders; what was added, removed, changed
    result = mgr.ingest(pdf_bytes, "scan (3).pdf", uploaded_by="ops-1")      # the document id comes from INSIDE the document
    mgr.library                                           # the current PolicyLibrary (every attached SnapshotBuilder uses it at once)

Identity: the document id (and version) is read from the document's own text. With a model (`--llm`) the MODEL reads it (policy/llm_identity.py: it names the identifier the document gives itself, quotes it,
and is checked); without one, or when it cannot decide, the fixed patterns of the pack (`policy_documents.yaml`) are tried, for example the line "Policy wording BASE-HOME-1 | Version
2025-01". The file name is never the id: a file may be called anything, and an uploaded file is stored as `<id>.pdf`. A document with no recognisable id is refused (or accepted with an explicit
`document_id`, for example a scan with no header); a folder file with no id inside falls back to its file name. Adding the same content again changes nothing; different content under an existing id is a
CONFLICT unless the caller says `replace` (a changed wording changes every claim that is assessed afterwards: that must be deliberate). A document that yields no clauses is refused: nothing usable was read.

Already-assessed claims are not touched: re-assess one from the policy node (`POST /claims/{id}/reassess?from_node=policy`) to pick up a new wording. The decomposition of a file is cached by its content hash, so
a reload only reads what changed. Uploaded files live in `upload_dir` with an index (original name, who, when); the built-in folder is read-only here.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pymupdf

from claimtower.policy.decompose import Decomposition, PolicyDecomposer
from claimtower.policy.snapshot import PolicyLibrary

SAFE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{1,63}$")


class PolicyDocumentError(ValueError):
    """The document cannot be added (not a PDF, no id, no clauses). The message says why."""


class PolicyConflict(PolicyDocumentError):
    """The id exists with different content and the caller did not say `replace`."""


@dataclass
class _Page:
    number: int
    text: str


@dataclass
class Identity:
    doc_id: str
    version: str | None
    line: str
    source: Literal["document", "model", "given", "file_name"]
    details: dict[str, str] = field(default_factory=dict)


@dataclass
class PolicyEntry:
    doc_id: str
    title: str | None
    kind: str
    version: str | None
    clauses: int
    unclassified: int
    warnings: list[str]
    source: Literal["built-in", "uploaded"]
    filename: str
    sha256: str
    id_from: str
    uploaded_by: str | None = None
    uploaded_at: str | None = None
    replaces_built_in: bool = False
    id_evidence: str = ""
    details: dict[str, str] = field(default_factory=dict)


@dataclass
class IngestResult:
    outcome: Literal["added", "replaced", "unchanged"]
    entry: PolicyEntry


@dataclass
class ReloadReport:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    total: int = 0


def read_pages(data: bytes) -> list[_Page]:
    try:
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            return [_Page(i + 1, p.get_text()) for i, p in enumerate(pdf)]
    except Exception as e:
        raise PolicyDocumentError(f"this is not a readable PDF ({type(e).__name__}: {e})") from None


class IdentityReader:
    def __init__(self, id_patterns: list[str], pages_scanned: int = 2, llm_reader: Any | None = None) -> None:
        self._rx = [re.compile(p, re.IGNORECASE) for p in id_patterns]
        self._n = pages_scanned
        self.llm = llm_reader
        self.last_problem: str | None = None

    @classmethod
    def load(cls, pack: Any, llm: Any | None = None) -> IdentityReader:
        import yaml

        from claimtower.policy.llm_identity import LLMIdentityReader

        raw = yaml.safe_load((pack.root / "policy_documents.yaml").read_text(encoding="utf-8"))
        return cls(raw["id_patterns"], raw.get("pages_scanned", 2), LLMIdentityReader(llm, pack) if llm is not None else None)

    def read(self, pages: list[_Page], use_llm: bool = False) -> Identity | None:
        """The model first (when there is one and `use_llm`), then the fixed patterns."""
        self.last_problem = None
        if use_llm and self.llm is not None:
            r = self.llm.read(pages)
            if r is not None:
                doc_id, version, quote, _kind, details = r
                return Identity(doc_id, version, quote, "model", details)
            self.last_problem = self.llm.last_problem
        return self._patterns(pages)

    def _patterns(self, pages: list[_Page]) -> Identity | None:
        for p in pages[: self._n]:
            for line in p.text.splitlines():
                line = line.strip()
                for rx in self._rx:
                    m = rx.match(line)
                    if m:
                        doc_id = m.group("id")
                        # the pattern is case-insensitive for its keywords; keep the id as written
                        return Identity(doc_id, (m.groupdict().get("version") or None), line, "document")
        return None


def _marks_to(m: Any) -> dict[str, Any] | None:
    return None if m is None else {"clauses": m.clauses, "sections": sorted(m.sections)}


def _marks_from(raw: Any) -> Any | None:
    from claimtower.policy.llm_segments import Marks

    return None if not raw else Marks(dict(raw["clauses"]), set(raw["sections"]))


def _apply_model_kind(d: Decomposition, ident: Identity) -> None:
    """What the document IS (endorsement or wording) as the model read it; the title pattern only decides when the model did not say."""
    kind = ident.details.get("model_kind")
    if ident.source == "model" and kind in ("endorsement", "policy_wording"):
        d.kind = kind


class PolicyDocumentManager:
    def __init__(self, decomposer: PolicyDecomposer, identity: IdentityReader, builtin_dir: Path | None, upload_dir: Path | None,
                 clock: Callable[[], datetime] | None = None) -> None:
        self._dec, self._identity = decomposer, identity
        self._builtin, self._uploads = builtin_dir, upload_dir
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[Decomposition, Identity]] = {}           # sha256 -> decomposition (a file is read once)
        self._entries: dict[str, PolicyEntry] = {}
        self._docs: dict[str, Decomposition] = {}
        self._builders: list[Any] = []
        self.library = PolicyLibrary([])
        if upload_dir is not None:
            upload_dir.mkdir(parents=True, exist_ok=True)
        self.reload()

    # ------------------------------------------------------------------ wiring
    def attach(self, builder: Any) -> None:
        """A snapshot builder that must always use the current library."""
        with self._lock:
            self._builders.append(builder)
            builder.set_library(self.library)

    def _publish(self) -> None:
        self.library = PolicyLibrary(self._docs.values())
        for b in self._builders:
            b.set_library(self.library)

    # ------------------------------------------------------------------ reading one document
    def _decompose(self, data: bytes, filename: str, *, given_id: str | None, allow_file_name: bool, use_llm: bool = False, marks: Any | None = None) -> tuple[Decomposition, Identity, str]:
        sha = hashlib.sha256(data).hexdigest()
        key = f"{sha}|{given_id or ''}|{int(allow_file_name)}|{Path(filename).stem if allow_file_name else ''}|{int(use_llm)}|{int(marks is not None)}"
        if key in self._cache:
            d, ident = self._cache[key]
            return d, ident, sha
        pages = read_pages(data)
        if not any(p.text.strip() for p in pages):
            raise PolicyDocumentError("the document has no text layer (a scan?): it cannot be read as a policy wording")
        ident = self._identity.read(pages, use_llm=use_llm and not given_id)
        if given_id:
            if not SAFE_ID.match(given_id):
                raise PolicyDocumentError(f"'{given_id}' is not a valid document id: use letters, digits, '.', '_' or '-'")
            ident = Identity(given_id, ident.version if ident else None, ident.line if ident else "", "given")
        elif ident is None and allow_file_name:
            ident = Identity(Path(filename).stem, None, "", "file_name")
        if ident is None:
            why = f" The model said: {self._identity.last_problem}." if self._identity.last_problem else ""
            raise PolicyDocumentError("no document id was found inside the document." + why + " Send it again with an explicit document_id.")
        if not SAFE_ID.match(ident.doc_id):
            raise PolicyDocumentError(f"the id found in the document ('{ident.doc_id}') is not usable as an id")
        d = self._dec.decompose(ident.doc_id, pages, use_segmenter=use_llm, marks=marks)
        if not d.clauses:
            why = f" The model said: {self._dec.last_problem}." if getattr(self._dec, "last_problem", None) else ""
            raise PolicyDocumentError("no clauses were found in the document." + why + " Without a model, clauses must be numbered like '2.2 Escape of water' or 'A1 Sudden' under their headings.")
        _apply_model_kind(d, ident)
        self._cache[key] = (d, ident)
        return d, ident, sha

    @staticmethod
    def _entry(d: Decomposition, ident: Identity, sha: str, source: str, filename: str, uploaded_by: str | None = None, uploaded_at: str | None = None) -> PolicyEntry:
        return PolicyEntry(d.doc_id, d.title, d.kind, ident.version, len(d.clauses), len(d.unclassified), list(d.warnings), source, filename, sha, ident.source,   # type: ignore[arg-type]
                           uploaded_by, uploaded_at, id_evidence=ident.line, details=dict(ident.details))

    # ------------------------------------------------------------------ the folders
    def _index_path(self) -> Path | None:
        return self._uploads / "index.json" if self._uploads else None

    def _read_index(self) -> dict[str, dict[str, Any]]:
        p = self._index_path()
        return json.loads(p.read_text(encoding="utf-8")) if p is not None and p.is_file() else {}

    def reload(self) -> ReloadReport:
        """Read both folders again. What was added, removed or changed since the last read; files that could not be read are reported, never silently dropped."""
        with self._lock:
            before = {i: e.sha256 for i, e in self._entries.items()}
            docs: dict[str, Decomposition] = {}
            entries: dict[str, PolicyEntry] = {}
            report = ReloadReport()
            if self._builtin is not None and self._builtin.is_dir():
                for f in sorted(self._builtin.glob("*.pdf")):
                    try:
                        d, ident, sha = self._decompose(f.read_bytes(), f.name, given_id=None, allow_file_name=True)
                    except PolicyDocumentError as e:
                        report.problems.append(f"{f.name}: {e}")
                        continue
                    docs[d.doc_id], entries[d.doc_id] = d, self._entry(d, ident, sha, "built-in", f.name)
            index = self._read_index()
            if self._uploads is not None:
                for f in sorted(self._uploads.glob("*.pdf")):
                    meta = index.get(f.stem, {})
                    try:
                        d, ident, sha = self._decompose(f.read_bytes(), f.name, given_id=f.stem if meta.get("id_from") in ("given", "model") else None, allow_file_name=True,
                                                        marks=_marks_from(meta.get("marks")))
                        if meta.get("id_from") == "model":                          # the model read it when it was uploaded; a reload does not ask it again
                            ident = Identity(d.doc_id, meta.get("version"), meta.get("id_evidence", ""), "model", meta.get("details", {}))
                            _apply_model_kind(d, ident)
                    except PolicyDocumentError as e:
                        report.problems.append(f"{f.name}: {e}")
                        continue
                    e = self._entry(d, ident, sha, "uploaded", meta.get("filename", f.name), meta.get("uploaded_by"), meta.get("uploaded_at"))
                    e.replaces_built_in = d.doc_id in entries
                    docs[d.doc_id], entries[d.doc_id] = d, e
            self._docs, self._entries = docs, entries
            self._publish()
            report.added = sorted(set(entries) - set(before))
            report.removed = sorted(set(before) - set(entries))
            report.changed = sorted(i for i in entries if i in before and before[i] != entries[i].sha256)
            report.total = len(entries)
            return report

    # ------------------------------------------------------------------ adding one
    def ingest(self, data: bytes, filename: str, *, uploaded_by: str | None = None, replace: bool = False, document_id: str | None = None) -> IngestResult:
        if self._uploads is None:
            raise PolicyDocumentError("this server has no folder for uploaded policy documents")
        with self._lock:
            d, ident, sha = self._decompose(data, filename, given_id=document_id, allow_file_name=False, use_llm=True)
            existing = self._entries.get(d.doc_id)
            if existing is not None and existing.sha256 == sha:
                return IngestResult("unchanged", existing)
            if existing is not None and not replace:
                raise PolicyConflict(f"a document with the id '{d.doc_id}' already exists ({existing.source}: {existing.filename}) with different content; "
                                     "send it with replace=true if the new one is meant to supersede it")
            (self._uploads / f"{d.doc_id}.pdf").write_bytes(data)
            index = self._read_index()
            stamp = self._clock().isoformat()
            index[d.doc_id] = {"filename": filename, "uploaded_by": uploaded_by, "uploaded_at": stamp, "sha256": sha, "id_from": ident.source, "version": ident.version,
                               "id_evidence": ident.line, "details": ident.details, "marks": _marks_to(d.marks)}
            self._index_path().write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")      # type: ignore[union-attr]
            e = self._entry(d, ident, sha, "uploaded", filename, uploaded_by, stamp)
            e.replaces_built_in = existing is not None and existing.source == "built-in"
            self._docs[d.doc_id], self._entries[d.doc_id] = d, e
            self._publish()
            return IngestResult("replaced" if existing is not None else "added", e)

    def entries(self) -> list[PolicyEntry]:
        with self._lock:
            return sorted(self._entries.values(), key=lambda e: e.doc_id)


__all__ = ["IdentityReader", "IngestResult", "PolicyConflict", "PolicyDocumentError", "PolicyDocumentManager", "PolicyEntry", "ReloadReport"]
