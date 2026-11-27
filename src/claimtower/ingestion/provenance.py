"""Facts with provenance (T2.4): extracted values -> `Fact`s that point at their evidence, and the checks that prove the pointers are true.

    built = build_facts(claim_id=..., document_id=..., doc_type=..., result=extraction, page_text=..., templates=pack.fact_templates)
    built.facts       only facts that PASSED the verbatim check: proposition, evidence_text, page, region, confidence, question tags
    built.rejected    anything that failed, with the reason: a fact that cannot cite its evidence is never emitted

Guarantees (BUILD_PLAN T2.4, guardrail 5)
  * `evidence_text` is a verbatim substring of the cited page. Basis: the whitespace-normalised page text (`loaders.normalise`), because a
    long line can wrap in a PDF. This is validated in code BEFORE a fact leaves this module.
  * `verify_fact_citations` re-checks facts against the ORIGINAL document, independently of the derived text the extractor used, and (for
    PDFs) that the stored region actually contains the evidence. That is the citation-correctness metric.
  * Facts leave here as evidence_type "neutral": whether a fact supports or contradicts an open question is decided later by the
    fact-question linker (T4.2). `related_question_ids` only says which questions a fact BEARS ON (pack: fact_templates.yaml).
  * Low-confidence values still become facts, with their low extraction_confidence: nothing is hidden, nothing is promoted.
  * Fact ids are stable and readable (`F-<document id>-<field>[-<n>]`) so a re-run after new evidence keeps the ids of unchanged facts.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

import pymupdf

from claimtower.core.config import FactTemplate, FactTemplates
from claimtower.core.schemas import DocumentRecord, Fact
from claimtower.core.schemas.common import BBox
from claimtower.ingestion.extractors.models import ExtractionResult, LineItem
from claimtower.ingestion.loaders import load, normalise
from claimtower.ingestion.store import DocumentStore

RULES_VERSION = "rules-1"
LLM_VERSION = "llm-field-filler-1"
REGION_MARGIN = 2.0                      # points of slack when checking that a region contains its evidence text
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "rs", "no", "approx", "st", "vs", "e.g", "i.e", "inr", "sr", "jr", "prof"}
_SENTENCE_END = re.compile(r"([.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? followed by space and a capital or digit, except after initials ('R. Gupta') and common abbreviations ('Mr.', 'Rs.').
    Every returned sentence is a verbatim slice of `text` (only stripped), so it stays a substring of the page."""
    parts: list[str] = []
    start = 0
    for m in _SENTENCE_END.finditer(text):
        token = re.search(r"(\S+)$", text[start:m.start(1)])
        word = token.group(1).lower().strip("(\"'") if token else ""
        if (len(word) == 1 and word.isalpha()) or word in _ABBREVIATIONS:
            continue
        parts.append(text[start:m.end(1)].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def money_text(amount: Decimal, currency: str | None) -> str:
    return f"{currency + ' ' if currency else ''}{amount:,}"


@dataclass(frozen=True)
class Rejected:
    field: str
    reason: str


@dataclass
class BuiltFacts:
    facts: list[Fact] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)          # e.g. a region dropped because it did not contain its text


def _region_contains(page: pymupdf.Page, region: BBox, evidence: str) -> bool:
    w, h = page.rect.width, page.rect.height
    m = REGION_MARGIN
    clip = pymupdf.Rect(region.x0 * w - m, region.y0 * h - m, region.x1 * w + m, region.y1 * h + m)
    return normalise(evidence) in normalise(page.get_text("text", clip=clip))


def build_facts(*, claim_id: str, document_id: str, doc_type: str, result: ExtractionResult, page_text: Mapping[int, str],
                templates: FactTemplates, pdf: pymupdf.Document | None = None) -> BuiltFacts:
    """Turn an ExtractionResult into verified Facts. `page_text` maps page number -> raw page text. If `pdf` is given (the original),
    each region is also checked against it and dropped (never invented) when it does not contain the evidence."""
    out = BuiltFacts()
    pages = {n: normalise(t) for n, t in page_text.items()}

    def emit(fact_id: str, proposition: str, evidence: str, page: int, region: BBox | None, confidence: float, version: str,
             questions: list[str], label: str) -> None:
        if not evidence.strip():
            out.rejected.append(Rejected(label, "empty evidence"))
            return
        if page not in pages or normalise(evidence) not in pages[page]:
            out.rejected.append(Rejected(label, f"evidence is not verbatim text of page {page}"))
            return
        if region is not None and pdf is not None and 1 <= page <= pdf.page_count and not _region_contains(pdf[page - 1], region, evidence):
            out.notes.append(f"{fact_id}: region did not contain the evidence text and was dropped")
            region = None
        out.facts.append(Fact(
            fact_id=fact_id, claim_id=claim_id, proposition=proposition, source_document_id=document_id, page=page, region=region,
            evidence_text=normalise(evidence), evidence_type="neutral", related_question_ids=list(questions),
            extraction_confidence=confidence, verification_status="unverified", extractor_version=version))

    for f in result.fields:
        tpl = templates.templates.get(f"{doc_type}.{f.name}")
        if tpl is None:
            out.rejected.append(Rejected(f.name, f"no fact template for '{doc_type}.{f.name}'"))
            continue
        version = LLM_VERSION if f.method == "llm" else RULES_VERSION
        base = dict(value=f.value, normalised=f.normalised or f.value, label=f.label, currency=f.currency or "")
        if tpl.split_sentences:
            sentences = [s for piece in f.value.split("\n") for s in split_sentences(piece)]          # a model's list of findings arrives one per line; a rule's text is one line
            for i, sentence in enumerate(sentences, start=1):
                emit(f"F-{document_id}-{f.name}-{i}", tpl.proposition.format(**base, sentence=sentence), sentence, f.page, f.region,
                     f.confidence, version, tpl.questions, f.name)
        else:
            emit(f"F-{document_id}-{f.name}", tpl.proposition.format(**base), f.evidence_text, f.page, f.region, f.confidence, version,
                 tpl.questions, f.name)
    if result.table is not None:
        tpl = templates.templates.get(f"{doc_type}.line_item")
        for i, item in enumerate(result.table.items, start=1):
            if tpl is None:
                out.rejected.append(Rejected(f"line_item {i}", f"no fact template for '{doc_type}.line_item'"))
                continue
            _line_item_fact(out, emit, tpl, document_id, i, item)
    return out


def _line_item_fact(out: BuiltFacts, emit: Callable[..., None], tpl: FactTemplate, document_id: str, i: int, item: LineItem) -> None:
    prop = tpl.proposition.format(description=item.description, amount=money_text(item.amount, item.currency), value=item.description,
                                  normalised=str(item.amount), label=item.description, currency=item.currency or "")
    emit(f"F-{document_id}-item-{i}", prop, item.evidence_text, item.page, item.region, item.confidence, RULES_VERSION, tpl.questions,
         f"line_item {i}")


# ---------------------------------------------------------------------------------------------------- independent verification
@dataclass(frozen=True)
class CitationProblem:
    fact_id: str
    problem: str


@dataclass
class CitationReport:
    """Citation correctness: do the facts' pointers tell the truth about the ORIGINAL documents?"""

    total: int = 0
    verbatim_ok: int = 0
    region_checked: int = 0
    region_ok: int = 0
    ocr_basis: int = 0                      # facts cited against OCR text (the original has no text layer to check)
    problems: list[CitationProblem] = field(default_factory=list)

    @property
    def verbatim_rate(self) -> float | None:
        return round(self.verbatim_ok / self.total, 4) if self.total else None

    @property
    def region_rate(self) -> float | None:
        return round(self.region_ok / self.region_checked, 4) if self.region_checked else None

    def summary(self) -> dict[str, object]:
        return {"facts": self.total, "verbatim_ok": self.verbatim_ok, "verbatim_rate": self.verbatim_rate,
                "region_checked": self.region_checked, "region_ok": self.region_ok, "region_rate": self.region_rate,
                "ocr_basis": self.ocr_basis,
                "problems": [f"{p.fact_id}: {p.problem}" for p in self.problems]}


def verify_fact_citations(facts: Sequence[Fact], store: DocumentStore) -> CitationReport:
    """Re-read each cited ORIGINAL (not the derived page text the extractor used) and check every fact: the document exists, the page
    exists, the evidence is verbatim text of that page, and - for PDFs - the stored region really contains the evidence. Text that only
    exists as OCR (a scanned page) is checked against the stored OCR text and counted in `ocr_basis`."""
    rep = CitationReport()
    by_doc: dict[str, list[Fact]] = {}
    for f in facts:
        by_doc.setdefault(f.source_document_id, []).append(f)
    for doc_id, group in by_doc.items():
        try:
            stored = store.stored(doc_id)
            data = store.original(doc_id)
            fresh = load(data, stored.filename)
        except Exception as e:                              # missing document, tampered bytes, unreadable file
            for f in group:
                rep.total += 1
                rep.problems.append(CitationProblem(f.fact_id, f"cannot re-read source document {doc_id}: {type(e).__name__}: {e}"))
            continue
        pdf = pymupdf.open(stream=data, filetype="pdf") if fresh.kind == "pdf" else None
        for f in group:
            rep.total += 1
            if f.page is None or not 1 <= f.page <= fresh.page_count:
                rep.problems.append(CitationProblem(f.fact_id, f"page {f.page} does not exist in {doc_id}"))
                continue
            text = normalise(fresh.pages[f.page - 1].text)
            on_ocr = not text and stored.pages[f.page - 1].source == "ocr"
            if on_ocr:
                text = normalise(stored.pages[f.page - 1].text)
                rep.ocr_basis += 1
            if not f.evidence_text.strip() or normalise(f.evidence_text) not in text:
                rep.problems.append(CitationProblem(f.fact_id, f"evidence is not verbatim text of {doc_id} page {f.page}"))
                continue
            rep.verbatim_ok += 1
            if f.region is not None and pdf is not None and not on_ocr:
                rep.region_checked += 1
                if _region_contains(pdf[f.page - 1], f.region, f.evidence_text):
                    rep.region_ok += 1
                else:
                    rep.problems.append(CitationProblem(f.fact_id, "region does not contain the evidence text"))
    return rep


# ---------------------------------------------------------------------------------------------------- the FactExtractor interface
class StoreFactExtractor:
    """Implements the frozen `FactExtractor` interface (core/interfaces.py): extract_facts(documents) -> list[Fact].

    It reads each document's stored extraction (running the extractor first if there is none and one was provided), builds facts, and
    returns only facts that passed the verbatim check. Whatever was refused is kept in `.rejected` / `.notes` for the caller to report."""

    def __init__(self, store: DocumentStore, templates: FactTemplates, extractor=None) -> None:
        self.store, self.templates, self.extractor = store, templates, extractor
        self.rejected: list[tuple[str, Rejected]] = []
        self.notes: list[str] = []

    def extract_facts(self, documents: list[DocumentRecord]) -> list[Fact]:
        self.rejected, self.notes = [], []
        facts: list[Fact] = []
        for rec in documents:
            stored = self.store.stored(rec.document_id)
            result = stored.extraction
            if result is None and self.extractor is not None:
                result = self.store.extract(rec.document_id, self.extractor)
            if result is None or result.status != "ok":
                continue                                                          # nothing was extracted (unknown type, no spec): no facts
            pdf = None
            if stored.kind == "pdf":
                try:
                    pdf = pymupdf.open(stream=self.store.original(rec.document_id), filetype="pdf")
                except Exception:
                    pdf = None
            built = build_facts(claim_id=rec.claim_id, document_id=rec.document_id, doc_type=result.document_type, result=result,
                                page_text={p.number: p.text for p in stored.pages}, templates=self.templates, pdf=pdf)
            facts.extend(built.facts)
            self.rejected.extend((rec.document_id, r) for r in built.rejected)
            self.notes.extend(built.notes)
        return facts
