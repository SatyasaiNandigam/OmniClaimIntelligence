"""Evidence agent (T2.4): the claim's documents -> classified, extracted, and turned into verified Facts with provenance.

Owns (writes) exactly `documents`, `facts` and `image_findings` (core.state.OWNERSHIP). It reads the claim's documents from the
DocumentStore and never touches the originals. `image_findings` is carried over unchanged: image analysis is T2.5 / T2.6.

Incremental by design (BUILD_PLAN T6.2): a document is (re)processed only if it is new or its
`(sha256, detected_type, needs_ocr, page_count)` changed since the state's `documents` (new upload, OCR finished, a human re-typed it).
Facts of unchanged documents are kept as they are, including their ids and any human `verification_status`; facts of removed documents
are dropped. A fact whose evidence changed loses its human verification, because the human verified different evidence.

Safety: every fact is re-checked against the ORIGINAL document before it is returned (verbatim text and, for PDFs, region). A fact that
fails is dropped and named in the patch rationale: the agent never emits a fact it cannot cite. A document it cannot classify stays in
`documents` with type 'unknown' (a human decides); no facts are guessed for it.
"""
from __future__ import annotations

from typing import ClassVar

from claimtower.core.config import Pack
from claimtower.core.schemas import ClaimState, DocumentRecord, Fact
from claimtower.core.state import OWNERSHIP, StatePatch
from claimtower.ingestion.classifier import HybridClassifier
from claimtower.ingestion.extractors import HybridExtractor
from claimtower.ingestion.ocr import OcrEngine
from claimtower.ingestion.provenance import StoreFactExtractor, verify_fact_citations
from claimtower.ingestion.store import DocumentStore


def _signature(d: DocumentRecord) -> tuple:
    return (d.sha256, d.detected_type, d.needs_ocr, d.page_count)


class EvidenceAgent:
    agent_id: ClassVar[str] = "evidence"
    owns: ClassVar[frozenset[str]] = OWNERSHIP["evidence"]

    def __init__(self, store: DocumentStore, pack: Pack, classifier: HybridClassifier, extractor: HybridExtractor, ocr: OcrEngine | None = None) -> None:
        self.store, self.pack, self.classifier, self.extractor, self.ocr = store, pack, classifier, extractor, ocr
        self._facts = StoreFactExtractor(store, pack.fact_templates)

    def run(self, state: ClaimState) -> StatePatch:
        previous = {d.document_id: d for d in state.documents}
        records: list[DocumentRecord] = []
        changed: list[DocumentRecord] = []
        scanned: list[str] = []
        for rec in self.store.records(state.claim_id):
            if self.ocr is not None and rec.needs_ocr:                    # a scan: the model transcribes it (the text is flagged `ocr`, never treated as verbatim)
                rec = self.store.run_ocr(rec.document_id, self.ocr)
                scanned.append(rec.document_id)
            stored = self.store.stored(rec.document_id)
            old = previous.get(rec.document_id)
            if old is not None and _signature(old) == _signature(rec) and old.detected_type != "unknown":
                records.append(rec)                                       # unchanged: keep its facts as they are
                continue
            if rec.detected_type == "unknown" and stored.type_method != "human":
                self.store.classify(rec.document_id, self.classifier)    # cheap rules; also re-tried after OCR completes
            self.store.extract(rec.document_id, self.extractor)
            fresh = self.store.record(rec.document_id)
            records.append(fresh)
            changed.append(fresh)                                         # new, changed, or 'unknown' being retried

        kept_ids = {r.document_id for r in records} - {c.document_id for c in changed}
        old_by_id = {f.fact_id: f for f in state.facts}
        facts: list[Fact] = [f for f in state.facts if f.source_document_id in kept_ids]

        new_facts = self._facts.extract_facts(changed)
        report = verify_fact_citations(new_facts, self.store)
        bad = {p.fact_id for p in report.problems}
        accepted: list[Fact] = []
        for f in new_facts:
            if f.fact_id in bad:
                continue
            old = old_by_id.get(f.fact_id)
            if old is not None and old.verification_status != "unverified" and old.evidence_text == f.evidence_text:
                f = f.model_copy(update={"verification_status": old.verification_status})      # a human already decided about this evidence
            accepted.append(f)
        facts.extend(accepted)
        facts.sort(key=lambda f: (f.source_document_id, f.fact_id))

        unknown = [r.document_id for r in records if r.detected_type == "unknown"]
        if scanned:
            left = [d for d in scanned if self.store.stored(d).ocr_errors]
            parts_ocr = f"REVIEW: {len(scanned)} scanned document(s) transcribed by the vision model: {', '.join(sorted(scanned))}" + (f" (OCR errors on {', '.join(sorted(left))})" if left else "")
        parts = [f"{len(changed)} document(s) processed, {len(records) - len(changed)} unchanged; "
                 f"{len(accepted)} new fact(s), {len(facts)} in total"]
        if scanned:
            parts.append(parts_ocr)
        floor = self.pack.document_types.extractor.low_confidence
        low = sorted(f.fact_id for f in facts if f.extraction_confidence < floor and f.verification_status == "unverified")
        if low:                                                                          # governance: a low-confidence reading is flagged for a person, never hidden or promoted
            parts.append(f"REVIEW: {len(low)} fact(s) read with low extraction confidence (below {floor:.2f}); a person should check them against the document: {', '.join(low[:8])}" + (" ..." if len(low) > 8 else ""))
        if unknown:
            parts.append(f"{len(unknown)} document(s) of unknown type need a human: {', '.join(sorted(unknown))}")
        if self._facts.rejected:
            refused = "; ".join(f"{d}/{r.field}: {r.reason}" for d, r in self._facts.rejected)
            parts.append("refused (no verifiable evidence): " + refused)
        if report.problems:
            dropped = "; ".join(f"{p.fact_id}: {p.problem}" for p in report.problems)
            parts.append("dropped after re-verification against the originals: " + dropped)
        return StatePatch(agent_id=self.agent_id, rationale=" | ".join(parts),
                          updates={"documents": sorted(records, key=lambda d: d.document_id), "facts": facts,
                                   "image_findings": list(state.image_findings)})
