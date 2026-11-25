"""Facts-with-provenance eval (T2.4; reused by the T8.1 harness).

Two questions:

1. CITATION CORRECTNESS (the done-when): is every fact's pointer true? Each fact is re-checked against the ORIGINAL document: the page
   exists, `evidence_text` is verbatim text of that page, and for PDFs the stored region really contains it. Measured over facts from
   (a) the five scenarios, run through the real Evidence agent, and (b) all labelled text documents, typed by a human and extracted.
2. COVERAGE of the hand-labelled ground truth (data/ground_truth/*.json): a ground-truth fact is COVERED when the pipeline produced a fact
   from the same document and page whose evidence contains the ground-truth evidence text. Question links are compared separately:
   the pipeline only says which questions a fact BEARS ON (direction is the linker's job, T4.2), so the ground-truth links are compared
   as a set against the fact's tags.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claimtower.agents.evidence import EvidenceAgent
from claimtower.core.config import Pack
from claimtower.core.schemas import ClaimState, Fact
from claimtower.core.state import apply_patch
from claimtower.ingestion.loaders import normalise
from claimtower.ingestion.provenance import CitationReport, StoreFactExtractor, verify_fact_citations
from claimtower.ingestion.store import DocumentStore
from evals.ground_truth import load_all_ground_truth, load_scenarios
from evals.metrics.extraction import item_cases

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FactsEval:
    scenario_facts: int = 0
    text_facts: int = 0
    citation: CitationReport = field(default_factory=CitationReport)
    gt_total: int = 0
    gt_covered: int = 0
    gt_uncovered: list[str] = field(default_factory=list)
    links_total: int = 0
    links_covered: int = 0
    links_missing: list[str] = field(default_factory=list)
    low_confidence_facts: int = 0
    neutral_only: bool = True
    ids_unique: bool = True
    refused: list[str] = field(default_factory=list)

    @property
    def gt_recall(self) -> float | None:
        return round(self.gt_covered / self.gt_total, 4) if self.gt_total else None

    @property
    def link_recall(self) -> float | None:
        return round(self.links_covered / self.links_total, 4) if self.links_total else None

    def summary(self) -> dict[str, Any]:
        return {"scenario_facts": self.scenario_facts, "text_document_facts": self.text_facts, "citation": self.citation.summary(),
                "ground_truth_facts": self.gt_total, "ground_truth_covered": self.gt_covered, "ground_truth_recall": self.gt_recall,
                "ground_truth_uncovered": self.gt_uncovered, "question_links": self.links_total,
                "question_links_covered": self.links_covered,
                "question_link_recall": self.link_recall, "question_links_missing": self.links_missing,
                "low_confidence_facts": self.low_confidence_facts, "all_neutral": self.neutral_only, "fact_ids_unique": self.ids_unique,
                "refused": self.refused}


def evaluate_facts(pack: Pack, classifier: Any, extractor: Any) -> tuple[FactsEval, dict[str, list[Fact]]]:
    ev = FactsEval()
    facts_by_claim: dict[str, list[Fact]] = {}
    gts = load_all_ground_truth()
    with tempfile.TemporaryDirectory() as tmp:
        store = DocumentStore(Path(tmp) / "store")
        # (a) the five scenarios through the real Evidence agent
        agent = EvidenceAgent(store, pack, classifier, extractor)
        all_scenario_facts: list[Fact] = []
        for sc in load_scenarios().scenarios:
            for d in sc.documents:
                data = (ROOT / "data" / "synthetic" / "claims" / sc.claim_id / d.filename).read_bytes()
                store.ingest(sc.claim_id, d.filename, data, document_id=d.document_id, uploaded_by=d.uploaded_by)
            state = ClaimState(claim_id=sc.claim_id)
            patch = agent.run(state)
            state = apply_patch(state, patch)
            facts_by_claim[sc.claim_id] = state.facts
            all_scenario_facts.extend(state.facts)
            ev.refused.extend(f"{doc}/{r.field}: {r.reason}" for doc, r in agent._facts.rejected)
            _compare_to_ground_truth(ev, sc.id, state.facts, gts[sc.id])
        ev.scenario_facts = len(all_scenario_facts)
        # (b) every labelled text document: type given by a human, then extraction -> facts
        types = list(pack.document_types.types)
        text_facts: list[Fact] = []
        extractor_facts = StoreFactExtractor(store, pack.fact_templates, extractor)
        for n, c in enumerate(item_cases(), start=1):
            doc_id = f"T-{n:04d}"
            store.ingest("TEXT-ITEMS", c.filename, c.data, document_id=doc_id)
            store.set_type(doc_id, c.doc_type, by="eval", allowed=types)
            text_facts.extend(extractor_facts.extract_facts([store.record(doc_id)]))
            ev.refused.extend(f"{doc_id}/{r.field}: {r.reason}" for _, r in extractor_facts.rejected)
        ev.text_facts = len(text_facts)
        everything = [*all_scenario_facts, *text_facts]
        ev.citation = verify_fact_citations(everything, store)
        ev.low_confidence_facts = sum(f.extraction_confidence < pack.document_types.extractor.low_confidence for f in everything)
        ev.neutral_only = all(f.evidence_type == "neutral" and f.verification_status == "unverified" for f in everything)
        ev.ids_unique = len({f.fact_id for f in everything}) == len(everything)
    return ev, facts_by_claim


def _compare_to_ground_truth(ev: FactsEval, sid: str, facts: list[Fact], gt: Any) -> None:
    for g in gt.facts:
        ev.gt_total += 1
        want = normalise(g.evidence_text)
        same_place = [f for f in facts if f.source_document_id == g.source_document_id and f.page == g.page]
        hits = [f for f in same_place if want in normalise(f.evidence_text)]
        if hits:
            ev.gt_covered += 1
        else:
            ev.gt_uncovered.append(f"{g.fact_id}: {g.evidence_text!r}")
        tags = {q for f in hits for q in f.related_question_ids}
        for q in g.related_question_ids:
            ev.links_total += 1
            if q in tags:
                ev.links_covered += 1
            else:
                ev.links_missing.append(f"{g.fact_id}->{q}" + ("" if hits else " (no matching fact)"))
