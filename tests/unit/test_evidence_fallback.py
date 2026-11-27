"""The Evidence agent's document-intelligence fallbacks (classifier tie-breaker, required-field filler, scan transcription) are on when a model is given, and only then."""
import json
from pathlib import Path

import pytest
import pymupdf

from claimtower.agents.evidence import EvidenceAgent
from claimtower.composition import SYNTHETIC, build_registry
from claimtower.core.config import load_settings
from claimtower.core.schemas import ClaimState
from claimtower.ingestion.classifier import HybridClassifier, LLMClassifier, RuleClassifier
from claimtower.ingestion.extractors import HybridExtractor, RuleExtractor
from claimtower.ingestion.extractors.llm import LLMFieldFiller
from claimtower.ingestion.ocr import VisionLLMOcr
from claimtower.ingestion.store import DocumentStore

SETTINGS = load_settings()
SCAN = (SYNTHETIC / "claims" / "CLM-0002" / "surveyor_report_scanned.pdf").read_bytes()


def surveyor_text() -> str:
    with pymupdf.open(SYNTHETIC / "claims" / "CLM-0002" / "surveyor_report.pdf") as pdf:
        return "\n".join(p.get_text() for p in pdf)


class Ocr:
    name = "test-ocr"

    def __init__(self, text=None, fail=False):
        self.text, self.fail, self.calls = text, fail, 0

    def recognise(self, image: bytes, mime: str) -> str:
        self.calls += 1
        if self.fail:
            raise RuntimeError("model unavailable")
        return self.text


def agent(store, ocr=None) -> EvidenceAgent:
    pack = SETTINGS.pack
    return EvidenceAgent(store, pack, HybridClassifier(RuleClassifier(pack.document_types)), HybridExtractor(RuleExtractor(pack.document_types)), ocr)


@pytest.fixture
def store(tmp_path):
    s = DocumentStore(tmp_path / "store")
    s.ingest("CLM-T", "scan.pdf", SCAN, document_id="DOC-SCAN")
    return s


def run(a, claim="CLM-T"):
    return a.run(ClaimState(claim_id=claim))


# ---------------------------------------------------------------------------------------------------------------- wiring
class FakeLLM:
    def derive(self, **kw):
        return self


def test_a_model_switches_the_three_fallbacks_on_and_no_model_leaves_them_off(tmp_path):
    s = DocumentStore(tmp_path / "s")
    on = build_registry(SETTINGS, document_store=s, llm=FakeLLM())["evidence"].agent
    assert isinstance(on.classifier.llm, LLMClassifier) and isinstance(on.extractor.llm, LLMFieldFiller) and isinstance(on.ocr, VisionLLMOcr)
    off = build_registry(SETTINGS, document_store=s)["evidence"].agent
    assert off.classifier.llm is None and off.extractor.llm is None and off.ocr is None


# ---------------------------------------------------------------------------------------------------------------- scan transcription
def test_without_an_ocr_engine_a_scan_stays_unread_and_is_flagged_for_a_human(store):
    p = run(agent(store))
    d = p.updates["documents"][0]
    assert d.needs_ocr and d.detected_type == "unknown" and p.updates["facts"] == [] and "REVIEW" not in p.rationale


def test_with_an_ocr_engine_the_scan_is_transcribed_typed_and_its_facts_cite_the_ocr_text(store):
    ocr = Ocr(surveyor_text())
    p = run(agent(store, ocr))
    d = p.updates["documents"][0]
    assert ocr.calls == 1 and not d.needs_ocr and d.detected_type == "surveyor_report" and p.updates["facts"]
    assert "REVIEW: 1 scanned document(s) transcribed by the vision model: DOC-SCAN" in p.rationale
    assert all(f.source_document_id == "DOC-SCAN" and f.evidence_text for f in p.updates["facts"])


def test_a_transcribed_scan_is_not_transcribed_again_on_the_next_run(store):
    ocr = Ocr(surveyor_text())
    a = agent(store, ocr)
    first = a.run(ClaimState(claim_id="CLM-T"))
    state = ClaimState(claim_id="CLM-T", documents=first.updates["documents"], facts=first.updates["facts"])
    second = a.run(state)
    assert ocr.calls == 1 and second.updates["facts"] == first.updates["facts"] and "scanned" not in second.rationale


def test_a_failing_ocr_engine_leaves_the_scan_flagged_and_says_so_and_is_retried_next_time(store):
    ocr = Ocr(fail=True)
    a = agent(store, ocr)
    p = run(a)
    d = p.updates["documents"][0]
    assert d.needs_ocr and d.detected_type == "unknown" and "OCR errors on DOC-SCAN" in p.rationale
    run(a)
    assert ocr.calls == 2


def test_a_text_document_is_never_sent_to_the_ocr_engine(tmp_path):
    s = DocumentStore(tmp_path / "s")
    s.ingest("CLM-T", "report.pdf", (SYNTHETIC / "claims" / "CLM-0002" / "surveyor_report.pdf").read_bytes(), document_id="DOC-T")
    ocr = Ocr("never used")
    p = run(agent(s, ocr))
    assert ocr.calls == 0 and p.updates["documents"][0].detected_type == "surveyor_report" and "scanned" not in p.rationale


def test_the_manifest_text_of_the_synthetic_claim_is_what_the_fake_engine_returns():
    assert "surveyor" in surveyor_text().lower() and json.loads((SYNTHETIC / "claims" / "CLM-0002" / "manifest.json").read_text(encoding="utf-8"))["claim_id"] == "CLM-0002"
    assert Path(SYNTHETIC / "claims" / "CLM-0002" / "surveyor_report_scanned.pdf").is_file()


# ---------------------------------------------------------------------------------------------------------------- the recorded vision transcription of the real scan
def test_the_recorded_model_transcription_of_the_scan_feeds_the_agent_offline(store):
    """data/cassettes/live_check.json holds the model's recorded transcription of this scan (T2.1 live check, E3): replayed here, no live call."""
    from claimtower.core.llm import Cassette, LLMClient
    llm = LLMClient.from_settings(SETTINGS)
    llm.mode, llm.cassette = "replay", Cassette(Path(__file__).resolve().parents[2] / "data" / "cassettes" / "live_check.json")
    p = run(agent(store, VisionLLMOcr(llm)))
    d = p.updates["documents"][0]
    assert not d.needs_ocr and d.detected_type == "surveyor_report" and p.updates["facts"] and "scanned document" in p.rationale
