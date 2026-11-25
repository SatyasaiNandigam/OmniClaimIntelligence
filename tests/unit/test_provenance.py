"""T2.4 facts with provenance: every fact cites verbatim evidence, the citation is re-verifiable, and the Evidence agent is incremental."""
import json
import os
import shutil
from pathlib import Path

import pymupdf
import pytest
import yaml

from claimtower.agents import evidence as evidence_module
from claimtower.agents.evidence import EvidenceAgent
from claimtower.core.config import FactTemplate, FactTemplates, QuestionTemplates, check_fact_templates, load_settings
from claimtower.core.errors import ConfigError
from claimtower.core.schemas import ClaimState, Fact
from claimtower.core.schemas.common import BBox
from claimtower.core.state import OWNERSHIP, apply_patch
from claimtower.ingestion.classifier import HybridClassifier, RuleClassifier
from claimtower.ingestion.extractors import HybridExtractor, RuleExtractor
from claimtower.ingestion.extractors.models import ExtractedField, ExtractionResult, LineItem
from claimtower.ingestion.loaders import normalise
from claimtower.ingestion.provenance import (
    BuiltFacts, StoreFactExtractor, build_facts, money_text, split_sentences, verify_fact_citations,
)
from claimtower.ingestion.store import DocumentStore

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data/synthetic"
PACK = load_settings().pack
DT = PACK.document_types
TPL = PACK.fact_templates
CLF = HybridClassifier(RuleClassifier(DT))
EXTRACTOR = HybridExtractor(RuleExtractor(DT))
NAMES = ["fnol.pdf", "plumber_report.pdf", "surveyor_report.pdf", "estimate.pdf"]


@pytest.fixture
def store(tmp_path):
    return DocumentStore(tmp_path / "s")


def ingest_claim(store, names=NAMES, claim="CLM-0001", start=1):
    for i, f in enumerate(names, start=start):
        store.ingest(claim, f, (SYN / "claims" / claim / f).read_bytes(), document_id=f"DOC-S01-{i}")


def agent(store):
    return EvidenceAgent(store, PACK, CLF, EXTRACTOR)


# ================================================================== sentence splitting
@pytest.mark.parametrize("text, expected", [
    ("The joint failed. It was replaced.", ["The joint failed.", "It was replaced."]),
    ("Claimant R. Gupta reported a leak. Water spread.", ["Claimant R. Gupta reported a leak.", "Water spread."]),
    ("Mr. R. Gupta called. He was upset.", ["Mr. R. Gupta called.", "He was upset."]),
    ("The cost was Rs. 5,000 in total. Paid later.", ["The cost was Rs. 5,000 in total.", "Paid later."]),
    ("Assessed INR 98,000. No part is for upgrades.", ["Assessed INR 98,000.", "No part is for upgrades."]),
    ("Pipe diameter 3.5 mm was found. Corroded.", ["Pipe diameter 3.5 mm was found.", "Corroded."]),
    ("No punctuation at the end", ["No punctuation at the end"]),
    ("Is it leaking? Yes it is! Fix it.", ["Is it leaking?", "Yes it is!", "Fix it."]),
    ("", []),
])
def test_sentences_split_at_real_boundaries_only(text, expected):
    assert split_sentences(text) == expected


def test_every_sentence_is_a_verbatim_slice_of_the_text():
    text = "Significant corrosion was found at the joint. The joint failed at the corroded section.  Replaced on 16 March 2026."
    for s in split_sentences(text):
        assert s in text


# ================================================================== building facts
def extraction(doc_type, text_by_page, fields=(), table=None):
    return ExtractionResult(document_type=doc_type, status="ok", fields=list(fields), table=table)


def fld(name, value, evidence, page=1, conf=0.95, norm=None, label="L", region=None, method="rule", currency=None):
    return ExtractedField(name=name, value=value, normalised=norm if norm is not None else value, currency=currency, label=label, page=page,
                          region=region, evidence_text=evidence, confidence=conf, low_confidence=conf < 0.7, method=method)


def test_a_field_becomes_a_fact_with_everything_a_reviewer_needs():
    page = "FIRST NOTICE OF LOSS\nDate of loss: 14 March 2026\n"
    r = extraction("fnol", {1: page}, [fld("loss_date", "14 March 2026", "Date of loss: 14 March 2026", norm="2026-03-14", label="Date of loss")])
    built = build_facts(claim_id="C1", document_id="D1", doc_type="fnol", result=r, page_text={1: page}, templates=TPL)
    assert built.rejected == [] and len(built.facts) == 1
    f = built.facts[0]
    assert isinstance(f, Fact)
    assert (f.fact_id, f.claim_id, f.source_document_id, f.page) == ("F-D1-loss_date", "C1", "D1", 1)
    assert f.proposition == "Reported date of loss is 14 March 2026" and f.evidence_text == "Date of loss: 14 March 2026"
    assert f.related_question_ids == ["Q-INFORCE", "Q-SUBLIMIT"] and f.extraction_confidence == 0.95
    assert (f.evidence_type, f.verification_status, f.extractor_version) == ("neutral", "unverified", "rules-1")


def test_narrative_fields_become_one_fact_per_sentence_each_with_its_own_evidence():
    text = "Findings: Significant corrosion was found at the joint. The joint failed at the corroded section."
    value = "Significant corrosion was found at the joint. The joint failed at the corroded section."
    r = extraction("plumber_report", {1: text}, [fld("findings", value, text)])
    built = build_facts(claim_id="C1", document_id="D2", doc_type="plumber_report", result=r, page_text={1: text}, templates=TPL)
    assert [f.fact_id for f in built.facts] == ["F-D2-findings-1", "F-D2-findings-2"]
    assert [f.evidence_text for f in built.facts] == ["Significant corrosion was found at the joint.", "The joint failed at the corroded section."]
    assert built.facts[1].proposition == "Plumber reports: The joint failed at the corroded section."
    assert all(f.related_question_ids == ["Q-SUDDEN", "Q-START"] for f in built.facts)


def test_table_line_items_become_facts_with_formatted_amounts():
    page = "Kitchen floor: INR 110,000\nTotal: INR 110,000"
    item = LineItem(description="Kitchen floor", amount=110000, currency="INR", page=1, evidence_text="Kitchen floor: INR 110,000",
                    confidence=0.95, low_confidence=False)
    from claimtower.ingestion.extractors.models import TableResult
    r = extraction("estimate", {1: page}, [fld("total_amount", "INR 110,000", "Total: INR 110,000", norm="110000", label="Total")],
                   table=TableResult(items=[item], status="reconciled"))
    built = build_facts(claim_id="C1", document_id="D4", doc_type="estimate", result=r, page_text={1: page}, templates=TPL)
    assert {f.fact_id for f in built.facts} == {"F-D4-total_amount", "F-D4-item-1"}
    assert next(f for f in built.facts if f.fact_id == "F-D4-item-1").proposition == "Estimate line 'Kitchen floor' is INR 110,000"
    assert money_text(__import__("decimal").Decimal("1250.5"), None) == "1,250.5"


def test_a_fact_that_cannot_cite_verbatim_evidence_is_refused_never_emitted():
    page = "Date of loss: 14 March 2026"
    r = extraction("fnol", {1: page}, [
        fld("loss_date", "14 March 2026", "Date of loss: 14 March 2026"),                      # fine
        fld("claimant_name", "R. Gupta", "Claimant: R. Gupta"),                                 # evidence not on the page
        fld("policy_number", "POL-1", "Policy number: POL-1", page=2),                          # page does not exist
        fld("estimated_loss", "x", "   "),                                                      # empty evidence
    ])
    built = build_facts(claim_id="C1", document_id="D1", doc_type="fnol", result=r, page_text={1: page}, templates=TPL)
    assert [f.fact_id for f in built.facts] == ["F-D1-loss_date"]
    reasons = {x.field: x.reason for x in built.rejected}
    assert "not verbatim" in reasons["claimant_name"] and "page 2" in reasons["policy_number"] and "empty" in reasons["estimated_loss"]


def test_a_field_without_a_template_is_refused_visibly():
    page = "Whatever: 1"
    r = extraction("fnol", {1: page}, [fld("mystery_field", "1", "Whatever: 1")])
    built = build_facts(claim_id="C1", document_id="D1", doc_type="fnol", result=r, page_text={1: page}, templates=TPL)
    assert built.facts == [] and "no fact template" in built.rejected[0].reason


def test_low_confidence_and_unparseable_values_still_become_facts_with_their_low_confidence():
    page = "Date of loss: last Tuesday"
    r = extraction("fnol", {1: page}, [fld("loss_date", "last Tuesday", page, conf=0.4, norm=None)])
    f = build_facts(claim_id="C1", document_id="D1", doc_type="fnol", result=r, page_text={1: page}, templates=TPL).facts[0]
    assert f.extraction_confidence == 0.4 and "last Tuesday" in f.proposition and f.verification_status == "unverified"


def test_llm_filled_fields_carry_their_own_extractor_version():
    page = "Cover reference - POL-2201"
    r = extraction("fnol", {1: page}, [fld("policy_number", "POL-2201", page, method="llm", norm="POL-2201")])
    assert build_facts(claim_id="C", document_id="D", doc_type="fnol", result=r, page_text={1: page}, templates=TPL).facts[0].extractor_version == "llm-field-filler-1"


def test_page_text_spanning_a_wrapped_line_still_verifies_on_the_normalised_basis():
    page = "Description: On 14 March 2026 the kitchen supply pipe burst suddenly and water flooded the kitchen and\nhallway.\n"
    evidence = "Description: On 14 March 2026 the kitchen supply pipe burst suddenly and water flooded the kitchen and hallway."
    r = extraction("fnol", {1: page}, [fld("loss_description", evidence.split(": ", 1)[1], evidence, label="Description")])
    built = build_facts(claim_id="C", document_id="D", doc_type="fnol", result=r, page_text={1: page}, templates=TPL)
    assert built.rejected == [] and built.facts[0].evidence_text == "On 14 March 2026 the kitchen supply pipe burst suddenly and water flooded the kitchen and hallway."


def test_a_region_that_does_not_contain_the_evidence_is_dropped_not_kept():
    data = (SYN / "claims/CLM-0001/fnol.pdf").read_bytes()
    pdf = pymupdf.open(stream=data, filetype="pdf")
    page_text = {1: pdf[0].get_text()}
    line = "Date of loss: 14 March 2026"
    spot = pdf[0].search_for(line)[0]
    good = BBox(x0=spot.x0 / 595, y0=spot.y0 / 842, x1=spot.x1 / 595, y1=spot.y1 / 842)
    bad = BBox(x0=0.1, y0=0.9, x1=0.2, y1=0.95)                                       # somewhere else on the page
    for region, kept in ((good, True), (bad, False)):
        r = extraction("fnol", page_text, [fld("loss_date", "14 March 2026", line, region=region)])
        built = build_facts(claim_id="C", document_id="D", doc_type="fnol", result=r, page_text=page_text, templates=TPL, pdf=pdf)
        assert (built.facts[0].region is not None) is kept
        assert bool(built.notes) is (not kept)


# ================================================================== independent citation verification
def facts_of(store, claim="CLM-0001"):
    fx = StoreFactExtractor(store, TPL, EXTRACTOR)
    for r in store.records(claim):
        if store.stored(r.document_id).detected_type == "unknown":
            store.classify(r.document_id, CLF)
    return fx.extract_facts(store.records(claim)), fx


def test_every_fact_of_a_claim_passes_the_independent_check_against_the_originals(store):
    ingest_claim(store)
    facts, fx = facts_of(store)
    rep = verify_fact_citations(facts, store)
    assert len(facts) == 16 and fx.rejected == [] and rep.problems == []
    assert (rep.verbatim_rate, rep.region_rate, rep.region_checked) == (1.0, 1.0, 16)


def test_tampered_facts_are_caught(store):
    ingest_claim(store)
    facts, _ = facts_of(store)
    good = facts[0]
    wrong_text = good.model_copy(update={"fact_id": "F-bad-1", "evidence_text": "Date of loss: 1 January 1999"})
    wrong_page = good.model_copy(update={"fact_id": "F-bad-2", "page": 7})
    no_doc = good.model_copy(update={"fact_id": "F-bad-3", "source_document_id": "DOC-NOPE"})
    bad_region = good.model_copy(update={"fact_id": "F-bad-4", "region": BBox(x0=0.1, y0=0.9, x1=0.2, y1=0.95)})
    empty = good.model_copy(update={"fact_id": "F-bad-5", "evidence_text": " "})
    rep = verify_fact_citations([good, wrong_text, wrong_page, no_doc, bad_region, empty], store)
    problems = {p.fact_id: p.problem for p in rep.problems}
    assert set(problems) == {"F-bad-1", "F-bad-2", "F-bad-3", "F-bad-4", "F-bad-5"}
    assert "not verbatim" in problems["F-bad-1"] and "page 7" in problems["F-bad-2"] and "cannot re-read" in problems["F-bad-3"]
    assert "region" in problems["F-bad-4"] and rep.verbatim_ok == 2                      # the good fact and the bad-region one (text is fine)


def test_a_tampered_original_makes_its_facts_unverifiable(store):
    ingest_claim(store, NAMES[:1])
    facts, _ = facts_of(store)
    path = store.root / store.stored("DOC-S01-1").blob_path
    os.chmod(path, 0o666)
    path.write_bytes(b"tampered")
    rep = verify_fact_citations(facts, store)
    assert rep.verbatim_ok == 0 and all("cannot re-read" in p.problem for p in rep.problems)


def test_facts_from_ocr_text_are_checked_against_the_stored_ocr_text_and_counted(store):
    class Ocr:
        name = "stub"

        def recognise(self, image, mime):
            return "SURVEYOR'S REPORT\nSite visit: 5 May 2026\nAssessed repair cost: INR 98,000\n"

    store.ingest("CLM-0002", "scan.pdf", (SYN / "claims/CLM-0002/surveyor_report_scanned.pdf").read_bytes(), document_id="DOC-SCAN")
    store.run_ocr("DOC-SCAN", Ocr())
    store.classify("DOC-SCAN", CLF)
    facts = StoreFactExtractor(store, TPL, EXTRACTOR).extract_facts([store.record("DOC-SCAN")])
    rep = verify_fact_citations(facts, store)
    assert facts and rep.problems == [] and rep.ocr_basis == len(facts) and rep.region_checked == 0
    assert all(f.extraction_confidence <= DT.extractor.ocr_confidence_cap for f in facts)


# ================================================================== the frozen FactExtractor interface
def test_store_fact_extractor_matches_the_frozen_interface_signature():
    import inspect
    frozen = json.loads((ROOT / "docs/contracts/protocols.json").read_text(encoding="utf-8"))["FactExtractor"]["methods"]["extract_facts"]
    assert str(inspect.signature(StoreFactExtractor.extract_facts)).replace("'", "") == frozen.replace("'", "")


def test_store_fact_extractor_runs_extraction_when_missing_and_skips_untyped_documents(store):
    ingest_claim(store, NAMES[:1])
    store.ingest("CLM-0001", "note.txt", b"Page intentionally left blank, nothing else to see here at all.\n", document_id="DOC-N")
    store.classify("DOC-S01-1", CLF)
    store.classify("DOC-N", CLF)
    assert store.stored("DOC-S01-1").extraction is None
    fx = StoreFactExtractor(store, TPL, EXTRACTOR)
    facts = fx.extract_facts(store.records("CLM-0001"))
    assert {f.source_document_id for f in facts} == {"DOC-S01-1"}                      # the unknown-type note yields nothing
    assert store.stored("DOC-S01-1").extraction is not None
    no_extractor = StoreFactExtractor(store, TPL)
    assert [f.fact_id for f in no_extractor.extract_facts([store.record("DOC-S01-1")])] == [f.fact_id for f in facts]


# ================================================================== the Evidence agent
def test_agent_follows_the_agent_protocol_and_writes_only_what_it_owns(store):
    ingest_claim(store)
    a = agent(store)
    assert (a.agent_id, a.owns) == ("evidence", OWNERSHIP["evidence"])
    state = ClaimState(claim_id="CLM-0001")
    patch = a.run(state)
    assert set(patch.updates) == set(OWNERSHIP["evidence"]) and patch.agent_id == "evidence"
    new = apply_patch(state, patch)                                                    # ownership + contract validation
    assert new.state_version == 1 and len(new.facts) == 16 and len(new.documents) == 4
    assert [d.detected_type for d in new.documents] == ["fnol", "plumber_report", "surveyor_report", "estimate"]


def test_second_run_processes_nothing_and_keeps_every_fact_and_id(store):
    ingest_claim(store)
    a = agent(store)
    s1 = apply_patch(ClaimState(claim_id="CLM-0001"), a.run(ClaimState(claim_id="CLM-0001")))
    calls = []
    real = a.extractor.extract
    a.extractor.extract = lambda **kw: calls.append(kw["doc_type"]) or real(**kw)
    p2 = a.run(s1)
    assert calls == [] and "0 document(s) processed, 4 unchanged" in p2.rationale
    assert [f.fact_id for f in p2.updates["facts"]] == [f.fact_id for f in s1.facts]


def test_a_new_upload_processes_only_that_document(store):
    ingest_claim(store, NAMES[:3])
    a = agent(store)
    s1 = apply_patch(ClaimState(claim_id="CLM-0001"), a.run(ClaimState(claim_id="CLM-0001")))
    before = {f.fact_id: f for f in s1.facts}
    store.ingest("CLM-0001", "estimate.pdf", (SYN / "claims/CLM-0001/estimate.pdf").read_bytes(), document_id="DOC-S01-4")
    calls = []
    real = a.extractor.extract
    a.extractor.extract = lambda **kw: calls.append(kw["doc_type"]) or real(**kw)
    s2 = apply_patch(s1, a.run(s1))
    assert calls == ["estimate"] and len(s2.documents) == 4
    assert all(before[i] == f for i, f in ((f.fact_id, f) for f in s2.facts) if i in before), "unchanged documents keep identical facts"
    assert {f.source_document_id for f in s2.facts} - {f.source_document_id for f in s1.facts} == {"DOC-S01-4"} and len(s2.facts) == 16


def test_human_verification_survives_a_rerun_but_not_a_change_of_evidence(store):
    ingest_claim(store, NAMES[:1])
    a = agent(store)
    s1 = apply_patch(ClaimState(claim_id="CLM-0001"), a.run(ClaimState(claim_id="CLM-0001")))
    verified = [f.model_copy(update={"verification_status": "human_verified"}) if f.fact_id.endswith("loss_date") else f for f in s1.facts]
    s1h = s1.model_copy(update={"facts": verified, "documents": []})                  # empty documents -> everything is reprocessed
    s2 = apply_patch(s1h, a.run(s1h))
    assert next(f for f in s2.facts if f.fact_id.endswith("loss_date")).verification_status == "human_verified"
    changed = [f.model_copy(update={"verification_status": "human_verified", "evidence_text": "Date of loss: 1 January 1999"})
               if f.fact_id.endswith("loss_date") else f for f in s1.facts]
    s1c = s1.model_copy(update={"facts": changed, "documents": []})
    s3 = apply_patch(s1c, a.run(s1c))
    assert next(f for f in s3.facts if f.fact_id.endswith("loss_date")).verification_status == "unverified"


def test_a_human_retyping_a_document_reprocesses_it_and_replaces_its_facts(store):
    ingest_claim(store, NAMES[:1])
    a = agent(store)
    s1 = apply_patch(ClaimState(claim_id="CLM-0001"), a.run(ClaimState(claim_id="CLM-0001")))
    assert any(f.fact_id.startswith("F-DOC-S01-1-claimant_name") for f in s1.facts)
    store.set_type("DOC-S01-1", "other", by="adjuster", allowed=list(DT.types))
    s2 = apply_patch(s1, a.run(s1))
    assert [d.detected_type for d in s2.documents] == ["other"] and s2.facts == []     # 'other' has no field spec: no facts, old ones gone


def test_documents_that_cannot_be_typed_stay_visible_with_no_facts_and_a_reason(store):
    ingest_claim(store, NAMES[:1])
    store.ingest("CLM-0001", "menu.txt", b"THE GARDEN CAFE\nStarters: soup, bread\nMains: fish, curry\nDesserts: ice cream\n", document_id="DOC-MENU")
    patch = agent(store).run(ClaimState(claim_id="CLM-0001"))
    docs = {d.document_id: d for d in patch.updates["documents"]}
    assert docs["DOC-MENU"].detected_type == "unknown" and not [f for f in patch.updates["facts"] if f.source_document_id == "DOC-MENU"]
    assert "1 document(s) of unknown type need a human: DOC-MENU" in patch.rationale


def test_a_fact_that_fails_re_verification_is_dropped_and_reported(store, monkeypatch):
    ingest_claim(store, NAMES[:1])
    real = evidence_module.verify_fact_citations

    def fake(facts, st):
        rep = real(facts, st)
        from claimtower.ingestion.provenance import CitationProblem
        rep.problems.append(CitationProblem("F-DOC-S01-1-loss_date", "simulated mismatch"))
        return rep

    monkeypatch.setattr(evidence_module, "verify_fact_citations", fake)
    patch = agent(store).run(ClaimState(claim_id="CLM-0001"))
    assert not any(f.fact_id == "F-DOC-S01-1-loss_date" for f in patch.updates["facts"])
    assert "dropped after re-verification" in patch.rationale and "simulated mismatch" in patch.rationale


def test_a_refused_fact_is_named_in_the_rationale_and_image_findings_are_carried_over(store):
    class BadExtractor:
        def extract(self, **kw):
            return ExtractionResult(document_type="fnol", status="ok",
                                    fields=[fld("loss_date", "14 March 2026", "Date of loss: 14 March 2026"),
                                            fld("claimant_name", "R. Gupta", "Claimant: not in the document")])

    ingest_claim(store, NAMES[:1])
    a = EvidenceAgent(store, PACK, CLF, BadExtractor())
    from claimtower.core.schemas import ImageFinding
    finding = ImageFinding(finding_id="IF-1", document_id="DOC-S01-1", observation="stain", confidence=0.8)
    patch = a.run(ClaimState(claim_id="CLM-0001", image_findings=[finding]))
    assert [f.fact_id for f in patch.updates["facts"]] == ["F-DOC-S01-1-loss_date"]
    assert "refused (no verifiable evidence)" in patch.rationale and "claimant_name" in patch.rationale
    assert patch.updates["image_findings"] == [finding]


# ================================================================== whole-pipeline evaluation gate
def test_pipeline_meets_the_targets_and_does_not_regress_the_recorded_baseline():
    from evals.metrics.provenance import evaluate_facts
    from evals.runners.eval_facts import BASELINES, extractor_hash

    ev, facts_by_claim = evaluate_facts(PACK, CLF, EXTRACTOR)
    c = ev.citation
    assert c.total >= 400 and c.verbatim_rate == 1.0 and c.problems == [] and c.region_rate == 1.0       # the done-when: 100% pass
    assert ev.refused == [] and ev.ids_unique and ev.neutral_only
    assert ev.gt_recall == 1.0 and ev.link_recall == 1.0, (ev.gt_uncovered, ev.links_missing)
    assert set(facts_by_claim) == {"CLM-0001", "CLM-0002", "CLM-0003", "CLM-0004", "CLM-0005"}
    entries = json.loads(BASELINES.read_text(encoding="utf-8"))["entries"]
    latest = entries[-1]["result"]
    assert entries[-1]["extractor_sha"] == extractor_hash(), "extractor config changed: re-run evals/runners/eval_facts.py --write-baseline"
    assert c.verbatim_rate >= latest["citation"]["verbatim_rate"] and ev.gt_recall >= latest["ground_truth_recall"]
    assert ev.link_recall >= latest["question_link_recall"]


def test_the_first_run_and_the_template_decision_are_on_record():
    from evals.runners.eval_facts import BASELINES

    e = json.loads(BASELINES.read_text(encoding="utf-8"))["entries"]
    assert e[0]["result"]["question_link_recall"] < 1.0 and e[0]["result"]["question_links_missing"] == ["F-S04-01->Q-SUBLIMIT"]
    assert e[-1]["result"]["question_link_recall"] == 1.0


# ================================================================== pack validation
def raw(name):
    return yaml.safe_load((ROOT / "packs/property_water_damage" / name).read_text(encoding="utf-8"))


@pytest.fixture
def pack_root(tmp_path):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    shutil.copytree(ROOT / "packs", tmp_path / "packs")
    return tmp_path


def write(root, name, data):
    (root / "packs/property_water_damage" / name).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


@pytest.mark.parametrize("mutate, match", [
    (lambda d: d["templates"].pop("fnol.loss_date"), "no template for 'fnol.loss_date'"),
    (lambda d: d["templates"].pop("estimate.line_item"), "no template for 'estimate.line_item'"),
    (lambda d: d["templates"].update({"fnol.nonexistent": {"proposition": "x {value}"}}), "does not match any extractable field"),
    (lambda d: d["templates"]["fnol.loss_date"].update(questions=["Q-NOPE"]), "unknown question 'Q-NOPE'"),
    (lambda d: d["templates"]["fnol.loss_date"].update(proposition="Date {when}"), "unknown placeholder"),
    (lambda d: d["templates"]["fnol.loss_date"].update(proposition="Date {sentence}"), "only for split_sentences"),
    (lambda d: d["templates"]["fnol.loss_description"].update(proposition="Reports {value}"), "must use {sentence}"),
])
def test_a_pack_with_an_incomplete_or_wrong_fact_template_file_fails_at_startup(pack_root, mutate, match):
    d = raw("fact_templates.yaml")
    mutate(d)
    write(pack_root, "fact_templates.yaml", d)
    with pytest.raises(ConfigError, match=match):
        load_settings(pack_root)


def test_missing_template_file_and_duplicate_question_ids_fail_at_startup(pack_root):
    (pack_root / "packs/property_water_damage/fact_templates.yaml").unlink()
    with pytest.raises(ConfigError, match="fact_templates.yaml: file not found"):
        load_settings(pack_root)
    d = raw("question_templates.yaml")
    d["questions"].append(dict(d["questions"][0]))
    write(pack_root, "question_templates.yaml", d)
    (pack_root / "packs/property_water_damage/fact_templates.yaml").write_text(
        (ROOT / "packs/property_water_damage/fact_templates.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate question ids"):
        load_settings(pack_root)


def test_template_models_validate_directly_and_every_question_is_used_or_known():
    assert FactTemplate(proposition="x {value}").questions == []
    problems = check_fact_templates(DT, QuestionTemplates(), FactTemplates(templates={}))
    assert any("no template for" in p for p in problems)
    used = {q for t in TPL.templates.values() for q in t.questions}
    assert used <= {q.id for q in PACK.question_templates.questions} and len(used) == len(PACK.question_templates.questions)


def test_evidence_in_facts_is_whitespace_normalised_text_of_the_page(store):
    ingest_claim(store, NAMES[:2])
    facts, _ = facts_of(store)
    pages = {d.document_id: normalise(" ".join(p.text for p in store.pages(d.document_id))) for d in store.records("CLM-0001")}
    assert facts and all(f.evidence_text == normalise(f.evidence_text) and f.evidence_text in pages[f.source_document_id] for f in facts)
    assert BuiltFacts().facts == []
