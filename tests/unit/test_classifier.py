"""T2.2 document type classifier: justified answers, explicit abstention, store integration, optional LLM tie-breaker."""
import io
import json
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path

import pymupdf
import pytest
import yaml
from PIL import Image
from pydantic import ValidationError

from claimtower.core.config import DocumentTypes, load_settings
from claimtower.core.llm import LLMClient, LLMResponse
from claimtower.ingestion.classifier import (
    HybridClassifier, LLMClassifier, RuleClassifier, image_page_metrics, looks_like_page,
)
from claimtower.ingestion.loaders import load, normalise
from claimtower.ingestion.store import DocumentStore, StoreError

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data/synthetic"
SETTINGS = load_settings()
PACK = SETTINGS.pack
RULES = RuleClassifier(PACK.document_types)
HYBRID = HybridClassifier(RULES)


@dataclass
class P:
    number: int
    text: str
    source: str = "text_layer"
    needs_ocr: bool = False


def run(text, kind="text", **kw):
    return RULES.classify(kind=kind, pages=[P(1, text)], **kw)


def classify_file(path, clf=RULES):
    data = Path(path).read_bytes()
    d = load(data, Path(path).name)
    return clf.classify(kind=d.kind, pages=d.pages, original=lambda: data)


# ------------------------------------------------------------------ answers are justified
@pytest.mark.parametrize("path, expected", [
    ("claims/CLM-0001/fnol.pdf", "fnol"), ("claims/CLM-0001/plumber_report.pdf", "plumber_report"),
    ("claims/CLM-0002/surveyor_report.pdf", "surveyor_report"), ("claims/CLM-0001/estimate.pdf", "estimate"),
    ("claims/CLM-0005/policy_schedule.pdf", "policy_schedule"), ("policies/BASE-HOME-2.pdf", "policy_wording"),
    ("policies/END-03.docx", "endorsement"), ("templates/invoice_sample.pdf", "invoice"), ("templates/email_sample.pdf", "email"),
])
def test_corpus_documents_are_classified_with_a_verbatim_evidence_snippet(path, expected):
    r = classify_file(SYN / path)
    assert r.type == expected and not r.needs_human and r.code == "ok" and 0 < r.confidence <= 0.99
    page_text = " ".join(p.text for p in load((SYN / path).read_bytes(), path).pages)
    assert normalise(r.evidence_snippet) in normalise(page_text), "the snippet must be quoted from the document"
    assert r.page == 1 and r.evidence_snippet in r.reason and r.scores[expected] == max(r.scores.values())


def test_the_answer_names_the_runner_up_and_scores_every_text_type():
    r = classify_file(SYN / "claims/CLM-0001/estimate.pdf")
    assert r.runner_up != "estimate" and set(r.scores) >= {"fnol", "invoice", "estimate", "policy_wording"}


def test_headings_are_short_lines_and_label_value_lines_are_fields_not_headings():
    # 'Endorsements attached: ...' is a schedule FIELD that mentions endorsement; it must not make the page an endorsement
    r = run("POLICY SCHEDULE\nPolicy number: POL-1\nEndorsements attached: END-01 Water Damage Sublimit Endorsement\nPeriod of insurance: 2026\n")
    assert r.type == "policy_schedule"
    long_line = "This is a long narrative sentence about an estimate of repair costs that is clearly not a heading at all " * 2
    assert run(long_line + "\nTotal: INR 5\nScope: x\nLabour: y\n").abstained


def test_a_keyword_that_appears_only_in_a_label_value_line_is_not_heading_evidence():
    """No heading, just fields: 'Invoice number: 12' must not count as a heading that says 'invoice'."""
    fields = "Invoice number: 12\nVendor: Acme Repairs\nTotal: INR 5,000\nPayment terms: 7 days\n"
    r = run("Notes\n" + fields)
    assert (r.type, r.code) == ("unknown", "no_heading")
    assert run("INVOICE\n" + fields).type == "invoice"


# ------------------------------------------------------------------ abstention is explicit, coded and never a guess
def test_scan_awaiting_ocr_is_unknown_with_a_clear_reason():
    r = classify_file(SYN / "claims/CLM-0002/surveyor_report_scanned.pdf")
    assert (r.type, r.code, r.needs_human, r.confidence, r.evidence_snippet) == ("unknown", "needs_ocr", True, 0.0, None)
    assert "OCR" in r.reason


def test_each_abstention_path_has_its_own_code():
    assert run("", ).code == "no_text"
    assert RULES.classify(kind="pdf", pages=[P(1, "", "none", False)]).code == "no_text"
    assert RULES.classify(kind="pdf", pages=[], load_error="CorruptDocument: empty file").code == "unreadable"
    assert run("The weather was pleasant and we walked along the river for an hour.\n").code == "low_score"
    assert run("Survey Report and Estimate\nSite visit: 17 October\nMoisture readings were high.\nScope: floors. Total: INR 52,000\n").code == "ambiguous"
    assert run("Terms and Conditions of Membership\nDefinitions\nWhat is covered by your membership\nWe cover the gym.\nWhat is not covered\n"
               "We do not cover personal training.\nConditions\nExclusions apply.\n").code == "no_heading"
    assert run("Estimate your reading time\nMost adults read about 250 words a minute.\nDivide the word count by 250.\n").code == "uncorroborated"
    for r in (run(""), run("hello world filler text, nothing relevant at all here")):
        assert r.type == "unknown" and r.confidence == 0.0 and r.evidence_snippet is None


def test_vocabulary_alone_never_decides_and_a_heading_alone_never_decides():
    phrases_only = "Claimant: A\nPolicy number: P1\nDate of loss: 1 May\nDescription: a leak\nEstimated loss: INR 5\n"
    assert run(phrases_only).code == "no_heading"
    heading_only = "FIRST NOTICE OF LOSS\nSee reverse side for instructions.\nThank you.\n"
    assert run(heading_only).code == "uncorroborated"
    assert run("FIRST NOTICE OF LOSS\n" + phrases_only).type == "fnol"


def test_classifier_never_outputs_other():
    assert "other" not in RULES.text_types()
    for text in ("a menu\nsoup\nbread\n" * 3, "random words about nothing in particular " * 5):
        assert run(text).type != "other"


def test_ocr_text_confidence_is_capped():
    text = "FIRST NOTICE OF LOSS\nClaimant: A\nPolicy number: P1\nDate of loss: 1 May\nDescription: leak\n"
    plain = RULES.classify(kind="pdf", pages=[P(1, text)])
    ocr = RULES.classify(kind="pdf", pages=[P(1, text, source="ocr")])
    assert plain.type == ocr.type == "fnol" and plain.confidence > ocr.confidence == PACK.document_types.classifier.ocr_confidence_cap


# ------------------------------------------------------------------ containers and images
def test_email_is_recognised_from_headers_even_when_the_body_mentions_other_documents():
    m = EmailMessage()
    m["From"], m["To"], m["Subject"] = "a@x.example", "b@y.example", "Fwd: plumber report attached"
    m.set_content("FIRST NOTICE OF LOSS\nInvoice number 5\nPlease see the plumber's report.")
    d = load(bytes(m), "m.eml")
    r = RULES.classify(kind=d.kind, pages=d.pages)
    assert r.type == "email" and r.method == "container" and r.evidence_snippet == "Subject: Fwd: plumber report attached"


def test_text_file_with_a_pasted_header_block_is_an_email():
    r = run("From: a@x.example\nTo: b@y.example\nDate: 18 October 2026\nSubject: Missing endorsement\n\nPlease resend the endorsement.\n")
    assert r.type == "email" and r.method == "rules"


def test_photos_are_told_apart_from_scanned_pages_by_the_image_itself():
    photos = sorted([*SYN.glob("claims/*/photos/*.jpg"), *SYN.glob("photos/quality_set/*.jpg")])
    assert len(photos) >= 12
    for p in photos:
        r = classify_file(p)
        assert (r.type, r.method) == ("photo", "visual"), p.name
    scan_pdf = pymupdf.open(SYN / "claims/CLM-0001/fnol.pdf")
    for dpi in (80, 110, 150, 200):
        for fmt in ("PNG", "JPEG"):
            buf = io.BytesIO()
            Image.open(io.BytesIO(scan_pdf[0].get_pixmap(dpi=dpi).tobytes("png"))).convert("RGB").save(buf, fmt)
            d = load(buf.getvalue(), "scan")
            r = RULES.classify(kind=d.kind, pages=d.pages, original=lambda b=buf.getvalue(): b)
            assert (r.type, r.code) == ("unknown", "needs_ocr"), (dpi, fmt)


def test_image_metrics_separate_pages_from_photos_and_the_original_is_needed():
    page = image_page_metrics(_page_png())
    photo = image_page_metrics((SYN / "claims/CLM-0001/photos/kitchen_wall_stain.jpg").read_bytes())
    assert looks_like_page(page) and not looks_like_page(photo)
    assert RULES.classify(kind="image", pages=[P(1, "", "none", True)]).code == "needs_ocr"       # cannot inspect without the bytes


def _page_png():
    return pymupdf.open(SYN / "claims/CLM-0001/fnol.pdf")[0].get_pixmap(dpi=100).tobytes("png")


# ------------------------------------------------------------------ store integration
@pytest.fixture
def store(tmp_path):
    return DocumentStore(tmp_path / "s")


def test_store_persists_the_classification_with_its_justification(tmp_path):
    st = DocumentStore(tmp_path / "s")
    data = (SYN / "claims/CLM-0001/estimate.pdf").read_bytes()
    rec = st.ingest("CLM-T", "estimate.pdf", data, document_id="DOC-1", classifier=HYBRID)
    assert (rec.detected_type, rec.type_evidence_snippet) == ("estimate", "REPAIR ESTIMATE - SYNTHETIC DOCUMENT") and rec.type_confidence > 0.9
    s = st.stored("DOC-1")
    assert s.type_method == "rules" and s.type_code == "ok" and s.type_page == 1 and "REPAIR ESTIMATE" in s.type_reason
    again = DocumentStore(tmp_path / "s").stored("DOC-1")                                   # survives a restart
    assert (again.detected_type, again.type_scores["estimate"]) == ("estimate", s.type_scores["estimate"])
    assert st.original("DOC-1") == data                                                      # classification never touches the original


def test_unknown_documents_are_listed_for_a_human_with_the_reason(store):
    store.ingest("CLM-T", "ok.pdf", (SYN / "claims/CLM-0001/fnol.pdf").read_bytes(), classifier=HYBRID)
    store.ingest("CLM-T", "scan.pdf", (SYN / "claims/CLM-0002/surveyor_report_scanned.pdf").read_bytes(), document_id="DOC-SCAN", classifier=HYBRID)
    store.ingest("CLM-T", "gym.txt", b"Terms and Conditions of Membership\nDefinitions\nWhat is covered by your membership\nWe cover the gym.\n"
                 b"What is not covered\nWe do not cover personal training.\nConditions\nExclusions apply.\n", document_id="DOC-GYM", classifier=HYBRID)
    store.ingest("CLM-T", "broken.pdf", b"%PDF-1.4 junk", document_id="DOC-BAD", classifier=HYBRID)
    held = {d.document_id: d.type_code for d in store.needs_human("CLM-T")}
    assert held == {"DOC-SCAN": "needs_ocr", "DOC-GYM": "no_heading", "DOC-BAD": "unreadable"}
    assert store.needs_human("CLM-OTHER") == []


def test_reclassifying_after_ocr_resolves_a_scan_and_caps_confidence(store):
    class Ocr:
        name = "stub"

        def recognise(self, image, mime):
            return "SURVEYOR'S REPORT\nSite visit: 5 May 2026\nMoisture was confined to the kitchen floor.\nAssessed repair cost: INR 98,000\n"

    rec = store.ingest("CLM-T", "scan.pdf", (SYN / "claims/CLM-0002/surveyor_report_scanned.pdf").read_bytes(), document_id="DOC-S", classifier=HYBRID)
    assert rec.detected_type == "unknown"
    store.run_ocr("DOC-S", Ocr())
    done = store.classify("DOC-S", HYBRID)
    assert done.detected_type == "surveyor_report" and done.type_confidence == PACK.document_types.classifier.ocr_confidence_cap
    assert store.needs_human("CLM-T") == []


def test_a_human_decision_survives_reclassification_and_is_validated(store):
    store.ingest("CLM-T", "n.txt", b"Page intentionally left blank, nothing else to see here at all.\n", document_id="DOC-H", classifier=HYBRID)
    allowed = list(PACK.document_types.types)
    with pytest.raises(StoreError, match="needs 'by'"):
        store.set_type("DOC-H", "other", by=" ", allowed=allowed)
    with pytest.raises(StoreError, match="not an allowed"):
        store.set_type("DOC-H", "unknown", by="adjuster", allowed=allowed)
    with pytest.raises(StoreError, match="not an allowed"):
        store.set_type("DOC-H", "made_up_type", by="adjuster", allowed=allowed)
    rec = store.set_type("DOC-H", "other", by="adjuster", allowed=allowed)
    assert (rec.detected_type, rec.type_confidence) == ("other", 1.0)
    s = store.stored("DOC-H")
    assert (s.type_method, s.type_by, s.type_reason) == ("human", "adjuster", "assigned by adjuster")
    assert store.classify("DOC-H", HYBRID).detected_type == "other"                          # a re-run never overwrites a human
    assert store.needs_human("CLM-T") == []


def test_email_attachments_are_classified_as_their_own_documents(store):
    m = EmailMessage()
    m["From"], m["To"], m["Subject"] = "p@x.example", "c@y.example", "Report"
    m.set_content("Attached.")
    m.add_attachment((SYN / "claims/CLM-0001/plumber_report.pdf").read_bytes(), maintype="application", subtype="pdf", filename="r.pdf")
    rec = store.ingest("CLM-T", "mail.eml", bytes(m), document_id="DOC-M", classifier=HYBRID)
    assert rec.detected_type == "email"
    assert [k.detected_type for k in store.children("DOC-M")] == ["plumber_report"]


# ------------------------------------------------------------------ optional LLM tie-breaker: checked, not trusted
class FakeLLM:
    name = "fake"

    def __init__(self, reply):
        self.reply, self.requests = reply, []

    def complete(self, req):
        self.requests.append(req)
        if isinstance(self.reply, Exception):
            raise self.reply
        return LLMResponse(text=json.dumps(self.reply), model="m", input_tokens=1, output_tokens=1)


def hybrid(reply):
    provider = FakeLLM(reply)
    llm = LLMClient(provider, text_model="t", vision_model="v")
    return HybridClassifier(RULES, LLMClassifier(llm, RULES, PACK)), provider


UNTITLED = "Site note\nVisited the property. Found the compression joint on the supply pipe corroded and leaking. Replaced the joint, supply restored.\n"


def test_the_model_decides_a_text_document_and_its_evidence_is_verified():
    clf, provider = hybrid({"document_type": "plumber_report", "confidence": 0.9, "evidence_snippet": "Found the compression joint on the supply pipe"})
    r = clf.classify(kind="text", pages=[P(1, UNTITLED)])
    assert (r.type, r.method, r.code) == ("plumber_report", "llm", "ok") and r.confidence == 0.8       # capped at llm_confidence_cap
    assert r.evidence_snippet == "Found the compression joint on the supply pipe" and r.page == 1 and "the model chose" in r.reason
    assert len(provider.requests) == 1
    assert "plumber_report" in provider.requests[0].messages[-1]["content"]                      # the allowed types are in the prompt


@pytest.mark.parametrize("reply, why", [
    ({"document_type": "plumber_report", "confidence": 0.95, "evidence_snippet": "The pipe was clearly burst by frost"}, "not verbatim"),
    ({"document_type": "plumber_report", "confidence": 0.95, "evidence_snippet": "short"}, "not verbatim"),
    ({"document_type": "invoice_from_mars", "confidence": 0.95, "evidence_snippet": "Found the compression joint"}, "not allowed"),
    ({"document_type": "other", "confidence": 0.95, "evidence_snippet": "Found the compression joint"}, "not allowed"),
    ({"document_type": "unknown", "confidence": 0.95, "evidence_snippet": ""}, "answered unknown"),
    ({"document_type": "plumber_report", "confidence": 0.40, "evidence_snippet": "Found the compression joint"}, "below"),
])
def test_llm_verdicts_that_do_not_check_out_are_rejected_and_the_document_stays_unknown(reply, why):
    clf, _ = hybrid(reply)
    r = clf.classify(kind="text", pages=[P(1, UNTITLED)])
    assert (r.type, r.code, r.needs_human) == ("unknown", "llm_rejected", True) and why in r.reason


FNOL_TEXT = "FIRST NOTICE OF LOSS\nClaimant: A\nPolicy number: P\nDate of loss: 1 May\n"


def test_the_model_is_not_asked_about_scans_and_unreadable_files_and_the_rules_never_overrule_it():
    clf, provider = hybrid({"document_type": "invoice", "confidence": 0.99, "evidence_snippet": "xxxxxxxxxx"})
    assert clf.classify(kind="pdf", pages=[P(1, "", "none", True)]).code == "needs_ocr"
    assert clf.classify(kind="pdf", pages=[], load_error="CorruptDocument: x").code == "unreadable"
    assert provider.requests == []
    r = clf.classify(kind="text", pages=[P(1, FNOL_TEXT)])                  # the rules would say fnol; the model's (rejected) answer stands: unknown, for a person
    assert (r.type, r.code) == ("unknown", "llm_rejected") and len(provider.requests) == 1
    assert "fnol" not in provider.requests[0].messages[-1]["content"].split("Document text")[1].lower().replace("first notice of loss", "")    # no rule answer is in the prompt


def test_a_photo_sheet_that_mentions_an_fnol_reference_is_a_photo_not_the_fnol():
    sheet = ("Photographic Evidence Submitted by Insured\nSubmission FNOL-20261014-0071 - Mr Rajesh Menon\nPhoto 1 (file photo1.jpg)\n"
             "Caption given by submitter\nKitchen wall and base cabinet after the burst\n")
    clf, _ = hybrid({"document_type": "photo", "confidence": 0.95, "evidence_snippet": "Photographic Evidence Submitted by Insured"})
    r = clf.classify(kind="pdf", pages=[P(1, sheet)])
    assert (r.type, r.method) == ("photo", "llm")


def test_an_unavailable_llm_falls_back_to_the_rules_not_to_a_crash():
    clf, _ = hybrid(TimeoutError("model unavailable"))
    assert clf.classify(kind="text", pages=[P(1, UNTITLED)]).type == "unknown"                         # the rules abstain too: a person decides
    r = clf.classify(kind="text", pages=[P(1, FNOL_TEXT)])
    assert (r.type, r.method) == ("fnol", "rules")                                                      # model down: the rules answer
    to_be_set = HybridClassifier(RULES, LLMClassifier(LLMClient(FakeLLM({}), text_model="TBD", vision_model="TBD"), RULES, PACK))
    assert to_be_set.classify(kind="text", pages=[P(1, FNOL_TEXT)]).method == "rules"                   # models not configured yet


def test_a_quote_padded_over_several_lines_falls_back_to_its_longest_verbatim_line_but_an_invented_one_is_rejected():
    text = "Plumbing Failure Inspection Report\nReport number SSP/1\nDate of attendance 14 October 2026\n"
    padded = "Plumbing Failure Inspection Report\nReport number SSP/1\nWords the document never said at all"
    clf, _ = hybrid({"document_type": "plumber_report", "confidence": 0.95, "evidence_snippet": padded})
    r = clf.classify(kind="pdf", pages=[P(1, text)])
    assert (r.type, r.evidence_snippet) == ("plumber_report", "Plumbing Failure Inspection Report")
    clf, _ = hybrid({"document_type": "plumber_report", "confidence": 0.95, "evidence_snippet": "Nothing\nthe document\nsaid anywhere"})
    assert clf.classify(kind="pdf", pages=[P(1, text)]).code == "llm_rejected"


# ------------------------------------------------------------------ config validation (a bad pack fails at startup)
def raw_types():
    return yaml.safe_load((ROOT / "packs/property_water_damage/document_types.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("mutate, match", [
    (lambda d: d["types"]["fnol"]["title_patterns"].append("(unclosed"), "bad pattern"),
    (lambda d: d["types"]["email"]["structure_patterns"].append("[bad"), "bad pattern"),
    (lambda d: d["classifier"].update(title_weight=0.9), "must equal 1"),
    (lambda d: d["classifier"].update(min_score=1.5), "min_score"),
    (lambda d: d["types"].pop("other"), "must include 'other'"),
    (lambda d: d["types"].update(unknown={"title": "x"}), "abstention value"),
    (lambda d: d["classifier"].pop("answer_without_heading"), "answer_without_heading"),
])
def test_invalid_document_type_config_is_rejected(mutate, match):
    d = raw_types()
    mutate(d)
    with pytest.raises(ValidationError, match=match):
        DocumentTypes.model_validate(d)


# ------------------------------------------------------------------ eval gate (targets proposed in evals/baselines/classification.json)
def test_classifier_meets_the_proposed_targets_and_does_not_regress_the_recorded_baseline():
    from evals.metrics.classification import all_cases, evaluate
    from evals.runners.eval_classifier import BASELINES, signals_hash

    reports = evaluate(HYBRID, all_cases())
    gated = [r for name, r in reports.items() if name not in ("adversarial", "llm1")]   # llm1: tracked, known rule limit l05 (blank form)
    for r in gated:                                                              # never a confident guess; never fails to abstain
        assert r.wrong_answers == [], [(o.case.id, o.got) for o in r.wrong_answers]
        assert r.answered_precision == 1.0 and r.abstain_recall in (None, 1.0), r.split
    assert reports["corpus"].accuracy == 1.0 and reports["dev"].accuracy == 1.0
    assert reports["fresh2"].accuracy >= 0.90
    pooled = [o for name, r in reports.items() if name in ("dev", "holdout", "fresh", "fresh2") for o in r.outcomes]
    assert sum(o.correct for o in pooled) / len(pooled) >= 0.85                    # pooled accuracy on non-corpus labelled documents

    latest = json.loads(BASELINES.read_text(encoding="utf-8"))["entries"][-1]
    assert latest["signals_sha"] == signals_hash(), "document_types.yaml changed: re-run evals/runners/eval_classifier.py --write-baseline"
    for name, r in reports.items():
        old = latest["splits"][name]
        assert r.accuracy >= old["accuracy"] - 1e-9, f"{name}: accuracy regressed"
        assert len(r.wrong_answers) <= len(old["wrong_answers"]), f"{name}: more wrong answers than the baseline"


def test_adversarial_lookalikes_are_tracked_and_the_limit_is_documented():
    """Keyword-rich look-alikes DO fool the rule classifier (known limit). They are tracked, not gated; guard against getting worse."""
    from evals.metrics.classification import all_cases, evaluate

    r = evaluate(HYBRID, all_cases())["adversarial"]
    assert len(r.wrong_answers) <= 4 and r.n == 4
