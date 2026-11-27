"""The model reads which policy document this is (no fixed header pattern needed); its answer is checked, and the fixed patterns are only the fallback."""
import shutil

import pymupdf
import pytest
from fastapi.testclient import TestClient

from claimtower.api.app import create_app
from claimtower.composition import SYNTHETIC, build_registry
from claimtower.core.config import load_settings
from claimtower.ingestion.store import DocumentStore
from claimtower.orchestration.graph import ClaimRunner
from claimtower.policy.decompose import PolicyDecomposer
from claimtower.policy.llm_identity import IdentityAnswer, LLMIdentityReader, safe_id
from claimtower.policy.manager import IdentityReader, PolicyDocumentError, PolicyDocumentManager

SETTINGS = load_settings()
POLICIES = SYNTHETIC / "policies"
PACK = SETTINGS.pack

ODD = ["Maple Mutual Insurance Company", "HOME SHIELD PLUS - Terms and Conditions", "Document ref HSP/2026/B  (edition 4)", "Policy number MM-55-0098213",
       "Period of insurance: 1 March 2026 to 28 February 2027", "1.1 Buildings", "Buildings means the dwelling and its fixed pipes.", "2.1 Escape of water",
       "We cover damage caused by water escaping from a fixed pipe."]


def make_pdf(lines: list[str]) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 60), "\n".join(lines), fontsize=10)
    return doc.tobytes()


class FakeLLM:
    def __init__(self, answer=None, fail=None):
        self.answer, self.fail, self.calls = answer, fail, 0

    def structured(self, cls, prompt, **kw):
        self.calls += 1
        if self.fail:
            raise self.fail
        return self.answer


GOOD = IdentityAnswer(document_id="HSP/2026/B", evidence_quote="Document ref HSP/2026/B (edition 4)", version="edition 4", title="HOME SHIELD PLUS - Terms and Conditions",
                      kind="policy_wording", policy_number="MM-55-0098213", period="1 March 2026 to 28 February 2027", confidence=0.93)


def mgr(tmp_path, llm, builtin=None):
    return PolicyDocumentManager(PolicyDecomposer(PACK.policy_structure), IdentityReader.load(PACK, llm), builtin, tmp_path / "up")


# ---------------------------------------------------------------------------------------------------------------- the checks on the model's answer
TEXT = "Maple Mutual HOME SHIELD PLUS Document ref HSP/2026/B (edition 4) Policy number MM-55-0098213 Period of insurance: 1 March 2026 to 28 February 2027"


def test_a_grounded_answer_is_accepted_the_id_is_made_safe_and_reported_values_are_verbatim():
    r = LLMIdentityReader(FakeLLM(), PACK)
    doc_id, version, quote, kind, details = r.check(GOOD, TEXT)
    assert (doc_id, version, kind) == ("HSP-2026-B", "edition 4", "policy_wording") and quote == "Document ref HSP/2026/B (edition 4)"
    assert details["policy_number"] == "MM-55-0098213" and details["id_as_written"] == "HSP/2026/B" and details["period"].startswith("1 March 2026")
    invented = GOOD.model_copy(update={"policy_number": "XX-99", "title": "A title not in the text"})
    assert "policy_number" not in r.check(invented, TEXT)[4] and "title" not in r.check(invented, TEXT)[4]


@pytest.mark.parametrize("change,why", [({"document_id": ""}, "no identifier"), ({"document_id": "HSP-9999"}, "does not appear in the text of the document"), ({"confidence": 0.3}, "not sure enough"),
                                        ({"document_id": "12"}, "does not appear in the text"), ({"document_id": "!!", "evidence_quote": "!!"}, "does not appear in the text")])
def test_an_ungrounded_answer_is_refused_with_the_reason(change, why):
    r = LLMIdentityReader(FakeLLM(), PACK)
    assert r.check(GOOD.model_copy(update=change), TEXT) is None and why in r.last_problem
    assert safe_id("HP / 2025 / A 12") == "HP-2025-A-12" and safe_id("  --x--  ") == "x"


def test_a_quote_that_does_not_hold_is_replaced_by_the_documents_own_words_around_the_identifier():
    r = LLMIdentityReader(FakeLLM(), PACK)
    doc_id, _, quote, _, _ = r.check(GOOD.model_copy(update={"evidence_quote": "words the document never said"}), TEXT)
    assert doc_id == "HSP-2026-B" and "HSP/2026/B" in quote and quote in TEXT


def test_a_model_that_is_down_is_reported_not_fatal():
    r = LLMIdentityReader(FakeLLM(fail=ConnectionError("down")), PACK)

    class P:
        number, text = 1, TEXT
    assert r.read([P()]) is None and "could not be reached" in r.last_problem


# ---------------------------------------------------------------------------------------------------------------- through the manager
def test_a_document_no_pattern_can_read_is_identified_by_the_model(tmp_path):
    llm = FakeLLM(GOOD)
    m = mgr(tmp_path, llm)
    res = m.ingest(make_pdf(ODD), "random file.pdf", uploaded_by="t")
    e = res.entry
    assert res.outcome == "added" and (e.doc_id, e.version, e.id_from) == ("HSP-2026-B", "edition 4", "model") and e.id_evidence == "Document ref HSP/2026/B (edition 4)"
    assert e.details["policy_number"] == "MM-55-0098213" and e.clauses >= 2 and llm.calls == 1 and m.library.get("HSP-2026-B") is not None


def test_the_patterns_are_the_fallback_when_the_model_refuses_or_is_down(tmp_path):
    for llm in (FakeLLM(IdentityAnswer(document_id="", note="not sure")), FakeLLM(fail=ConnectionError("down"))):
        e = mgr(tmp_path / str(id(llm)), llm).ingest((POLICIES / "END-01.pdf").read_bytes(), "x.pdf").entry
        assert (e.doc_id, e.id_from) == ("END-01", "document")


def test_when_both_fail_the_error_says_what_the_model_said(tmp_path):
    m = mgr(tmp_path, FakeLLM(IdentityAnswer(document_id="", note="the document names no reference")))
    with pytest.raises(PolicyDocumentError, match="The model said: the model found no identifier .*names no reference"):
        m.ingest(make_pdf(["Some paper", "1.1 Buildings", "Buildings means the dwelling."]), "x.pdf")
    assert m.ingest(make_pdf(["Some paper", "1.1 Buildings", "Buildings means the dwelling."]), "x.pdf", document_id="MY-DOC-1").entry.id_from == "given"


def test_without_a_model_the_patterns_alone_decide(tmp_path):
    m = mgr(tmp_path, None)
    with pytest.raises(PolicyDocumentError, match="no document id was found"):
        m.ingest(make_pdf(ODD), "x.pdf")
    assert m.ingest((POLICIES / "END-02.pdf").read_bytes(), "x.pdf").entry.doc_id == "END-02"


def test_a_reload_keeps_the_model_read_identity_and_does_not_ask_the_model_again(tmp_path):
    llm = FakeLLM(GOOD)
    mgr(tmp_path, llm).ingest(make_pdf(ODD), "random file.pdf")
    again = FakeLLM(fail=AssertionError("the model must not be asked on a reload"))
    m2 = mgr(tmp_path, again)
    e = next(x for x in m2.entries() if x.doc_id == "HSP-2026-B")
    assert again.calls == 0 and e.id_from == "model" and e.details["policy_number"] == "MM-55-0098213" and e.id_evidence.startswith("Document ref") and e.version == "edition 4"


def test_the_built_in_folder_is_read_without_the_model(tmp_path):
    builtin = tmp_path / "b"
    builtin.mkdir()
    shutil.copy(POLICIES / "BASE-HOME-1.pdf", builtin / "weird name.pdf")
    llm = FakeLLM(fail=AssertionError("no model at start-up"))
    m = mgr(tmp_path, llm, builtin)
    assert llm.calls == 0 and [e.doc_id for e in m.entries()] == ["BASE-HOME-1"]


def test_the_api_shows_how_the_id_was_read_and_what_else_the_document_said(tmp_path):
    m = mgr(tmp_path, FakeLLM(GOOD))
    store = DocumentStore(tmp_path / "store")
    runner = ClaimRunner(build_registry(SETTINGS, force_mock=("intake", "gap", "coverage", "settlement", "action", "audit", "routing")))
    c = TestClient(create_app(runner, store, synthetic_claims=SYNTHETIC / "claims", policies=m))
    r = c.post("/policies/documents", files={"file": ("Some name (2).pdf", make_pdf(ODD), "application/pdf")})
    d = r.json()["document"]
    assert r.status_code == 201 and d["doc_id"] == "HSP-2026-B" and d["id_from"] == "model" and d["id_evidence"] and d["details"]["period"].startswith("1 March 2026")
