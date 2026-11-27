"""T8.3 - the audit trail and the decision pack: append-only and hash-chained, any final decision can be reconstructed from the trail alone."""
import pymupdf
import pytest
from fastapi.testclient import TestClient

from claimtower.api.app import create_app
from claimtower.audit.explain_pack import build_pack, pack_to_pdf, to_html
from claimtower.audit.trail import GENESIS, AuditTrail, TrailRecord
from claimtower.composition import SYNTHETIC, build_commit_service, build_registry
from claimtower.core.config import load_settings
from claimtower.core.schemas import ClaimState
from claimtower.ingestion.store import DocumentStore
from claimtower.orchestration.graph import ClaimRunner
from claimtower.orchestration.overrides import OverrideStore

SETTINGS = load_settings()
MOCKS = ("intake", "gap", "coverage", "action", "audit", "routing")        # the real settlement agent calculates; the rest are the labelled fixtures


def runner(trail, **kw):
    ov = OverrideStore()
    reg = build_registry(SETTINGS, force_mock=MOCKS, overrides=ov)
    return ClaimRunner(reg, commit_service=build_commit_service(SETTINGS), overrides=ov, settings=SETTINGS, trail=trail, **kw)


def finished(r, claim="CLM-0002"):
    r.start(ClaimState(claim_id=claim))
    r.decide(claim, {"decision": "edit", "reviewer_id": "rev-1", "reviewer_role": "adjuster", "reason": "upgrade confirmed", "edits": {"non_covered_items_total": "28000"}})
    r.decide(claim, {"decision": "approve", "reviewer_id": "rev-9", "reviewer_role": "supervisor"})
    return r


# ---------------------------------------------------------------------------------------------------------------- the chain
def test_the_trail_is_hash_chained_and_a_fresh_one_verifies():
    t = AuditTrail()
    a, b = t.append("C1", "event", {"x": 1}), t.append("C2", "event", {"x": 2})
    assert a.prev_hash == GENESIS and b.prev_hash == a.hash and t.head == b.hash and t.verify() == [] and len(t) == 2
    assert [r.claim_id for r in t.records()] == ["C1", "C2"] and t.records("C1", "event")[0].data == {"x": 1}


def test_an_edited_removed_or_reordered_record_is_detected():
    t = AuditTrail()
    for i in range(4):
        t.append("C1", "event", {"i": i})
    recs = t.records()
    edited = [recs[0], TrailRecord(recs[1].seq, recs[1].ts, recs[1].claim_id, recs[1].kind, {"i": 99}, recs[1].prev_hash, recs[1].hash), *recs[2:]]
    assert any("was edited" in p for p in t.verify(edited))
    assert any("removed or inserted" in p or "chain is broken" in p for p in t.verify([recs[0], recs[2], recs[3]]))
    assert t.verify([recs[0], recs[2], recs[1], recs[3]]) != []


def test_the_trail_survives_a_restart_and_a_tampered_file_fails_verification(tmp_path):
    path = tmp_path / "trail.jsonl"
    t = AuditTrail(path)
    t.append("C1", "event", {"a": "one"})
    t.append("C1", "event", {"a": "two"})
    again = AuditTrail(path)
    assert len(again) == 2 and again.head == t.head and again.verify() == []
    again.append("C1", "event", {"a": "three"})
    assert len(AuditTrail(path)) == 3
    path.write_text(path.read_text(encoding="utf-8").replace('"two"', '"2"'), encoding="utf-8")
    assert AuditTrail(path).verify() != []


def test_the_trail_has_no_way_to_change_or_delete_a_record():
    public = {n for n in dir(AuditTrail) if not n.startswith("_")}
    assert not {n for n in public if any(w in n for w in ("delete", "remove", "update", "edit", "clear", "pop"))} and "append" in public


# ---------------------------------------------------------------------------------------------------------------- reconstruction from the trail alone
def test_the_final_state_is_rebuilt_from_the_trail_exactly_after_an_edit_and_an_approval():
    t = AuditTrail()
    r = finished(runner(t))
    live = r.state("CLM-0002").model_dump(mode="json")
    assert t.reconstruct("CLM-0002").model_dump(mode="json") == live and t.verify() == []
    assert live["settlement"]["payable"] == "60000" and live["resolution_status"] == "approved"
    versions = t.versions("CLM-0002")
    assert versions == sorted(versions) and versions[-1] == r.state("CLM-0002").state_version


def test_any_earlier_version_can_be_rebuilt_and_matches_what_the_system_held_then():
    t = AuditTrail()
    r = finished(runner(t))
    for v in t.versions("CLM-0002"):
        assert t.reconstruct("CLM-0002", v).model_dump(mode="json") == r.state_at("CLM-0002", v).model_dump(mode="json"), v


def test_a_reassessment_and_a_second_claim_are_in_the_trail_too_and_kept_apart():
    t = AuditTrail()
    r = runner(t)
    r.start(ClaimState(claim_id="CLM-0002"))
    r.decide("CLM-0002", {"decision": "request_more_evidence", "reviewer_id": "rev-1", "reviewer_role": "senior_adjuster", "reason": "need more"})
    r.reassess("CLM-0002", from_node="gap")
    r.start(ClaimState(claim_id="CLM-0004"))
    for cid in ("CLM-0002", "CLM-0004"):
        assert t.reconstruct(cid).model_dump(mode="json") == r.state(cid).model_dump(mode="json"), cid
    assert {x.claim_id for x in t.records()} == {"CLM-0002", "CLM-0004"}


def test_the_trail_records_the_human_decisions_the_mock_flags_and_the_model_calls():
    class Calls:
        calls = []

    llm = Calls()
    r = runner(AuditTrail(), llm=llm)
    llm.calls.append(type("C", (), {"kind": "structured", "model": "m-1", "prompt_name": "p", "prompt_version": "7", "input_tokens": 3, "output_tokens": 4, "replayed": False})())
    t = r.trail
    r.start(ClaimState(claim_id="CLM-0002"))
    patches = t.records("CLM-0002", "state_patch")
    assert any(p.data["mock"] and p.data["agent_id"] == "intake" for p in patches) and any(p.data["agent_id"] == "settlement" and not p.data["mock"] for p in patches)
    assert patches[0].data["agent_id"] == "supervisor" and t.records("CLM-0002", "claim_created")
    r.decide("CLM-0002", {"decision": "approve", "reviewer_id": "rev-9", "reviewer_role": "supervisor"})
    decisions = [e for e in t.records("CLM-0002", "event") if e.data["kind"] == "human_decision"]
    assert decisions[0].data["data"]["reviewer_id"] == "rev-9" and decisions[0].data["data"]["reviewer_role"] == "supervisor"


def test_a_model_call_made_inside_an_agent_is_written_with_its_model_and_prompt_version():
    class Calls:
        calls = []

    llm = Calls()
    t = AuditTrail()
    r = runner(t, llm=llm)

    class Noisy:
        agent_id, owns = "coverage", frozenset({"coverage_issues"})

        def run(self, state):
            from claimtower.core.state import StatePatch
            llm.calls.append(type("C", (), {"kind": "structured", "model": "gpt-x", "prompt_name": "coverage_reading", "prompt_version": "2", "input_tokens": 10, "output_tokens": 5,
                                            "replayed": True})())
            return StatePatch(agent_id="coverage", updates={"coverage_issues": []}, rationale="noisy")

    from claimtower.orchestration.registry import AgentRegistry
    slots = {s.node: (s.agent, s.status) for s in r.registry}
    slots["coverage"] = (Noisy(), "implemented")
    r2 = ClaimRunner(AgentRegistry(slots), trail=t, llm=llm)
    r2.start(ClaimState(claim_id="CLM-0002"))
    p = next(x for x in t.records("CLM-0002", "state_patch") if x.data["agent_id"] == "coverage")
    assert p.data["model_calls"] == [{"kind": "structured", "model": "gpt-x", "prompt": "coverage_reading", "prompt_version": "2", "input_tokens": 10, "output_tokens": 5, "replayed": True}]


# ---------------------------------------------------------------------------------------------------------------- the decision pack
def test_the_decision_pack_is_built_from_the_trail_alone_and_names_the_decision_the_calculation_and_the_overrides():
    t = AuditTrail()
    finished(runner(t))
    md = build_pack(t, "CLM-0002")
    for needle in ["# Decision pack: claim CLM-0002", "**approve** by rev-9 (supervisor)", "**edit** by rev-1 (adjuster)", "upgrade confirmed", "non_covered_items_total", "payable 60000",
                   "Payable: INR 60,000", "human override by rev-1", "## 5. Policy wording used", "CL-COV-WATER", "## 8. How the assessment was made", "Chain verified",
                   "The system only recommends", "Mock agents took part", "tier **simple**"]:
        assert needle in md, needle


def test_the_pack_says_when_nothing_has_been_decided_and_when_the_chain_is_broken():
    t = AuditTrail()
    r = runner(t)
    r.start(ClaimState(claim_id="CLM-0002"))
    assert "No human decision has been recorded yet." in build_pack(t, "CLM-0002")
    k = next(n for n, x in enumerate(t._records) if x.kind == "event")
    old = t._records[k]
    t._records[k] = TrailRecord(old.seq, old.ts, old.claim_id, old.kind, {**old.data, "message": "edited after the fact"}, old.prev_hash, old.hash)
    assert "CHAIN BROKEN" in build_pack(t, "CLM-0002")


def test_the_pack_is_available_as_a_pdf_with_the_same_content():
    t = AuditTrail()
    finished(runner(t))
    pdf = pack_to_pdf(build_pack(t, "CLM-0002"))
    doc = pymupdf.open(stream=pdf, filetype="pdf")
    text = " ".join(p.get_text() for p in doc)
    assert pdf[:4] == b"%PDF" and doc.page_count >= 2 and "Decision pack" in text and "rev-9" in text and "60,000" in text
    assert "<table" in to_html("| a | b |\n|---|---|\n| 1 | 2 |\n") and "<b>x</b>" in to_html("**x**")


def test_an_unknown_claim_has_no_pack():
    with pytest.raises(KeyError):
        build_pack(AuditTrail(), "CLM-NOPE")


# ---------------------------------------------------------------------------------------------------------------- the API
def client(tmp_path, trail):
    store = DocumentStore(tmp_path / "s")
    r = finished(runner(trail))
    return TestClient(create_app(r, store, synthetic_claims=SYNTHETIC / "claims"))


def test_the_api_serves_the_trail_and_the_decision_pack(tmp_path):
    t = AuditTrail()
    c = client(tmp_path, t)
    tr = c.get("/claims/CLM-0002/trail").json()
    assert tr["chain_ok"] and tr["problems"] == [] and tr["head"] == t.head and tr["records"][0]["kind"] == "claim_created" and any("human_decision" in x["summary"] for x in tr["records"])
    md = c.get("/claims/CLM-0002/decision-pack")
    assert md.status_code == 200 and md.headers["content-type"].startswith("text/markdown") and "**approve** by rev-9" in md.text
    pdf = c.get("/claims/CLM-0002/decision-pack", params={"format": "pdf"})
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf" and pdf.content[:4] == b"%PDF"
    assert c.get("/claims/CLM-0002/decision-pack", params={"format": "docx"}).status_code == 422 and c.get("/claims/CLM-NOPE/trail").status_code == 404


def test_without_a_trail_the_endpoints_say_so(tmp_path):
    store = DocumentStore(tmp_path / "s")
    r = ClaimRunner(build_registry(SETTINGS, force_mock=("intake", "gap", "coverage", "settlement", "action", "audit", "routing")))
    r.start(ClaimState(claim_id="CLM-0002"))
    c = TestClient(create_app(r, store, synthetic_claims=SYNTHETIC / "claims"))
    assert c.get("/claims/CLM-0002/trail").status_code == 404 and "not enabled" in c.get("/claims/CLM-0002/decision-pack").text
