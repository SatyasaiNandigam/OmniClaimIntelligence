"""T7.1 - the API contract: every endpoint, its errors, and the guarantees the UI relies on (mock flag on every response, the gate only opened by a named, authorised reviewer)."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from claimtower.api.app import create_app
from claimtower.composition import SYNTHETIC, build_registry, policy_library_from_pdfs
from claimtower.core.config import load_settings
from claimtower.ingestion.store import DocumentStore
from claimtower.orchestration.graph import ClaimRunner

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()
MOCKS = "intake,coverage,settlement,action,audit,routing"
PDF = (SYNTHETIC / "claims" / "CLM-0001" / "plumber_report.pdf").read_bytes()


@pytest.fixture(scope="module")
def library():
    return policy_library_from_pdfs(SETTINGS)


def make_client(tmp_path, library, **kw):
    store = DocumentStore(tmp_path / "store")
    runner = ClaimRunner(build_registry(SETTINGS, document_store=store, policy_library=library))
    return TestClient(create_app(runner, store, synthetic_claims=SYNTHETIC / "claims", **kw)), runner


@pytest.fixture
def client(tmp_path, library):
    return make_client(tmp_path, library)[0]


def create(client, claim="CLM-0001", **body):
    return client.post("/claims", json={"claim_id": claim, "use_synthetic_documents": True, **body})


def decision(role="senior_adjuster", kind="approve", reason="", reviewer="rev-1"):
    return {"decision": kind, "reviewer_id": reviewer, "reviewer_role": role, "reason": reason}


# ------------------------------------------------------------------------------------------------ service
def test_health_says_what_is_real_and_what_is_a_mock(client):
    r = client.get("/health")
    assert r.status_code == 200
    h = r.json()
    assert h["status"] == "ok" and h["prototype"] and h["synthetic_data_only"] and h["claims"] == 0
    assert h["mock_agents"] == MOCKS.split(",") and h["implemented_agents"] == ["policy", "evidence", "gap"]


def test_every_response_carries_the_mock_flag_even_an_error(client):
    assert client.get("/health").headers["x-mock-agents"] == MOCKS
    missing = client.get("/claims/NOPE-1")
    assert missing.status_code == 404 and missing.headers["x-mock-agents"] == MOCKS


def test_the_roster_lists_each_agent_with_its_status_and_task(client):
    rows = {r["node"]: r for r in client.get("/agents").json()}
    assert rows["gap"]["status"] == "implemented" and rows["coverage"]["status"] == "YET TO IMPLEMENT (mock)" and rows["coverage"]["task"] == "T5.1"


# ------------------------------------------------------------------------------------------------ claims
def test_creating_a_claim_runs_it_to_the_human_gate(client):
    r = create(client)
    assert r.status_code == 201
    v = r.json()
    assert v["claim_id"] == "CLM-0001" and v["state_version"] == 10 and v["state"]["claim_id"] == "CLM-0001" and v["state"]["state_version"] == 10
    s = v["status"]
    assert s["phase"] == "awaiting_human_decision" and s["paused_at"] == "human_gate" and s["gate"]["tier"] == "complex"
    assert s["gate"]["allowed_decisions"] == ["approve", "reject", "request_more_evidence"] and set(s["mock_nodes_run"]) == set(MOCKS.split(","))
    assert v["state"]["facts"] and v["state"]["policy_snapshot"]["mock"] is True and v["state"]["correlation_id"] == "corr-CLM-0001"


def test_a_claim_without_an_id_gets_one_and_a_supplied_header_reaches_the_policy_agent(client):
    r = client.post("/claims", json={"policy_ref": "POL-1001", "loss_date": "2026-03-14", "loss_type": "water_damage", "loss_description": "Kitchen pipe burst"})
    assert r.status_code == 201
    v = r.json()
    assert v["claim_id"].startswith("CLM-") and len(v["claim_id"]) == 12
    assert v["state"]["claim_header"]["policy_ref"] == "POL-1001" and v["state"]["claim_header"]["claim_id"] == v["claim_id"]      # the mock intake kept it
    assert v["state"]["policy_snapshot"]["pas_status"] == "in_force" and v["state"]["facts"] == []                               # no documents yet


def test_a_lapsed_policy_claim_skips_coverage_and_settlement(client):
    v = create(client, "CLM-0003").json()
    assert "coverage" not in v["status"]["nodes_run"] and v["state"]["policy_snapshot"] is None and v["status"]["gate"]["reason_codes"] == ["POLICY_NOT_IN_FORCE"]


@pytest.mark.parametrize("body,code", [({"claim_id": "CLM-0001", "use_synthetic_documents": True}, 409), ({"claim_id": "CLM-9999", "use_synthetic_documents": True}, 404),
                                       ({"claim_id": "x"}, 422), ({"claim_id": "bad id!"}, 422), ({"loss_date": "not a date"}, 422)])
def test_creating_a_claim_refuses_a_duplicate_an_unknown_demo_claim_and_a_bad_body(client, body, code):
    create(client)
    assert client.post("/claims", json=body).status_code == code


def test_claims_are_listed_one_line_each_and_read_one_by_one(client):
    create(client, "CLM-0001")
    create(client, "CLM-0002")
    rows = {r["claim_id"]: r for r in client.get("/claims").json()}
    assert set(rows) == {"CLM-0001", "CLM-0002"} and rows["CLM-0001"]["tier"] == "complex" and rows["CLM-0002"]["tier"] == "simple"
    assert rows["CLM-0001"]["phase"] == "awaiting_human_decision" and sum(rows["CLM-0001"]["open_questions"].values()) == 6
    assert client.get("/claims/CLM-0002").json()["state"]["routing"]["tier"] == "simple"
    s = client.get("/claims/CLM-0002/status").json()
    assert s["phase"] == "awaiting_human_decision" and s["claim_id"] == "CLM-0002"
    assert client.get("/claims/CLM-0404").status_code == 404 and client.get("/claims/CLM-0404/status").status_code == 404


def test_versions_and_diffs_trace_what_each_node_changed(client):
    create(client)
    versions = client.get("/claims/CLM-0001/versions").json()
    assert [v["node"] for v in versions] == ["start", "intake", "policy", "evidence", "gap", "coverage", "settlement", "action", "audit", "routing"]
    assert versions[1]["mock"] is True and versions[2]["mock"] is False and [v["version"] for v in versions] == list(range(1, 11))
    d = client.get("/claims/CLM-0001/diff", params={"from": 3, "to": 6}).json()
    assert d["from_version"] == 3 and d["to_version"] == 6 and {"facts", "open_questions", "coverage_issues"} <= {c["field"] for c in d["changes"]}
    assert client.get("/claims/CLM-0001/diff", params={"from": 4, "to": 4}).json()["changes"] == []
    assert client.get("/claims/CLM-0001/diff", params={"from": 1, "to": 99}).status_code == 404
    assert client.get("/claims/CLM-0001/diff", params={"from": 1}).status_code == 422
    assert client.get("/claims/CLM-0404/versions").status_code == 404


# ------------------------------------------------------------------------------------------------ documents
def test_a_claim_lists_its_documents_and_serves_the_original(client):
    create(client)
    docs = client.get("/claims/CLM-0001/documents").json()
    assert {d["detected_type"] for d in docs} >= {"fnol", "plumber_report", "surveyor_report", "estimate"}
    first = docs[0]
    f = client.get(f"/claims/CLM-0001/documents/{first['document_id']}/file")
    assert f.status_code == 200 and f.content[:4] == b"%PDF" and f.headers["content-type"].startswith("application/pdf")
    assert client.get("/claims/CLM-0001/documents/DOC-NOPE/file").status_code == 404


def test_uploading_a_document_re_assesses_the_claim_from_the_evidence_node_and_supersedes_the_open_gate(client):
    v1 = create(client).json()
    r = client.post("/claims/CLM-0001/documents", files={"file": ("plumber_supplementary.pdf", PDF, "application/pdf")}, data={"uploaded_by": "plumber"})
    assert r.status_code == 201
    out = r.json()
    assert out["document"]["filename"] == "plumber_supplementary.pdf" and out["document"]["uploaded_by"] == "plumber" and out["reassessment"]["phase"] == "awaiting_human_decision"
    v2 = client.get("/claims/CLM-0001").json()
    assert v2["state_version"] > v1["state_version"] and len(v2["state"]["facts"]) > len(v1["state"]["facts"]) and len(v2["state"]["documents"]) == 5
    events = client.get("/claims/CLM-0001/audit").json()["events"]
    re = next(e for e in events if e["kind"] == "reassessment")
    assert re["data"]["superseded_gate"] is True and re["node"] == "evidence"
    nodes_after = [e["node"] for e in events if e["seq"] > re["seq"] and e["kind"] == "agent_finished"]
    assert nodes_after[0] == "evidence" and "intake" not in nodes_after and "policy" not in nodes_after


def test_an_upload_can_be_stored_without_re_assessing(client):
    v1 = create(client).json()
    r = client.post("/claims/CLM-0001/documents", params={"reassess": "false"}, files={"file": ("late.pdf", PDF, "application/pdf")})
    assert r.status_code == 201 and r.json()["reassessment"] is None and "not re-assessed" in r.json()["note"]
    assert client.get("/claims/CLM-0001").json()["state_version"] == v1["state_version"]
    assert len(client.get("/claims/CLM-0001/documents").json()) == 5


def test_uploads_are_refused_when_empty_too_large_or_for_an_unknown_claim(tmp_path, library):
    c, _ = make_client(tmp_path, library, max_upload_bytes=1000)
    create(c)
    assert c.post("/claims/CLM-0001/documents", files={"file": ("e.pdf", b"", "application/pdf")}).status_code == 422
    assert c.post("/claims/CLM-0001/documents", files={"file": ("big.pdf", PDF, "application/pdf")}).status_code == 413
    assert c.post("/claims/CLM-0404/documents", files={"file": ("x.pdf", b"abc", "application/pdf")}).status_code == 404
    assert c.post("/claims/CLM-0001/documents").status_code == 422


# ------------------------------------------------------------------------------------------------ the human gate
def test_an_authorised_approval_completes_the_claim_and_a_second_decision_is_a_conflict(client):
    create(client)
    r = client.post("/claims/CLM-0001/decision", json=decision("senior_adjuster", reason="reviewed"))
    assert r.status_code == 200
    v = r.json()
    assert v["status"]["phase"] == "completed" and v["state"]["resolution_status"] == "approved" and v["status"]["gate"] is None
    again = client.post("/claims/CLM-0001/decision", json=decision("supervisor"))
    assert again.status_code == 409 and "not waiting" in again.json()["detail"]


@pytest.mark.parametrize("body,fragment", [(decision("adjuster"), "senior_adjuster"), (decision("intern"), "unknown role"), (decision("supervisor", "reject"), "needs a reason"),
                                           (decision("supervisor", "request_more_evidence"), "needs a reason")])
def test_a_refused_decision_returns_422_with_the_reason_and_leaves_the_gate_open(client, body, fragment):
    create(client)
    r = client.post("/claims/CLM-0001/decision", json=body)
    assert r.status_code == 422 and fragment in r.json()["detail"]["error"] and r.json()["detail"]["gate"] == "still open"
    s = client.get("/claims/CLM-0001/status").json()
    assert s["phase"] == "awaiting_human_decision" and s["gate"]["error"]
    assert client.get("/claims/CLM-0001").json()["state"]["resolution_status"] == "awaiting_evidence"       # nothing was decided


@pytest.mark.parametrize("body", [{"decision": "pay", "reviewer_id": "u", "reviewer_role": "supervisor"}, {"decision": "approve"}, {"decision": "approve", "reviewer_id": " ", "reviewer_role": "supervisor"}])
def test_a_malformed_decision_is_rejected_before_it_reaches_the_graph(client, body):
    create(client)
    assert client.post("/claims/CLM-0001/decision", json=body).status_code == 422
    assert client.get("/claims/CLM-0001/status").json()["phase"] == "awaiting_human_decision"


def test_a_simple_claim_may_be_decided_by_an_adjuster_and_an_unknown_claim_is_404(client):
    create(client, "CLM-0002")
    assert client.post("/claims/CLM-0002/decision", json=decision("adjuster")).status_code == 200
    assert client.post("/claims/CLM-0404/decision", json=decision()).status_code == 404


def test_request_more_evidence_then_a_new_document_brings_the_claim_back_to_the_gate(client):
    create(client)
    r = client.post("/claims/CLM-0001/decision", json=decision("supervisor", "request_more_evidence", "need the plumber's statement"))
    assert r.status_code == 200 and r.json()["state"]["resolution_status"] == "awaiting_evidence" and r.json()["status"]["phase"] == "completed"
    up = client.post("/claims/CLM-0001/documents", files={"file": ("plumber_supplementary.pdf", PDF, "application/pdf")})
    assert up.status_code == 201 and up.json()["reassessment"]["phase"] == "awaiting_human_decision"
    assert client.post("/claims/CLM-0001/decision", json=decision("senior_adjuster")).json()["state"]["resolution_status"] == "approved"


# ------------------------------------------------------------------------------------------------ audit
def test_the_audit_log_names_every_agent_every_mock_and_every_human_decision(client):
    create(client)
    client.post("/claims/CLM-0001/decision", json=decision("adjuster"))                                   # refused
    client.post("/claims/CLM-0001/decision", json=decision("senior_adjuster", reviewer="rev-9", reason="ok"))
    a = client.get("/claims/CLM-0001/audit").json()
    kinds = [e["kind"] for e in a["events"]]
    assert kinds[0] == "run_started" and kinds.count("agent_finished") == 9 and kinds[-2:] == ["human_decision", "commit"]
    finished = {e["node"]: e for e in a["events"] if e["kind"] == "agent_finished"}
    assert finished["coverage"]["mock"] is True and finished["coverage"]["message"].startswith("YET TO IMPLEMENT (mock, T5.1)") and finished["gap"]["mock"] is False
    assert [e["kind"] for e in a["human_decisions"]] == ["human_decision_refused", "human_decision"] and a["human_decisions"][1]["data"]["reviewer_id"] == "rev-9"
    assert all("key" not in e["data"] for e in a["events"]) and {r["node"] for r in a["agents"]} >= {"gap", "coverage"}
    assert client.get("/claims/CLM-0404/audit").status_code == 404


# ------------------------------------------------------------------------------------------------ the OpenAPI contract
def test_the_openapi_document_has_every_endpoint_of_t7_1(client):
    spec = client.get("/openapi.json").json()
    paths = {(m.upper(), p) for p, item in spec["paths"].items() for m in item}
    assert {("POST", "/claims"), ("POST", "/claims/{claim_id}/documents"), ("GET", "/claims/{claim_id}"), ("GET", "/claims/{claim_id}/versions"),
            ("GET", "/claims/{claim_id}/diff"), ("POST", "/claims/{claim_id}/decision"), ("GET", "/claims/{claim_id}/audit"), ("GET", "/health")} <= paths
    assert {"ClaimView", "ClaimState", "HumanDecision", "RunStatus", "StateDiff", "AuditView"} <= set(spec["components"]["schemas"])
    assert "YET TO IMPLEMENT" in spec["info"]["description"] and "SYNTHETIC" in spec["info"]["description"]


def test_the_committed_openapi_file_matches_the_service(client):
    committed = json.loads((ROOT / "docs" / "contracts" / "openapi.json").read_text(encoding="utf-8"))
    assert committed == client.get("/openapi.json").json(), "the API changed: run python scripts/export_openapi.py and review the diff"
