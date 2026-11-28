"""Claims survive a restart of the server when it is started with the same store folder: state, versions, pending human decision, decisions, audit events and the audit chain."""
import pytest
from fastapi.testclient import TestClient

from claimtower.api.main import build_default_app


@pytest.fixture
def start(tmp_path, monkeypatch):
    """A function that starts the default server on the SAME store folder every time it is called (a restart)."""
    monkeypatch.setenv("CLAIMTOWER_STORE", str(tmp_path / "store"))
    monkeypatch.delenv("CLAIMTOWER_LLM", raising=False)
    monkeypatch.delenv("CLAIMTOWER_POLICIES", raising=False)
    monkeypatch.delenv("CLAIMTOWER_AUDIT", raising=False)
    return lambda: TestClient(build_default_app())


def test_an_assessed_claim_is_back_after_a_restart_exactly_as_it_was(start):
    a = start()
    assert a.post("/claims", json={"claim_id": "CLM-0001", "use_synthetic_documents": True}).status_code == 201
    before = a.get("/claims/CLM-0001").json()
    versions = a.get("/claims/CLM-0001/versions").json()
    events = a.get("/claims/CLM-0001/audit").json()["events"]
    assert before["status"]["phase"] == "awaiting_human_decision"

    b = start()                                                                         # the restart
    after = b.get("/claims/CLM-0001").json()
    assert after["state"] == before["state"] and after["status"]["phase"] == "awaiting_human_decision" and after["status"]["gate"]["allowed_decisions"] == before["status"]["gate"]["allowed_decisions"]
    assert [c["claim_id"] for c in b.get("/claims").json()] == ["CLM-0001"]
    assert b.get("/claims/CLM-0001/versions").json() == versions
    assert [e["seq"] for e in b.get("/claims/CLM-0001/audit").json()["events"]][:len(events)] == [e["seq"] for e in events]
    assert b.get("/claims/CLM-0001/diff", params={"from": versions[0]["version"], "to": versions[-1]["version"]}).json()["changes"]
    assert b.get("/claims/CLM-0001/documents").json() and b.get("/claims/CLM-0001/trail").json()["chain_ok"]


def test_a_claim_waiting_at_the_gate_can_be_decided_after_a_restart_and_that_decision_also_survives(start):
    a = start()
    a.post("/claims", json={"claim_id": "CLM-0001", "use_synthetic_documents": True})
    b = start()
    r = b.post("/claims/CLM-0001/decision", json={"decision": "approve", "reviewer_id": "rev-1", "reviewer_role": "senior_adjuster"})
    assert r.status_code == 200 and r.json()["status"]["phase"] == "completed"
    c = start()
    done = c.get("/claims/CLM-0001").json()
    assert done["status"]["phase"] == "completed" and done["state"]["resolution_status"] == r.json()["state"]["resolution_status"]
    d = c.get("/claims/CLM-0001/audit").json()["human_decisions"]
    assert d and d[-1]["data"]["reviewer_id"] == "rev-1"
    seqs = [e["seq"] for e in c.get("/claims/CLM-0001/audit").json()["events"]]
    assert seqs == sorted(set(seqs))                                                    # no sequence number was used twice after the restarts


def test_several_claims_keep_their_own_state_and_order_and_a_reassessment_works_after_a_restart(start):
    a = start()
    for cid in ("CLM-0002", "CLM-0001"):
        a.post("/claims", json={"claim_id": cid, "use_synthetic_documents": True})
    b = start()
    assert [c["claim_id"] for c in b.get("/claims").json()] == ["CLM-0002", "CLM-0001"]                  # in the order they were started
    assert b.get("/claims/CLM-0002").json()["state"]["claim_header"]["claim_id"] == "CLM-0002"
    r = b.post("/claims/CLM-0001/reassess", params={"from_node": "policy"})
    assert r.status_code == 200 and r.json()["status"]["phase"] == "awaiting_human_decision"
    c = start()
    assert c.get("/claims/CLM-0001").json()["status"]["state_version"] >= r.json()["status"]["state_version"]
    assert c.post("/claims", json={"claim_id": "CLM-0001", "use_synthetic_documents": True}).status_code == 409           # a restored claim is not created twice


def test_reviewer_edits_survive_a_restart(start):
    a = start()
    a.post("/claims", json={"claim_id": "CLM-0002", "use_synthetic_documents": True})
    r = a.post("/claims/CLM-0002/decision", json={"decision": "edit", "reviewer_id": "rev-1", "reviewer_role": "senior_adjuster", "reason": "surveyor confirmed",
                                                  "edits": {"non_covered_items_total": "28000"}})
    assert r.status_code == 200
    b = start()
    status = b.get("/claims/CLM-0002").json()["status"]
    assert status["phase"] == "awaiting_human_decision" and status["gate"]["overrides"] == {"non_covered_items_total": "28000"}


def test_a_finished_claim_can_be_read_while_another_claim_is_being_assessed(start):
    import threading

    a = start()
    a.post("/claims", json={"claim_id": "CLM-0001", "use_synthetic_documents": True})
    a.post("/claims", json={"claim_id": "CLM-0002", "use_synthetic_documents": True})
    errors: list[str] = []
    done = threading.Event()

    def assess() -> None:
        try:
            for _ in range(3):
                a.post("/claims/CLM-0002/reassess", params={"from_node": "evidence"})
        except Exception as e:                                                              # noqa: BLE001
            errors.append(f"assess: {e}")
        finally:
            done.set()

    t = threading.Thread(target=assess)
    t.start()
    reads = 0
    while not done.is_set() or reads < 5:
        for path in ("", "/versions", "/status", "/audit", "/trail"):
            r = a.get(f"/claims/CLM-0001{path}")
            if r.status_code != 200:
                errors.append(f"{path}: {r.status_code} {r.text[:120]}")
        a.get("/claims")
        reads += 1
    t.join()
    assert not errors and reads >= 5


def _gap(client):
    runner = client.app.state.runner
    return next(s.agent for s in runner.registry if s.node == "gap")


def test_the_gap_agent_re_assesses_incrementally_after_a_restart(start, tmp_path):
    a = start()
    a.post("/claims", json={"claim_id": "CLM-0001", "use_synthetic_documents": True})
    assert _gap(a).last_run.cold                                                       # the first assessment is a full one
    assert (tmp_path / "store" / "gap" / "CLM-0001.json").is_file()
    before = a.get("/claims/CLM-0001").json()["state"]

    b = start()
    assert _gap(b).memory_of("CLM-0001") is None                                       # not loaded until the claim is assessed again
    assert b.post("/claims/CLM-0001/reassess", params={"from_node": "gap"}).status_code == 200
    run = _gap(b).last_run
    assert not run.cold and run.affected == set() and run.kept                         # nothing new: every question carried over, none asked again
    after = b.get("/claims/CLM-0001").json()["state"]
    for k in ("open_questions", "contradictions", "missing_evidence"):                  # (scenarios are written by the first re-assessment of any claim)
        assert after[k] == before[k]


def test_a_saved_gap_memory_that_is_ahead_of_the_claim_is_not_used(start, tmp_path):
    a = start()
    a.post("/claims", json={"claim_id": "CLM-0001", "use_synthetic_documents": True})
    path = tmp_path / "store" / "gap" / "CLM-0001.json"
    import json
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["state"]["open_questions"][0]["status"] = "missing"                           # as if the server stopped between the agent's run and the save of the claim
    path.write_text(json.dumps(doc), encoding="utf-8")
    b = start()
    b.post("/claims/CLM-0001/reassess", params={"from_node": "gap"})
    assert _gap(b).last_run.cold                                                       # a full run: slower, never wrong
    path.write_text("not json", encoding="utf-8")
    c = start()
    assert c.post("/claims/CLM-0001/reassess", params={"from_node": "gap"}).status_code == 200 and _gap(c).last_run.cold
