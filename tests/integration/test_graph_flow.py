"""T6.1 - the ENTIRE graph flow with the agents that exist (Evidence, Policy, Gap) running for real on the synthetic claims, and the rest ("YET TO IMPLEMENT")
as mocks. This is the test to extend as each mock is replaced: the flow, the gate and the real agents' outputs are already exercised together.
"""
from pathlib import Path

import pytest

from claimtower.composition import SYNTHETIC, build_registry, ingest_claim_folder, policy_library_from_pdfs
from claimtower.core.config import load_settings
from claimtower.core.schemas import ClaimState
from claimtower.ingestion.store import DocumentStore
from claimtower.orchestration.graph import ClaimRunner
from claimtower.orchestration.registry import PIPELINE
from claimtower.orchestration.supervisor import HumanDecision

SETTINGS = load_settings()
CLAIMS = {"S-01": "CLM-0001", "S-02": "CLM-0002", "S-03": "CLM-0003", "S-04": "CLM-0004", "S-05": "CLM-0005"}
REAL = {"policy", "evidence", "gap"}


@pytest.fixture(scope="module")
def library():
    return policy_library_from_pdfs(SETTINGS)


def make_runner(tmp_path: Path, library, claim: str, *, without_wording: tuple[str, ...] = ()):
    store = DocumentStore(tmp_path / "store")
    ingest_claim_folder(store, SYNTHETIC / "claims" / claim)
    lib = library.without(*without_wording) if without_wording else library
    return ClaimRunner(build_registry(SETTINGS, document_store=store, policy_library=lib)), store


def decision(role="senior_adjuster", kind="approve", reason=""):
    return HumanDecision(decision=kind, reviewer_id="adjuster-7", reviewer_role=role, reason=reason)


def test_the_status_table_names_what_is_real_and_what_is_yet_to_implement(tmp_path, library):
    r, _ = make_runner(tmp_path, library, "CLM-0001")
    table = {row["node"]: row["status"] for row in r.status("CLM-0001").agents}
    assert {n for n, s in table.items() if s == "implemented"} == REAL
    assert {n for n, s in table.items() if s.startswith("YET TO IMPLEMENT")} == set(PIPELINE) - REAL


def test_a_new_claim_runs_through_real_and_mock_agents_to_the_gate_and_an_approval_completes_it(tmp_path, library):
    r, _ = make_runner(tmp_path, library, "CLM-0001")
    status = r.start(ClaimState(claim_id="CLM-0001"))

    # the flow: every node, in order, to the gate; only the three real agents are not mocks
    assert status.phase == "awaiting_human_decision" and status.nodes_run == list(PIPELINE)
    assert set(status.mock_nodes_run) == set(PIPELINE) - REAL
    state = r.state("CLM-0001")

    # policy (real): the contract in force on the loss date, from the mock PAS and the decomposed wording
    assert state.policy_snapshot.pas_status == "in_force" and state.policy_snapshot.mock
    ids = {c.clause_id for c in state.policy_snapshot.clauses}
    assert all("/" in i for i in ids) and any(i.startswith("BASE-HOME-1/") for i in ids) and any(i.startswith("END-01/") for i in ids)

    # evidence (real): facts with provenance, from the stored documents
    assert len(state.facts) > 10 and {d.detected_type for d in state.documents} >= {"fnol", "plumber_report", "surveyor_report", "estimate"}
    assert all(f.evidence_text and f.page for f in state.facts)

    # gap (real): the questions, linked to the REAL clauses, assessed without a model (fact questions abstain, they are not guessed)
    qs = {q.question_id: q for q in state.open_questions}
    assert set(qs) >= {"Q-INFORCE", "Q-SUDDEN", "Q-START", "Q-EXTENT", "Q-AMOUNT", "Q-SUBLIMIT"}
    assert qs["Q-INFORCE"].status == "established" and qs["Q-SUBLIMIT"].status == "established" and qs["Q-SUDDEN"].status == "unverified"
    assert any(c.startswith("BASE-HOME-1/") for c in qs["Q-SUDDEN"].linked_clause_ids) and qs["Q-SUDDEN"].linked_fact_ids
    gap_event = next(e for e in r.events_for("CLM-0001") if e.node == "gap")
    assert not gap_event.mock and gap_event.message.startswith("FULL assessment") and "REVIEW" in gap_event.message

    # the mocks say so: their output is the labelled fixture, not derived from the real agents above
    coverage = next(e for e in r.events_for("CLM-0001") if e.node == "coverage")
    assert coverage.mock and coverage.message.startswith("YET TO IMPLEMENT (mock, T5.1)")
    assert state.routing.tier == "complex" and status.gate["tier"] == "complex" and set(status.gate["mock_agents"]) == set(PIPELINE) - REAL

    # the gate: nothing happens until a person decides; the decision is recorded; only an approval reaches the (mock) commit
    assert r.status("CLM-0001").resolution_status == "awaiting_evidence" and not r.events.of_kind("CLM-0001", "human_decision")
    refused = r.decide("CLM-0001", decision("adjuster"))
    assert refused.phase == "awaiting_human_decision" and "senior_adjuster" in refused.gate["error"]
    done = r.decide("CLM-0001", decision("senior_adjuster", reason="reviewed with the surveyor's report"))
    assert done.phase == "completed" and done.resolution_status == "approved"
    assert r.events.of_kind("CLM-0001", "human_decision")[0].data["reviewer_id"] == "adjuster-7" and r.events.of_kind("CLM-0001", "commit")[0].mock
    versions = [v for v, _ in r.history("CLM-0001")]
    assert versions == sorted(versions) and versions[-1] == r.state("CLM-0001").state_version


@pytest.mark.parametrize("sid", list(CLAIMS))
def test_every_synthetic_claim_goes_through_the_real_agents_to_the_gate(tmp_path, library, sid):
    claim = CLAIMS[sid]
    r, _ = make_runner(tmp_path, library, claim, without_wording=("END-01",) if sid == "S-05" else ())      # S-05: the endorsement wording is missing
    status = r.start(ClaimState(claim_id=claim))
    state = r.state(claim)
    assert status.phase == "awaiting_human_decision" and not r.events.of_kind(claim, "agent_failed")
    assert state.facts and state.open_questions and state.routing is not None
    if sid == "S-03":                                                          # lapsed: no policy in force, so coverage and settlement are skipped
        assert state.policy_snapshot is None and [q.question_id for q in state.open_questions] == ["Q-INFORCE"]
        assert "coverage" not in status.nodes_run and "settlement" not in status.nodes_run
    else:
        assert state.policy_snapshot is not None and "coverage" in status.nodes_run
    if sid == "S-05":
        assert {q.question_id: q.status for q in state.open_questions}["Q-SUBLIMIT"] == "missing"       # the real gap agent finds the missing wording
    if sid == "S-04":
        clause_ids = {c.clause_id for c in state.policy_snapshot.clauses}                               # loss 20 Jan 2026: END-02 only takes effect on 1 Feb 2026
        assert any(i.startswith("END-01/") for i in clause_ids) and not any(i.startswith("END-02/") for i in clause_ids)


def test_late_evidence_re_enters_at_the_evidence_node_and_only_the_touched_questions_are_re_run(tmp_path, library):
    r, store = make_runner(tmp_path, library, "CLM-0001")
    r.start(ClaimState(claim_id="CLM-0001"))
    first = r.state("CLM-0001")
    r.decide("CLM-0001", decision("supervisor", "request_more_evidence", "the plumber must explain the corrosion"))
    assert r.status("CLM-0001").resolution_status == "awaiting_evidence"

    # a supplementary plumber report arrives: stored as a new document of the claim
    before_events = len(r.events_for("CLM-0001"))
    store.ingest("CLM-0001", "plumber_supplementary.pdf", (SYNTHETIC / "claims" / "CLM-0001" / "plumber_report.pdf").read_bytes(), document_id="DOC-LATE-1",
                 uploaded_by="plumber")
    status = r.reassess("CLM-0001", from_node="evidence")
    new_events = r.events_for("CLM-0001")[before_events:]

    assert status.phase == "awaiting_human_decision"
    ran = [e.node for e in new_events if e.kind == "agent_finished"]
    assert ran == ["evidence", "gap", "coverage", "settlement", "action", "audit", "routing"]            # intake and policy were NOT re-run
    state = r.state("CLM-0001")
    old_facts = {f.fact_id: f for f in first.facts}
    assert all(state_f == old_facts[state_f.fact_id] for state_f in state.facts if state_f.fact_id in old_facts)       # unchanged documents keep their facts
    assert any(f.source_document_id == "DOC-LATE-1" for f in state.facts) and len(state.facts) > len(first.facts)
    evidence_msg = next(e.message for e in new_events if e.node == "evidence" and e.kind == "agent_finished")
    assert "1 document(s) processed" in evidence_msg and "unchanged" in evidence_msg
    gap = r.registry["gap"].agent
    assert not gap.last_run.cold and gap.last_run.kept                                                    # the Gap agent was incremental: some question was carried over
    assert next(e.message for e in new_events if e.node == "gap" and e.kind == "agent_finished").startswith("INCREMENTAL")
    assert state.state_version > first.state_version and r.status("CLM-0001").gate is not None


def test_a_failing_real_agent_halts_the_claim_with_the_reason(tmp_path, library):
    r, store = make_runner(tmp_path, library, "CLM-0001")
    original = r.registry["policy"].agent._snapshots

    class Down:
        def get_snapshot(self, *a, **k):
            raise ConnectionError("the policy administration system is not reachable")

        build = get_snapshot

    r.registry["policy"].agent._snapshots = Down()
    status = r.start(ClaimState(claim_id="CLM-0001"))
    r.registry["policy"].agent._snapshots = original
    assert status.phase == "halted" and "not reachable" in status.halted_reason and status.paused_at == "policy"
    assert status.nodes_run == ["intake"] and status.resolution_status == "awaiting_human"
