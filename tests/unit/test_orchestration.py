"""T6.1 - the claim graph with every agent that is not built yet as a MOCK ("YET TO IMPLEMENT"): flow, routing, the human gate, failure, re-entry."""
import json
import logging
from collections import Counter
from pathlib import Path

import pytest

from claimtower.composition import build_registry
from claimtower.core.config import load_settings
from claimtower.core.schemas import ClaimState
from claimtower.core.state import OwnershipViolation, StatePatch, apply_patch
from claimtower.orchestration.checkpoints import contract_types, make_checkpointer
from claimtower.orchestration.graph import ClaimRunner
from claimtower.orchestration.mocks import MARK, FixtureSource, MockAgent, neutral_slice
from claimtower.orchestration.registry import PIPELINE, ROSTER, AgentRegistry
from claimtower.orchestration.supervisor import HumanDecision, policy_in_force

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = load_settings()
CLAIMS = {"S-01": "CLM-0001", "S-02": "CLM-0002", "S-03": "CLM-0003", "S-04": "CLM-0004", "S-05": "CLM-0005"}


def fixture(sid) -> ClaimState:
    return ClaimState(**json.loads((ROOT / f"data/fixtures/{sid}/claim_state.json").read_text(encoding="utf-8")))


def all_mock_registry(**kw):
    """Every node a mock: the graph runs on the labelled fixtures alone (the gap and coverage agents are real in `build_registry` by default)."""
    return build_registry(SETTINGS, force_mock=("intake", "gap", "coverage", "settlement", "action", "audit", "routing"), **kw)


def runner(registry=None, **kw):
    return ClaimRunner(registry or all_mock_registry(), **kw)


def approve(role="senior_adjuster", **kw):
    return HumanDecision(decision=kw.pop("decision", "approve"), reviewer_id=kw.pop("reviewer_id", "u-1"), reviewer_role=role, **kw)


# ------------------------------------------------------------------------------------------------ the roster and the mocks
def test_the_roster_is_the_whole_pipeline_and_every_agent_names_its_task():
    assert PIPELINE == ("intake", "policy", "evidence", "gap", "coverage", "settlement", "action", "audit", "routing")
    assert all(task.startswith("T") for _, task, _ in ROSTER.values())


def test_a_registry_needs_every_node_and_each_node_its_own_agent():
    ok = {n: (MockAgent(n), "mock") for n in PIPELINE}
    AgentRegistry(ok)
    with pytest.raises(ValueError, match="missing"):
        AgentRegistry({k: v for k, v in ok.items() if k != "gap"})
    with pytest.raises(ValueError, match="written by agent"):
        AgentRegistry({**ok, "gap": (MockAgent("policy"), "mock")})


def test_the_default_registry_has_the_real_gap_coverage_settlement_and_routing_agents_and_two_mocks_that_say_they_are_yet_to_implement():
    reg = build_registry(SETTINGS)
    assert reg.implemented == ["intake", "gap", "coverage", "settlement", "action", "audit", "routing"] and len(reg.mocks) == 2
    table = {r["node"]: r for r in reg.describe()}
    assert table["gap"]["status"] == "implemented" and table["coverage"]["status"] == "implemented"
    assert table["settlement"]["status"] == "implemented" and table["audit"]["status"] == "implemented" and table["audit"]["task"] == "T8.5"
    assert build_registry(SETTINGS, force_mock=("intake", "gap", "coverage", "settlement", "action", "audit", "routing")).implemented == []


@pytest.mark.parametrize("node", PIPELINE)
def test_a_mock_returns_the_fixture_slice_it_owns_and_says_it_is_not_implemented(node):
    s = fixture("S-01")
    blank = ClaimState(claim_id=s.claim_id)
    m = MockAgent(node)
    patch = m.run(blank)
    assert isinstance(patch, StatePatch) and set(patch.updates) == set(m.owns)
    assert patch.rationale.startswith(f"{MARK} (mock, {ROSTER[node][1]})") and "scenario S-01" in patch.rationale and "nothing was computed" in patch.rationale
    after = apply_patch(blank, patch)                                    # ownership enforced
    assert all(getattr(after, f) == getattr(s, f) for f in m.owns)


def test_applying_the_mocks_in_order_reproduces_every_labelled_fixture_state():
    for sid, claim in CLAIMS.items():
        state = ClaimState(claim_id=claim)
        for node in PIPELINE:
            state = apply_patch(state, MockAgent(node).run(state))
        want = fixture(sid)
        skip = ("state_version", "correlation_id")                       # written at claim creation, not by an agent
        assert {k: v for k, v in state.model_dump().items() if k not in skip} == {k: v for k, v in want.model_dump().items() if k not in skip}, sid


def test_a_mock_for_a_claim_without_a_fixture_changes_nothing_and_says_so():
    blank = ClaimState(claim_id="CLM-NEW")
    patch = MockAgent("coverage").run(blank)
    assert patch.updates == neutral_slice("coverage") and "no change (no fixture for claim CLM-NEW" in patch.rationale
    from claimtower.core.schemas import ClaimHeader
    seeded = blank.model_copy(update={"claim_header": ClaimHeader(claim_id="CLM-NEW", policy_ref="POL-9")})
    assert MockAgent("intake").run(seeded).updates["claim_header"].policy_ref == "POL-9"                      # a header the caller supplied is kept
    assert FixtureSource(ROOT / "no-such-dir").scenario_of("CLM-0001") is None


# ------------------------------------------------------------------------------------------------ the flow
def test_a_new_claim_runs_every_node_in_order_and_pauses_at_the_human_gate():
    r = runner()
    status = r.start(ClaimState(claim_id="CLM-0001"))
    assert status.phase == "awaiting_human_decision" and status.paused_at == "human_gate"
    assert status.nodes_run == list(PIPELINE) and set(status.mock_nodes_run) == set(PIPELINE)
    state = r.state("CLM-0001")
    want = fixture("S-01")
    assert {k: v for k, v in state.model_dump().items() if k != "state_version"} == {k: v for k, v in want.model_dump().items() if k != "state_version"}
    assert r.events.of_kind("CLM-0001", "agent_finished")[0].mock and all(MARK in e.message for e in r.events.of_kind("CLM-0001", "agent_finished"))


@pytest.mark.parametrize("sid", list(CLAIMS))
def test_every_labelled_scenario_reaches_the_gate_with_its_labelled_tier(sid):
    r = runner()
    status = r.start(ClaimState(claim_id=CLAIMS[sid]))
    want = fixture(sid)
    assert status.phase == "awaiting_human_decision" and status.gate["tier"] == want.routing.tier and status.gate["reason_codes"] == want.routing.reason_codes
    assert status.gate["required_approver_role"] == want.routing.required_approver_role and status.resolution_status == want.resolution_status


def test_a_claim_with_no_policy_in_force_runs_coverage_but_skips_settlement():
    r = runner()
    status = r.start(ClaimState(claim_id="CLM-0003"))
    assert "settlement" not in status.nodes_run and status.nodes_run[3:7] == ["gap", "coverage", "action", "audit"]
    skip = r.events.of_kind("CLM-0003", "routed")[0]
    assert skip.data["skipped"] == ["settlement"] and "no policy in force" in skip.message
    assert not policy_in_force(r.state("CLM-0003")) and policy_in_force(fixture("S-01"))
    assert policy_in_force(ClaimState(claim_id="x"))                                              # not yet known is not "not in force"


def test_the_gate_shows_the_reviewer_what_is_needed_and_which_agents_are_mocks():
    g = runner().start(ClaimState(claim_id="CLM-0001")).gate
    assert g["claim_id"] == "CLM-0001" and g["tier"] == "complex" and g["allowed_decisions"] == ["approve", "edit", "reject", "request_more_evidence"]
    assert set(g["mock_agents"]) == set(PIPELINE) and g["open_questions"]["Q-START"] == "missing" and g["contradictions"] == 1
    assert "only recommends" in g["note"] and g["settlement"]["status"] in ("calculated", "cannot_calculate")


# ------------------------------------------------------------------------------------------------ the human gate
def test_an_approval_by_an_authorised_reviewer_is_recorded_and_reaches_the_mock_commit_only():
    r = runner()
    r.start(ClaimState(claim_id="CLM-0001"))
    status = r.decide("CLM-0001", approve("senior_adjuster", reason="evidence reviewed"))
    assert status.phase == "completed" and status.resolution_status == "approved" and status.paused_at is None
    d = r.events.of_kind("CLM-0001", "human_decision")[0]
    assert d.data["reviewer_id"] == "u-1" and d.data["reviewer_role"] == "senior_adjuster" and d.data["decision"] == "approve" and d.data["reason"] == "evidence reviewed"
    c = r.events.of_kind("CLM-0001", "commit")[0]
    assert c.mock and MARK in c.message and "nothing was sent" in c.message


@pytest.mark.parametrize("decision,status", [("reject", "closed"), ("request_more_evidence", "awaiting_evidence")])
def test_a_reject_or_a_request_for_evidence_needs_a_reason_and_never_reaches_the_commit(decision, status):
    r = runner()
    r.start(ClaimState(claim_id="CLM-0001"))
    refused = r.decide("CLM-0001", approve("supervisor", decision=decision))
    assert refused.phase == "awaiting_human_decision" and "needs a reason" in refused.gate["error"]
    done = r.decide("CLM-0001", approve("supervisor", decision=decision, reason="because"))
    assert done.phase == "completed" and done.resolution_status == status and not r.events.of_kind("CLM-0001", "commit")
    assert len(r.events.of_kind("CLM-0001", "human_decision_refused")) == 1


def test_a_reviewer_whose_role_is_too_low_or_unknown_is_refused_and_the_gate_stays_open():
    r = runner()
    r.start(ClaimState(claim_id="CLM-0001"))                                     # complex: needs senior_adjuster or higher
    low = r.decide("CLM-0001", approve("adjuster"))
    assert low.phase == "awaiting_human_decision" and "needs 'senior_adjuster' or higher" in low.gate["error"] and low.resolution_status == "awaiting_evidence"
    unknown = r.decide("CLM-0001", {"decision": "approve", "reviewer_id": "u", "reviewer_role": "intern"})
    assert "unknown role" in unknown.gate["error"]
    anonymous = r.decide("CLM-0001", {"decision": "approve", "reviewer_id": " ", "reviewer_role": "supervisor"})
    assert "must be named" in anonymous.gate["error"]
    bad = r.decide("CLM-0001", {"decision": "pay", "reviewer_id": "u", "reviewer_role": "supervisor"})
    assert "not a valid decision" in bad.gate["error"]
    assert r.status("CLM-0001").resolution_status == "awaiting_evidence" and not r.events.of_kind("CLM-0001", "human_decision")
    assert r.decide("CLM-0001", approve("senior_adjuster")).phase == "completed"   # a valid one still goes through


def test_a_simple_claim_may_be_decided_by_an_adjuster():
    r = runner()
    r.start(ClaimState(claim_id="CLM-0002"))
    assert r.decide("CLM-0002", approve("adjuster")).resolution_status == "approved"


def test_deciding_when_nothing_is_waiting_or_starting_twice_is_an_error():
    r = runner()
    with pytest.raises(ValueError, match="not waiting"):
        r.decide("CLM-0001", approve())
    r.start(ClaimState(claim_id="CLM-0001"))
    with pytest.raises(ValueError, match="already been started"):
        r.start(ClaimState(claim_id="CLM-0001"))
    assert runner().status("CLM-9999").phase == "not_started"
    with pytest.raises(KeyError):
        runner().state("CLM-9999")


# ------------------------------------------------------------------------------------------------ failure
class Boom:
    agent_id = "evidence"
    owns = frozenset({"documents", "facts", "image_findings"})

    def run(self, state):
        raise RuntimeError("the document store is down")


class Rogue:
    """The policy agent writing a field it does not own."""
    agent_id = "policy"
    owns = frozenset({"policy_snapshot"})

    def run(self, state):
        return StatePatch(agent_id="policy", updates={"settlement": None})


def with_agent(node, agent):
    reg = all_mock_registry()
    slots = {s.node: (s.agent, s.status) for s in reg}
    slots[node] = (agent, "implemented")
    return AgentRegistry(slots)


def test_a_failing_agent_parks_the_claim_with_its_reason_and_the_flow_halts():
    r = runner(with_agent("evidence", Boom()))
    status = r.start(ClaimState(claim_id="CLM-0001"))
    assert status.phase == "halted" and status.paused_at == "evidence" and "the document store is down" in status.halted_reason
    assert status.resolution_status == "awaiting_human" and status.nodes_run == ["intake", "policy"]
    ev = r.events.last_failure("CLM-0001")
    assert ev.node == "evidence" and ev.data["reason"] == "RuntimeError: the document store is down"
    with pytest.raises(ValueError, match="not waiting"):
        r.decide("CLM-0001", approve())                                          # a halted claim has no gate to decide at


def test_an_ownership_violation_is_a_bug_so_the_run_stops_loudly_and_is_logged():
    r = runner(with_agent("policy", Rogue()))
    with pytest.raises(OwnershipViolation, match="policy tried to write"):
        r.start(ClaimState(claim_id="CLM-0001"))
    assert r.events.of_kind("CLM-0001", "ownership_violation")[0].node == "policy"


def test_a_patch_returned_under_another_agents_id_is_refused():
    class Impostor(Rogue):
        def run(self, state):
            return StatePatch(agent_id="gap", updates={})

    r = runner(with_agent("policy", Impostor()))
    with pytest.raises(OwnershipViolation, match="returned a patch as 'gap'"):
        r.start(ClaimState(claim_id="CLM-0001"))


# ------------------------------------------------------------------------------------------------ re-entry and history
class Counting:
    def __init__(self, inner):
        self.inner, self.calls = inner, 0
        self.agent_id, self.owns = inner.agent_id, inner.owns

    def run(self, state):
        self.calls += 1
        return self.inner.run(state)


def counting_registry():
    reg = all_mock_registry()
    counters = {s.node: Counting(s.agent) for s in reg}
    return AgentRegistry({n: (c, "mock") for n, c in counters.items()}), counters


def test_new_evidence_re_runs_from_the_chosen_node_and_skips_everything_before_it():
    reg, counters = counting_registry()
    r = ClaimRunner(reg)
    r.start(ClaimState(claim_id="CLM-0001"))
    r.decide("CLM-0001", approve("supervisor", decision="request_more_evidence", reason="need the plumber's statement"))
    v = r.state("CLM-0001").state_version
    status = r.reassess("CLM-0001", from_node="gap")
    assert status.phase == "awaiting_human_decision" and r.state("CLM-0001").state_version > v
    assert {n: c.calls for n, c in counters.items()} == {"intake": 1, "policy": 1, "evidence": 1, "gap": 2, "coverage": 1, "settlement": 1, "action": 1, "audit": 1, "routing": 1}      # only gap re-ran (it also reads the coverage issues, new since its first run); the rest had unchanged inputs
    assert [e.node for e in r.events.of_kind("CLM-0001", "node_skipped")] == ["coverage", "settlement", "action", "audit", "routing"] and "unchanged" in r.events.of_kind("CLM-0001", "node_skipped")[0].message
    assert r.events.of_kind("CLM-0001", "reassessment")[0].data["from_node"] == "gap"
    assert r.decide("CLM-0001", approve("senior_adjuster")).resolution_status == "approved"


def test_reassessing_needs_a_started_claim_a_known_node_and_no_open_gate():
    r = runner()
    with pytest.raises(ValueError, match="not been started"):
        r.reassess("CLM-0001")
    r.start(ClaimState(claim_id="CLM-0001"))
    with pytest.raises(ValueError, match="unknown node"):
        r.reassess("CLM-0001", from_node="payment")


def test_the_checkpoint_history_is_a_monotonic_trace_and_claims_are_isolated():
    r = runner()
    r.start(ClaimState(claim_id="CLM-0001"))
    r.start(ClaimState(claim_id="CLM-0002"))
    versions = [v for v, _ in r.history("CLM-0001")]
    assert versions == sorted(versions) and versions[-1] == r.state("CLM-0001").state_version and r.history("CLM-0001")[-1][1] == ("human_gate",)
    assert r.state("CLM-0002").claim_header.claim_id == "CLM-0002" and r.state("CLM-0001").claim_header.claim_id == "CLM-0001"
    assert all(e.claim_id == "CLM-0001" for e in r.events_for("CLM-0001"))


def test_a_second_runner_on_the_same_checkpointer_can_resume_a_paused_claim():
    saver = make_checkpointer("memory")
    a = ClaimRunner(all_mock_registry(), checkpointer=saver)
    a.start(ClaimState(claim_id="CLM-0001"))
    b = ClaimRunner(all_mock_registry(), checkpointer=saver)
    assert b.status("CLM-0001").phase == "awaiting_human_decision"
    assert b.decide("CLM-0001", approve("senior_adjuster")).resolution_status == "approved"


def test_the_contract_models_are_registered_with_the_checkpoint_serializer(caplog):
    assert ("claimtower.core.schemas.evidence", "Fact") in contract_types() and len(contract_types()) > 30
    with caplog.at_level(logging.WARNING):
        r = runner()
        r.start(ClaimState(claim_id="CLM-0001"))
        r.state("CLM-0001")
    assert not [m for m in caplog.messages if "unregistered type" in m]
    with pytest.raises(ValueError, match="unknown checkpointer"):
        make_checkpointer("postgres")


def test_the_graph_can_be_drawn_with_every_node():
    d = runner().diagram()
    for n in (*PIPELINE, "start", "human_gate", "commit"):
        assert n in d
    _ = Counter


def test_new_evidence_while_the_gate_is_open_supersedes_the_pending_recommendation():
    reg, counters = counting_registry()
    r = ClaimRunner(reg)
    r.start(ClaimState(claim_id="CLM-0001"))
    assert r.status("CLM-0001").phase == "awaiting_human_decision"
    status = r.reassess("CLM-0001", from_node="evidence")                      # nobody had decided: the gate is presented again after the re-assessment
    assert status.phase == "awaiting_human_decision" and counters["intake"].calls == 1 and counters["evidence"].calls == 2
    ev = r.events.of_kind("CLM-0001", "reassessment")[0]
    assert ev.data["superseded_gate"] is True and "superseded" in ev.message
    assert r.decide("CLM-0001", approve("senior_adjuster")).phase == "completed"


def test_versions_and_diffs_come_from_the_checkpoints():
    r = runner()
    r.start(ClaimState(claim_id="CLM-0001"))
    versions = r.versions("CLM-0001")
    assert [v.node for v in versions] == ["start", *PIPELINE] and [v.version for v in versions] == list(range(1, 11)) and versions[1].mock
    assert r.state_at("CLM-0001", 5).state_version == 5
    d = r.diff("CLM-0001", 3, 6)                                              # evidence, gap and coverage ran in between
    assert {c.field for c in d.changes} >= {"facts", "open_questions", "coverage_issues"} and d.from_version == 3 and d.to_version == 6
    assert r.diff("CLM-0001", 4, 4).empty and r.knows("CLM-0001") and not r.knows("CLM-X") and r.claims() == ["CLM-0001"]
    with pytest.raises(KeyError):
        r.state_at("CLM-0001", 99)
