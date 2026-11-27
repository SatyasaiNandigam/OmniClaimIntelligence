"""T6.3 - the human approval gate: approve / edit / reject / request more evidence by a named, authorised reviewer; overrides with a reason; no path to a payment without a recorded decision."""
from decimal import Decimal

import pytest

from claimtower.composition import build_commit_service, build_registry
from claimtower.core.config import load_settings
from claimtower.core.schemas import ClaimState
from claimtower.integrations import Adapters
from claimtower.integrations.dispatcher import Dispatcher
from claimtower.integrations.mock.claims_mock import MockClaimsAdapter
from claimtower.integrations.mock.pas_mock import MockPasAdapter
from claimtower.integrations.mock.recon_mock import MockReconAdapter
from claimtower.integrations.retry import RetryPolicy
from claimtower.orchestration.graph import ClaimRunner
from claimtower.orchestration.overrides import OverrideStore

SETTINGS = load_settings()
MOCK_ALL_BUT_SETTLEMENT = ("intake", "gap", "coverage", "action", "audit", "routing")


def build(claim="CLM-0002", **runner_kw):
    """The labelled claim on the fixtures with the REAL settlement agent, so an edit really changes a calculation."""
    claims, recon, overrides = MockClaimsAdapter(), MockReconAdapter(), OverrideStore()
    adapters = Adapters(MockPasAdapter(), claims, recon)
    disp = Dispatcher(policy=RetryPolicy(max_attempts=2), sleeper=lambda s: None)
    reg = build_registry(SETTINGS, force_mock=MOCK_ALL_BUT_SETTLEMENT, adapters=adapters, dispatcher=disp, overrides=overrides)
    r = ClaimRunner(reg, commit_service=build_commit_service(SETTINGS, adapters, disp), overrides=overrides, settings=SETTINGS, **runner_kw)
    r.start(ClaimState(claim_id=claim))
    return r, recon, claims, overrides


def decision(role, kind="approve", reason="", reviewer="rev-1", **edits):
    return {"decision": kind, "reviewer_id": reviewer, "reviewer_role": role, "reason": reason, "edits": edits}


def refused(r, claim, **kw):
    st = r.decide(claim, decision(**kw))
    assert st.phase == "awaiting_human_decision"
    return st.gate["error"]


# ---------------------------------------------------------------------------------------------------------------- edit
def test_an_edit_changes_a_settlement_input_the_engine_calculates_again_and_the_gate_opens_again():
    r, recon, _, ov = build()
    assert r.state("CLM-0002").settlement.payable == Decimal(80000)
    st = r.decide("CLM-0002", decision("adjuster", "edit", "the surveyor confirmed a second upgrade", non_covered_items_total="28000"))
    s = r.state("CLM-0002").settlement
    assert st.phase == "awaiting_human_decision" and s.payable == Decimal(60000)                          # 98,000 - 28,000 - 10,000: the engine, not the reviewer, computes
    assert "human override by rev-1" in s.explanation and st.gate["overrides"] == {"non_covered_items_total": "28000"}
    ev = r.events.of_kind("CLM-0002", "human_decision")[0]
    assert ev.data["decision"] == "edit" and ev.data["edits"] == {"non_covered_items_total": "28000"} and ev.data["reason"] == "the surveyor confirmed a second upgrade"
    assert ov.get("CLM-0002") == {"non_covered_items_total": Decimal(28000)} and ov.last("CLM-0002").by == "rev-1" and recon.instruction_count == 0
    assert "REVIEW" in [e for e in r.events.of_kind("CLM-0002", "agent_finished") if e.node == "settlement"][-1].message


def test_an_edit_re_runs_settlement_and_what_depends_on_it_and_nothing_before():
    r, *_ = build()
    before = len(r.events.of_kind("CLM-0002", "agent_finished"))
    r.decide("CLM-0002", decision("adjuster", "edit", "r", applicable_deductible="20000"))
    nodes = [e.node for e in r.events.of_kind("CLM-0002", "agent_finished")][before:]
    assert nodes[0] == "settlement" and "intake" not in nodes and "evidence" not in nodes and "coverage" not in nodes
    assert r.state("CLM-0002").settlement.payable == Decimal(70000)


def test_a_second_edit_replaces_the_first_for_the_same_input_and_every_edit_is_kept():
    r, _, _, ov = build()
    r.decide("CLM-0002", decision("adjuster", "edit", "first", non_covered_items_total="28000"))
    r.decide("CLM-0002", decision("adjuster", "edit", "second", non_covered_items_total="8000", reviewer="rev-2"))
    assert r.state("CLM-0002").settlement.payable == Decimal(80000) and [h.reason for h in ov.history("CLM-0002")] == ["first", "second"]
    assert len(r.events.of_kind("CLM-0002", "human_decision")) == 2


@pytest.mark.parametrize("kw,fragment", [
    (dict(role="adjuster", kind="edit", reason="r"), "needs the settlement inputs"),
    (dict(role="adjuster", kind="edit", reason="r", payable="1"), "cannot edit"),
    (dict(role="adjuster", kind="edit", reason="r", applicable_deductible="-5"), "not negative"),
    (dict(role="adjuster", kind="edit", reason="r", applicable_deductible="lots"), "not a valid amount"),
    (dict(role="adjuster", kind="edit", reason="", applicable_deductible="5"), "needs a reason"),
    (dict(role="adjuster", kind="approve", applicable_deductible="5"), "only an 'edit' decision"),
])
def test_an_invalid_edit_is_refused_with_the_reason_and_changes_nothing(kw, fragment):
    r, _, _, ov = build()
    assert fragment in refused(r, "CLM-0002", **kw)
    assert ov.get("CLM-0002") == {} and r.state("CLM-0002").settlement.payable == Decimal(80000)


# ---------------------------------------------------------------------------------------------------------------- roles (rbac.yaml + the autonomy matrix)
def test_the_role_hierarchy_comes_from_rbac_yaml():
    assert SETTINGS.rbac.approval_order == ["adjuster", "senior_adjuster", "supervisor"]
    r, *_ = build()
    assert r.supervisor.rank == {"adjuster": 1, "senior_adjuster": 2, "supervisor": 3} and "finance" not in r.supervisor.rank
    assert "unknown role 'finance'" in refused(r, "CLM-0002", role="finance")


def test_rejecting_a_claim_needs_the_role_the_autonomy_matrix_names_even_on_a_simple_claim():
    r, *_ = build()
    assert r.state("CLM-0002").routing.tier == "simple"                                                      # the fixture routing: an adjuster may approve it
    assert "autonomy matrix: deny_claim" in refused(r, "CLM-0002", role="adjuster", kind="reject", reason="no")
    st = r.decide("CLM-0002", decision("senior_adjuster", "reject", "outside the period"))
    assert st.phase == "completed" and st.resolution_status == "closed"


def test_a_decision_without_a_named_reviewer_or_a_known_role_is_refused():
    r, *_ = build()
    assert "unknown role" in refused(r, "CLM-0002", role="intern")
    assert "must be named" in refused(r, "CLM-0002", role="adjuster", reviewer=" ")


# ---------------------------------------------------------------------------------------------------------------- no path to a payment without a recorded authorised decision
def test_nothing_reaches_the_recon_system_unless_a_supervisor_approves():
    for kind, role, reason in [("reject", "senior_adjuster", "r"), ("request_more_evidence", "adjuster", "need more"), ("edit", "adjuster", "r"), ("approve", "adjuster", ""),
                               ("approve", "senior_adjuster", "")]:
        r, recon, claims, _ = build()
        r.decide("CLM-0002", decision(role, kind, reason, **({"applicable_deductible": "1"} if kind == "edit" else {})))
        assert recon.instruction_count == 0 and recon.calls == [], (kind, role)
    r, recon, *_ = build()
    r.decide("CLM-0002", decision("supervisor", "approve", reviewer="rev-9"))
    ev = r.events.of_kind("CLM-0002", "human_decision")[-1]
    assert recon.instruction_count == 1 and recon.calls == ["CLM-0002:pay:1"]
    pay = r.events.of_kind("CLM-0002", "commit")[0]
    assert "accepted" in pay.data["payment"] and recon._instructions["RECON-0001"].approval_ref == f"EVT-{ev.seq}" and recon._instructions["RECON-0001"].approved_by == "rev-9"


def test_an_edited_amount_is_what_is_paid_and_only_after_the_approval_that_follows_the_edit():
    r, recon, *_ = build()
    r.decide("CLM-0002", decision("supervisor", "edit", "r", non_covered_items_total="28000", reviewer="rev-9"))
    assert recon.instruction_count == 0
    r.decide("CLM-0002", decision("supervisor", "approve", reviewer="rev-9"))
    assert recon._instructions["RECON-0001"].amount == Decimal(60000)


def test_the_commit_node_is_reachable_only_from_the_gate_and_refuses_to_run_without_a_recorded_approval():
    r, recon, *_ = build()
    edges = {(e.source, e.target) for e in r.graph.get_graph().edges}
    assert {s for s, t in edges if t == "commit"} == {"human_gate"}
    with pytest.raises(RuntimeError, match="no recorded human approval"):
        r.supervisor.commit(r.state("CLM-0002"))                       # called directly, with no decision recorded
    assert recon.instruction_count == 0
    r.decide("CLM-0002", decision("supervisor", "reject", "r", reviewer="rev-9"))
    with pytest.raises(RuntimeError, match="no recorded human approval"):
        r.supervisor.commit(r.state("CLM-0002"))                       # the last decision was a rejection
    assert recon.instruction_count == 0 and r.events.of_kind("CLM-0002", "commit") == []
