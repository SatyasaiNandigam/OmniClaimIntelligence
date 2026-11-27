"""T8.4 - guards and the governance test group (PDF 11.1): missing citation, low-confidence extraction, model unavailable, human override, action without authority."""
import copy
from decimal import Decimal

import pytest

from claimtower.audit.guards import Actor, AutonomyGuard, GuardedRecon, NotAuthorised
from claimtower.composition import build_commit_service, build_registry, ingest_claim_folder, policy_library_from_pdfs, SYNTHETIC
from claimtower.core.config import AutonomyMatrix, load_settings
from claimtower.core.schemas import ClaimState, CoverageIssue, Fact, PaymentInstruction, ProposedAction
from claimtower.core.state import OWNERSHIP
from claimtower.ingestion.store import DocumentStore
from claimtower.integrations import Adapters
from claimtower.integrations.dispatcher import Dispatcher
from claimtower.integrations.mock.claims_mock import MockClaimsAdapter
from claimtower.integrations.mock.pas_mock import MockPasAdapter
from claimtower.integrations.mock.recon_mock import MockReconAdapter
from claimtower.integrations.retry import RetryPolicy
from claimtower.orchestration.graph import ClaimRunner
from claimtower.orchestration.overrides import OverrideStore
from claimtower.orchestration.registry import AgentRegistry

SETTINGS = load_settings()
ORDER = SETTINGS.rbac.approval_order
HUMAN = lambda role, i="rev-1": Actor("human", i, role)      # noqa: E731
AGENT = Actor("agent", "action")


def guard(sink=None):
    return AutonomyGuard(SETTINGS.autonomy, ORDER, sink)


def instruction(claim="CLM-G"):
    return PaymentInstruction(claim_id=claim, amount=Decimal(80000), payee_ref="P", approved_by="rev-9", approval_ref="EVT-1", idempotency_key=f"{claim}:pay:1")


# ---------------------------------------------------------------------------------------------------------------- the guard
def test_an_authorised_human_only_operation_is_for_the_named_roles_and_those_above_never_an_agent():
    g = guard()
    for role, want in [("adjuster", False), ("senior_adjuster", False), ("supervisor", True), ("finance", True), ("intake", False)]:
        assert g.decide("release_payment", HUMAN(role))[0] is want, role
    for role, want in [("adjuster", False), ("senior_adjuster", True), ("supervisor", True)]:
        assert g.decide("deny_claim", HUMAN(role))[0] is want, role
    for actor in (AGENT, Actor("system", "scheduler"), Actor("agent", "x", "supervisor")):
        ok, why = g.decide("release_payment", actor)
        assert not ok and "authorised person only" in why
    assert not g.decide("release_payment", Actor("human", " ", "supervisor"))[0]                      # a person with no name is not a person


def test_the_other_modes_limit_what_an_agent_may_do_with_them():
    g = guard()
    assert g.decide("draft_information_request", AGENT, mode="draft")[0] and not g.decide("draft_information_request", AGENT, mode="controlled_api")[0]
    assert g.decide("create_investigation_task", AGENT, mode="controlled_api")[0] and not g.decide("create_investigation_task", AGENT, mode="execute")[0]
    assert g.decide("extract_document_info", AGENT, mode="execute")[0]                                  # autonomous
    assert g.decide("recommend_settlement", AGENT, mode="draft")[0] and not g.decide("recommend_settlement", AGENT, mode="execute")[0]     # a person decides
    assert g.decide("create_investigation_task", HUMAN("adjuster"), mode="execute")[0]


def test_anything_not_in_the_matrix_is_denied():
    ok, why = guard().decide("wire_money_abroad", HUMAN("supervisor"))
    assert not ok and "not in the autonomy matrix" in why


def test_a_denial_is_recorded_and_sent_to_the_sink():
    seen = []
    g = guard(seen.append)
    with pytest.raises(NotAuthorised) as e:
        g.require("deny_claim", AGENT, claim_id="CLM-G")
    assert seen == g.denials and seen[0].operation == "deny_claim" and seen[0].actor == AGENT and seen[0].claim_id == "CLM-G" and "authorised person only" in str(e.value)
    g.require("deny_claim", HUMAN("senior_adjuster"))
    assert len(g.denials) == 1                                                                          # an allowed call leaves no denial


def test_the_decorator_guards_a_tool_call_and_treats_a_missing_actor_as_an_unidentified_agent():
    g, calls = guard(), []

    @g.guarded("release_payment")
    def pay(amount):
        calls.append(amount)
        return "paid"

    assert pay(5, actor=HUMAN("supervisor")) == "paid" and calls == [5]
    for kw in ({}, {"actor": AGENT}, {"actor": HUMAN("adjuster")}):
        with pytest.raises(NotAuthorised):
            pay(6, **kw)
    assert calls == [5] and g.denials[0].actor.id == "unidentified"


# ---------------------------------------------------------------------------------------------------------------- agent attempts: denial and payment release are blocked
def test_an_agent_cannot_reach_the_payment_system_only_an_authorised_person_can():
    recon = MockReconAdapter()
    g = guard()
    wrapped = GuardedRecon(recon, g)
    for actor in (None, AGENT, HUMAN("adjuster"), HUMAN("senior_adjuster")):
        with pytest.raises(NotAuthorised):
            wrapped.submit_payment_instruction(instruction(), "CLM-G:pay:1", actor=actor)
    assert recon.instruction_count == 0 and recon.calls == [] and len(g.denials) == 4
    assert wrapped.submit_payment_instruction(instruction(), "CLM-G:pay:1", actor=HUMAN("supervisor")).status == "accepted" and recon.instruction_count == 1
    assert wrapped.list_breaks() == []                                                                  # everything else passes straight through


def test_a_blocked_attempt_leaves_a_trace_in_the_run_log():
    claims, recon = MockClaimsAdapter(), MockReconAdapter()
    adapters = Adapters(MockPasAdapter(), claims, recon)
    svc = build_commit_service(SETTINGS, adapters, Dispatcher())
    reg = build_registry(SETTINGS, force_mock=("intake", "gap", "coverage", "action", "audit", "routing"))
    r = ClaimRunner(reg, commit_service=svc, settings=SETTINGS)
    r.start(ClaimState(claim_id="CLM-0002"))
    with pytest.raises(NotAuthorised):
        svc.adapters.recon.submit_payment_instruction(instruction("CLM-0002"), "CLM-0002:pay:1", actor=AGENT)       # an agent tries to release a payment
    ev = r.events.of_kind("CLM-0002", "action_denied")
    assert len(ev) == 1 and ev[0].data["operation"] == "release_payment" and ev[0].data["actor_type"] == "agent" and recon.instruction_count == 0
    assert "blocked: agent 'action' tried 'release_payment'" in ev[0].message


def test_rejecting_a_claim_without_the_role_is_blocked_and_logged_once_however_often_the_gate_is_replayed():
    reg = build_registry(SETTINGS, force_mock=("intake", "gap", "coverage", "settlement", "action", "audit", "routing"))
    r = ClaimRunner(reg, settings=SETTINGS)
    r.start(ClaimState(claim_id="CLM-0002"))
    for _ in range(2):
        st = r.decide("CLM-0002", {"decision": "reject", "reviewer_id": "rev-3", "reviewer_role": "adjuster", "reason": "no"})
        assert st.phase == "awaiting_human_decision" and "deny_claim" in st.gate["error"]
    assert len(r.events.of_kind("CLM-0002", "action_denied")) == 1 and r.events.of_kind("CLM-0002", "action_denied")[0].data["actor_role"] == "adjuster"


# ---------------------------------------------------------------------------------------------------------------- governance group: action without authority
def draft_only_matrix() -> AutonomyMatrix:
    raw = copy.deepcopy(SETTINGS.autonomy.model_dump())
    raw["operations"]["create_investigation_task"] = {"mode": "draft_only", "roles": []}
    return AutonomyMatrix.model_validate(raw)


def test_an_action_the_matrix_does_not_allow_to_be_carried_out_is_blocked_failed_and_not_sent():
    claims = MockClaimsAdapter()
    adapters = Adapters(MockPasAdapter(), claims, MockReconAdapter())
    svc = build_commit_service(SETTINGS, adapters, Dispatcher(policy=RetryPolicy(max_attempts=1), sleeper=lambda s: None))
    svc.guard = AutonomyGuard(draft_only_matrix(), ORDER)                                                # the matrix says a task may only be DRAFTED
    s = ClaimState(claim_id="CLM-0005", proposed_actions=[ProposedAction(action_id="ACT-1", action_type="create_task", responsible_party="request_endorsement_END-01_from_policy_administration",
                                                                         execution_mode="controlled_api", idempotency_key="CLM-0005:create_task:x")])
    res = svc.commit(s, reviewer_id="rev-9", reviewer_role="senior_adjuster", approval_ref="EVT-1")
    assert [a.status for a in res.actions] == ["failed"] and claims.task_count == 0 and len(svc.guard.denials) == 1
    assert "draft_only" in svc.guard.denials[0].reason


# ---------------------------------------------------------------------------------------------------------------- governance group: missing citation
class StubCoverage:
    agent_id = "coverage"
    owns = OWNERSHIP["coverage"]

    def run(self, state):
        from claimtower.core.state import StatePatch
        return StatePatch(agent_id="coverage", updates={"coverage_issues": [CoverageIssue(issue_id="ISS-COVER-WATER", issue="x", status="established", clause_citations=[])]}, rationale="stub")


def test_a_missing_citation_blocks_the_release_to_the_gate():
    reg = build_registry(SETTINGS, force_mock=("intake", "gap", "action", "routing"))
    slots = {s.node: (s.agent, s.status) for s in reg}
    slots["coverage"] = (StubCoverage(), "implemented")
    st = ClaimRunner(AgentRegistry(slots)).start(ClaimState(claim_id="CLM-0002"))
    assert st.phase == "halted" and st.paused_at == "audit" and "citation_missing" in st.halted_reason and st.gate is None


# ---------------------------------------------------------------------------------------------------------------- governance group: human override
def test_a_human_override_needs_a_reason_names_the_reviewer_and_is_recorded():
    reg = build_registry(SETTINGS, force_mock=("intake", "gap", "coverage", "action", "audit", "routing"), overrides=(ov := OverrideStore()))
    r = ClaimRunner(reg, overrides=ov, settings=SETTINGS)
    r.start(ClaimState(claim_id="CLM-0002"))
    st = r.decide("CLM-0002", {"decision": "edit", "reviewer_id": "rev-1", "reviewer_role": "adjuster", "reason": " ", "edits": {"applicable_deductible": "20000"}})
    assert "needs a reason" in st.gate["error"] and ov.get("CLM-0002") == {}
    r.decide("CLM-0002", {"decision": "edit", "reviewer_id": "rev-1", "reviewer_role": "adjuster", "reason": "policy schedule was updated", "edits": {"applicable_deductible": "20000"}})
    rec = ov.last("CLM-0002")
    assert (rec.by, rec.reason) == ("rev-1", "policy schedule was updated") and r.events.of_kind("CLM-0002", "human_decision")[0].data["reason"] == "policy schedule was updated"


# ---------------------------------------------------------------------------------------------------------------- governance group: low-confidence extraction
def test_a_low_confidence_fact_is_flagged_for_a_person_never_hidden_or_promoted():
    floor = SETTINGS.pack.document_types.extractor.low_confidence
    fact = lambda fid, c: Fact(fact_id=fid, claim_id="CLM-0002", proposition="p", source_document_id="D", page=1, evidence_text="p", evidence_type="neutral",   # noqa: E731
                               extraction_confidence=c)
    reg = build_registry(SETTINGS, force_mock=("intake", "gap", "coverage", "settlement", "action", "audit", "routing"))
    r = ClaimRunner(reg, settings=SETTINGS)
    state = ClaimState(claim_id="CLM-0002")
    r.start(state)
    gate = r.status("CLM-0002").gate
    assert gate["low_confidence_facts"] == sorted(f.fact_id for f in r.state("CLM-0002").facts if f.extraction_confidence < floor)
    low = r.supervisor.gate_payload(r.state("CLM-0002").model_copy(update={"facts": [fact("F-LOW", floor - 0.2), fact("F-OK", floor + 0.2)]}))
    assert low["low_confidence_facts"] == ["F-LOW"]


def test_the_evidence_agent_says_so_when_it_read_a_value_with_low_confidence(tmp_path):
    from claimtower.agents.evidence import EvidenceAgent
    from claimtower.ingestion.classifier import HybridClassifier, RuleClassifier
    from claimtower.ingestion.extractors import HybridExtractor, RuleExtractor
    pack = SETTINGS.pack
    store = DocumentStore(tmp_path / "s")
    ingest_claim_folder(store, SYNTHETIC / "claims" / "CLM-0002")
    a = EvidenceAgent(store, pack, HybridClassifier(RuleClassifier(pack.document_types)), HybridExtractor(RuleExtractor(pack.document_types)))
    p = a.run(ClaimState(claim_id="CLM-0002"))
    low = [f.fact_id for f in p.updates["facts"] if f.extraction_confidence < pack.document_types.extractor.low_confidence]
    assert ("REVIEW" in p.rationale and "low extraction confidence" in p.rationale) == bool(low) and all(f.verification_status == "unverified" for f in p.updates["facts"])


# ---------------------------------------------------------------------------------------------------------------- governance group: model unavailable
class DownLLM:
    """A model that is unreachable: every call fails."""

    text_model = vision_model = "down"
    calls: list = []
    mode = "auto"

    def derive(self, **kw):
        return self

    def __getattr__(self, name):
        def fail(*a, **k):
            raise ConnectionError("the model is not reachable")
        return fail


def test_with_the_model_unreachable_the_claim_still_reaches_the_gate_on_rules_and_nothing_is_invented(tmp_path):
    store = DocumentStore(tmp_path / "s")
    ingest_claim_folder(store, SYNTHETIC / "claims" / "CLM-0002")
    reg = build_registry(SETTINGS, document_store=store, policy_library=policy_library_from_pdfs(SETTINGS), llm=DownLLM())
    r = ClaimRunner(reg, settings=SETTINGS)
    st = r.start(ClaimState(claim_id="CLM-0002"))
    state = r.state("CLM-0002")
    assert st.phase == "awaiting_human_decision" and not r.events.of_kind("CLM-0002", "agent_failed")
    assert state.settlement.status == "calculated" and state.settlement.payable == Decimal(80000)                  # from rules: the same number as without a model
    assert state.qa_report.passed
    assert {i.issue_id: i.status for i in state.coverage_issues}["ISS-COVER-WATER"] != "established"                  # no reading: nothing is concluded
    assert state.routing.tier == "complex"                                                                         # and a person looks at it
